"""
extraction.py — Model loading, forward-hook activation capture, and .npy caching.

Captures per-sample mean-pooled activations for:
  - FFN gate (post-SwiGLU): [8960] per layer
  - Residual stream (block output): [1536] per layer

Saves per-layer .npy files (float16) to the cache directory so downstream
analyses never need to re-run inference.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _atomic_np_save(path: Path, arr: np.ndarray, allow_pickle: bool = False) -> None:
    """Write a .npy file atomically: write to .tmp, fsync, then os.replace.

    Prevents zero-filled / header-less files when the process is killed
    mid-save (Windows NTFS doesn't always flush write buffers on hard kill).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=allow_pickle)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODEL_NAME, TARGET_LAYERS, FFN_DIM, HIDDEN_DIM,
    MAX_SEQ_LEN, BATCH_SIZE, CHECKPOINT_FNAME,
)
from utils import strip_brackets


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    device: torch.device, logger: logging.Logger
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen2.5-1.5B-Instruct in bfloat16."""
    logger.info(f"Loading tokeniser: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {MODEL_NAME} (bfloat16)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model ready — {n_params:.0f}M parameters on {device}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION STORE + HOOKS
# ─────────────────────────────────────────────────────────────────────────────

class ActivationStore:
    """Collects mean-pooled activations from forward hooks (batch-aware)."""

    def __init__(self):
        self.ffn_gates: Dict[int, torch.Tensor] = {}     # layer → [batch, 8960]
        self.residuals: Dict[int, torch.Tensor] = {}     # layer → [batch, 1536]
        self.attention_mask: Optional[torch.Tensor] = None  # [batch, seq]

    def clear(self):
        self.ffn_gates.clear()
        self.residuals.clear()
        self.attention_mask = None


def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool across the sequence dimension using the attention mask.
    tensor: [batch, seq, dim]
    mask  : [batch, seq]
    Returns: [batch, dim]
    """
    mask_f = mask.unsqueeze(-1).float()                  # [batch, seq, 1]
    summed = (tensor.float() * mask_f).sum(dim=1)        # [batch, dim]
    lengths = mask_f.sum(dim=1).clamp(min=1)             # [batch, 1]
    return summed / lengths


def register_hooks(
    model: AutoModelForCausalLM,
    store: ActivationStore,
    target_layers: List[int],
) -> List[torch.utils.hooks.RemovableHook]:
    """Register FFN gate + residual hooks on each target layer."""
    hooks = []

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]

        # ── FFN gate (post-SwiGLU) ───────────────────────────────────────
        def make_ffn_hook(idx):
            def hook_fn(module, inputs, output):
                x = inputs[0]                                # [batch, seq, 1536]
                gate = module.act_fn(module.gate_proj(x))    # [batch, seq, 8960]
                up   = module.up_proj(x)                     # [batch, seq, 8960]
                intermediate = gate * up                     # [batch, seq, 8960]
                pooled = _masked_mean(intermediate, store.attention_mask)
                store.ffn_gates[idx] = pooled.detach().cpu()
            return hook_fn

        # ── Residual stream ──────────────────────────────────────────────
        def make_res_hook(idx):
            def hook_fn(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                pooled = _masked_mean(hidden, store.attention_mask)
                store.residuals[idx] = pooled.detach().cpu()
            return hook_fn

        hooks.append(layer.mlp.register_forward_hook(make_ffn_hook(layer_idx)))
        hooks.append(layer.register_forward_hook(make_res_hook(layer_idx)))

    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> int:
    if path.exists():
        with open(path) as f:
            return json.load(f).get("next_batch", 0)
    return 0


def save_checkpoint(path: Path, next_batch: int):
    with open(path, "w") as f:
        json.dump({"next_batch": next_batch}, f)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_cache(
    df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    cache_dir: Path,
    logger: logging.Logger,
    batch_size: int = BATCH_SIZE,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """
    Run batched inference over the entire dataset and cache activations.

    Saves per-layer files:
      cache/ffn_layer_{L}.npy      — float16, shape [N, FFN_DIM]
      cache/residual_layer_{L}.npy — float16, shape [N, HIDDEN_DIM]
      cache/emotions.npy           — object array of emotion strings
      cache/metadata.json          — bookkeeping

    Returns:
      ffn_acts     : {layer_idx: np.ndarray [N, FFN_DIM]}
      res_acts     : {layer_idx: np.ndarray [N, HIDDEN_DIM]}
      emotions_arr : np.ndarray of emotion strings [N]
    """
    meta_path = cache_dir / "metadata.json"

    # ── Check for existing cache ──────────────────────────────────────────
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("n_samples") == len(df) and meta.get("complete", False):
            logger.info("Cache found and complete — loading from disk.")
            return _load_cache(cache_dir, logger)
        else:
            logger.info("Incomplete cache found — resuming extraction.")

    n_samples = len(df)
    logger.info(f"Starting activation extraction: {n_samples} samples, "
                f"batch_size={batch_size}")

    # ── Pre-allocate numpy arrays ─────────────────────────────────────────
    ffn_arrays  = {l: np.zeros((n_samples, FFN_DIM),    dtype=np.float16)
                   for l in TARGET_LAYERS}
    res_arrays  = {l: np.zeros((n_samples, HIDDEN_DIM), dtype=np.float16)
                   for l in TARGET_LAYERS}
    emotions_arr = df["primary_emotion"].values.copy()

    # ── Hook setup ────────────────────────────────────────────────────────
    store = ActivationStore()
    hooks = register_hooks(model, store, TARGET_LAYERS)

    # ── Checkpoint ────────────────────────────────────────────────────────
    ckpt_path = cache_dir / CHECKPOINT_FNAME
    start_batch = load_checkpoint(ckpt_path)
    start_idx   = start_batch * batch_size

    if start_idx > 0:
        # Reload partial results already saved
        for l in TARGET_LAYERS:
            ffn_path = cache_dir / f"ffn_layer_{l}.npy"
            res_path = cache_dir / f"residual_layer_{l}.npy"
            if ffn_path.exists():
                ffn_arrays[l] = np.load(ffn_path)
            if res_path.exists():
                res_arrays[l] = np.load(res_path)
        logger.info(f"Resumed from batch {start_batch} (row {start_idx})")

    texts = [strip_brackets(str(t)) for t in df["chinese_text"].values]

    n_batches = (n_samples + batch_size - 1) // batch_size

    try:
        for batch_idx in tqdm(range(start_batch, n_batches),
                              initial=start_batch, total=n_batches,
                              desc="Extracting activations"):
            lo = batch_idx * batch_size
            hi = min(lo + batch_size, n_samples)
            batch_texts = texts[lo:hi]

            store.clear()

            inputs = tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=MAX_SEQ_LEN, padding=True,
            )
            store.attention_mask = inputs["attention_mask"].to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs)

            # ── Collect activations ───────────────────────────────────────
            for l in TARGET_LAYERS:
                if l in store.ffn_gates:
                    ffn_arrays[l][lo:hi] = store.ffn_gates[l].half().numpy()
                if l in store.residuals:
                    res_arrays[l][lo:hi] = store.residuals[l].half().numpy()

            # ── Periodic save ─────────────────────────────────────────────
            if (batch_idx + 1) % 100 == 0 or hi == n_samples:
                for l in TARGET_LAYERS:
                    _atomic_np_save(cache_dir / f"ffn_layer_{l}.npy", ffn_arrays[l])
                    _atomic_np_save(cache_dir / f"residual_layer_{l}.npy", res_arrays[l])
                save_checkpoint(ckpt_path, batch_idx + 1)

    finally:
        remove_hooks(hooks)

    # ── Final save ────────────────────────────────────────────────────────
    _atomic_np_save(cache_dir / "emotions.npy", emotions_arr, allow_pickle=True)
    for l in TARGET_LAYERS:
        _atomic_np_save(cache_dir / f"ffn_layer_{l}.npy", ffn_arrays[l])
        _atomic_np_save(cache_dir / f"residual_layer_{l}.npy", res_arrays[l])

    meta = {
        "n_samples": n_samples,
        "model": MODEL_NAME,
        "target_layers": TARGET_LAYERS,
        "ffn_dim": FFN_DIM,
        "hidden_dim": HIDDEN_DIM,
        "batch_size": batch_size,
        "complete": True,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Remove checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()

    logger.info("Extraction complete. Cache saved.")
    return ffn_arrays, res_arrays, emotions_arr


def _load_cache(
    cache_dir: Path, logger: logging.Logger
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """Load cached activation arrays from disk."""
    ffn_acts = {}
    res_acts = {}
    for l in TARGET_LAYERS:
        ffn_acts[l] = np.load(cache_dir / f"ffn_layer_{l}.npy")
        res_acts[l] = np.load(cache_dir / f"residual_layer_{l}.npy")
        logger.info(f"  Loaded layer {l}: FFN {ffn_acts[l].shape}, "
                     f"Residual {res_acts[l].shape}")
    emotions_arr = np.load(cache_dir / "emotions.npy", allow_pickle=True)
    logger.info(f"  Loaded {len(emotions_arr)} emotion labels")
    return ffn_acts, res_acts, emotions_arr
