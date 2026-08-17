"""
Emotion Neuron Activation Analysis
====================================
Reads a CSV of Chinese text + primary emotion pairs, runs each row through
Qwen/Qwen2.5-1.5B-Instruct, captures FFN gate activations AND residual stream
hidden states from layers 14-22, stores activated neuron IDs in the CSV, then
generates UMAP / t-SNE / PCA cluster plots coloured by the 6 primary emotions.

Hardware target : Apple M3 MacBook Air (MPS backend)
Activation capture : Layers 14-22
                     - FFN gate (post-SwiGLU): shape [seq_len, 8960]
                     - Residual stream (post-block): shape [seq_len, 1536]
Token pooling    : Mean-pool across all token positions
Neuron threshold : Top-5 % of neurons by absolute activation value

CSV output columns added
  ffn_neuron_ids      → "[layer14:[n1,n2,...]; layer15:[...]; ...]"
  ffn_activated_layers→ "[14,15,...]"
  residual_neuron_ids → "[layer14:[n1,n2,...]; ...]"
  residual_activated_layers → "[14,15,...]"

Checkpoint file : emotion_neuron_checkpoint.json  (same folder as CSV)

Plots saved (high-DPI PNG, same folder as CSV):
  cluster_umap.png
  cluster_tsne.png
  cluster_pca.png
  heatmap_emotion_layer.png
  violin_ffn_sparsity.png

Usage
-----
  pip install transformers torch accelerate umap-learn scikit-learn
              matplotlib seaborn pandas tqdm

  python emotion_neuron_analysis.py --csv /path/to/emotion_dataset_translated.csv
"""

# ─── standard library ────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ─── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── plotting (imported lazily where needed so the script still loads if
#     umap-learn is missing and user just wants to run extraction) ───────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all systems
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← adjust if needed
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LAYERS    = list(range(14, 23))       # layers 14-22 inclusive (0-indexed)
FFN_DIM          = 8960                      # SwiGLU intermediate size
HIDDEN_DIM       = 1536                      # residual stream size
TOP_K_PERCENT    = 5.0                       # top-5 % threshold for "activated"
BATCH_SIZE       = 1                         # one row at a time (safe on 8 GB)
CHECKPOINT_FNAME = "emotion_neuron_checkpoint.json"

# Emotion → colour + marker for plots
EMOTION_STYLE: Dict[str, Dict] = {
    "ANGER":     {"color": "#E24B4A", "marker": "o", "label": "Anger"},
    "DISGUST":   {"color": "#854F0B", "marker": "s", "label": "Disgust"},
    "FEAR":      {"color": "#534AB7", "marker": "^", "label": "Fear"},
    "HAPPINESS": {"color": "#3B6D11", "marker": "D", "label": "Happiness"},
    "SADNESS":   {"color": "#185FA5", "marker": "v", "label": "Sadness"},
    "SURPRISE":  {"color": "#D4537E", "marker": "P", "label": "Surprise"},
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("emotion_neuron")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SELECTION  (MPS → CUDA → CPU)
# ─────────────────────────────────────────────────────────────────────────────

def select_device(logger: logging.Logger) -> torch.device:
    if torch.backends.mps.is_available():
        logger.info("Hardware: Apple MPS (M-series GPU) selected")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.info("Hardware: CUDA GPU selected")
        return torch.device("cuda")
    logger.info("Hardware: CPU selected (will be slow)")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    device: torch.device,
    logger: logging.Logger
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen2.5-1.5B-Instruct in bfloat16 onto the target device."""
    logger.info(f"Loading tokeniser from {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    logger.info(f"Loading model {MODEL_NAME} in bfloat16 …")
    # MPS does not support device_map='auto' — load to CPU first then move
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    logger.info("Model ready.")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# HOOK MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class ActivationStore:
    """Thread-safe store for forward-hook captures."""

    def __init__(self):
        self.ffn_gates: Dict[int, torch.Tensor] = {}       # layer → [seq, 8960]
        self.residuals: Dict[int, torch.Tensor] = {}       # layer → [seq, 1536]

    def clear(self):
        self.ffn_gates.clear()
        self.residuals.clear()


def register_hooks(
    model: AutoModelForCausalLM,
    store: ActivationStore,
    target_layers: List[int]
) -> List[torch.utils.hooks.RemovableHook]:
    """
    Register two forward hooks per target layer:
      1. FFN gate hook  — captures post-SwiGLU intermediate activations
      2. Residual hook  — captures the full decoder-block output

    Qwen2 / Qwen2.5 layer layout (each element of model.model.layers):
        self_attn  → grouped-query attention
        mlp.gate_proj / mlp.up_proj / mlp.down_proj / mlp.act_fn
        input_layernorm / post_attention_layernorm
    The SwiGLU intermediate = act_fn(gate_proj(x)) * up_proj(x)
    We capture act_fn output * up_proj output via a hook on mlp itself.
    """
    hooks: List[torch.utils.hooks.RemovableHook] = []

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]

        # ── Hook 1: FFN gate activation (post-SwiGLU) ────────────────────
        # Qwen2 MLP: output = down_proj( act_fn(gate_proj(x)) * up_proj(x) )
        # We hook the MLP module's forward and capture the intermediate product
        def make_ffn_hook(idx: int):
            def ffn_hook(module, inputs, output):
                # inputs[0] is the hidden state going into MLP
                x = inputs[0]                               # [batch, seq, 1536]
                gate = module.act_fn(module.gate_proj(x))  # [batch, seq, 8960]
                up   = module.up_proj(x)                   # [batch, seq, 8960]
                intermediate = gate * up                   # [batch, seq, 8960]
                # mean-pool across tokens, detach to CPU float32
                store.ffn_gates[idx] = intermediate[0].float().mean(dim=0).detach().cpu()
            return ffn_hook

        # ── Hook 2: Residual stream (full block output) ───────────────────
        def make_residual_hook(idx: int):
            def residual_hook(module, inputs, output):
                # output is a tuple; first element is the hidden state
                hidden = output[0] if isinstance(output, tuple) else output
                # [batch, seq, 1536] → mean-pool → [1536]
                store.residuals[idx] = hidden[0].float().mean(dim=0).detach().cpu()
            return residual_hook

        hooks.append(layer.mlp.register_forward_hook(make_ffn_hook(layer_idx)))
        hooks.append(layer.register_forward_hook(make_residual_hook(layer_idx)))

    return hooks


def remove_hooks(hooks: List[torch.utils.hooks.RemovableHook]):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# NEURON SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def top_k_percent_indices(vector: torch.Tensor, percent: float) -> List[int]:
    """
    Return indices of the top `percent` % neurons by absolute activation value.
    E.g. percent=5.0 on an 8960-dim vector → top 448 neuron indices.
    """
    k = max(1, int(len(vector) * percent / 100.0))
    _, indices = torch.topk(vector.abs(), k)
    return sorted(indices.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-ROW INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def strip_brackets(text: str) -> str:
    """Remove leading/trailing [] wrappers added by the earlier pipeline."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text


def run_inference(
    chinese_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    store: ActivationStore,
    hooks: List[torch.utils.hooks.RemovableHook],
    target_layers: List[int],
    top_k_pct: float,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Tokenise `chinese_text`, run a forward pass, and return four dicts:
      ffn_ids   : {layer_idx: [neuron_id, ...]}
      ffn_layers: list of layer indices that had any activation
      res_ids   : {layer_idx: [neuron_id, ...]}
      res_layers: list of layer indices

    Also returns the raw mean-pooled vectors for downstream analysis.
    """
    store.clear()
    text = strip_brackets(chinese_text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        model(**inputs)

    # ── FFN gate neurons ──────────────────────────────────────────────────
    ffn_ids: Dict[int, List[int]] = {}
    for layer_idx in target_layers:
        vec = store.ffn_gates.get(layer_idx)
        if vec is not None:
            ffn_ids[layer_idx] = top_k_percent_indices(vec, top_k_pct)

    # ── Residual stream neurons ───────────────────────────────────────────
    res_ids: Dict[int, List[int]] = {}
    for layer_idx in target_layers:
        vec = store.residuals.get(layer_idx)
        if vec is not None:
            res_ids[layer_idx] = top_k_percent_indices(vec, top_k_pct)

    ffn_layers = sorted(ffn_ids.keys())
    res_layers = sorted(res_ids.keys())

    return ffn_ids, ffn_layers, res_ids, res_layers


# ─────────────────────────────────────────────────────────────────────────────
# SERIALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def format_neuron_ids(ids_by_layer: Dict[int, List[int]]) -> str:
    """
    Produce a compact string stored in the CSV cell, e.g.:
    [layer14:[0,5,99,...]; layer15:[1,2,...]]
    """
    parts = [f"layer{l}:{ids_by_layer[l]}" for l in sorted(ids_by_layer)]
    return "[" + "; ".join(parts) + "]"


def format_layers(layers: List[int]) -> str:
    return "[" + ",".join(str(l) for l in layers) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: Path) -> int:
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            data = json.load(f)
        return data.get("next_row", 0)
    return 0


def save_checkpoint(ckpt_path: Path, next_row: int):
    with open(ckpt_path, "w") as f:
        json.dump({"next_row": next_row}, f)


# ─────────────────────────────────────────────────────────────────────────────
# CSV PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

NEW_COLS = [
    "ffn_neuron_ids",
    "ffn_activated_layers",
    "residual_neuron_ids",
    "residual_activated_layers",
]


def process_csv(
    csv_path: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    logger: logging.Logger,
):
    """
    Main loop: read CSV, run inference per row, write activations back,
    checkpoint after every row.
    """
    logger.info(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")

    # Ensure new columns exist
    for col in NEW_COLS:
        if col not in df.columns:
            df[col] = ""

    ckpt_path = csv_path.parent / CHECKPOINT_FNAME
    start_row = load_checkpoint(ckpt_path)
    if start_row > 0:
        logger.info(f"Resuming from row {start_row} / {len(df)}")
    else:
        logger.info(f"Starting fresh — {len(df)} rows to process")

    # Set up activation store + hooks (persistent across rows)
    store = ActivationStore()
    hooks = register_hooks(model, store, TARGET_LAYERS)

    try:
        for idx in tqdm(range(start_row, len(df)), initial=start_row,
                        total=len(df), desc="Extracting activations"):
            row = df.iloc[idx]
            chinese_text   = str(row.get("chinese_text", "")).strip()
            primary_emotion = str(row.get("primary_emotion", "")).strip().upper()

            if not chinese_text or primary_emotion not in EMOTION_STYLE:
                # Skip rows with missing text or unknown emotion
                save_checkpoint(ckpt_path, idx + 1)
                continue

            try:
                ffn_ids, ffn_layers, res_ids, res_layers = run_inference(
                    chinese_text, model, tokenizer, device,
                    store, hooks, TARGET_LAYERS, TOP_K_PERCENT,
                )

                df.at[idx, "ffn_neuron_ids"]          = format_neuron_ids(ffn_ids)
                df.at[idx, "ffn_activated_layers"]     = format_layers(ffn_layers)
                df.at[idx, "residual_neuron_ids"]      = format_neuron_ids(res_ids)
                df.at[idx, "residual_activated_layers"] = format_layers(res_layers)

            except Exception as e:
                logger.warning(f"Row {idx} failed ({e}) — leaving blank.")

            # Save checkpoint + flush CSV after every row
            save_checkpoint(ckpt_path, idx + 1)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    finally:
        remove_hooks(hooks)

    logger.info("Extraction complete. Removing checkpoint.")
    if ckpt_path.exists():
        ckpt_path.unlink()

    logger.info(f"CSV saved: {csv_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    csv_path: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[np.ndarray, List[str]]:
    """
    Re-run inference (or reload stored activations) to build a 2D numpy array
    of shape [N_rows, num_layers * hidden_dim] for dimensionality reduction.

    We use the residual stream vectors (1536 per layer × 9 layers = 13,824 dims)
    as they are dense and well-suited for UMAP / t-SNE / PCA.
    Rows with unknown/missing emotion are skipped.
    """
    logger.info("Building feature matrix for clustering …")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")

    store = ActivationStore()
    hooks = register_hooks(model, store, TARGET_LAYERS)

    features: List[np.ndarray] = []
    emotions: List[str]        = []

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building features"):
            chinese_text    = str(row.get("chinese_text", "")).strip()
            primary_emotion = str(row.get("primary_emotion", "")).strip().upper()

            if not chinese_text or primary_emotion not in EMOTION_STYLE:
                continue

            store.clear()
            text   = strip_brackets(chinese_text)
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs)

            # Concatenate residual vectors across all target layers → 1 feature row
            vecs = []
            for layer_idx in TARGET_LAYERS:
                vec = store.residuals.get(layer_idx)
                if vec is not None:
                    vecs.append(vec.numpy())
                else:
                    vecs.append(np.zeros(HIDDEN_DIM, dtype=np.float32))

            features.append(np.concatenate(vecs))   # shape: [9*1536] = [13824]
            emotions.append(primary_emotion)

    finally:
        remove_hooks(hooks)

    X = np.array(features, dtype=np.float32)
    logger.info(f"Feature matrix shape: {X.shape}  |  {len(emotions)} samples")
    return X, emotions


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def scatter_plot(
    coords: np.ndarray,
    emotions: List[str],
    title: str,
    save_path: Path,
    logger: logging.Logger,
):
    """Generic 2-D scatter coloured by emotion."""
    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    for emotion, style in EMOTION_STYLE.items():
        mask = [e == emotion for e in emotions]
        if not any(mask):
            continue
        pts = coords[np.array(mask)]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=style["color"],
            marker=style["marker"],
            label=style["label"],
            alpha=0.65,
            s=22,
            linewidths=0.3,
            edgecolors="white",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Dimension 1", fontsize=11)
    ax.set_ylabel("Dimension 2", fontsize=11)
    ax.legend(
        title="Primary emotion",
        title_fontsize=10,
        fontsize=9,
        framealpha=0.9,
        loc="best",
    )
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_umap(X: np.ndarray, emotions: List[str], out_dir: Path,
              logger: logging.Logger):
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed — skipping UMAP plot. "
                       "Install with: pip install umap-learn")
        return

    logger.info("Running UMAP …")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="cosine", random_state=42, verbose=False)
    coords = reducer.fit_transform(X)
    scatter_plot(
        coords, emotions,
        "UMAP — Qwen2.5-1.5B residual stream activations (layers 14–22)\nColoured by primary emotion",
        out_dir / "cluster_umap.png",
        logger,
    )


def plot_tsne(X: np.ndarray, emotions: List[str], out_dir: Path,
              logger: logging.Logger):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA as skPCA

    logger.info("Running t-SNE (PCA pre-reduction to 50 dims) …")
    # Pre-reduce with PCA for speed (standard practice for t-SNE on high-dim data)
    n_components_pre = min(50, X.shape[1], X.shape[0] - 1)
    Xr = skPCA(n_components=n_components_pre, random_state=42).fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
                init="pca", random_state=42, n_iter=1000, verbose=0)
    coords = tsne.fit_transform(Xr)
    scatter_plot(
        coords, emotions,
        "t-SNE — Qwen2.5-1.5B residual stream activations (layers 14–22)\nColoured by primary emotion",
        out_dir / "cluster_tsne.png",
        logger,
    )


def plot_pca(X: np.ndarray, emotions: List[str], out_dir: Path,
             logger: logging.Logger):
    from sklearn.decomposition import PCA as skPCA

    logger.info("Running PCA …")
    pca   = skPCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    for emotion, style in EMOTION_STYLE.items():
        mask = [e == emotion for e in emotions]
        if not any(mask):
            continue
        pts = coords[np.array(mask)]
        ax.scatter(pts[:, 0], pts[:, 1], c=style["color"],
                   marker=style["marker"], label=style["label"],
                   alpha=0.65, s=22, linewidths=0.3, edgecolors="white")

    ax.set_title(
        "PCA — Qwen2.5-1.5B residual stream activations (layers 14–22)\n"
        "Coloured by primary emotion",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% variance)", fontsize=11)
    ax.legend(title="Primary emotion", title_fontsize=10, fontsize=9,
              framealpha=0.9, loc="best")
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    save_path = out_dir / "cluster_pca.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_heatmap_emotion_layer(
    csv_path: Path, out_dir: Path, logger: logging.Logger
):
    """
    Heatmap: rows = 6 emotions, cols = layers 14-22.
    Cell value = mean number of activated FFN neurons for that (emotion, layer).
    This reveals which layers light up most per emotion.
    """
    logger.info("Building emotion × layer activation heatmap …")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")

    emotions_list = list(EMOTION_STYLE.keys())
    # Matrix: [emotion, layer]
    counts = {e: {l: [] for l in TARGET_LAYERS} for e in emotions_list}

    for _, row in df.iterrows():
        emotion = str(row.get("primary_emotion", "")).strip().upper()
        ffn_str = str(row.get("ffn_neuron_ids", ""))
        if emotion not in emotions_list or not ffn_str:
            continue

        # Parse "[layer14:[1,2,3]; layer15:[...]]"
        for match in re.finditer(r"layer(\d+):\[([^\]]*)\]", ffn_str):
            layer_idx = int(match.group(1))
            neurons   = [n for n in match.group(2).split(",") if n.strip()]
            if layer_idx in TARGET_LAYERS:
                counts[emotion][layer_idx].append(len(neurons))

    # Build mean matrix
    matrix = np.zeros((len(emotions_list), len(TARGET_LAYERS)), dtype=np.float32)
    for i, emotion in enumerate(emotions_list):
        for j, layer in enumerate(TARGET_LAYERS):
            vals = counts[emotion][layer]
            matrix[i, j] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.heatmap(
        matrix,
        xticklabels=[f"L{l}" for l in TARGET_LAYERS],
        yticklabels=emotions_list,
        annot=True, fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean activated FFN neurons"},
    )
    ax.set_title(
        "Mean FFN neuron activations per emotion × layer\n"
        "(Qwen2.5-1.5B, top-5% threshold)",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Primary emotion", fontsize=11)
    plt.tight_layout()
    save_path = out_dir / "heatmap_emotion_layer.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_violin_ffn_sparsity(
    csv_path: Path, out_dir: Path, logger: logging.Logger
):
    """
    Violin plot: distribution of total activated FFN neurons per sample,
    split by emotion. Shows whether some emotions activate more neurons overall.
    """
    logger.info("Building FFN sparsity violin plot …")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")

    records = []
    for _, row in df.iterrows():
        emotion = str(row.get("primary_emotion", "")).strip().upper()
        ffn_str = str(row.get("ffn_neuron_ids", ""))
        if emotion not in EMOTION_STYLE or not ffn_str:
            continue

        total_neurons = sum(
            len(m.group(2).split(","))
            for m in re.finditer(r"layer\d+:\[([^\]]*)\]", ffn_str)
            if m.group(2).strip()
        )
        records.append({"emotion": emotion, "total_ffn_neurons": total_neurons})

    if not records:
        logger.warning("No data for violin plot — skipping.")
        return

    plot_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    palette = {e: s["color"] for e, s in EMOTION_STYLE.items()}

    emotion_order = [e for e in EMOTION_STYLE if e in plot_df["emotion"].unique()]
    sns.violinplot(
        data=plot_df,
        x="emotion", y="total_ffn_neurons",
        order=emotion_order,
        palette=palette,
        inner="box",
        ax=ax,
        linewidth=0.8,
        cut=0,
    )
    ax.set_title(
        "Distribution of total activated FFN neurons per sample\n"
        "Grouped by primary emotion (layers 14–22, top-5%)",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Primary emotion", fontsize=11)
    ax.set_ylabel("Total activated FFN neurons", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    save_path = out_dir / "violin_ffn_sparsity.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emotion neuron activation extraction for Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to emotion_dataset_translated.csv"
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip the CSV extraction phase (e.g. already done) and go straight to plotting"
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip cluster plot generation (extraction only)"
    )
    parser.add_argument(
        "--top-k-pct", type=float, default=TOP_K_PERCENT,
        help=f"Top-K percent threshold for activated neurons (default: {TOP_K_PERCENT})"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    csv_path = Path(args.csv).resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    out_dir  = csv_path.parent
    log_path = out_dir / "emotion_neuron_analysis.log"
    logger   = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("Emotion Neuron Activation Analysis")
    logger.info(f"CSV          : {csv_path}")
    logger.info(f"Target layers: {TARGET_LAYERS}")
    logger.info(f"Top-K pct    : {args.top_k_pct}%")
    logger.info(f"FFN dim      : {FFN_DIM}  →  top {int(FFN_DIM * args.top_k_pct / 100)} neurons/layer")
    logger.info(f"Hidden dim   : {HIDDEN_DIM} →  top {int(HIDDEN_DIM * args.top_k_pct / 100)} neurons/layer")
    logger.info("=" * 60)

    device = select_device(logger)

    # ── PHASE 1: model loading ────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(device, logger)

    # ── PHASE 2: extraction ───────────────────────────────────────────────
    if not args.skip_extraction:
        df = process_csv(csv_path, model, tokenizer, device, logger)
    else:
        logger.info("Skipping extraction phase (--skip-extraction flag set)")

    # ── PHASE 3: cluster plots ────────────────────────────────────────────
    if not args.skip_plots:
        X, emotions = build_feature_matrix(
            csv_path, model, tokenizer, device, logger
        )

        if len(X) == 0:
            logger.error("No valid rows found for clustering — check CSV content.")
            sys.exit(1)

        plot_pca(X, emotions, out_dir, logger)
        plot_tsne(X, emotions, out_dir, logger)
        plot_umap(X, emotions, out_dir, logger)

        # ── PHASE 4: bonus interpretability plots ─────────────────────────
        plot_heatmap_emotion_layer(csv_path, out_dir, logger)
        plot_violin_ffn_sparsity(csv_path, out_dir, logger)

    logger.info("=" * 60)
    logger.info("All done!")
    logger.info(f"Outputs written to: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
