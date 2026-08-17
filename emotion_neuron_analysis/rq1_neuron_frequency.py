"""
rq1_neuron_frequency.py
========================
RQ1: "When an LLM tries to understand an emotion, is there a specific set
      of neurons that get activated?"

Analyses
--------
1. Per-emotion neuron activation frequency (what % of samples activate each neuron)
2. Identification of "consistently activated" neurons (freq > 50%)
3. Chi-square test of independence: neuron_active × emotion_label
4. Permutation test: shuffle labels → null distribution of frequency counts
5. Volcano plot: effect size vs -log10(p-value)

Outputs  →  results/rq1_neuron_frequency/
  - freq_heatmap_top50_layer{L}.png
  - consistent_neurons_per_emotion.png
  - volcano_layer{L}.png
  - chi2_significant_neurons.png
  - rq1_summary.txt
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TARGET_LAYERS, FFN_DIM, EMOTION_ORDER, EMOTION_STYLE,
    TOP_K_PERCENT, SIGNIFICANCE_ALPHA, N_PERMUTATIONS, RANDOM_SEED,
)
from utils import new_figure, save_figure, style_axis


def _binary_masks(ffn_layer: np.ndarray, top_k_pct: float) -> np.ndarray:
    """
    Convert raw FFN activations to binary masks at the given threshold.
    ffn_layer: [N, FFN_DIM] float16
    Returns  : [N, FFN_DIM] bool
    """
    k = max(1, int(FFN_DIM * top_k_pct / 100.0))
    abs_vals = np.abs(ffn_layer.astype(np.float32))
    # Threshold per sample: kth-largest value
    thresholds = np.partition(abs_vals, -k, axis=1)[:, -k]  # [N]
    masks = abs_vals >= thresholds[:, None]
    return masks


def _compute_frequencies(
    masks: np.ndarray, emotions: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Per-emotion frequency: fraction of samples where each neuron is active.
    Returns {emotion: [FFN_DIM] float array of frequencies}.
    """
    freqs = {}
    for emotion in EMOTION_ORDER:
        idx = emotions == emotion
        if idx.sum() == 0:
            freqs[emotion] = np.zeros(masks.shape[1], dtype=np.float32)
            continue
        freqs[emotion] = masks[idx].astype(np.float32).mean(axis=0)
    return freqs


# ─────────────────────────────────────────────────────────────────────────────
# CHI-SQUARE TEST
# ─────────────────────────────────────────────────────────────────────────────

def _chi2_per_neuron(
    masks: np.ndarray, emotions: np.ndarray, emotion: str
) -> tuple:
    """
    For each neuron run a 2×2 chi-square test:
      rows: neuron active / not active
      cols: emotion = target / emotion != target

    Returns (chi2_stats [D], p_values [D], effect_sizes [D])
    Effect size = (observed_freq_in_emotion - overall_freq) / overall_freq
    """
    is_target = (emotions == emotion)
    n_target  = is_target.sum()
    n_other   = (~is_target).sum()
    n_total   = len(emotions)

    active_target = masks[is_target].sum(axis=0)      # [D]
    active_other  = masks[~is_target].sum(axis=0)     # [D]

    # Build contingency counts
    # cell: [active&target, active&other, inactive&target, inactive&other]
    a = active_target
    b = active_other
    c = n_target - a
    d = n_other - b

    chi2_vals = np.zeros(masks.shape[1], dtype=np.float64)
    p_vals    = np.ones(masks.shape[1], dtype=np.float64)

    for i in range(masks.shape[1]):
        table = np.array([[a[i], b[i]], [c[i], d[i]]])
        if table.min() < 0:
            continue
        # Skip if a row or column is all zero
        if table.sum(axis=0).min() == 0 or table.sum(axis=1).min() == 0:
            continue
        chi2_val, p_val, _, _ = stats.chi2_contingency(table, correction=True)
        chi2_vals[i] = chi2_val
        p_vals[i]    = p_val

    # Effect size: relative frequency difference
    freq_target  = active_target / max(n_target, 1)
    freq_overall = (active_target + active_other) / max(n_total, 1)
    effect_sizes = np.where(freq_overall > 0,
                            (freq_target - freq_overall) / freq_overall,
                            0.0)

    return chi2_vals, p_vals, effect_sizes


# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def _permutation_test(
    masks: np.ndarray, emotions: np.ndarray, emotion: str,
    n_perms: int = N_PERMUTATIONS, seed: int = RANDOM_SEED,
) -> tuple:
    """
    Permutation test for the number of 'consistently activated' neurons
    (freq > 0.5) for a given emotion.

    Returns (observed_count, null_distribution [n_perms], p_value).
    """
    rng = np.random.RandomState(seed)
    is_target = emotions == emotion
    n_target  = is_target.sum()

    # Observed
    freq_obs = masks[is_target].astype(np.float32).mean(axis=0)
    observed = (freq_obs > 0.5).sum()

    null_dist = np.zeros(n_perms, dtype=int)
    for p in range(n_perms):
        perm_idx = rng.choice(len(emotions), size=n_target, replace=False)
        freq_perm = masks[perm_idx].astype(np.float32).mean(axis=0)
        null_dist[p] = (freq_perm > 0.5).sum()

    p_value = (null_dist >= observed).mean()
    return observed, null_dist, p_value


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _plot_freq_heatmap(
    freqs: Dict[str, np.ndarray], layer_idx: int,
    out_dir: Path, logger: logging.Logger, top_n: int = 50,
):
    """Heatmap of top-N most frequently activated neurons per emotion."""
    # For each emotion, find its top-N neurons by frequency
    all_top_indices = set()
    for emotion in EMOTION_ORDER:
        top_idx = np.argsort(freqs[emotion])[-top_n:]
        all_top_indices.update(top_idx.tolist())
    all_top_indices = sorted(all_top_indices)

    # Build matrix
    matrix = np.zeros((len(EMOTION_ORDER), len(all_top_indices)), dtype=np.float32)
    for i, emotion in enumerate(EMOTION_ORDER):
        matrix[i] = freqs[emotion][all_top_indices]

    fig, ax = plt.subplots(figsize=(20, 6), dpi=300)
    sns.heatmap(
        matrix, xticklabels=[str(idx) for idx in all_top_indices],
        yticklabels=EMOTION_ORDER, cmap="YlOrRd",
        linewidths=0.1, ax=ax, vmin=0, vmax=1,
        cbar_kws={"label": "Activation frequency"},
    )
    ax.set_title(
        f"Top-{top_n} FFN neuron activation frequency per emotion — Layer {layer_idx}",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Neuron index", fontsize=10)
    ax.set_ylabel("Primary emotion", fontsize=10)
    ax.tick_params(axis="x", labelsize=5, rotation=90)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    save_figure(fig, out_dir / f"freq_heatmap_top{top_n}_layer{layer_idx}.png", logger)


def _plot_consistent_neurons(
    consistent_counts: Dict[str, Dict[int, int]],
    out_dir: Path, logger: logging.Logger,
):
    """Bar chart: number of consistently activated neurons per emotion per layer."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    x = np.arange(len(TARGET_LAYERS))
    width = 0.12
    offsets = np.linspace(-(len(EMOTION_ORDER)-1)*width/2,
                           (len(EMOTION_ORDER)-1)*width/2,
                           len(EMOTION_ORDER))

    for i, emotion in enumerate(EMOTION_ORDER):
        counts = [consistent_counts[emotion].get(l, 0) for l in TARGET_LAYERS]
        ax.bar(x + offsets[i], counts, width,
               color=EMOTION_STYLE[emotion]["color"],
               label=EMOTION_STYLE[emotion]["label"], edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in TARGET_LAYERS])
    style_axis(ax,
               "Number of consistently activated FFN neurons (freq > 50%) per emotion × layer",
               "Layer", "Neuron count")
    ax.legend(title="Emotion", fontsize=9, title_fontsize=10, loc="upper left")
    plt.tight_layout()
    save_figure(fig, out_dir / "consistent_neurons_per_emotion.png", logger)


def _plot_volcano(
    p_vals: np.ndarray, effect_sizes: np.ndarray,
    emotion: str, layer_idx: int,
    out_dir: Path, logger: logging.Logger,
):
    """Volcano plot: effect size vs -log10(p-value)."""
    neg_log_p = -np.log10(np.clip(p_vals, 1e-300, 1.0))
    significant = p_vals < SIGNIFICANCE_ALPHA
    large_effect = np.abs(effect_sizes) > 0.5

    fig, ax = new_figure(figsize=(10, 8))
    ax.scatter(effect_sizes[~(significant & large_effect)],
               neg_log_p[~(significant & large_effect)],
               c="#CCCCCC", s=4, alpha=0.3, label="Not significant")
    ax.scatter(effect_sizes[significant & ~large_effect],
               neg_log_p[significant & ~large_effect],
               c="#5DA5DA", s=6, alpha=0.5, label="Significant (small effect)")
    ax.scatter(effect_sizes[significant & large_effect],
               neg_log_p[significant & large_effect],
               c=EMOTION_STYLE[emotion]["color"], s=10, alpha=0.7,
               label="Significant + large effect")
    ax.axhline(-np.log10(SIGNIFICANCE_ALPHA), color="red", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.axvline(-0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.axvline( 0.5, color="grey", linestyle=":", linewidth=0.5)

    n_sig = (significant & large_effect).sum()
    style_axis(
        ax,
        f"Volcano — {EMOTION_STYLE[emotion]['label']} FFN neurons, Layer {layer_idx}\n"
        f"({n_sig} neurons with p < {SIGNIFICANCE_ALPHA} and |effect| > 0.5)",
        "Effect size (relative freq difference)", "-log10(p-value)",
    )
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    save_figure(fig, out_dir / f"volcano_{emotion.lower()}_layer{layer_idx}.png", logger)


def _plot_chi2_summary(
    sig_counts: Dict[str, Dict[int, int]],
    out_dir: Path, logger: logging.Logger,
):
    """Heatmap: number of significant chi2 neurons per emotion × layer."""
    matrix = np.zeros((len(EMOTION_ORDER), len(TARGET_LAYERS)), dtype=int)
    for i, emotion in enumerate(EMOTION_ORDER):
        for j, layer in enumerate(TARGET_LAYERS):
            matrix[i, j] = sig_counts.get(emotion, {}).get(layer, 0)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.heatmap(
        matrix,
        xticklabels=[f"L{l}" for l in TARGET_LAYERS],
        yticklabels=EMOTION_ORDER,
        annot=True, fmt="d", cmap="YlGnBu",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Significant neurons (p < 0.05)"},
    )
    style_axis(ax,
               "Chi-square significant FFN neurons per emotion × layer",
               "Layer", "Primary emotion")
    plt.tight_layout()
    save_figure(fig, out_dir / "chi2_significant_neurons.png", logger)


def _plot_permutation(
    emotion: str, observed: int, null_dist: np.ndarray, p_value: float,
    layer_idx: int, out_dir: Path, logger: logging.Logger,
):
    """Histogram of null distribution with observed value marked."""
    fig, ax = new_figure(figsize=(10, 6))
    ax.hist(null_dist, bins=40, color="#CCCCCC", edgecolor="white", alpha=0.8,
            label="Null distribution")
    ax.axvline(observed, color=EMOTION_STYLE[emotion]["color"], linewidth=2,
               linestyle="--", label=f"Observed = {observed} (p = {p_value:.4f})")
    style_axis(
        ax,
        f"Permutation test — {EMOTION_STYLE[emotion]['label']}, Layer {layer_idx}\n"
        f"Consistently activated neurons (freq > 50%)",
        "Number of consistent neurons", "Frequency",
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_figure(fig, out_dir / f"permutation_{emotion.lower()}_layer{layer_idx}.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RQ1 RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_rq1(
    ffn_acts: Dict[int, np.ndarray],
    emotions: np.ndarray,
    out_dir: Path,
    logger: logging.Logger,
    top_k_pct: float = TOP_K_PERCENT,
):
    """Execute all RQ1 analyses and save outputs."""
    logger.info("=" * 60)
    logger.info("RQ1: Neuron activation frequency analysis")
    logger.info("=" * 60)

    summary_lines = ["RQ1: Neuron Activation Frequency Analysis",
                     "=" * 50, ""]

    consistent_counts = {e: {} for e in EMOTION_ORDER}
    sig_counts        = {e: {} for e in EMOTION_ORDER}

    # Analyse a representative subset of layers for detailed plots
    detail_layers = [TARGET_LAYERS[0], TARGET_LAYERS[len(TARGET_LAYERS)//2],
                     TARGET_LAYERS[-1]]  # first, middle, last

    for layer_idx in TARGET_LAYERS:
        logger.info(f"--- Layer {layer_idx} ---")
        masks = _binary_masks(ffn_acts[layer_idx], top_k_pct)
        freqs = _compute_frequencies(masks, emotions)

        # ── Consistent neurons ────────────────────────────────────────────
        for emotion in EMOTION_ORDER:
            n_consistent = (freqs[emotion] > 0.5).sum()
            consistent_counts[emotion][layer_idx] = int(n_consistent)
            logger.info(f"  {emotion:<12s}: {n_consistent:>4d} neurons with freq > 50%")

        summary_lines.append(f"Layer {layer_idx}:")
        for emotion in EMOTION_ORDER:
            summary_lines.append(
                f"  {emotion}: {consistent_counts[emotion][layer_idx]} consistent neurons"
            )
        summary_lines.append("")

        # ── Frequency heatmap (detailed layers only) ──────────────────────
        if layer_idx in detail_layers:
            _plot_freq_heatmap(freqs, layer_idx, out_dir, logger)

        # ── Chi-square tests ──────────────────────────────────────────────
        for emotion in EMOTION_ORDER:
            chi2_vals, p_vals, effect_sizes = _chi2_per_neuron(
                masks, emotions, emotion
            )
            n_sig = (p_vals < SIGNIFICANCE_ALPHA).sum()
            n_sig_large = ((p_vals < SIGNIFICANCE_ALPHA) &
                           (np.abs(effect_sizes) > 0.5)).sum()
            sig_counts[emotion][layer_idx] = int(n_sig)
            logger.info(f"  {emotion} chi2: {n_sig} significant, "
                        f"{n_sig_large} with large effect")

            # Volcano for detailed layers
            if layer_idx in detail_layers:
                _plot_volcano(p_vals, effect_sizes, emotion, layer_idx,
                              out_dir, logger)

        # ── Permutation test (middle layer only to save time) ─────────────
        if layer_idx == detail_layers[1]:
            for emotion in EMOTION_ORDER:
                observed, null_dist, p_val = _permutation_test(
                    masks, emotions, emotion
                )
                logger.info(f"  Permutation {emotion}: observed={observed}, "
                            f"p={p_val:.4f}")
                summary_lines.append(
                    f"Permutation test L{layer_idx} {emotion}: "
                    f"observed={observed}, p={p_val:.4f}"
                )
                _plot_permutation(emotion, observed, null_dist, p_val,
                                  layer_idx, out_dir, logger)
            summary_lines.append("")

    # ── Summary plots ─────────────────────────────────────────────────────
    _plot_consistent_neurons(consistent_counts, out_dir, logger)
    _plot_chi2_summary(sig_counts, out_dir, logger)

    # ── Save summary text ─────────────────────────────────────────────────
    summary_path = out_dir / "rq1_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    logger.info(f"RQ1 summary saved: {summary_path}")
    logger.info("RQ1 complete.")
