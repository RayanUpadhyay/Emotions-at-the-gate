"""
rq4_overlap_analysis.py
========================
RQ4: "Is there any overlap in such clusters?"

Analyses
--------
1. Pairwise Jaccard similarity between emotion signature neuron sets.
2. Shared neuron analysis: which neurons are shared between which emotions.
3. Exclusive neuron analysis: which neurons are unique to each emotion.
4. Hierarchical clustering / dendrogram of emotions based on neuron overlap.
5. UpSet-style intersection size plot.

Outputs  →  results/rq4_overlap_analysis/
  - jaccard_heatmap.png
  - jaccard_heatmap_per_layer.png
  - shared_exclusive_bar.png
  - emotion_dendrogram.png
  - upset_intersections.png
  - overlap_matrix.csv
  - rq4_summary.txt
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TARGET_LAYERS, FFN_DIM, EMOTION_ORDER, EMOTION_STYLE,
    TOP_K_PERCENT, SIGNIFICANCE_ALPHA, RANDOM_SEED,
)
from utils import new_figure, save_figure, style_axis


def _binary_masks(ffn_layer: np.ndarray, top_k_pct: float) -> np.ndarray:
    k = max(1, int(FFN_DIM * top_k_pct / 100.0))
    abs_vals = np.abs(ffn_layer.astype(np.float32))
    thresholds = np.partition(abs_vals, -k, axis=1)[:, -k]
    return abs_vals >= thresholds[:, None]


def _signature_set(
    masks: np.ndarray, emotions: np.ndarray, emotion: str,
    alpha: float = SIGNIFICANCE_ALPHA,
) -> Set[int]:
    """Return set of neuron indices that are signature neurons for `emotion`."""
    is_target = emotions == emotion
    n_t = is_target.sum()
    n_o = (~is_target).sum()
    if n_t == 0 or n_o == 0:
        return set()

    p_t = masks[is_target].astype(np.float64).mean(axis=0)
    p_o = masks[~is_target].astype(np.float64).mean(axis=0)
    p_pool = masks.astype(np.float64).mean(axis=0)

    se = np.sqrt(p_pool * (1 - p_pool) * (1.0/n_t + 1.0/n_o))
    se = np.where(se > 0, se, 1.0)
    z = (p_t - p_o) / se
    p_vals = 1.0 - stats.norm.cdf(z)

    sig_mask = (p_vals < alpha) & (p_t > p_o)
    return set(np.where(sig_mask)[0].tolist())


def _jaccard(a: Set, b: Set) -> float:
    if len(a | b) == 0:
        return 0.0
    return len(a & b) / len(a | b)


# ─────────────────────────────────────────────────────────────────────────────
# PAIRWISE JACCARD
# ─────────────────────────────────────────────────────────────────────────────

def _compute_jaccard_matrix(
    sig_sets: Dict[str, Set[int]],
) -> np.ndarray:
    """Compute 6×6 Jaccard similarity matrix."""
    n = len(EMOTION_ORDER)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, ei in enumerate(EMOTION_ORDER):
        for j, ej in enumerate(EMOTION_ORDER):
            matrix[i, j] = _jaccard(sig_sets[ei], sig_sets[ej])
    return matrix


def _plot_jaccard_heatmap(
    matrix: np.ndarray, title: str, save_path: Path, logger: logging.Logger,
):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    mask_upper = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sns.heatmap(
        matrix, xticklabels=EMOTION_ORDER, yticklabels=EMOTION_ORDER,
        annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
        mask=mask_upper,
        cbar_kws={"label": "Jaccard similarity"},
    )
    style_axis(ax, title, "", "")
    plt.tight_layout()
    save_figure(fig, save_path, logger)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED / EXCLUSIVE NEURONS
# ─────────────────────────────────────────────────────────────────────────────

def _shared_exclusive_analysis(
    sig_sets: Dict[str, Set[int]], logger: logging.Logger,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Set[int]]]:
    """
    For each emotion, count:
      - exclusive neurons: in that emotion only
      - shared neurons: in that emotion AND at least one other

    Also return per-emotion exclusive sets for downstream use.
    """
    exclusive_counts = {}
    shared_counts = {}
    exclusive_sets = {}

    all_other_union = {}
    for emotion in EMOTION_ORDER:
        others = set()
        for e2 in EMOTION_ORDER:
            if e2 != emotion:
                others |= sig_sets[e2]
        all_other_union[emotion] = others

    for emotion in EMOTION_ORDER:
        exclusive = sig_sets[emotion] - all_other_union[emotion]
        shared = sig_sets[emotion] & all_other_union[emotion]
        exclusive_counts[emotion] = len(exclusive)
        shared_counts[emotion] = len(shared)
        exclusive_sets[emotion] = exclusive
        logger.info(f"  {emotion}: {len(exclusive)} exclusive, "
                    f"{len(shared)} shared, {len(sig_sets[emotion])} total")

    return exclusive_counts, shared_counts, exclusive_sets


def _plot_shared_exclusive(
    exclusive_counts: Dict[str, int],
    shared_counts: Dict[str, int],
    sig_sets: Dict[str, Set[int]],
    out_dir: Path, logger: logging.Logger,
):
    """Stacked bar chart: exclusive vs shared neurons per emotion."""
    fig, ax = new_figure(figsize=(12, 7))

    labels = [EMOTION_STYLE[e]["label"] for e in EMOTION_ORDER]
    excl = [exclusive_counts[e] for e in EMOTION_ORDER]
    shared = [shared_counts[e] for e in EMOTION_ORDER]
    colors = [EMOTION_STYLE[e]["color"] for e in EMOTION_ORDER]

    x = np.arange(len(EMOTION_ORDER))
    width = 0.6

    bars_excl = ax.bar(x, excl, width, color=colors, edgecolor="white",
                       linewidth=0.5, label="Exclusive")
    bars_shared = ax.bar(x, shared, width, bottom=excl,
                         color=[c + "80" for c in [EMOTION_STYLE[e]["color"]
                                                   for e in EMOTION_ORDER]],
                         edgecolor="white", linewidth=0.5, alpha=0.5,
                         label="Shared with other emotions")

    for i, (e_val, s_val) in enumerate(zip(excl, shared)):
        total = e_val + s_val
        if total > 0:
            ax.text(i, total + 5, str(total), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axis(ax,
               "Exclusive vs shared signature neurons per emotion\n"
               "(Aggregated across layers 14–22)",
               "Primary emotion", "Number of signature neurons")
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    save_figure(fig, out_dir / "shared_exclusive_bar.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
# DENDROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def _plot_dendrogram(
    jaccard_matrix: np.ndarray, out_dir: Path, logger: logging.Logger,
):
    """Hierarchical clustering of emotions based on Jaccard distance."""
    # Convert similarity to distance
    dist_matrix = 1.0 - jaccard_matrix
    np.fill_diagonal(dist_matrix, 0)

    # Make symmetric and extract condensed form
    dist_condensed = squareform(dist_matrix, checks=False)
    Z = linkage(dist_condensed, method="average")

    fig, ax = new_figure(figsize=(10, 7))
    colors = [EMOTION_STYLE[e]["color"] for e in EMOTION_ORDER]
    labels = [EMOTION_STYLE[e]["label"] for e in EMOTION_ORDER]

    dendrogram(
        Z, labels=labels, ax=ax,
        leaf_font_size=12, color_threshold=0,
        above_threshold_color="#555555",
    )
    style_axis(ax,
               "Hierarchical clustering of emotions by neuron signature overlap\n"
               "(Average linkage, Jaccard distance)",
               "", "Jaccard distance")
    plt.tight_layout()
    save_figure(fig, out_dir / "emotion_dendrogram.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
# UPSET-STYLE INTERSECTION PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _plot_upset_intersections(
    sig_sets: Dict[str, Set[int]], out_dir: Path, logger: logging.Logger,
    top_n: int = 20,
):
    """
    Bar chart of the top-N largest set intersections (UpSet-style).
    Each bar represents a unique combination of emotions that share neurons.
    """
    from itertools import combinations

    # Compute all possible intersections (including singles)
    intersection_sizes = []

    for r in range(1, len(EMOTION_ORDER) + 1):
        for combo in combinations(EMOTION_ORDER, r):
            # Intersection of these emotions, exclusive of all others
            in_set = sig_sets[combo[0]]
            for e in combo[1:]:
                in_set = in_set & sig_sets[e]
            # Subtract neurons also in emotions NOT in this combo
            others = set(EMOTION_ORDER) - set(combo)
            for e in others:
                in_set = in_set - sig_sets[e]
            if len(in_set) > 0:
                label = " ∩ ".join(EMOTION_STYLE[e]["label"] for e in combo)
                intersection_sizes.append({
                    "label": label,
                    "size": len(in_set),
                    "n_emotions": r,
                    "combo": combo,
                })

    if not intersection_sizes:
        logger.warning("No intersections found for UpSet plot.")
        return

    # Sort by size descending, take top N
    intersection_sizes.sort(key=lambda x: x["size"], reverse=True)
    top = intersection_sizes[:top_n]

    fig, ax = new_figure(figsize=(16, 8))
    x = np.arange(len(top))
    colors_bar = []
    for item in top:
        if item["n_emotions"] == 1:
            colors_bar.append(EMOTION_STYLE[item["combo"][0]]["color"])
        else:
            colors_bar.append("#888888")

    bars = ax.bar(x, [item["size"] for item in top],
                  color=colors_bar, edgecolor="white", linewidth=0.5)

    for i, (bar, item) in enumerate(zip(bars, top)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(item["size"]), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([item["label"] for item in top],
                       rotation=45, ha="right", fontsize=8)
    style_axis(ax,
               f"Top-{top_n} neuron set intersections (UpSet-style)\n"
               "Exclusive membership: neurons counted only in their exact overlap group",
               "", "Number of neurons")
    plt.tight_layout()
    save_figure(fig, out_dir / "upset_intersections.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
# PER-LAYER JACCARD HEATMAPS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_per_layer_jaccard(
    ffn_acts: Dict[int, np.ndarray],
    emotions: np.ndarray,
    out_dir: Path, logger: logging.Logger,
    top_k_pct: float = TOP_K_PERCENT,
):
    """Small-multiples: Jaccard heatmap per layer."""
    n_layers = len(TARGET_LAYERS)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), dpi=300)
    axes = axes.flatten()

    for i, layer_idx in enumerate(TARGET_LAYERS):
        masks = _binary_masks(ffn_acts[layer_idx], top_k_pct)
        sig_sets = {}
        for emotion in EMOTION_ORDER:
            sig_sets[emotion] = _signature_set(masks, emotions, emotion)

        matrix = _compute_jaccard_matrix(sig_sets)
        mask_upper = np.triu(np.ones_like(matrix, dtype=bool), k=1)

        sns.heatmap(
            matrix, xticklabels=EMOTION_ORDER, yticklabels=EMOTION_ORDER,
            annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.3, ax=axes[i], vmin=0, vmax=0.5,
            mask=mask_upper, cbar=False,
            annot_kws={"size": 7},
        )
        axes[i].set_title(f"Layer {layer_idx}", fontsize=10, fontweight="bold")
        axes[i].tick_params(labelsize=7)

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Pairwise Jaccard similarity of signature neurons per layer",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure(fig, out_dir / "jaccard_heatmap_per_layer.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RQ4 RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_rq4(
    ffn_acts: Dict[int, np.ndarray],
    emotions: np.ndarray,
    out_dir: Path,
    logger: logging.Logger,
    top_k_pct: float = TOP_K_PERCENT,
):
    """Execute all RQ4 analyses."""
    logger.info("=" * 60)
    logger.info("RQ4: Overlap analysis of emotion neuron clusters")
    logger.info("=" * 60)

    summary_lines = ["RQ4: Overlap Analysis", "=" * 50, ""]

    # ── 1. Compute signature sets (aggregated across all layers) ──────────
    logger.info("Step 1: Computing aggregated signature neuron sets")
    sig_sets_agg = {e: set() for e in EMOTION_ORDER}

    for layer_idx in TARGET_LAYERS:
        masks = _binary_masks(ffn_acts[layer_idx], top_k_pct)
        for emotion in EMOTION_ORDER:
            layer_sigs = _signature_set(masks, emotions, emotion)
            # Encode as (layer, neuron_idx) to distinguish across layers
            sig_sets_agg[emotion] |= {(layer_idx, n) for n in layer_sigs}

    for emotion in EMOTION_ORDER:
        logger.info(f"  {emotion}: {len(sig_sets_agg[emotion])} total "
                    f"signature (layer, neuron) pairs")

    # ── 2. Pairwise Jaccard (aggregated) ──────────────────────────────────
    logger.info("Step 2: Pairwise Jaccard similarity")
    jaccard_matrix = _compute_jaccard_matrix(sig_sets_agg)
    _plot_jaccard_heatmap(
        jaccard_matrix,
        "Pairwise Jaccard similarity of emotion signature neurons\n"
        "(Aggregated across layers 14–22)",
        out_dir / "jaccard_heatmap.png", logger,
    )

    summary_lines.append("Pairwise Jaccard similarity (aggregated):")
    for i, ei in enumerate(EMOTION_ORDER):
        for j, ej in enumerate(EMOTION_ORDER):
            if j > i:
                summary_lines.append(
                    f"  {ei} vs {ej}: {jaccard_matrix[i,j]:.4f}"
                )
    summary_lines.append("")

    # Save as CSV
    jac_df = pd.DataFrame(jaccard_matrix, index=EMOTION_ORDER,
                           columns=EMOTION_ORDER)
    jac_df.to_csv(out_dir / "overlap_matrix.csv")

    # ── 3. Shared / exclusive analysis ────────────────────────────────────
    logger.info("Step 3: Shared vs exclusive neurons")
    excl_counts, shared_counts, excl_sets = _shared_exclusive_analysis(
        sig_sets_agg, logger
    )
    _plot_shared_exclusive(excl_counts, shared_counts, sig_sets_agg,
                           out_dir, logger)

    summary_lines.append("Exclusive vs shared signature neurons:")
    for emotion in EMOTION_ORDER:
        total = excl_counts[emotion] + shared_counts[emotion]
        summary_lines.append(
            f"  {emotion}: {excl_counts[emotion]} exclusive, "
            f"{shared_counts[emotion]} shared (total {total})"
        )
    summary_lines.append("")

    # ── 4. Dendrogram ─────────────────────────────────────────────────────
    logger.info("Step 4: Hierarchical clustering dendrogram")
    _plot_dendrogram(jaccard_matrix, out_dir, logger)

    # ── 5. UpSet intersection plot ────────────────────────────────────────
    logger.info("Step 5: UpSet intersection plot")
    _plot_upset_intersections(sig_sets_agg, out_dir, logger)

    # ── 6. Per-layer Jaccard heatmaps ─────────────────────────────────────
    logger.info("Step 6: Per-layer Jaccard heatmaps")
    _plot_per_layer_jaccard(ffn_acts, emotions, out_dir, logger, top_k_pct)

    # ── Save summary ──────────────────────────────────────────────────────
    summary_path = out_dir / "rq4_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    logger.info(f"RQ4 summary saved: {summary_path}")
    logger.info("RQ4 complete.")
