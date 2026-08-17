"""
rq3_cluster_enumeration.py
============================
RQ3: "Can we find out all such clusters for all types of primary emotions?"

Analyses
--------
1. Dimensionality reduction on residual stream vectors:
   - PCA (with explained variance)
   - t-SNE (with PCA pre-reduction)
   - UMAP
2. Silhouette analysis: how well do the 6-emotion labels match the actual
   cluster structure in activation space?
3. Per-emotion cluster purity via nearest-neighbour analysis.
4. Complete enumeration table: top signature neurons per emotion per layer
   with their mean activation values.

Outputs  →  results/rq3_cluster_enumeration/
  - cluster_pca.png
  - cluster_tsne.png
  - cluster_umap.png
  - silhouette_scores.png
  - purity_per_emotion.png
  - signature_enumeration.csv
  - rq3_summary.txt
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TARGET_LAYERS, FFN_DIM, HIDDEN_DIM, EMOTION_ORDER, EMOTION_STYLE,
    TOP_K_PERCENT, SIGNIFICANCE_ALPHA, CV_FOLDS, RANDOM_SEED,
)
from utils import new_figure, save_figure, style_axis, scatter_by_emotion


def _build_residual_matrix(res_acts: Dict[int, np.ndarray]) -> np.ndarray:
    """Concatenate residual stream vectors across layers → [N, 9*1536]."""
    parts = [res_acts[l].astype(np.float32) for l in TARGET_LAYERS]
    return np.concatenate(parts, axis=1)


def _binary_masks(ffn_layer: np.ndarray, top_k_pct: float) -> np.ndarray:
    k = max(1, int(FFN_DIM * top_k_pct / 100.0))
    abs_vals = np.abs(ffn_layer.astype(np.float32))
    thresholds = np.partition(abs_vals, -k, axis=1)[:, -k]
    return abs_vals >= thresholds[:, None]


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _plot_pca(X: np.ndarray, emotions: list, out_dir: Path,
              logger: logging.Logger):
    logger.info("Running PCA …")
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    scatter_by_emotion(
        coords, emotions,
        "PCA — Residual stream activations (layers 14–22)\n"
        f"PC1: {var[0]*100:.1f}% var, PC2: {var[1]*100:.1f}% var",
        out_dir / "cluster_pca.png", logger,
        xlabel=f"PC1 ({var[0]*100:.1f}%)",
        ylabel=f"PC2 ({var[1]*100:.1f}%)",
    )
    return pca


def _plot_tsne(X: np.ndarray, emotions: list, out_dir: Path,
               logger: logging.Logger):
    from sklearn.manifold import TSNE

    logger.info("Running t-SNE (PCA pre-reduction to 50 dims) …")
    n_pre = min(50, X.shape[1], X.shape[0] - 1)
    Xr = PCA(n_components=n_pre, random_state=RANDOM_SEED).fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
                init="pca", random_state=RANDOM_SEED, max_iter=1500, verbose=0)
    coords = tsne.fit_transform(Xr)

    scatter_by_emotion(
        coords, emotions,
        "t-SNE — Residual stream activations (layers 14–22)",
        out_dir / "cluster_tsne.png", logger,
    )


def _plot_umap(X: np.ndarray, emotions: list, out_dir: Path,
               logger: logging.Logger):
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed — skipping UMAP.")
        return

    logger.info("Running UMAP …")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="cosine", random_state=RANDOM_SEED, verbose=False)
    coords = reducer.fit_transform(X)

    scatter_by_emotion(
        coords, emotions,
        "UMAP — Residual stream activations (layers 14–22)",
        out_dir / "cluster_umap.png", logger,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SILHOUETTE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_silhouette(X: np.ndarray, emotions: list, out_dir: Path,
                     logger: logging.Logger):
    """Per-emotion silhouette score distribution."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(emotions)

    # Use PCA-reduced features for faster silhouette computation
    n_pre = min(50, X.shape[1], X.shape[0] - 1)
    Xr = PCA(n_components=n_pre, random_state=RANDOM_SEED).fit_transform(X)

    logger.info("Computing silhouette scores …")
    overall_sil = silhouette_score(Xr, y)
    sample_sils = silhouette_samples(Xr, y)
    logger.info(f"  Overall silhouette score: {overall_sil:.4f}")

    fig, ax = new_figure(figsize=(12, 8))

    y_lower = 0
    for i, emotion in enumerate(EMOTION_ORDER):
        mask = np.array(emotions) == emotion
        sil_vals = sample_sils[mask]
        sil_vals.sort()
        y_upper = y_lower + len(sil_vals)

        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, sil_vals,
            facecolor=EMOTION_STYLE[emotion]["color"], alpha=0.7,
            edgecolor="white", linewidth=0.2,
        )
        ax.text(-0.05, y_lower + 0.5 * len(sil_vals),
                EMOTION_STYLE[emotion]["label"], fontsize=10, va="center")
        y_lower = y_upper + 10  # gap between groups

    ax.axvline(overall_sil, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {overall_sil:.3f}")
    style_axis(ax,
               "Silhouette analysis of emotion clusters\n"
               "(Residual stream, PCA-reduced)",
               "Silhouette coefficient", "Samples (grouped by emotion)")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_yticks([])
    plt.tight_layout()
    save_figure(fig, out_dir / "silhouette_scores.png", logger)

    return overall_sil


# ─────────────────────────────────────────────────────────────────────────────
# NEAREST-NEIGHBOUR PURITY
# ─────────────────────────────────────────────────────────────────────────────

def _knn_purity(
    X: np.ndarray, emotions: list, out_dir: Path, logger: logging.Logger,
    k: int = 15,
) -> Dict[str, float]:
    """
    For each sample, look at its k nearest neighbours in activation space
    and compute what fraction share the same emotion label.
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(emotions)

    n_pre = min(50, X.shape[1], X.shape[0] - 1)
    Xr = PCA(n_components=n_pre, random_state=RANDOM_SEED).fit_transform(X)

    logger.info(f"Computing {k}-NN purity …")
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(Xr, y)
    neighbours = knn.kneighbors(Xr, return_distance=False)  # [N, k]

    purities = {}
    for emotion in EMOTION_ORDER:
        mask = np.array(emotions) == emotion
        emotion_label = le.transform([emotion])[0]
        indices = np.where(mask)[0]
        if len(indices) == 0:
            purities[emotion] = 0.0
            continue

        # For each sample of this emotion, fraction of k-NN with same label
        sample_purities = []
        for idx in indices:
            neighbour_labels = y[neighbours[idx]]
            purity = (neighbour_labels == emotion_label).mean()
            sample_purities.append(purity)
        purities[emotion] = float(np.mean(sample_purities))
        logger.info(f"  {emotion}: purity = {purities[emotion]:.4f}")

    # Plot
    fig, ax = new_figure(figsize=(12, 7))
    colors = [EMOTION_STYLE[e]["color"] for e in EMOTION_ORDER]
    labels = [EMOTION_STYLE[e]["label"] for e in EMOTION_ORDER]
    vals = [purities[e] for e in EMOTION_ORDER]

    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(1.0/6, color="grey", linestyle=":", linewidth=1,
               label="Random baseline (1/6)")
    style_axis(ax,
               f"{k}-NN cluster purity per emotion\n"
               "(Fraction of neighbours sharing the same label)",
               "Primary emotion", "Purity")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    save_figure(fig, out_dir / "purity_per_emotion.png", logger)

    return purities


# ─────────────────────────────────────────────────────────────────────────────
# SIGNATURE ENUMERATION TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _enumerate_signatures(
    ffn_acts: Dict[int, np.ndarray],
    emotions: np.ndarray,
    out_dir: Path,
    logger: logging.Logger,
    top_k_pct: float = TOP_K_PERCENT,
    top_n_per_layer: int = 20,
):
    """
    For each emotion × layer, list the top-N most frequently activated
    signature neurons with their activation frequencies and mean values.
    Saved as CSV for inclusion in the paper.
    """
    logger.info("Enumerating top signature neurons per emotion × layer …")
    rows = []

    for layer_idx in TARGET_LAYERS:
        masks = _binary_masks(ffn_acts[layer_idx], top_k_pct)
        raw_vals = ffn_acts[layer_idx].astype(np.float32)

        for emotion in EMOTION_ORDER:
            is_target = emotions == emotion
            freq = masks[is_target].astype(np.float32).mean(axis=0)
            mean_val = np.abs(raw_vals[is_target]).mean(axis=0)

            # Top-N by frequency
            top_idx = np.argsort(freq)[-top_n_per_layer:][::-1]
            for rank, neuron_idx in enumerate(top_idx, 1):
                rows.append({
                    "emotion": emotion,
                    "layer": layer_idx,
                    "neuron_index": int(neuron_idx),
                    "activation_freq": round(float(freq[neuron_idx]), 4),
                    "mean_abs_activation": round(float(mean_val[neuron_idx]), 4),
                    "rank": rank,
                })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "signature_enumeration.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  Enumeration table saved: {csv_path} ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RQ3 RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_rq3(
    ffn_acts: Dict[int, np.ndarray],
    res_acts: Dict[int, np.ndarray],
    emotions: np.ndarray,
    out_dir: Path,
    logger: logging.Logger,
    top_k_pct: float = TOP_K_PERCENT,
):
    """Execute all RQ3 analyses."""
    logger.info("=" * 60)
    logger.info("RQ3: Cluster enumeration for all primary emotions")
    logger.info("=" * 60)

    summary_lines = ["RQ3: Cluster Enumeration", "=" * 50, ""]
    emotions_list = emotions.tolist()

    # Build residual feature matrix
    X = _build_residual_matrix(res_acts)
    logger.info(f"Residual feature matrix: {X.shape}")

    # ── 1. Dimensionality reduction plots ─────────────────────────────────
    pca_model = _plot_pca(X, emotions_list, out_dir, logger)
    var = pca_model.explained_variance_ratio_
    summary_lines.append(f"PCA: PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}%")

    _plot_tsne(X, emotions_list, out_dir, logger)
    _plot_umap(X, emotions_list, out_dir, logger)

    # ── 2. Silhouette analysis ────────────────────────────────────────────
    sil_score = _plot_silhouette(X, emotions_list, out_dir, logger)
    summary_lines.append(f"Overall silhouette score: {sil_score:.4f}")
    summary_lines.append("")

    # ── 3. KNN purity ─────────────────────────────────────────────────────
    purities = _knn_purity(X, emotions_list, out_dir, logger)
    summary_lines.append("KNN purity (k=15):")
    for emotion in EMOTION_ORDER:
        summary_lines.append(f"  {emotion}: {purities[emotion]:.4f}")
    summary_lines.append("")

    # ── 4. Signature enumeration ──────────────────────────────────────────
    _enumerate_signatures(ffn_acts, emotions, out_dir, logger, top_k_pct)

    # ── Save summary ──────────────────────────────────────────────────────
    summary_path = out_dir / "rq3_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    logger.info(f"RQ3 summary saved: {summary_path}")
    logger.info("RQ3 complete.")
