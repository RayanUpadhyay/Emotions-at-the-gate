"""
sensitivity.py
===============
Threshold sensitivity analysis: re-run key metrics from RQ1–RQ4 at multiple
top-K thresholds (1%, 3%, 5%, 10%) to demonstrate robustness.

Analyses
--------
1. Consistent neuron counts vs threshold per emotion.
2. Classifier accuracy vs threshold.
3. Jaccard overlap stability vs threshold.
4. Silhouette score vs threshold (using FFN binary masks for consistency).

Outputs  →  results/sensitivity_analysis/
  - consistent_vs_threshold.png
  - accuracy_vs_threshold.png
  - jaccard_vs_threshold.png
  - sensitivity_summary.txt
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    TARGET_LAYERS, FFN_DIM, EMOTION_ORDER, EMOTION_STYLE,
    SENSITIVITY_THRESHOLDS, SIGNIFICANCE_ALPHA, CV_FOLDS, RANDOM_SEED,
)
from utils import new_figure, save_figure, style_axis


def _binary_masks(ffn_layer: np.ndarray, top_k_pct: float) -> np.ndarray:
    k = max(1, int(FFN_DIM * top_k_pct / 100.0))
    abs_vals = np.abs(ffn_layer.astype(np.float32))
    thresholds = np.partition(abs_vals, -k, axis=1)[:, -k]
    return abs_vals >= thresholds[:, None]


def _signature_set(masks, emotions, emotion, alpha=SIGNIFICANCE_ALPHA):
    is_target = emotions == emotion
    n_t, n_o = is_target.sum(), (~is_target).sum()
    if n_t == 0 or n_o == 0:
        return set()
    p_t = masks[is_target].astype(np.float64).mean(axis=0)
    p_o = masks[~is_target].astype(np.float64).mean(axis=0)
    p_pool = masks.astype(np.float64).mean(axis=0)
    se = np.sqrt(p_pool * (1 - p_pool) * (1.0/n_t + 1.0/n_o))
    se = np.where(se > 0, se, 1.0)
    z = (p_t - p_o) / se
    p_vals = 1.0 - stats.norm.cdf(z)
    return set(np.where((p_vals < alpha) & (p_t > p_o))[0].tolist())


# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS (per threshold)
# ─────────────────────────────────────────────────────────────────────────────

def _consistent_counts_at_threshold(
    ffn_acts: Dict[int, np.ndarray], emotions: np.ndarray, pct: float,
) -> Dict[str, int]:
    """Total consistent neurons (freq > 50%) per emotion across all layers."""
    counts = {e: 0 for e in EMOTION_ORDER}
    for layer_idx in TARGET_LAYERS:
        masks = _binary_masks(ffn_acts[layer_idx], pct)
        for emotion in EMOTION_ORDER:
            is_target = emotions == emotion
            freq = masks[is_target].astype(np.float32).mean(axis=0)
            counts[emotion] += int((freq > 0.5).sum())
    return counts


def _classifier_accuracy_at_threshold(
    ffn_acts: Dict[int, np.ndarray], emotions: np.ndarray, pct: float,
) -> float:
    """Cross-validated logistic regression accuracy."""
    parts = [_binary_masks(ffn_acts[l], pct).astype(np.float32) for l in TARGET_LAYERS]
    X = np.concatenate(parts, axis=1)
    le = LabelEncoder()
    y = le.fit_transform(emotions)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(solver="lbfgs", max_iter=200, tol=1e-3,
                                 random_state=RANDOM_SEED, verbose=1)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    return float(np.mean(accs))


def _mean_jaccard_at_threshold(
    ffn_acts: Dict[int, np.ndarray], emotions: np.ndarray, pct: float,
) -> float:
    """Mean pairwise Jaccard of aggregated signature sets."""
    from itertools import combinations

    sig_sets = {e: set() for e in EMOTION_ORDER}
    for layer_idx in TARGET_LAYERS:
        masks = _binary_masks(ffn_acts[layer_idx], pct)
        for emotion in EMOTION_ORDER:
            sigs = _signature_set(masks, emotions, emotion)
            sig_sets[emotion] |= {(layer_idx, n) for n in sigs}

    jaccards = []
    for e1, e2 in combinations(EMOTION_ORDER, 2):
        union = len(sig_sets[e1] | sig_sets[e2])
        inter = len(sig_sets[e1] & sig_sets[e2])
        jaccards.append(inter / union if union > 0 else 0.0)
    return float(np.mean(jaccards))


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _plot_consistent_vs_threshold(
    results: Dict[float, Dict[str, int]], out_dir: Path, logger: logging.Logger,
):
    fig, ax = new_figure(figsize=(12, 7))
    for emotion in EMOTION_ORDER:
        vals = [results[t][emotion] for t in SENSITIVITY_THRESHOLDS]
        ax.plot(SENSITIVITY_THRESHOLDS, vals,
                color=EMOTION_STYLE[emotion]["color"],
                marker=EMOTION_STYLE[emotion]["marker"],
                label=EMOTION_STYLE[emotion]["label"],
                linewidth=2, markersize=8)
    style_axis(ax,
               "Consistent neuron count vs activation threshold\n"
               "(Neurons with freq > 50% across all layers)",
               "Top-K threshold (%)", "Total consistent neurons")
    ax.legend(fontsize=10, title="Emotion", title_fontsize=10)
    plt.tight_layout()
    save_figure(fig, out_dir / "consistent_vs_threshold.png", logger)


def _plot_accuracy_vs_threshold(
    results: Dict[float, float], out_dir: Path, logger: logging.Logger,
):
    fig, ax = new_figure(figsize=(10, 7))
    thresholds = list(results.keys())
    accs = list(results.values())
    ax.plot(thresholds, accs, "o-", color="#2C3E50", linewidth=2.5, markersize=10)
    for t, a in zip(thresholds, accs):
        ax.text(t, a + 0.005, f"{a:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    style_axis(ax,
               "Classifier accuracy vs activation threshold\n"
               f"({CV_FOLDS}-fold CV, Logistic Regression on binary FFN masks)",
               "Top-K threshold (%)", "Accuracy")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_figure(fig, out_dir / "accuracy_vs_threshold.png", logger)


def _plot_jaccard_vs_threshold(
    results: Dict[float, float], out_dir: Path, logger: logging.Logger,
):
    fig, ax = new_figure(figsize=(10, 7))
    thresholds = list(results.keys())
    jacs = list(results.values())
    ax.plot(thresholds, jacs, "s-", color="#E67E22", linewidth=2.5, markersize=10)
    for t, j in zip(thresholds, jacs):
        ax.text(t, j + 0.005, f"{j:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    style_axis(ax,
               "Mean pairwise Jaccard overlap vs activation threshold\n"
               "(Aggregated signature neurons across layers 14–22)",
               "Top-K threshold (%)", "Mean pairwise Jaccard")
    plt.tight_layout()
    save_figure(fig, out_dir / "jaccard_vs_threshold.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SENSITIVITY RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity(
    ffn_acts: Dict[int, np.ndarray],
    emotions: np.ndarray,
    out_dir: Path,
    logger: logging.Logger,
):
    """Run sensitivity analysis across multiple thresholds."""
    logger.info("=" * 60)
    logger.info("Sensitivity analysis across thresholds: "
                f"{SENSITIVITY_THRESHOLDS}")
    logger.info("=" * 60)

    summary_lines = ["Sensitivity Analysis", "=" * 50, ""]

    # ── 1. Consistent neuron counts ───────────────────────────────────────
    logger.info("Metric 1: Consistent neuron counts")
    consistent_results = {}
    for pct in SENSITIVITY_THRESHOLDS:
        logger.info(f"  Threshold {pct}% …")
        consistent_results[pct] = _consistent_counts_at_threshold(
            ffn_acts, emotions, pct
        )
        for e in EMOTION_ORDER:
            logger.info(f"    {e}: {consistent_results[pct][e]}")
    _plot_consistent_vs_threshold(consistent_results, out_dir, logger)

    summary_lines.append("Consistent neurons per threshold:")
    for pct in SENSITIVITY_THRESHOLDS:
        summary_lines.append(f"  {pct}%: " + ", ".join(
            f"{e}={consistent_results[pct][e]}" for e in EMOTION_ORDER
        ))
    summary_lines.append("")

    # ── 2. Classifier accuracy ────────────────────────────────────────────
    logger.info("Metric 2: Classifier accuracy")
    acc_results = {}
    for pct in SENSITIVITY_THRESHOLDS:
        logger.info(f"  Threshold {pct}% …")
        acc = _classifier_accuracy_at_threshold(ffn_acts, emotions, pct)
        acc_results[pct] = acc
        logger.info(f"    Accuracy: {acc:.4f}")
    _plot_accuracy_vs_threshold(acc_results, out_dir, logger)

    summary_lines.append("Classifier accuracy per threshold:")
    for pct, acc in acc_results.items():
        summary_lines.append(f"  {pct}%: {acc:.4f}")
    summary_lines.append("")

    # ── 3. Mean pairwise Jaccard ──────────────────────────────────────────
    logger.info("Metric 3: Mean pairwise Jaccard")
    jac_results = {}
    for pct in SENSITIVITY_THRESHOLDS:
        logger.info(f"  Threshold {pct}% …")
        jac = _mean_jaccard_at_threshold(ffn_acts, emotions, pct)
        jac_results[pct] = jac
        logger.info(f"    Mean Jaccard: {jac:.4f}")
    _plot_jaccard_vs_threshold(jac_results, out_dir, logger)

    summary_lines.append("Mean pairwise Jaccard per threshold:")
    for pct, jac in jac_results.items():
        summary_lines.append(f"  {pct}%: {jac:.4f}")
    summary_lines.append("")

    # ── Save summary ──────────────────────────────────────────────────────
    summary_path = out_dir / "sensitivity_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    logger.info(f"Sensitivity summary saved: {summary_path}")
    logger.info("Sensitivity analysis complete.")
