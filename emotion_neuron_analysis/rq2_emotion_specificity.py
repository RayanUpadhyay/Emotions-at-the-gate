"""
rq2_emotion_specificity.py
===========================
RQ2: "Is that specific set the same for specific types of primary emotions?"

Analyses
--------
1. Signature neuron identification: neurons whose activation frequency for a
   given emotion is significantly higher than for all other emotions combined.
2. Split-half reliability: randomly split each emotion's samples, compute
   signature sets on each half, measure Jaccard overlap.  Repeat N times.
3. Cross-validated classifier: logistic regression on binary FFN masks →
   if high accuracy, the neuron signatures are discriminative and consistent.
4. Per-emotion classification metrics + confusion matrix.

Outputs  →  results/rq2_emotion_specificity/
  - signature_neuron_counts.png
  - split_half_jaccard.png
  - classifier_confusion_matrix.png
  - classifier_accuracy_per_emotion.png
  - rq2_summary.txt
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TARGET_LAYERS, FFN_DIM, EMOTION_ORDER, EMOTION_STYLE,
    TOP_K_PERCENT, SIGNIFICANCE_ALPHA, SPLIT_HALF_REPEATS,
    CV_FOLDS, RANDOM_SEED,
)
from utils import new_figure, save_figure, style_axis


def _binary_masks(ffn_layer: np.ndarray, top_k_pct: float) -> np.ndarray:
    k = max(1, int(FFN_DIM * top_k_pct / 100.0))
    abs_vals = np.abs(ffn_layer.astype(np.float32))
    thresholds = np.partition(abs_vals, -k, axis=1)[:, -k]
    return abs_vals >= thresholds[:, None]


def _signature_neurons(
    masks: np.ndarray, emotions: np.ndarray, emotion: str,
    alpha: float = SIGNIFICANCE_ALPHA,
) -> np.ndarray:
    """
    Identify signature neurons for `emotion`: neurons whose activation rate
    in samples of `emotion` is significantly higher (one-sided proportion
    z-test) than in all other samples.

    Returns boolean array [FFN_DIM] — True for signature neurons.
    """
    is_target = emotions == emotion
    n_t = is_target.sum()
    n_o = (~is_target).sum()
    if n_t == 0 or n_o == 0:
        return np.zeros(masks.shape[1], dtype=bool)

    p_t = masks[is_target].astype(np.float64).mean(axis=0)   # freq in emotion
    p_o = masks[~is_target].astype(np.float64).mean(axis=0)  # freq in others
    p_pool = masks.astype(np.float64).mean(axis=0)            # pooled freq

    # Z-test for two proportions (one-sided: p_t > p_o)
    se = np.sqrt(p_pool * (1 - p_pool) * (1.0/n_t + 1.0/n_o))
    se = np.where(se > 0, se, 1.0)  # avoid division by zero
    z = (p_t - p_o) / se
    p_vals = 1.0 - stats.norm.cdf(z)

    return (p_vals < alpha) & (p_t > p_o)


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT-HALF RELIABILITY
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between two boolean arrays."""
    intersection = (a & b).sum()
    union = (a | b).sum()
    return intersection / union if union > 0 else 0.0


def _split_half_reliability(
    masks: np.ndarray, emotions: np.ndarray,
    n_repeats: int = SPLIT_HALF_REPEATS, seed: int = RANDOM_SEED,
) -> Dict[str, List[float]]:
    """
    For each emotion, randomly split its samples into two halves,
    compute signature neurons on each half, measure Jaccard overlap.
    Repeat n_repeats times.

    Returns {emotion: [jaccard_scores]}.
    """
    rng = np.random.RandomState(seed)
    results = {e: [] for e in EMOTION_ORDER}

    for _ in range(n_repeats):
        for emotion in EMOTION_ORDER:
            target_idx = np.where(emotions == emotion)[0]
            if len(target_idx) < 10:
                results[emotion].append(0.0)
                continue

            rng.shuffle(target_idx)
            mid = len(target_idx) // 2
            half_a = target_idx[:mid]
            half_b = target_idx[mid:]

            # Build subsets — keep all "other" samples in both
            other_idx = np.where(emotions != emotion)[0]
            idx_a = np.concatenate([half_a, other_idx])
            idx_b = np.concatenate([half_b, other_idx])

            # Labels for subsets
            emo_a = emotions[idx_a]
            emo_b = emotions[idx_b]

            sig_a = _signature_neurons(masks[idx_a], emo_a, emotion)
            sig_b = _signature_neurons(masks[idx_b], emo_b, emotion)

            results[emotion].append(_jaccard(sig_a, sig_b))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def _run_classifier(
    ffn_acts: Dict[int, np.ndarray],
    emotions: np.ndarray,
    top_k_pct: float,
    n_folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Train a logistic regression classifier on concatenated binary FFN masks
    across all target layers.  Uses stratified k-fold CV.

    Returns (y_true_all, y_pred_all, metrics_dict).
    """
    # Build feature matrix: concatenate binary masks across layers
    feature_parts = []
    for l in TARGET_LAYERS:
        masks = _binary_masks(ffn_acts[l], top_k_pct)
        feature_parts.append(masks.astype(np.float32))
    X = np.concatenate(feature_parts, axis=1)  # [N, 9*FFN_DIM]

    le = LabelEncoder()
    y = le.fit_transform(emotions)

    if logger:
        logger.info(f"  Classifier feature matrix: {X.shape}")
        logger.info(f"  Classes: {le.classes_.tolist()}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(
            solver="lbfgs", max_iter=200, tol=1e-3,
            random_state=seed, C=1.0, verbose=1,
        )
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(preds)

        acc = accuracy_score(y[test_idx], preds)
        if logger:
            logger.info(f"  Fold {fold+1}/{n_folds}: accuracy = {acc:.4f}")

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    report = classification_report(
        y_true_all, y_pred_all, target_names=le.classes_, output_dict=True
    )

    metrics = {
        "overall_accuracy": overall_acc,
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }
            for cls in le.classes_
        },
        "label_encoder_classes": le.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
    }

    return y_true_all, y_pred_all, metrics, le


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _plot_signature_counts(
    sig_counts: Dict[str, Dict[int, int]],
    out_dir: Path, logger: logging.Logger,
):
    """Heatmap: number of signature neurons per emotion × layer."""
    matrix = np.zeros((len(EMOTION_ORDER), len(TARGET_LAYERS)), dtype=int)
    for i, emotion in enumerate(EMOTION_ORDER):
        for j, layer in enumerate(TARGET_LAYERS):
            matrix[i, j] = sig_counts[emotion].get(layer, 0)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    sns.heatmap(
        matrix,
        xticklabels=[f"L{l}" for l in TARGET_LAYERS],
        yticklabels=EMOTION_ORDER,
        annot=True, fmt="d", cmap="PuBuGn",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Signature neurons"},
    )
    style_axis(ax, "Signature FFN neurons per emotion × layer (p < 0.05, one-sided z-test)",
               "Layer", "Primary emotion")
    plt.tight_layout()
    save_figure(fig, out_dir / "signature_neuron_counts.png", logger)


def _plot_split_half(
    jaccard_results: Dict[str, List[float]],
    out_dir: Path, logger: logging.Logger,
):
    """Box plot of split-half Jaccard scores per emotion."""
    fig, ax = new_figure(figsize=(12, 7))

    data = []
    labels = []
    for emotion in EMOTION_ORDER:
        scores = jaccard_results[emotion]
        data.append(scores)
        labels.append(EMOTION_STYLE[emotion]["label"])

    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markersize=6),
    )
    for i, emotion in enumerate(EMOTION_ORDER):
        bp["boxes"][i].set_facecolor(EMOTION_STYLE[emotion]["color"])
        bp["boxes"][i].set_alpha(0.7)

    style_axis(
        ax,
        f"Split-half reliability of signature neurons ({SPLIT_HALF_REPEATS} repeats)\n"
        f"Jaccard similarity between signature sets from two random halves",
        "Primary emotion", "Jaccard similarity",
    )
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    save_figure(fig, out_dir / "split_half_jaccard.png", logger)


def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, le,
    out_dir: Path, logger: logging.Logger,
):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(
        cm_norm, xticklabels=le.classes_, yticklabels=le.classes_,
        annot=True, fmt=".2f", cmap="Blues",
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
        cbar_kws={"label": "Normalised frequency"},
    )
    style_axis(ax,
               f"Classifier confusion matrix ({CV_FOLDS}-fold CV)\n"
               "Logistic regression on binary FFN masks (all layers)",
               "Predicted", "True")
    plt.tight_layout()
    save_figure(fig, out_dir / "classifier_confusion_matrix.png", logger)


def _plot_per_class_f1(
    metrics: dict, out_dir: Path, logger: logging.Logger,
):
    """Bar chart of per-class F1 scores."""
    fig, ax = new_figure(figsize=(12, 7))

    f1_scores = [metrics["per_class"][e]["f1"] for e in EMOTION_ORDER]
    colors = [EMOTION_STYLE[e]["color"] for e in EMOTION_ORDER]
    labels = [EMOTION_STYLE[e]["label"] for e in EMOTION_ORDER]

    bars = ax.bar(labels, f1_scores, color=colors, edgecolor="white", linewidth=0.5)
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(metrics["overall_accuracy"], color="black", linestyle="--",
               linewidth=1, alpha=0.5, label=f"Overall acc: {metrics['overall_accuracy']:.3f}")
    style_axis(ax,
               f"Per-emotion F1 score ({CV_FOLDS}-fold CV, Logistic Regression)",
               "Primary emotion", "F1 score")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    save_figure(fig, out_dir / "classifier_accuracy_per_emotion.png", logger)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RQ2 RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_rq2(
    ffn_acts: Dict[int, np.ndarray],
    emotions: np.ndarray,
    out_dir: Path,
    logger: logging.Logger,
    top_k_pct: float = TOP_K_PERCENT,
):
    """Execute all RQ2 analyses."""
    logger.info("=" * 60)
    logger.info("RQ2: Emotion specificity — signature neurons and consistency")
    logger.info("=" * 60)

    summary_lines = ["RQ2: Emotion Specificity Analysis", "=" * 50, ""]

    # ── 1. Signature neuron identification ────────────────────────────────
    logger.info("Step 1: Identifying signature neurons per emotion per layer")
    sig_counts = {e: {} for e in EMOTION_ORDER}

    for layer_idx in TARGET_LAYERS:
        masks = _binary_masks(ffn_acts[layer_idx], top_k_pct)
        for emotion in EMOTION_ORDER:
            sig = _signature_neurons(masks, emotions, emotion)
            sig_counts[emotion][layer_idx] = int(sig.sum())
            logger.info(f"  L{layer_idx} {emotion}: {sig.sum()} signature neurons")

    _plot_signature_counts(sig_counts, out_dir, logger)

    summary_lines.append("Signature neuron counts per emotion × layer:")
    for emotion in EMOTION_ORDER:
        total = sum(sig_counts[emotion].values())
        summary_lines.append(f"  {emotion}: total = {total} across all layers")
        for l in TARGET_LAYERS:
            summary_lines.append(f"    L{l}: {sig_counts[emotion][l]}")
    summary_lines.append("")

    # ── 2. Split-half reliability (use middle layer for tractability) ─────
    mid_layer = TARGET_LAYERS[len(TARGET_LAYERS) // 2]
    logger.info(f"Step 2: Split-half reliability (layer {mid_layer})")
    masks_mid = _binary_masks(ffn_acts[mid_layer], top_k_pct)
    jaccard_results = _split_half_reliability(masks_mid, emotions)
    _plot_split_half(jaccard_results, out_dir, logger)

    summary_lines.append(f"Split-half reliability (Layer {mid_layer}, "
                         f"{SPLIT_HALF_REPEATS} repeats):")
    for emotion in EMOTION_ORDER:
        scores = jaccard_results[emotion]
        mean_j = np.mean(scores)
        std_j  = np.std(scores)
        logger.info(f"  {emotion}: Jaccard = {mean_j:.4f} +/- {std_j:.4f}")
        summary_lines.append(f"  {emotion}: {mean_j:.4f} +/- {std_j:.4f}")
    summary_lines.append("")

    # ── 3. Classifier ─────────────────────────────────────────────────────
    logger.info("Step 3: Cross-validated logistic regression classifier")
    y_true, y_pred, metrics, le = _run_classifier(
        ffn_acts, emotions, top_k_pct, logger=logger
    )
    _plot_confusion_matrix(y_true, y_pred, le, out_dir, logger)
    _plot_per_class_f1(metrics, out_dir, logger)

    logger.info(f"  Overall accuracy: {metrics['overall_accuracy']:.4f}")
    summary_lines.append(f"Classifier ({CV_FOLDS}-fold CV):")
    summary_lines.append(f"  Overall accuracy: {metrics['overall_accuracy']:.4f}")
    for emotion in EMOTION_ORDER:
        m = metrics["per_class"][emotion]
        summary_lines.append(
            f"  {emotion}: P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} (n={m['support']})"
        )
    summary_lines.append("")

    # ── Save summary ──────────────────────────────────────────────────────
    summary_path = out_dir / "rq2_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    logger.info(f"RQ2 summary saved: {summary_path}")
    logger.info("RQ2 complete.")
