"""
utils.py — Shared utilities: dual logging, device selection, plot helpers.
"""

import logging
import sys
import os
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import (
    EMOTION_STYLE, EMOTION_ORDER, FIGURE_DPI, FIGURE_FORMAT,
    RESULTS_ROOT, CACHE_DIR, RQ_DIRS, LOG_FNAME,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING (dual: terminal + file)
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(base_dir: Path) -> logging.Logger:
    """Create a logger that writes to both stdout and a log file."""
    log_path = base_dir / LOG_FNAME
    logger = logging.getLogger("emotion_neuron")
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers on re-init
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(base_dir: Path) -> dict:
    """Create all output directories. Returns dict of Path objects."""
    dirs = {}
    results_dir = base_dir / RESULTS_ROOT
    results_dir.mkdir(parents=True, exist_ok=True)
    dirs["results"] = results_dir

    cache_dir = base_dir / CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    dirs["cache"] = cache_dir

    for key, name in RQ_DIRS.items():
        d = results_dir / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[key] = d

    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SELECTION (CUDA → MPS → CPU)
# ─────────────────────────────────────────────────────────────────────────────

def select_device(logger: logging.Logger):
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Hardware: CUDA — {name} ({mem:.1f} GB VRAM)")
        return dev
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Hardware: Apple MPS (M-series GPU)")
        return torch.device("mps")
    logger.info("Hardware: CPU (will be slow)")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def style_axis(ax, title: str, xlabel: str = "", ylabel: str = ""):
    """Apply consistent styling to a matplotlib axis."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)


def emotion_legend(ax, loc="best"):
    """Add a standardised emotion legend."""
    handles = [
        mpatches.Patch(color=EMOTION_STYLE[e]["color"], label=EMOTION_STYLE[e]["label"])
        for e in EMOTION_ORDER
        if e in EMOTION_STYLE
    ]
    ax.legend(
        handles=handles, title="Primary Emotion",
        title_fontsize=10, fontsize=9, framealpha=0.9, loc=loc,
    )


def save_figure(fig, path: Path, logger: logging.Logger):
    """Save figure at configured DPI and log the path."""
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved plot: {path}")


def new_figure(figsize=(12, 8)):
    """Create a figure with consistent background."""
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")
    return fig, ax


def scatter_by_emotion(
    coords: np.ndarray,
    emotions: List[str],
    title: str,
    save_path: Path,
    logger: logging.Logger,
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
):
    """2-D scatter coloured by emotion with consistent styling."""
    fig, ax = new_figure(figsize=(12, 9))
    for emotion in EMOTION_ORDER:
        style = EMOTION_STYLE[emotion]
        mask = np.array([e == emotion for e in emotions])
        if not mask.any():
            continue
        pts = coords[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=style["color"], marker=style["marker"], label=style["label"],
            alpha=0.6, s=18, linewidths=0.3, edgecolors="white",
        )
    style_axis(ax, title, xlabel, ylabel)
    emotion_legend(ax)
    save_figure(fig, save_path, logger)


def strip_brackets(text: str) -> str:
    """Remove leading/trailing [] wrappers from the translation pipeline."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text
