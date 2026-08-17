"""
preprocessing.py — Load, deduplicate, and validate the emotion CSV dataset.
"""

import logging
from pathlib import Path

import pandas as pd

from config import EMOTION_ORDER


def load_and_clean(csv_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load the emotion CSV, deduplicate, strip bracket wrappers, and validate.

    Returns a clean DataFrame with columns:
      chinese_text, primary_emotion, secondary_emotions, english_text
    and a sequential integer index.
    """
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")

    original_len = len(df)
    logger.info(f"  Raw rows          : {original_len}")

    # ── Deduplicate on (chinese_text, primary_emotion) ────────────────────
    df = df.drop_duplicates(subset=["chinese_text", "primary_emotion"], keep="first")
    deduped_len = len(df)
    removed = original_len - deduped_len
    logger.info(f"  Duplicates removed : {removed}")
    logger.info(f"  Rows after dedup   : {deduped_len}")

    # ── Normalise primary_emotion to uppercase ────────────────────────────
    df["primary_emotion"] = df["primary_emotion"].str.strip().str.upper()

    # ── Keep only valid Ekman emotions ────────────────────────────────────
    valid_mask = df["primary_emotion"].isin(EMOTION_ORDER)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"  Dropping {invalid_count} rows with unknown emotions")
    df = df[valid_mask].copy()

    # ── Drop rows with empty chinese_text ─────────────────────────────────
    empty_text = df["chinese_text"].str.strip().eq("")
    if empty_text.any():
        logger.warning(f"  Dropping {empty_text.sum()} rows with empty text")
        df = df[~empty_text].copy()

    # ── Reset index ──────────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    # ── Log final distribution ────────────────────────────────────────────
    logger.info(f"  Final dataset size : {len(df)}")
    logger.info("  Emotion distribution:")
    for emotion in EMOTION_ORDER:
        count = (df["primary_emotion"] == emotion).sum()
        pct = 100.0 * count / len(df)
        logger.info(f"    {emotion:<12s} : {count:>5d}  ({pct:5.1f}%)")

    return df
