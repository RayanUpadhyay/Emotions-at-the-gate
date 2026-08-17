"""
main.py — Orchestrator for the Emotion Neuron Analysis pipeline.
============================================================

Usage
-----
  # Full pipeline (extraction + all analyses)
  python main.py --csv /path/to/emotion_dataset_translated.csv

  # Skip extraction (use cached activations)
  python main.py --csv /path/to/data.csv --skip-extraction

  # Run specific RQs only
  python main.py --csv /path/to/data.csv --skip-extraction --rq 1 3

  # Extraction only, no analysis
  python main.py --csv /path/to/data.csv --skip-analysis

  # Custom batch size
  python main.py --csv /path/to/data.csv --batch-size 16
"""

import argparse
import sys
import time
from pathlib import Path

from config import (
    TARGET_LAYERS, FFN_DIM, HIDDEN_DIM, TOP_K_PERCENT,
    BATCH_SIZE, SENSITIVITY_THRESHOLDS, MODEL_NAME,
)
from utils import setup_logging, ensure_dirs, select_device
from preprocessing import load_and_clean
from extraction import load_model_and_tokenizer, extract_and_cache
from rq1_neuron_frequency import run_rq1
from rq2_emotion_specificity import run_rq2
from rq3_cluster_enumeration import run_rq3
from rq4_overlap_analysis import run_rq4
from sensitivity import run_sensitivity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Emotion Neuron Activation Analysis — full pipeline"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to emotion_dataset_translated.csv"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Base output directory (default: same folder as CSV)"
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip model inference; use cached .npy activations"
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Run extraction only, skip all RQ analyses"
    )
    parser.add_argument(
        "--rq", nargs="+", type=int, default=None,
        help="Run only specific RQs, e.g. --rq 1 3 (default: all)"
    )
    parser.add_argument(
        "--skip-sensitivity", action="store_true",
        help="Skip the threshold sensitivity analysis"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size for inference (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--top-k-pct", type=float, default=TOP_K_PERCENT,
        help=f"Top-K percent threshold (default: {TOP_K_PERCENT})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv).resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    base_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent
    dirs = ensure_dirs(base_dir)
    logger = setup_logging(base_dir)

    t_start = time.time()

    logger.info("=" * 70)
    logger.info("  EMOTION NEURON ACTIVATION ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"CSV            : {csv_path}")
    logger.info(f"Output dir     : {base_dir}")
    logger.info(f"Model          : {MODEL_NAME}")
    logger.info(f"Target layers  : {TARGET_LAYERS}")
    logger.info(f"FFN dim        : {FFN_DIM}")
    logger.info(f"Hidden dim     : {HIDDEN_DIM}")
    logger.info(f"Top-K pct      : {args.top_k_pct}%")
    logger.info(f"Batch size     : {args.batch_size}")
    logger.info(f"Thresholds     : {SENSITIVITY_THRESHOLDS}")
    logger.info("=" * 70)

    # ── PHASE 0: Preprocessing ────────────────────────────────────────────
    df = load_and_clean(csv_path, logger)

    # ── PHASE 1: Model + Extraction ───────────────────────────────────────
    device = select_device(logger)
    model, tokenizer = load_model_and_tokenizer(device, logger)

    if not args.skip_extraction:
        ffn_acts, res_acts, emotions = extract_and_cache(
            df, model, tokenizer, device, dirs["cache"], logger,
            batch_size=args.batch_size,
        )
    else:
        logger.info("Skipping extraction (--skip-extraction)")
        from extraction import _load_cache
        ffn_acts, res_acts, emotions = _load_cache(dirs["cache"], logger)

    # Free GPU memory after extraction
    del model, tokenizer
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    logger.info("Model unloaded, GPU memory freed.")

    if args.skip_analysis:
        logger.info("Skipping analysis (--skip-analysis)")
        logger.info(f"Total time: {time.time() - t_start:.1f}s")
        return

    # ── PHASE 2: RQ Analyses ──────────────────────────────────────────────
    run_rqs = args.rq if args.rq else [1, 2, 3, 4]

    if 1 in run_rqs:
        run_rq1(ffn_acts, emotions, dirs["rq1"], logger, args.top_k_pct)

    if 2 in run_rqs:
        run_rq2(ffn_acts, emotions, dirs["rq2"], logger, args.top_k_pct)

    if 3 in run_rqs:
        run_rq3(ffn_acts, res_acts, emotions, dirs["rq3"], logger, args.top_k_pct)

    if 4 in run_rqs:
        run_rq4(ffn_acts, emotions, dirs["rq4"], logger, args.top_k_pct)

    # ── PHASE 3: Sensitivity ──────────────────────────────────────────────
    if not args.skip_sensitivity:
        run_sensitivity(ffn_acts, emotions, dirs["sensitivity"], logger)

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"  ALL DONE — total time: {elapsed/60:.1f} minutes")
    logger.info(f"  Results in: {dirs['results']}")
    logger.info(f"  Cache in  : {dirs['cache']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
