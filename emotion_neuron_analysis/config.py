"""
config.py — Central configuration for the Emotion Neuron Analysis pipeline.
All tunable parameters live here. Nothing else imports torch or heavy libs.
"""

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LAYERS    = list(range(14, 23))       # layers 14-22 inclusive (0-indexed)
FFN_DIM          = 8960                      # SwiGLU intermediate size
HIDDEN_DIM       = 1536                      # residual stream width
MAX_SEQ_LEN      = 512                       # tokeniser truncation limit

# ─── Extraction ──────────────────────────────────────────────────────────────
BATCH_SIZE       = 8                         # rows per forward pass (RTX 2080 Ti)
TOP_K_PERCENT    = 5.0                       # default threshold for "activated"
SENSITIVITY_THRESHOLDS = [1.0, 3.0, 5.0, 10.0]   # for sensitivity analysis

# ─── Checkpointing ──────────────────────────────────────────────────────────
CHECKPOINT_FNAME     = "extraction_checkpoint.json"
CSV_FLUSH_INTERVAL   = 50                    # write CSV every N rows

# ─── Output ──────────────────────────────────────────────────────────────────
RESULTS_ROOT         = "results"
CACHE_DIR            = "cache"
LOG_FNAME            = "analysis.log"
FIGURE_DPI           = 300
FIGURE_FORMAT        = "png"

# Sub-folders under RESULTS_ROOT
RQ_DIRS = {
    "rq1": "rq1_neuron_frequency",
    "rq2": "rq2_emotion_specificity",
    "rq3": "rq3_cluster_enumeration",
    "rq4": "rq4_overlap_analysis",
    "sensitivity": "sensitivity_analysis",
}

# ─── Emotion palette ────────────────────────────────────────────────────────
EMOTION_STYLE = {
    "ANGER":     {"color": "#E24B4A", "marker": "o", "label": "Anger"},
    "DISGUST":   {"color": "#854F0B", "marker": "s", "label": "Disgust"},
    "FEAR":      {"color": "#534AB7", "marker": "^", "label": "Fear"},
    "HAPPINESS": {"color": "#3B6D11", "marker": "D", "label": "Happiness"},
    "SADNESS":   {"color": "#185FA5", "marker": "v", "label": "Sadness"},
    "SURPRISE":  {"color": "#D4537E", "marker": "P", "label": "Surprise"},
}

EMOTION_ORDER = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE"]

# ─── Statistical ─────────────────────────────────────────────────────────────
SIGNIFICANCE_ALPHA   = 0.05
N_PERMUTATIONS       = 1000       # for permutation tests
SPLIT_HALF_REPEATS   = 100        # for split-half reliability
CV_FOLDS             = 5          # for classifier cross-validation
RANDOM_SEED          = 42
