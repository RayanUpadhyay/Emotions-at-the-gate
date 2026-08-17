"""
Chinese Emotion Classification Pipeline
========================================
Reads two CSV files containing Chinese text, combines them into 20,000 rows,
then uses the GLM-4.5 model via Z.ai API to classify each row into
primary and secondary emotions based on the Emotion Wheel (Paul Ekman).

For rows that already have an emotion label in the CSV, the label is checked
against the Emotion Wheel. If it maps directly to a valid primary emotion,
that primary is used and only secondary emotions are requested from the LLM.
If the label doesn't map, the LLM classifies from scratch.

Output CSV columns:
  1. chinese_text  -> Chinese text enclosed in []
  2. primary_emotion -> Single best-fitting primary emotion
  3. secondary_emotions -> Semicolon-separated secondary emotions enclosed in []

Usage:
  python emotion_classifier.py

Configuration:
  - Set API_KEY below before running.
  - Adjust INPUT_FILE_1, INPUT_FILE_2, OUTPUT_FILE, CHECKPOINT_FILE paths if needed.
"""

import csv
import json
import re
import time
import os
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION  ← edit .env for API_KEY
# ─────────────────────────────────────────────
API_KEY        = os.getenv("GLM_API_KEY", "")
if not API_KEY:
    raise RuntimeError("GLM_API_KEY not found. Set it in .env file.")
API_BASE_URL   = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL          = "glm-4.5"

INPUT_FILE_1   = "./chinese_sentiment_dataset_large 2.csv"
INPUT_FILE_2   = "./transcription.csv"
OUTPUT_FILE    = "emotion_dataset_output.csv"
CHECKPOINT_FILE = "checkpoint.json"
LOG_FILE       = "skipped_rows.log"

TOTAL_ROWS     = 20_000
BATCH_SIZE     = 20       # reduced from 100 to avoid API timeouts
BATCH_DELAY    = 3        # seconds between batches
MAX_RETRIES    = 3
RETRY_DELAY    = 10       # seconds between retries
API_TIMEOUT    = 120      # seconds per API call

# ─────────────────────────────────────────────
# EMOTION WHEEL  (hardcoded from emotion_wheel.docx)
# ─────────────────────────────────────────────
EMOTION_WHEEL = {
    "ANGER": [
        "Hostile", "Mad", "Aggressive", "Frustrated", "Critical",
        "Irritated", "Hateful", "Threatened", "Hurt", "Rage",
        "Offended", "Jealous", "Disgusted", "Provoked", "Withdrawn",
    ],
    "DISGUST": [
        "Disapproval", "Disappointed", "Awful", "Avoidance",
        "Contempt", "Revulsion", "Loathing",
    ],
    "FEAR": [
        "Scared", "Anxious", "Insecure", "Submissive", "Rejected",
        "Humiliated", "Confused", "Embarrassed", "Startled", "Exposed",
        "Paranoid", "Stressed",
    ],
    "HAPPINESS": [
        "Joyful", "Excited", "Content", "Optimistic", "Proud",
        "Peaceful", "Powerful", "Playful", "Loving", "Grateful",
        "Creative", "Accepted", "Interested", "Trusting", "Strong",
        "Blissful", "Mellow",
    ],
    "SADNESS": [
        "Lonely", "Depressed", "Guilty", "Hurt", "Bored",
        "Disappointed", "Despair", "Rejected", "Betrayed", "Pensive", "Tired",
    ],
    "SURPRISE": [
        "Startled", "Confused", "Amazed", "Excited", "Moved",
        "Overwhelmed", "Curious",
    ],
}

# Map CSV emotion labels (from both files) to Emotion Wheel primaries
CSV_EMOTION_TO_PRIMARY = {
    # File 1 (emotion_type column)
    "joy":       "HAPPINESS",
    "disgust":   "DISGUST",
    "surprise":  "SURPRISE",
    "anger":     "ANGER",
    "fear":      "FEAR",
    "sadness":   "SADNESS",
    # File 2 (emotion column)
    "angry":     "ANGER",
    "disgusted": "DISGUST",
    "fearful":   "FEAR",
    "happy":     "HAPPINESS",
    "sad":       "SADNESS",
    "surprised": "SURPRISE",
    # "neutral" is intentionally NOT mapped — those go to the LLM
}

VALID_PRIMARIES = set(EMOTION_WHEEL.keys())

EMOTION_WHEEL_PROMPT_STR = "\n".join(
    f"  {primary}: {', '.join(secondaries)}"
    for primary, secondaries in EMOTION_WHEEL.items()
)

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

logger.addHandler(file_handler)


def log(msg: str, level: str = "info"):
    """Print to terminal AND log to file."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)
    getattr(logger, level)(msg)


# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are an expert multilingual emotion analyst specialising in Chinese text.
Your task is to classify Chinese sentences into emotions using the following Emotion Wheel, which is based on Paul Ekman's six primary emotions.

EMOTION WHEEL (Primary → Secondary options):
{EMOTION_WHEEL_PROMPT_STR}

RULES:
1. For each sentence, identify ONE primary emotion that best fits the context (choose from: ANGER, DISGUST, FEAR, HAPPINESS, SADNESS, SURPRISE).
2. Then identify one or more secondary emotions from the list belonging to that chosen primary.
3. If the text is neutral, factual, or emotionless and cannot be mapped to any primary emotion, return null for both fields.
4. When a secondary emotion could belong to multiple primaries (e.g. "Disappointed" appears under DISGUST and SADNESS), always choose the primary that best fits the full context of the sentence.
5. Return ONLY a valid JSON array — no markdown, no explanation, no extra text whatsoever.
6. Each element in the array corresponds to the input sentence at the same index.

OUTPUT FORMAT (strict JSON array):
[
  {{"primary": "HAPPINESS", "secondary": ["Joyful", "Excited"]}},
  {{"primary": "SADNESS",   "secondary": ["Lonely"]}},
  {{"primary": null,        "secondary": []}}
]
"""

def build_user_prompt(texts: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return (
        f"Classify the emotions in each of the following {len(texts)} Chinese sentences.\n"
        f"Return a JSON array with exactly {len(texts)} elements in the same order.\n\n"
        f"{numbered}"
    )


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def strip_markup(text: str) -> str:
    """Remove [over/]...[/over] and [interrupted] markup from transcription.csv."""
    text = re.sub(r'\[over/\]|\[/over\]|\[interrupted\]', '', text)
    return text.strip()


def map_csv_emotion(label: str) -> Optional[str]:
    """Map a CSV emotion label to an Emotion Wheel primary. Returns None if unmappable."""
    if not label:
        return None
    mapped = CSV_EMOTION_TO_PRIMARY.get(label.strip().lower())
    return mapped


def load_chinese_texts() -> tuple[list[str], list[Optional[str]]]:
    """
    Load and combine Chinese text from both CSVs.
    Returns (texts, existing_primaries) where existing_primaries[i] is
    the mapped Emotion Wheel primary if the CSV had a valid label, else None.
    """
    texts = []
    existing_primaries = []

    # ── File 1 ──────────────────────────────
    log(f"[LOAD] Opening File 1: {INPUT_FILE_1}")
    with open(INPUT_FILE_1, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        log(f"[LOAD] File 1 columns: {reader.fieldnames}")
        for row in reader:
            t = row.get("text", "").strip()
            if t:
                texts.append(t)
                csv_label = row.get("emotion_type", "")
                existing_primaries.append(map_csv_emotion(csv_label))
            if len(texts) >= TOTAL_ROWS:
                break

    log(f"[LOAD] File 1 done — {len(texts)} rows loaded")

    # ── File 2 ──────────────────────────────
    if len(texts) < TOTAL_ROWS:
        remaining = TOTAL_ROWS - len(texts)
        log(f"[LOAD] Opening File 2: {INPUT_FILE_2} (need {remaining} more rows)")
        with open(INPUT_FILE_2, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            log(f"[LOAD] File 2 columns: {reader.fieldnames}")
            file2_count = 0
            for row in reader:
                t = strip_markup(row.get("chinese", "").strip())
                if t:
                    texts.append(t)
                    csv_label = row.get("emotion", "")
                    existing_primaries.append(map_csv_emotion(csv_label))
                    file2_count += 1
                if len(texts) >= TOTAL_ROWS:
                    break
        log(f"[LOAD] File 2 done — {file2_count} rows loaded")
    else:
        log("[LOAD] File 1 already filled TOTAL_ROWS, skipping File 2")

    # Stats on pre-mapped emotions
    mapped_count = sum(1 for p in existing_primaries if p is not None)
    unmapped_count = len(existing_primaries) - mapped_count
    log(f"[LOAD] Combined total: {len(texts)} rows (cap: {TOTAL_ROWS})")
    log(f"[LOAD] Pre-mapped emotions: {mapped_count} | Need LLM classification: {unmapped_count}")

    return texts[:TOTAL_ROWS], existing_primaries[:TOTAL_ROWS]


def load_checkpoint() -> int:
    """Return the index of the next unprocessed batch (0-based row index)."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
            return data.get("next_row", 0)
    return 0


def save_checkpoint(next_row: int) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"next_row": next_row}, f)


def call_llm(texts: list[str]) -> Optional[list[dict]]:
    """
    Call GLM-4.5 via Z.ai API with retry logic.
    Returns a list of dicts: [{"primary": str|None, "secondary": [str]}, ...]
    Returns None if all retries exhausted.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(texts)},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"  [API] Attempt {attempt}/{MAX_RETRIES} — sending {len(texts)} texts to {MODEL}")
            response = requests.post(API_BASE_URL, headers=headers,
                                     json=payload, timeout=API_TIMEOUT)
            log(f"  [API] Response status: {response.status_code}")
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()

            content = re.sub(r"^```json\s*|^```\s*|```$", "", content, flags=re.MULTILINE).strip()

            parsed = json.loads(content)

            if not isinstance(parsed, list) or len(parsed) != len(texts):
                raise ValueError(
                    f"Expected list of {len(texts)}, got {type(parsed).__name__} "
                    f"of length {len(parsed) if isinstance(parsed, list) else '?'}"
                )
            log(f"  [API] Successfully parsed {len(parsed)} results")
            return parsed

        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            log(f"  [API] Attempt {attempt}/{MAX_RETRIES} failed: {e}", "warning")
            if attempt < MAX_RETRIES:
                log(f"  [API] Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

    log(f"  [API] All {MAX_RETRIES} retries exhausted. Skipping this batch.", "error")
    return None


def format_primary(primary: Optional[str]) -> str:
    if not primary:
        return ""
    return primary.upper()


def format_secondary(secondary: Optional[list]) -> str:
    if not secondary:
        return ""
    valid = [s for s in secondary if isinstance(s, str) and s.strip()]
    if not valid:
        return ""
    return "[" + "; ".join(valid) + "]"


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    log("=" * 60)
    log("  Chinese Emotion Classification Pipeline")
    log("=" * 60)

    # ── 1. Load all Chinese texts ────────────
    log("\n[STEP 1/5] Loading Chinese texts from CSV files...")
    all_texts, all_existing_primaries = load_chinese_texts()
    log(f"[STEP 1/5] Total texts to classify: {len(all_texts)}")
    if all_texts:
        log(f"[STEP 1/5] First text preview: {all_texts[0][:80]}...")

    # ── 2. Determine resume point ────────────
    log("\n[STEP 2/5] Checking for checkpoint...")
    start_row = load_checkpoint()
    if start_row > 0:
        log(f"[STEP 2/5] Checkpoint found — resuming from row {start_row}")
    else:
        log("[STEP 2/5] No checkpoint — starting fresh run")

    # ── 3. Open output CSV ───────────────────
    write_mode = "a" if start_row > 0 else "w"
    log(f"\n[STEP 3/5] Opening output file: {OUTPUT_FILE} (mode='{write_mode}')")
    out_f = open(OUTPUT_FILE, write_mode, encoding="utf-8", newline="")
    writer = csv.writer(out_f)

    if start_row == 0:
        writer.writerow(["chinese_text", "primary_emotion", "secondary_emotions"])
        log("[STEP 3/5] Wrote CSV header row")

    # ── 4. Process in batches ────────────────
    total = len(all_texts)
    batch_num = start_row // BATCH_SIZE + 1
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    log(f"\n[STEP 4/5] Starting batch processing — {total_batches} batches of {BATCH_SIZE}")
    log(f"[STEP 4/5] API timeout: {API_TIMEOUT}s | Retry delay: {RETRY_DELAY}s | Batch delay: {BATCH_DELAY}s")
    log("")

    for batch_start in range(start_row, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = all_texts[batch_start:batch_end]
        batch_primaries = all_existing_primaries[batch_start:batch_end]

        # Split: rows that need LLM vs rows with pre-mapped primaries
        llm_indices = []
        llm_texts = []
        pre_mapped = {}  # index_in_batch -> primary
        for i, (text, existing_p) in enumerate(zip(batch_texts, batch_primaries)):
            if existing_p:
                pre_mapped[i] = existing_p
            else:
                llm_indices.append(i)
                llm_texts.append(text)

        log(f"[BATCH {batch_num}/{total_batches}] Rows {batch_start+1}-{batch_end} "
            f"({len(batch_texts)} texts: {len(pre_mapped)} pre-mapped, {len(llm_texts)} need LLM)")

        # Call LLM only for texts that need classification
        llm_results = {}
        if llm_texts:
            raw_results = call_llm(llm_texts)
            if raw_results is None:
                log(f"[BATCH {batch_num}/{total_batches}] SKIPPED LLM rows — all retries failed", "warning")
            else:
                for idx, result in zip(llm_indices, raw_results):
                    llm_results[idx] = result
        else:
            log(f"  [BATCH {batch_num}/{total_batches}] All rows pre-mapped, no LLM call needed")

        # Write results
        classified = 0
        null_count = 0
        pre_used = 0
        for i, text in enumerate(batch_texts):
            if i in pre_mapped:
                primary = pre_mapped[i]
                secondary = ""  # LLM wasn't called for secondaries on pre-mapped rows
                pre_used += 1
            elif i in llm_results:
                result = llm_results[i]
                primary = format_primary(result.get("primary"))
                secondary = format_secondary(result.get("secondary"))
            else:
                primary = ""
                secondary = ""

            writer.writerow([f"[{text}]", primary, secondary])
            if primary:
                classified += 1
            else:
                null_count += 1

        log(f"[BATCH {batch_num}/{total_batches}] Done — "
            f"{classified} classified ({pre_used} pre-mapped), {null_count} neutral/null")

        out_f.flush()
        save_checkpoint(batch_end)
        log(f"[BATCH {batch_num}/{total_batches}] Checkpoint saved at row {batch_end}")

        batch_num += 1

        if batch_end < total:
            log(f"[DELAY] Waiting {BATCH_DELAY}s before next batch...\n")
            time.sleep(BATCH_DELAY)

    # ── 5. Cleanup ───────────────────────────
    log("\n[STEP 5/5] Cleaning up...")
    out_f.close()
    log(f"[STEP 5/5] Output file closed: {OUTPUT_FILE}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log("[STEP 5/5] Checkpoint file removed — run complete")

    log("")
    log("=" * 60)
    log(f"  DONE — Output: {OUTPUT_FILE}")
    log(f"  Skipped rows log: {LOG_FILE}")
    log("=" * 60)


if __name__ == "__main__":
    main()
