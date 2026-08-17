"""
Chinese → English Translation Pipeline (Concurrent)
====================================================
Reads emotion_dataset_filtered.csv, translates the chinese_text column
to English using the GLM-4.5 model via Z.ai API, and writes a new CSV
with an added english_text column.

Uses ThreadPoolExecutor for parallel batch API calls to speed up
processing of ~16,667 rows.

Supports checkpoint-based resumption — safe to interrupt and restart.

Usage:
  python translate_chinese.py

Configuration:
  - Set GLM_API_KEY in .env before running.
"""

import csv
import json
import re
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
API_KEY = os.getenv("GLM_API_KEY", "")
if not API_KEY:
    raise RuntimeError("GLM_API_KEY not found. Set it in .env file.")
API_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-4.5"

INPUT_FILE = "emotion_dataset_filtered.csv"
OUTPUT_FILE = "emotion_dataset_translated.csv"
CHECKPOINT_FILE = "translate_checkpoint.json"
LOG_FILE = "translate.log"

BATCH_SIZE = 25           # sentences per API call
MAX_WORKERS = 5           # concurrent API calls in flight
MAX_RETRIES = 3
RETRY_DELAY = 10          # seconds between retries
API_TIMEOUT = 120         # seconds per API call

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(file_handler)


def log(msg: str, level: str = "info"):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)
    getattr(logger, level)(msg)


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Chinese-to-English translator.

RULES:
1. Translate each Chinese sentence into natural, fluent English.
2. Preserve the original meaning and emotional tone as closely as possible.
3. Return ONLY a valid JSON array of strings — no markdown, no explanation, no extra text.
4. Each element in the array is the English translation of the input sentence at the same index.
5. If a sentence is already in English or is untranslatable, return it as-is.

OUTPUT FORMAT (strict JSON array of strings):
["The product quality is terrible, very disappointing.", "This store has awful service."]
"""


def build_user_prompt(texts: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return (
        f"Translate the following {len(texts)} Chinese sentences into English.\n"
        f"Return a JSON array with exactly {len(texts)} string elements in the same order.\n\n"
        f"{numbered}"
    )


# ─────────────────────────────────────────────
# API CALL WITH RETRIES
# ─────────────────────────────────────────────
def call_translate_api(batch_idx: int, texts: list[str]) -> tuple[int, list[str] | None]:
    """
    Translate a batch of texts. Returns (batch_idx, translations) or (batch_idx, None) on failure.
    batch_idx is passed through so results can be reassembled in order.
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
            response = requests.post(
                API_BASE_URL, headers=headers, json=payload, timeout=API_TIMEOUT
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```json\s*|^```\s*|```$", "", content, flags=re.MULTILINE).strip()

            parsed = json.loads(content)
            if not isinstance(parsed, list) or len(parsed) != len(texts):
                raise ValueError(
                    f"Expected list of {len(texts)}, got {type(parsed).__name__} "
                    f"of length {len(parsed) if isinstance(parsed, list) else '?'}"
                )
            return (batch_idx, [str(t) for t in parsed])

        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            log(f"  [BATCH {batch_idx}] Attempt {attempt}/{MAX_RETRIES} failed: {e}", "warning")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    log(f"  [BATCH {batch_idx}] All retries exhausted — returning None", "error")
    return (batch_idx, None)


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f).get("next_row", 0)
    return 0


def save_checkpoint(next_row: int):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"next_row": next_row}, f)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log("=" * 60)
    log("  Chinese → English Translation Pipeline (Concurrent)")
    log("=" * 60)

    # ── 1. Load input CSV ────────────────────
    log("\n[STEP 1] Loading input CSV...")
    rows = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    log(f"[STEP 1] Loaded {len(rows)} rows from {INPUT_FILE}")

    # ── 2. Check checkpoint ──────────────────
    start_row = load_checkpoint()
    if start_row > 0:
        log(f"\n[STEP 2] Resuming from row {start_row} (checkpoint found)")
    else:
        log("\n[STEP 2] Starting fresh run")

    # ── 3. Prepare output file ───────────────
    fieldnames = ["chinese_text", "primary_emotion", "secondary_emotions", "english_text"]
    write_mode = "a" if start_row > 0 else "w"
    out_f = open(OUTPUT_FILE, write_mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if start_row == 0:
        writer.writeheader()

    # ── 4. Build batches from remaining rows ─
    remaining_rows = rows[start_row:]
    batches = []
    for i in range(0, len(remaining_rows), BATCH_SIZE):
        batches.append(remaining_rows[i : i + BATCH_SIZE])

    total_batches = len(batches)
    log(f"\n[STEP 3] Processing {len(remaining_rows)} rows in {total_batches} batches "
        f"(batch_size={BATCH_SIZE}, workers={MAX_WORKERS})")

    # ── 5. Process with concurrent workers ───
    global_row = start_row
    wave_num = 0

    for wave_start in range(0, total_batches, MAX_WORKERS):
        wave_end = min(wave_start + MAX_WORKERS, total_batches)
        wave_batches = batches[wave_start:wave_end]
        wave_num += 1

        log(f"\n[WAVE {wave_num}] Dispatching batches {wave_start+1}-{wave_end} of {total_batches} "
            f"({len(wave_batches)} concurrent calls)")

        # Extract just the chinese_text for each batch
        batch_texts = []
        for batch in wave_batches:
            texts = [row["chinese_text"].strip("[]") for row in batch]
            batch_texts.append(texts)

        results: dict[int, list[str] | None] = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(call_translate_api, wave_start + i, texts): wave_start + i
                for i, texts in enumerate(batch_texts)
            }
            for future in as_completed(futures):
                batch_idx, translations = future.result()
                results[batch_idx] = translations
                status = f"{len(translations)} translations" if translations else "FAILED"
                log(f"  [BATCH {batch_idx + 1}/{total_batches}] {status}")

        # Write results in order
        for i, batch in enumerate(wave_batches):
            batch_idx = wave_start + i
            translations = results.get(batch_idx)

            for j, row in enumerate(batch):
                if translations and j < len(translations):
                    english = f"[{translations[j]}]"
                else:
                    english = ""

                writer.writerow({
                    "chinese_text": row["chinese_text"],
                    "primary_emotion": row["primary_emotion"],
                    "secondary_emotions": row["secondary_emotions"],
                    "english_text": english,
                })

            global_row += len(batch)

        out_f.flush()
        save_checkpoint(global_row)
        log(f"[WAVE {wave_num}] Checkpoint saved at row {global_row}/{len(rows)}")

    # ── 6. Cleanup ───────────────────────────
    out_f.close()
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    log("\n" + "=" * 60)
    log(f"  DONE — Output: {OUTPUT_FILE}")
    log(f"  Total rows translated: {len(rows)}")
    log("=" * 60)


if __name__ == "__main__":
    main()
