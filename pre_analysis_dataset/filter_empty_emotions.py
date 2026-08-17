import csv
import re

INPUT_FILE = "emotion_dataset_output.csv"
OUTPUT_FILE = "emotion_dataset_filtered.csv"

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

VALID_PRIMARIES = set(EMOTION_WHEEL.keys())
ALL_VALID_SECONDARIES = {s.lower() for secs in EMOTION_WHEEL.values() for s in secs}


def parse_secondary_emotions(raw: str) -> list[str]:
    """Extract individual secondary emotion strings from the bracket-wrapped field."""
    raw = raw.strip()
    if not raw:
        return []
    inner = raw.strip("[]")
    return [s.strip() for s in re.split(r"[;,]", inner) if s.strip()]


with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()

    total = 0
    kept = 0
    skipped_empty = 0
    skipped_bad_primary = 0
    skipped_bad_secondary = 0

    for row in reader:
        total += 1
        primary = row["primary_emotion"].strip()

        if not primary:
            skipped_empty += 1
            continue

        if primary.upper() not in VALID_PRIMARIES:
            skipped_bad_primary += 1
            continue

        secondaries = parse_secondary_emotions(row["secondary_emotions"])
        if secondaries:
            bad = [s for s in secondaries if s.lower() not in ALL_VALID_SECONDARIES]
            if bad:
                skipped_bad_secondary += 1
                continue

        writer.writerow(row)
        kept += 1

    print(f"Total rows:            {total}")
    print(f"Kept:                  {kept}")
    print(f"Skipped (empty):       {skipped_empty}")
    print(f"Skipped (bad primary): {skipped_bad_primary}")
    print(f"Skipped (bad secondary): {skipped_bad_secondary}")
    print(f"Total skipped:         {skipped_empty + skipped_bad_primary + skipped_bad_secondary}")
    print(f"Output saved to {OUTPUT_FILE}")
