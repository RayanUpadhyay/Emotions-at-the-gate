import csv
import re
from collections import Counter

INPUT_FILE = "emotion_dataset_filtered.csv"
OUTPUT_FILE = "summary.txt"


def parse_secondary_emotions(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    inner = raw.strip("[]")
    return [s.strip() for s in re.split(r"[;,]", inner) if s.strip()]


primary_counts = Counter()
secondary_counts = Counter()
total_rows = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_rows += 1
        primary_counts[row["primary_emotion"].strip()] += 1
        for sec in parse_secondary_emotions(row["secondary_emotions"]):
            secondary_counts[sec] += 1

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    out.write("=" * 50 + "\n")
    out.write("  EMOTION DATASET SUMMARY\n")
    out.write("=" * 50 + "\n\n")
    out.write(f"Total rows: {total_rows}\n\n")

    out.write("-" * 50 + "\n")
    out.write("  PRIMARY EMOTIONS\n")
    out.write("-" * 50 + "\n\n")
    for emotion, count in primary_counts.most_common():
        pct = count / total_rows * 100
        out.write(f"  {emotion:<12} {count:>6}  ({pct:5.1f}%)\n")
    out.write(f"\n  {'TOTAL':<12} {sum(primary_counts.values()):>6}\n\n")

    out.write("-" * 50 + "\n")
    out.write("  SECONDARY EMOTIONS\n")
    out.write("-" * 50 + "\n\n")
    total_secondary = sum(secondary_counts.values())
    for emotion, count in secondary_counts.most_common():
        pct = count / total_secondary * 100
        out.write(f"  {emotion:<16} {count:>6}  ({pct:5.1f}%)\n")
    out.write(f"\n  {'TOTAL':<16} {total_secondary:>6}\n")
    out.write("\n" + "=" * 50 + "\n")

print(f"Summary written to {OUTPUT_FILE}")
