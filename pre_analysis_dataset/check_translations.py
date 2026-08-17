import csv

INPUT_FILE = "emotion_dataset_translated.csv"

missing = []
total = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        english = row.get("english_text", "").strip()
        if not english:
            missing.append((total + 1, row["chinese_text"]))

print(f"Total rows checked: {total}")
print(f"Missing translations: {len(missing)}")

if missing:
    print(f"\nRows missing english_text ({len(missing)} total):")
    print("-" * 60)
    for line_num, chinese in missing:
        preview = chinese[:60] + ("..." if len(chinese) > 60 else "")
        print(f"  Line {line_num}: {preview}")
else:
    print("\nAll rows have translations. No issues found.")
