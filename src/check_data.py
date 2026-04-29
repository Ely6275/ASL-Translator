"""
check_data.py

Shows how many samples you have per ASL letter and
prints the exact commands to run for any missing ones.

Usage:
    python src/check_data.py
"""

import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gestures.csv")
LETTERS   = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TARGET    = 300

def check():
    if not os.path.exists(DATA_PATH):
        print(" No dataset found yet. Start collecting:\n")
        for letter in LETTERS:
            print(f"   python src/collect_data.py --gesture {letter} --samples {TARGET}")
        return

    df     = pd.read_csv(DATA_PATH)
    counts = df["label"].value_counts()

    print("\n ASL Dataset Status")
    print(" " + "=" * 45)

    complete = []
    missing  = []
    partial  = []

    # Normalize all labels to uppercase for comparison
    counts.index = counts.index.str.upper()

    for letter in LETTERS:
        count = counts.get(letter, 0)
        bar   = "█" * (count // 15)
        status = "✓" if count >= TARGET else ("~" if count > 0 else "✗")
        color_hint = "" if count >= TARGET else " ← needs more" if count > 0 else " ← missing"
        print(f"  {status} {letter}  {count:>4} samples  {bar}{color_hint}")

        if count >= TARGET:
            complete.append(letter)
        elif count > 0:
            partial.append((letter, count))
        else:
            missing.append(letter)

    print(f"\n  Complete ({len(complete)}/26): {' '.join(complete) if complete else 'none'}")

    if partial:
        print(f"\n  Partial — needs top-up:")
        for letter, count in partial:
            needed = TARGET - count
            print(f"    python src/collect_data.py --gesture {letter} --samples {needed}")

    if missing:
        print(f"\n  Missing — run these:")
        for letter in missing:
            print(f"    python src/collect_data.py --gesture {letter} --samples {TARGET}")

    print(f"\n  Total samples: {len(df)}")
    pct = len(complete) / 26 * 100
    print(f"  Progress: {len(complete)}/26 letters ({pct:.0f}%)\n")

if __name__ == "__main__":
    check()
