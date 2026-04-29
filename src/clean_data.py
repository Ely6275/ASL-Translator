"""
clean_data.py - Removes old robot gesture labels from gestures.csv
Run once before training the ASL model.
"""
import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gestures.csv")

OLD_LABELS = {"open_palm", "fist", "index_up", "peace", "thumbs_up"}

df = pd.read_csv(DATA_PATH)
before = len(df)

removed = df[df["label"].isin(OLD_LABELS)]["label"].value_counts()
df = df[~df["label"].isin(OLD_LABELS)]

df.to_csv(DATA_PATH, index=False)

print("Removed old robot gesture data:")
for label, count in removed.items():
    print(f"  {label}: {count} samples removed")

print(f"\nBefore: {before} samples")
print(f"After:  {len(df)} samples")
print(f"\nRemaining labels:")
print(df["label"].value_counts().to_string())
