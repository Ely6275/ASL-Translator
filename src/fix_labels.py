"""
fix_labels.py - One-time script to uppercase single-letter labels in gestures.csv
Run once then delete.
"""
import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gestures.csv")

df = pd.read_csv(DATA_PATH)
before = df["label"].value_counts().to_dict()

def fix_label(label):
    label = str(label).strip()
    return label.upper() if len(label) == 1 else label.lower()

df["label"] = df["label"].apply(fix_label)

after = df["label"].value_counts().to_dict()
df.to_csv(DATA_PATH, index=False)

print("Labels fixed:")
for k, v in sorted(after.items()):
    print(f"  {k}: {v} samples")
