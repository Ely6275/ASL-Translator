"""
train_model.py (v2)

Trains a classifier on windowed feature vectors (115-dim).

Usage:
    python src/train_model.py
    python src/train_model.py --model svm
    python src/train_model.py --model knn --k 5
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import get_feature_size

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(BASE_DIR, "data", "gestures.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "gesture_classifier.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "models", "training_report.json")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    expected = get_feature_size() + 1
    if len(df.columns) != expected:
        raise ValueError(
            f"Dataset has {len(df.columns)} columns, expected {expected}. "
            f"Re-collect data with the current pipeline."
        )
    print(f" Loaded {len(df)} samples from {DATA_PATH}")
    print(f" Feature columns: {len(df.columns) - 1}")
    print("\n Samples per gesture:")
    counts = df["label"].value_counts()
    for label, count in counts.items():
        bar = "#" * (count // 20)
        print(f"   {label:<10} {count:>4}  {bar}")
    return df


def build_model(model_type, k=5):
    if model_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    elif model_type == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])


def train(model_type="svm", k=5):
    df = load_data()

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print(f"\n Classes: {list(label_encoder.classes_)}")
    print(f" Feature size: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n Train: {len(X_train)}  |  Test: {len(X_test)}")

    print(f"\n Training {model_type.upper()}...")
    pipeline = build_model(model_type, k)
    pipeline.fit(X_train, y_train)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f" 5-fold CV: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    class_names = label_encoder.classes_
    print("\n" + "=" * 55)
    print(" EVALUATION RESULTS")
    print("=" * 55)
    print(f"\n Overall accuracy: {acc * 100:.2f}%\n")
    print(" Per-gesture breakdown:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    report = classification_report(y_test, y_pred,
                                     target_names=class_names, output_dict=True)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bundle = {
        "pipeline":      pipeline,
        "label_encoder": label_encoder,
        "model_type":    model_type,
        "feature_size":  X.shape[1],
        "classes":       list(label_encoder.classes_),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    training_report = {
        "model_type":    model_type,
        "accuracy":      float(acc),
        "cv_mean":       float(cv_scores.mean()),
        "cv_std":        float(cv_scores.std()),
        "n_samples":     int(len(df)),
        "n_train":       int(len(X_train)),
        "n_test":        int(len(X_test)),
        "n_classes":     int(len(label_encoder.classes_)),
        "n_features":    int(X.shape[1]),
        "classes":       list(label_encoder.classes_),
        "per_class":     report,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(training_report, f, indent=2)

    print(f"\n Saved model:  {MODEL_PATH}")
    print(f" Saved report: {REPORT_PATH}\n")
    return training_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="svm", choices=["knn", "svm"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    train(args.model, args.k)
