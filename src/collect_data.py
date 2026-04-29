"""
collect_data.py (v2)

Records hand gesture samples with windowed motion features.

Each saved row contains 115 features: 85 static pose + 30 motion. The motion
features are computed over the previous ~0.5 sec of frames, so the recorder
needs a brief warm-up period before saving begins.

For motion letters (J, Z), the user is prompted to repeat the motion in a
loop during recording so the buffer captures real movement.

Usage:
    python src/collect_data.py --gesture A --samples 400
    python src/collect_data.py --gesture J --samples 400
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import argparse
import os
import time
import urllib.request
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import (
    WindowedFeatureExtractor,
    get_feature_size,
    get_feature_names,
    get_window_size,
)

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe import Image, ImageFormat

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH       = os.path.join(os.path.dirname(__file__), "..", "data", "gestures.csv")
LANDMARK_MODEL  = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
MODEL_URL       = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

FEATURE_SIZE = get_feature_size()
COLUMNS      = get_feature_names() + ["label"]

# Static letter prompt = let your hand drift naturally
# Motion letter prompt = repeat the motion continuously
MOTION_LETTERS = {"J", "Z"}

STATIC_VARIATION_PHASES = [
    ("Hold steady (let hand drift naturally — don't freeze)", 0.30),
    ("Move hand slowly around the frame",                       0.25),
    ("Tilt hand at slight angles",                              0.25),
    ("Vary distance — closer and farther",                      0.20),
]

MOTION_PROMPT = "Repeat the motion continuously — don't pause between reps"


def ensure_model():
    os.makedirs(os.path.dirname(LANDMARK_MODEL), exist_ok=True)
    if not os.path.exists(LANDMARK_MODEL):
        print(" Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, LANDMARK_MODEL)
        print(" Done.")


def load_existing_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if len(df.columns) != len(COLUMNS):
            print(f" Existing CSV has {len(df.columns)} columns, expected {len(COLUMNS)}.")
            print(" Old format detected — old data will be ignored.")
            return pd.DataFrame(columns=COLUMNS)
        return df
    return pd.DataFrame(columns=COLUMNS)


def draw_landmarks_manual(frame, landmarks):
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


def get_static_phase(progress):
    cumulative = 0.0
    for text, fraction in STATIC_VARIATION_PHASES:
        cumulative += fraction
        if progress < cumulative:
            return text
    return STATIC_VARIATION_PHASES[-1][0]


def draw_ui(frame, gesture, recording, count, target, countdown,
            warming_up, is_motion):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Gesture: {gesture}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    motion_tag = "  [MOTION letter]" if is_motion else ""
    cv2.putText(frame, f"Samples: {count}/{target}{motion_tag}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if countdown > 0:
        text, color = f"Starting in {countdown}...", (0, 200, 255)
    elif warming_up:
        text, color = "WARMING UP...", (255, 200, 0)
    elif recording:
        text, color = "RECORDING", (0, 0, 255)
    else:
        text, color = "PAUSED - Press SPACE", (180, 180, 180)
    cv2.putText(frame, text, (w - 380, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Progress bar
    if target > 0:
        bar_w = int((count / target) * w)
        cv2.rectangle(frame, (0, 76), (bar_w, 80), (0, 220, 100), -1)

    # Variation prompt (mid-screen)
    if recording or warming_up:
        if is_motion:
            prompt = MOTION_PROMPT
        else:
            progress = count / max(1, target)
            prompt = get_static_phase(progress)

        v_overlay = frame.copy()
        cv2.rectangle(v_overlay, (0, h - 110), (w, h - 50), (10, 10, 10), -1)
        cv2.addWeighted(v_overlay, 0.75, frame, 0.25, 0, frame)
        label_color = (100, 200, 255) if not is_motion else (255, 150, 100)
        cv2.putText(frame, "PROMPT:", (10, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)
        cv2.putText(frame, prompt, (10, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, "SPACE: start/pause   Q: quit & save", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return frame


def collect(gesture_name, target_samples):
    ensure_model()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    is_motion = gesture_name.upper() in MOTION_LETTERS
    window_size = get_window_size()

    existing_df = load_existing_data()
    existing_count = len(existing_df[existing_df["label"] == gesture_name]) \
        if not existing_df.empty and "label" in existing_df.columns else 0

    print(f"\n Collecting: '{gesture_name}'  {'(motion letter)' if is_motion else '(static letter)'}")
    print(f" Existing: {existing_count}  |  Target new: {target_samples}")
    print(f" Window size: {window_size} frames")
    if is_motion:
        print(" Tip: repeat the motion continuously during recording.")
    else:
        print(" Tip: hold the sign but DO NOT freeze your hand — natural drift is good.")
    print(" SPACE to start, Q to quit\n")

    extractor = WindowedFeatureExtractor()
    new_samples     = []
    recording       = False
    countdown       = 0
    countdown_start = None
    warmup_frames   = 0          # frames since recording started

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_idx = 0
    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_idx += 1
            timestamp_ms = int(frame_idx * (1000 / 30))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB,
                             data=np.ascontiguousarray(rgb))
            result = detector.detect_for_video(mp_image, timestamp_ms)

            features_now = None
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                draw_landmarks_manual(frame, lms)
                features_now = extractor.update(lms)
            else:
                # Hand left frame — clear the buffer so motion features stay clean
                extractor.reset()
                warmup_frames = 0

            # Countdown
            if countdown > 0:
                elapsed = time.time() - countdown_start
                countdown = max(0, 3 - int(elapsed))
                if countdown == 0:
                    recording = True
                    warmup_frames = 0
                    extractor.reset()  # fresh buffer

            # Recording
            warming_up = recording and warmup_frames < window_size
            if recording and features_now is not None and len(new_samples) < target_samples:
                warmup_frames += 1
                if warmup_frames >= window_size and extractor.is_warmed_up():
                    new_samples.append(features_now.tolist() + [gesture_name])
                    if len(new_samples) >= target_samples:
                        recording = False
                        print(f" Done! Collected {target_samples} samples.")

            frame = draw_ui(frame, gesture_name, recording,
                            len(new_samples), target_samples,
                            countdown, warming_up, is_motion)
            cv2.imshow("Gesture Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if not recording and countdown == 0:
                    countdown = 3
                    countdown_start = time.time()
                elif recording:
                    recording = False
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if new_samples:
        new_df = pd.DataFrame(new_samples, columns=COLUMNS)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.to_csv(DATA_PATH, index=False)
        print(f"\n Saved {len(new_samples)} samples -> {DATA_PATH}")
        print(f" Total in dataset: {len(combined)}")
    else:
        print("\n No samples collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", type=str, required=True)
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()

    label = args.gesture.strip()
    if len(label) == 1:
        label = label.upper()
    else:
        label = label.lower().replace(" ", "_")
    collect(label, args.samples)
