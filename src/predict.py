"""
predict.py (v2)

Real-time inference using windowed feature extraction.

Maintains a rolling buffer of recent landmark frames and produces
combined static + motion feature vectors that match the format used
during training. This is what makes I vs J discrimination work — the
classifier sees the same temporal context at inference time as in training.
"""

import os
import pickle
import numpy as np
import urllib.request
import sys

sys.path.insert(0, os.path.dirname(__file__))
from features import WindowedFeatureExtractor

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe import Image, ImageFormat

LANDMARK_MODEL  = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gesture_classifier.pkl")
MODEL_URL       = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model():
    os.makedirs(os.path.dirname(LANDMARK_MODEL), exist_ok=True)
    if not os.path.exists(LANDMARK_MODEL):
        print(" Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, LANDMARK_MODEL)


class GesturePredictor:
    def __init__(self, confidence_threshold=0.75):
        ensure_model()
        if not os.path.exists(CLASSIFIER_PATH):
            raise FileNotFoundError(
                f"Classifier not found at {CLASSIFIER_PATH}. Run train_model.py first."
            )

        with open(CLASSIFIER_PATH, "rb") as f:
            bundle = pickle.load(f)

        self.pipeline      = bundle["pipeline"]
        self.label_encoder = bundle["label_encoder"]
        self.classes       = bundle["classes"]
        self.threshold     = confidence_threshold

        # Windowed feature extractor — keeps its own rolling buffer
        self.extractor = WindowedFeatureExtractor()

        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = HandLandmarker.create_from_options(options)
        self.frame_idx = 0

    def detect(self, bgr_frame):
        """Run hand detection on a BGR frame. Returns the mediapipe result."""
        self.frame_idx += 1
        timestamp_ms = int(self.frame_idx * (1000 / 30))
        rgb = bgr_frame[:, :, ::-1].copy()
        mp_image = Image(image_format=ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def predict_from_landmarks(self, landmarks):
        """
        Update the rolling buffer with this frame's landmarks and return a
        prediction. Returns ('warming_up', 0.0) until the buffer is full.
        """
        features = self.extractor.update(landmarks)

        # Need a full window before predictions are valid
        if not self.extractor.is_warmed_up():
            return "warming_up", 0.0

        features = features.reshape(1, -1)
        try:
            proba = self.pipeline.predict_proba(features)[0]
            confidence = float(np.max(proba))
            class_idx = int(np.argmax(proba))
        except AttributeError:
            class_idx  = int(self.pipeline.predict(features)[0])
            confidence = 1.0

        gesture = self.label_encoder.inverse_transform([class_idx])[0]
        if confidence < self.threshold:
            return "unknown", confidence
        return gesture, confidence

    def reset_buffer(self):
        """Reset the rolling window — call when hand leaves the frame."""
        self.extractor.reset()

    def get_all_classes(self):
        return self.classes

    def close(self):
        self.detector.close()
