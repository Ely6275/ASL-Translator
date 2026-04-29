"""
app.py (v2) — CLI version of the ASL translator using windowed features.

Usage:
    python app.py
    python app.py --camera 1 --threshold 0.80
"""

import os
import sys
import time
import argparse
import threading
import json
import urllib.request
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import GesturePredictor

FONT             = cv2.FONT_HERSHEY_SIMPLEX
SMOOTHING_LEN    = 12
HOLD_FRAMES      = 20
LETTER_COOLDOWN  = 0.8
PANEL_H          = 140


def get_ai_suggestion(text):
    if not text.strip():
        return ""
    try:
        from api_config import get_headers
        prompt = (
            f"The user is finger-spelling in ASL. They have typed: \"{text}\". "
            f"Suggest a short natural completion (max 8 words). "
            f"Reply with ONLY the completion, no quotes."
        )
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 40,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers=get_headers(),
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"].strip()
    except ValueError:
        return "[API key not set — see src/api_config.py]"
    except Exception:
        return ""


class WordBuilder:
    def __init__(self):
        self.text = ""
        self.suggestion = ""
        self.fetching = False

    def add_letter(self, letter):
        self.text += letter.upper()
        self.suggestion = ""

    def add_space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "
            self._fetch()

    def backspace(self):
        if self.text:
            self.text = self.text[:-1]

    def clear(self):
        self.text = ""
        self.suggestion = ""

    def accept_suggestion(self):
        if self.suggestion:
            if not self.text.endswith(" "):
                self.text += " "
            self.text += self.suggestion + " "
            self.suggestion = ""

    def _fetch(self):
        if self.fetching:
            return
        self.fetching = True
        snap = self.text.strip()
        def worker():
            try:
                self.suggestion = get_ai_suggestion(snap)
            finally:
                self.fetching = False
        threading.Thread(target=worker, daemon=True).start()


class LetterAcceptor:
    def __init__(self, hold_frames=HOLD_FRAMES, cooldown=LETTER_COOLDOWN):
        self.hold_frames = hold_frames
        self.cooldown    = cooldown
        self.current     = None
        self.hold_count  = 0
        self.last_added  = {}
        self.just_added  = None

    def update(self, gesture):
        self.just_added = None
        if gesture is None or gesture in ("unknown", "warming_up"):
            self.current    = None
            self.hold_count = 0
            return None

        if gesture != self.current:
            self.current    = gesture
            self.hold_count = 0
            return None

        self.hold_count += 1
        if self.hold_count >= self.hold_frames:
            now = time.time()
            if now - self.last_added.get(gesture, 0) >= self.cooldown:
                self.last_added[gesture] = now
                self.hold_count = 0
                self.just_added = gesture
                return gesture
        return None

    @property
    def hold_progress(self):
        if self.current is None:
            return 0.0
        return min(self.hold_count / self.hold_frames, 1.0)


def draw_landmarks(frame, landmarks):
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


def draw_top_bar(frame, gesture, confidence, hold_progress, warming_up):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if warming_up:
        cv2.putText(frame, "...", (16, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 165, 255), 3)
        cv2.putText(frame, "Building motion buffer", (90, 35),
                    FONT, 0.5, (180, 180, 180), 1)
    elif gesture and gesture != "unknown":
        cv2.putText(frame, gesture, (16, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 220, 100), 3)
        cv2.putText(frame, f"{int(confidence*100)}%", (90, 35),
                    FONT, 0.6, (180, 180, 180), 1)
        bar_x, bar_y, bar_w, bar_h = 90, 45, 180, 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 50), -1)
        fill = int(bar_w * hold_progress)
        if fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill, bar_y + bar_h), (0, 220, 100), -1)
        cv2.putText(frame, "Hold to add", (bar_x + bar_w + 8, bar_y + 9),
                    FONT, 0.38, (130, 130, 130), 1)
    else:
        cv2.putText(frame, "Show a letter", (16, 45),
                    FONT, 0.8, (80, 80, 80), 2)


def draw_bottom_panel(frame, builder, just_added):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - PANEL_H), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (0, h - PANEL_H), (5, h), (0, 200, 100), -1)
    cv2.putText(frame, "TRANSLATED TEXT", (14, h - PANEL_H + 22),
                FONT, 0.45, (100, 100, 100), 1)

    display = builder.text
    if len(display) > 55:
        display = "..." + display[-52:]
    text_color = (0, 255, 150) if just_added else (230, 230, 230)
    cv2.putText(frame, display if display else "_", (14, h - PANEL_H + 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, text_color, 2)

    if builder.suggestion:
        cv2.putText(frame, f"Suggestion: {builder.suggestion}",
                    (14, h - PANEL_H + 90), FONT, 0.48, (150, 200, 255), 1)


class ASLApp:
    def __init__(self, camera_index=0, confidence_threshold=0.75):
        print("\n Loading gesture model...")
        self.predictor = GesturePredictor(confidence_threshold=confidence_threshold)
        self.camera_idx = camera_index
        self.buffer    = deque(maxlen=SMOOTHING_LEN)
        self.acceptor  = LetterAcceptor()
        self.builder   = WordBuilder()

    def smooth(self, gesture):
        self.buffer.append(gesture)
        return max(set(self.buffer), key=self.buffer.count)

    def run(self):
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            print(f"ERROR: cannot open camera {self.camera_idx}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(" Recognizing:", self.predictor.get_all_classes())
        print(" Q quit | SPACE word break | ENTER use suggestion | BACKSPACE delete | C clear\n")

        just_added_timer = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            result = self.predictor.detect(frame)
            gesture     = None
            confidence  = 0.0
            warming_up  = False

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                draw_landmarks(frame, lms)
                raw, confidence = self.predictor.predict_from_landmarks(lms)
                if raw == "warming_up":
                    warming_up = True
                else:
                    gesture = self.smooth(raw)
            else:
                self.buffer.clear()
                self.predictor.reset_buffer()

            accepted = self.acceptor.update(
                gesture if gesture and gesture != "unknown" else None
            )
            if accepted:
                self.builder.add_letter(accepted)
                just_added_timer = 8

            just_flash = self.acceptor.just_added if just_added_timer > 0 else None
            just_added_timer = max(0, just_added_timer - 1)

            draw_top_bar(frame, gesture, confidence,
                         self.acceptor.hold_progress, warming_up)
            draw_bottom_panel(frame, self.builder, just_flash)

            cv2.imshow("ASL Finger-Spelling Translator (v2)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                self.builder.clear()
            elif key == 13:
                self.builder.accept_suggestion()
            elif key == 8 or key == 127:
                self.builder.backspace()
            elif key == 32:
                self.builder.add_space()

        cap.release()
        cv2.destroyAllWindows()
        self.predictor.close()
        if self.builder.text:
            print(f"\n Final: {self.builder.text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    ASLApp(camera_index=args.camera, confidence_threshold=args.threshold).run()


if __name__ == "__main__":
    main()
