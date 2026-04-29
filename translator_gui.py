"""
translator_gui.py (v2)

ASL translator using windowed motion+pose classification.
The classifier handles I vs J (and similar pairs) on its own — there is no
hardcoded motion-detection branching here.

Usage:
    python translator_gui.py
"""

import os
import sys
import threading
import time
import json
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import GesturePredictor

# ── Config ────────────────────────────────────────────────────────────────────
SMOOTHING_LEN   = 12
HOLD_FRAMES     = 20
LETTER_COOLDOWN = 0.8

# ── Theme ─────────────────────────────────────────────────────────────────────
BG          = "#1a1a1a"
BG_PANEL    = "#252525"
BG_CARD     = "#2d2d2d"
FG          = "#e0e0e0"
FG_DIM      = "#888888"
ACCENT      = "#1D9E75"
ACCENT_HOV  = "#22B585"
DANGER      = "#D85A30"
WARN        = "#BA7517"


def get_ai_suggestion(text):
    if not text.strip():
        return ""
    try:
        from api_config import get_headers
        prompt = (
            f"The user is finger-spelling in ASL. They have typed: \"{text}\". "
            f"Suggest a short natural completion (max 8 words). "
            f"Reply with ONLY the completion, no quotes, no explanation."
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
    except ValueError as e:
        # Missing API key — show message to user
        return f"[API key not set — see src/api_config.py]"
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return f"[API error {e.code}]"
    except Exception:
        return ""


class LetterAcceptor:
    def __init__(self, hold_frames=HOLD_FRAMES, cooldown=LETTER_COOLDOWN):
        self.hold_frames = hold_frames
        self.cooldown    = cooldown
        self.current     = None
        self.hold_count  = 0
        self.last_added  = {}

    def update(self, gesture):
        if gesture is None or gesture == "unknown" or gesture == "warming_up":
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
                return gesture
        return None

    @property
    def progress(self):
        if self.current is None:
            return 0.0
        return min(self.hold_count / self.hold_frames, 1.0)


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
            self._fetch_suggestion()

    def backspace(self):
        if self.text:
            self.text = self.text[:-1]
            self.suggestion = ""

    def clear(self):
        self.text = ""
        self.suggestion = ""

    def accept_suggestion(self):
        if self.suggestion:
            if not self.text.endswith(" "):
                self.text += " "
            self.text += self.suggestion + " "
            self.suggestion = ""

    def _fetch_suggestion(self):
        if self.fetching or not self.text.strip():
            return
        self.fetching = True
        text_snapshot = self.text.strip()

        def worker():
            try:
                self.suggestion = get_ai_suggestion(text_snapshot)
            finally:
                self.fetching = False
        threading.Thread(target=worker, daemon=True).start()


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


class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Finger-Spelling Translator")
        self.root.configure(bg=BG)
        self.root.geometry("1100x780")
        self.root.minsize(900, 700)

        self.predictor    = None
        self.cap          = None
        self.running      = False
        self.buffer       = deque(maxlen=SMOOTHING_LEN)
        self.acceptor     = LetterAcceptor()
        self.builder      = WordBuilder()
        self.last_gesture = None
        self.fps_counter  = 0
        self.fps_start    = time.time()
        self.fps          = 0

        self._build_ui()
        self._init_model()

    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG, padx=20, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="ASL Translator (v2)",
                 font=("Helvetica", 22, "bold"),
                 bg=BG, fg=FG).pack(side="left")
        self.status_label = tk.Label(header, text="● Initializing...",
                                      font=("Helvetica", 11),
                                      bg=BG, fg=WARN)
        self.status_label.pack(side="right")

        main = tk.Frame(self.root, bg=BG, padx=20, pady=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left: video
        video_card = tk.Frame(main, bg=BG_CARD, padx=12, pady=12)
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(video_card, text="CAMERA FEED",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(0, 8))
        self.video_label = tk.Label(video_card, bg="#000")
        self.video_label.pack(fill="both", expand=True)

        info = tk.Frame(video_card, bg=BG_CARD, pady=10)
        info.pack(fill="x")

        self.gesture_label = tk.Label(info, text="—",
                                       font=("Helvetica", 32, "bold"),
                                       bg=BG_CARD, fg=ACCENT, width=4)
        self.gesture_label.pack(side="left", padx=(0, 15))

        info_right = tk.Frame(info, bg=BG_CARD)
        info_right.pack(side="left", fill="x", expand=True)
        self.conf_label = tk.Label(info_right, text="Confidence: —",
                                    font=("Helvetica", 10),
                                    bg=BG_CARD, fg=FG_DIM, anchor="w")
        self.conf_label.pack(fill="x")

        self.hold_canvas = tk.Canvas(info_right, height=8, bg="#404040",
                                      highlightthickness=0)
        self.hold_canvas.pack(fill="x", pady=(8, 4))

        self.hold_label = tk.Label(info_right, text="Hold to add",
                                    font=("Helvetica", 8),
                                    bg=BG_CARD, fg=FG_DIM, anchor="w")
        self.hold_label.pack(fill="x")
        self.fps_label = tk.Label(info_right, text="0 fps",
                                   font=("Helvetica", 8),
                                   bg=BG_CARD, fg=FG_DIM, anchor="w")
        self.fps_label.pack(fill="x")

        # Right: text + controls
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        text_card = tk.Frame(right, bg=BG_CARD, padx=12, pady=12)
        text_card.pack(fill="both", expand=True)
        tk.Label(text_card, text="TRANSLATED TEXT",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(0, 8))

        self.text_display = tk.Text(text_card, height=8, wrap="word",
                                     font=("Helvetica", 18),
                                     bg="#1f1f1f", fg=FG,
                                     insertbackground=FG,
                                     relief="flat", padx=12, pady=12,
                                     highlightthickness=0)
        self.text_display.pack(fill="both", expand=True)
        self.text_display.config(state="disabled")

        sugg_frame = tk.Frame(text_card, bg=BG_CARD, pady=10)
        sugg_frame.pack(fill="x")
        tk.Label(sugg_frame, text="AI SUGGESTION",
                 font=("Helvetica", 8, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w")
        self.suggestion_label = tk.Label(sugg_frame, text="—",
                                          font=("Helvetica", 12, "italic"),
                                          bg=BG_CARD, fg="#7DB8FF", anchor="w",
                                          wraplength=300, justify="left")
        self.suggestion_label.pack(fill="x", pady=(4, 0))

        controls = tk.Frame(right, bg=BG, pady=15)
        controls.pack(fill="x")
        btn_style = {"font": ("Helvetica", 10, "bold"),
                     "relief": "flat", "borderwidth": 0,
                     "padx": 14, "pady": 10, "cursor": "hand2"}

        self.start_btn = tk.Button(controls, text="▶  Start Camera",
                                    bg=ACCENT, fg="white",
                                    activebackground=ACCENT_HOV,
                                    command=self.toggle_camera, **btn_style)
        self.start_btn.pack(fill="x", pady=4)

        row1 = tk.Frame(controls, bg=BG)
        row1.pack(fill="x", pady=4)
        tk.Button(row1, text="Space", bg=BG_CARD, fg=FG, activebackground="#3a3a3a",
                  command=self.add_space, **btn_style).pack(side="left", expand=True, fill="x", padx=(0, 4))
        tk.Button(row1, text="Backspace", bg=BG_CARD, fg=FG, activebackground="#3a3a3a",
                  command=self.backspace, **btn_style).pack(side="left", expand=True, fill="x", padx=(4, 0))

        row2 = tk.Frame(controls, bg=BG)
        row2.pack(fill="x", pady=4)
        tk.Button(row2, text="Use Suggestion", bg=BG_CARD, fg="#7DB8FF", activebackground="#3a3a3a",
                  command=self.accept_suggestion, **btn_style).pack(side="left", expand=True, fill="x", padx=(0, 4))
        tk.Button(row2, text="Copy", bg=BG_CARD, fg=FG, activebackground="#3a3a3a",
                  command=self.copy_text, **btn_style).pack(side="left", expand=True, fill="x", padx=(4, 0))

        tk.Button(controls, text="Clear All", bg=DANGER, fg="white",
                  activebackground="#E06A40", command=self.clear_text,
                  **btn_style).pack(fill="x", pady=(8, 0))

        # Settings
        settings = tk.Frame(right, bg=BG_CARD, padx=12, pady=12)
        settings.pack(fill="x", pady=(15, 0))
        tk.Label(settings, text="SETTINGS",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(0, 8))

        thresh_frame = tk.Frame(settings, bg=BG_CARD)
        thresh_frame.pack(fill="x", pady=4)
        tk.Label(thresh_frame, text="Confidence threshold:",
                 font=("Helvetica", 9), bg=BG_CARD, fg=FG).pack(side="left")
        self.threshold_var = tk.DoubleVar(value=0.75)
        self.thresh_value = tk.Label(thresh_frame, text="75%",
                                      font=("Helvetica", 9, "bold"),
                                      bg=BG_CARD, fg=ACCENT, width=5)
        self.thresh_value.pack(side="right")
        tk.Scale(settings, from_=0.5, to=0.95, resolution=0.05,
                  orient="horizontal", variable=self.threshold_var,
                  bg=BG_CARD, fg=FG, troughcolor="#404040",
                  highlightthickness=0,
                  command=self._on_threshold_change,
                  showvalue=False, sliderrelief="flat").pack(fill="x")

        self.root.bind("<space>",     lambda e: self.add_space())
        self.root.bind("<BackSpace>", lambda e: self.backspace())
        self.root.bind("<Return>",    lambda e: self.accept_suggestion())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_threshold_change(self, val):
        v = float(val)
        self.thresh_value.config(text=f"{int(v*100)}%")
        if self.predictor:
            self.predictor.threshold = v

    def _init_model(self):
        def worker():
            try:
                self.predictor = GesturePredictor(
                    confidence_threshold=self.threshold_var.get()
                )
                self.root.after(0, lambda: self._set_status("● Ready", ACCENT))
            except FileNotFoundError:
                self.root.after(0, lambda: self._set_status("● Model not found", DANGER))
                self.root.after(0, lambda: messagebox.showerror(
                    "Model Not Found",
                    "No trained model found.\n\nOpen the Training GUI to collect data and train."
                ))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._set_status("● Init failed", DANGER))
                self.root.after(0, lambda: messagebox.showerror("Error", err))
        threading.Thread(target=worker, daemon=True).start()

    def _set_status(self, text, color):
        self.status_label.config(text=text, fg=color)

    def toggle_camera(self):
        if not self.predictor:
            messagebox.showwarning("Not Ready", "Model is still loading.")
            return
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        self.predictor.reset_buffer()
        self.start_btn.config(text="■  Stop Camera", bg=DANGER, activebackground="#E06A40")
        self._set_status("● Live", ACCENT)
        self._update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(text="▶  Start Camera", bg=ACCENT, activebackground=ACCENT_HOV)
        self._set_status("● Stopped", FG_DIM)
        self.video_label.config(image="")
        self.gesture_label.config(text="—")
        self.conf_label.config(text="Confidence: —")
        self.hold_canvas.delete("all")

    def _update_frame(self):
        if not self.running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(33, self._update_frame)
            return

        frame = cv2.flip(frame, 1)

        result = self.predictor.detect(frame)
        gesture     = None
        confidence  = 0.0
        warming_up  = False

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks)
            raw, confidence = self.predictor.predict_from_landmarks(landmarks)
            if raw == "warming_up":
                warming_up = True
                gesture = None
            else:
                self.buffer.append(raw)
                gesture = max(set(self.buffer), key=self.buffer.count)
        else:
            self.buffer.clear()
            self.predictor.reset_buffer()

        # Letter acceptor
        accepted = self.acceptor.update(gesture if gesture and gesture != "unknown" else None)
        if accepted:
            self.builder.add_letter(accepted)
            self._refresh_text()

        # Display
        if warming_up:
            self.gesture_label.config(text="...", fg=WARN)
            self.conf_label.config(text="Building motion buffer...")
        elif gesture and gesture != "unknown":
            self.gesture_label.config(text=gesture, fg=ACCENT)
            self.conf_label.config(text=f"Confidence: {int(confidence*100)}%")
        else:
            self.gesture_label.config(text="—", fg=FG_DIM)
            self.conf_label.config(text="Confidence: —")

        self._draw_hold_progress(self.acceptor.progress)

        # FPS
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start
        if elapsed > 0.5:
            self.fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start = time.time()
            self.fps_label.config(text=f"{self.fps:.0f} fps")

        # Display frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target_w = self.video_label.winfo_width() or 640
        target_h = self.video_label.winfo_height() or 480
        if target_w > 50 and target_h > 50:
            aspect = rgb.shape[1] / rgb.shape[0]
            if target_w / target_h > aspect:
                new_h = target_h
                new_w = int(new_h * aspect)
            else:
                new_w = target_w
                new_h = int(new_w / aspect)
            rgb = cv2.resize(rgb, (max(1, new_w), max(1, new_h)))

        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=photo)
        self.video_label.image = photo

        sugg_text = self.builder.suggestion or ("Loading..." if self.builder.fetching else "—")
        self.suggestion_label.config(text=sugg_text)

        self.root.after(33, self._update_frame)

    def _draw_hold_progress(self, progress):
        self.hold_canvas.delete("all")
        w = self.hold_canvas.winfo_width()
        if w > 1:
            fill = int(w * progress)
            if fill > 0:
                self.hold_canvas.create_rectangle(0, 0, fill, 10,
                                                    fill=ACCENT, outline="")

    def add_space(self):
        self.builder.add_space()
        self._refresh_text()

    def backspace(self):
        self.builder.backspace()
        self._refresh_text()

    def accept_suggestion(self):
        self.builder.accept_suggestion()
        self._refresh_text()

    def copy_text(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.builder.text.strip())
        self._set_status("● Copied", ACCENT)
        self.root.after(1500, lambda: self._set_status(
            "● Live" if self.running else "● Ready", ACCENT))

    def clear_text(self):
        self.builder.clear()
        self._refresh_text()

    def _refresh_text(self):
        self.text_display.config(state="normal")
        self.text_display.delete("1.0", "end")
        self.text_display.insert("1.0", self.builder.text)
        self.text_display.config(state="disabled")

    def _on_close(self):
        if self.cap:
            self.cap.release()
        if self.predictor:
            self.predictor.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = TranslatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
