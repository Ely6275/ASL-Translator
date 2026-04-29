"""
training_gui.py

GUI for data collection, training, and analysis.
Three tabs: Collect Data | Train Model | Analysis

Usage:
    python training_gui.py
"""

import os
import sys
import json
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Theme ─────────────────────────────────────────────────────────────────────
BG          = "#1a1a1a"
BG_PANEL    = "#252525"
BG_CARD     = "#2d2d2d"
BG_HIGHLIGHT = "#353535"
FG          = "#e0e0e0"
FG_DIM      = "#888888"
ACCENT      = "#1D9E75"
ACCENT_HOV  = "#22B585"
DANGER      = "#D85A30"
WARN        = "#BA7517"
INFO        = "#7DB8FF"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "gestures.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "gesture_classifier.pkl")
REPORT_PATH      = os.path.join(BASE_DIR, "models", "training_report.json")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")
PRETRAINED_CSV   = os.path.join(BASE_DIR, "data", "pretrained", "gestures.csv")
FIRST_LAUNCH_FLAG = os.path.join(BASE_DIR, "data", ".firstlaunch_done")


def make_button(parent, text, command, color=ACCENT, hover=ACCENT_HOV, fg="white", **kwargs):
    return tk.Button(parent, text=text, command=command,
                     bg=color, fg=fg, activebackground=hover, activeforeground=fg,
                     font=("Helvetica", 10, "bold"),
                     relief="flat", borderwidth=0,
                     padx=14, pady=8, cursor="hand2", **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
class CollectionTab:
    def __init__(self, parent, app):
        self.app = app
        self.frame = tk.Frame(parent, bg=BG)

        self.collection_proc = None

        # Header
        header = tk.Frame(self.frame, bg=BG, padx=20, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="Data Collection",
                 font=("Helvetica", 18, "bold"),
                 bg=BG, fg=FG).pack(anchor="w")
        tk.Label(header,
                 text="Record gesture samples through the webcam. Vary your hand pose for robust models.",
                 font=("Helvetica", 10), bg=BG, fg=FG_DIM).pack(anchor="w", pady=(2, 0))

        # Two columns: settings (left) + status (right)
        body = tk.Frame(self.frame, bg=BG, padx=20, pady=10)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ── LEFT: collection settings ────────────────────────────────────────
        left = tk.Frame(body, bg=BG_CARD, padx=18, pady=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(left, text="COLLECT NEW SAMPLES",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(0, 12))

        # Letter input
        tk.Label(left, text="Letter / Gesture",
                 font=("Helvetica", 10), bg=BG_CARD, fg=FG).pack(anchor="w")

        self.letter_var = tk.StringVar(value="A")
        letter_entry = tk.Entry(left, textvariable=self.letter_var,
                                  font=("Helvetica", 14, "bold"),
                                  bg="#1f1f1f", fg=ACCENT,
                                  insertbackground=FG,
                                  relief="flat", width=10)
        letter_entry.pack(anchor="w", pady=(4, 12), ipady=6)

        # Quick pick row
        quick_frame = tk.Frame(left, bg=BG_CARD)
        quick_frame.pack(fill="x", pady=(0, 12))
        tk.Label(quick_frame, text="Quick pick:",
                 font=("Helvetica", 9), bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(0, 4))

        quick_grid = tk.Frame(quick_frame, bg=BG_CARD)
        quick_grid.pack(fill="x")
        for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            r, c = divmod(i, 13)
            btn = tk.Label(quick_grid, text=letter,
                           bg=BG_HIGHLIGHT, fg=FG,
                           font=("Helvetica", 9),
                           width=2, padx=2, pady=4, cursor="hand2")
            btn.grid(row=r, column=c, padx=1, pady=1, sticky="ew")
            btn.bind("<Button-1>", lambda e, l=letter: self.letter_var.set(l))
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=ACCENT))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=BG_HIGHLIGHT))
            quick_grid.columnconfigure(c, weight=1)

        # Sample count
        tk.Label(left, text="Number of samples",
                 font=("Helvetica", 10), bg=BG_CARD, fg=FG).pack(anchor="w", pady=(8, 2))
        sample_frame = tk.Frame(left, bg=BG_CARD)
        sample_frame.pack(fill="x", pady=(0, 8))
        self.samples_var = tk.IntVar(value=400)
        sample_scale = tk.Scale(sample_frame, from_=100, to=800, resolution=50,
                                  orient="horizontal", variable=self.samples_var,
                                  bg=BG_CARD, fg=FG, troughcolor="#404040",
                                  highlightthickness=0,
                                  showvalue=True, sliderrelief="flat",
                                  font=("Helvetica", 9))
        sample_scale.pack(fill="x")

        # Variation
        self.variation_var = tk.BooleanVar(value=True)
        var_check = tk.Checkbutton(left,
                                     text="Use variation prompts (recommended)",
                                     variable=self.variation_var,
                                     bg=BG_CARD, fg=FG, selectcolor="#404040",
                                     activebackground=BG_CARD, activeforeground=FG,
                                     font=("Helvetica", 9))
        var_check.pack(anchor="w", pady=(8, 16))

        # Action button
        self.collect_btn = make_button(left, "📷  Open Camera & Record",
                                         self.start_collection)
        self.collect_btn.pack(fill="x", pady=(0, 12))

        # Tools section
        tk.Label(left, text="DATA TOOLS",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(16, 8))

        tools_row = tk.Frame(left, bg=BG_CARD)
        tools_row.pack(fill="x")
        make_button(tools_row, "Refresh", self.refresh_status,
                    color=BG_HIGHLIGHT, hover="#404040", fg=FG).pack(side="left", expand=True, fill="x", padx=(0, 4))
        make_button(tools_row, "Clear ALL data", self.clear_data,
                    color=DANGER, hover="#E06A40").pack(side="left", expand=True, fill="x", padx=(4, 0))

        # Motion letter info
        tk.Label(left, text="MOTION LETTERS (J / Z)",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(16, 8))
        tk.Label(left,
                 text=("J and Z are now learned by the same classifier.\n"
                       "When collecting them, repeat the motion continuously\n"
                       "during the recording period. The motion features\n"
                       "will be captured automatically."),
                 font=("Helvetica", 9), bg=BG_CARD, fg=FG, justify="left",
                 wraplength=320).pack(anchor="w", pady=(0, 4))

        # ── RIGHT: dataset status ───────────────────────────────────────────
        right = tk.Frame(body, bg=BG_CARD, padx=18, pady=18)
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(right, text="DATASET STATUS",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(0, 12))

        # Summary stats
        stats_frame = tk.Frame(right, bg=BG_CARD)
        stats_frame.pack(fill="x", pady=(0, 12))

        self.total_label = tk.Label(stats_frame, text="0",
                                      font=("Helvetica", 28, "bold"),
                                      bg=BG_CARD, fg=ACCENT)
        self.total_label.pack(anchor="w")
        tk.Label(stats_frame, text="Total samples",
                 font=("Helvetica", 9), bg=BG_CARD, fg=FG_DIM).pack(anchor="w")

        # Per-letter counts header
        row_hdr = tk.Frame(right, bg=BG_CARD)
        row_hdr.pack(fill="x", pady=(8, 4))
        tk.Label(row_hdr, text="Per-letter counts  (click to select, Ctrl+click for multi)",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(side="left")

        # Listbox with scrollbar
        list_frame = tk.Frame(right, bg=BG_CARD)
        list_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical",
                                  bg=BG_CARD, troughcolor=BG_HIGHLIGHT)
        scrollbar.pack(side="right", fill="y")

        self.letter_listbox = tk.Listbox(
            list_frame,
            font=("Consolas", 11),
            bg="#1f1f1f", fg=FG,
            selectbackground=ACCENT, selectforeground="white",
            relief="flat", borderwidth=0,
            highlightthickness=0,
            activestyle="none",
            selectmode="extended",
            yscrollcommand=scrollbar.set,
        )
        self.letter_listbox.pack(side="left", fill="both", expand=True,
                                  padx=(0, 4), pady=4)
        scrollbar.config(command=self.letter_listbox.yview)

        # Double-click to load letter into input field
        self.letter_listbox.bind("<Double-Button-1>", self._on_listbox_select)

        # Clear selected / select all buttons
        clear_sel_row = tk.Frame(right, bg=BG_CARD)
        clear_sel_row.pack(fill="x", pady=(10, 0))
        make_button(clear_sel_row, "Clear selected letters",
                    self.clear_selected_letters,
                    color=DANGER, hover="#E06A40").pack(side="left", expand=True, fill="x", padx=(0, 4))
        make_button(clear_sel_row, "Select all",
                    self._select_all_letters,
                    color=BG_HIGHLIGHT, hover="#404040", fg=FG).pack(side="left")

        # Refresh on first load
        self.refresh_status()

    def refresh_status(self):
        self.letter_listbox.delete(0, "end")
        self._listbox_labels = []

        if not os.path.exists(DATA_PATH):
            self.total_label.config(text="0")
            self.letter_listbox.insert("end", "  No data collected yet.")
            self._listbox_labels.append(None)
            return

        try:
            df = pd.read_csv(DATA_PATH)
        except Exception as e:
            self.letter_listbox.insert("end", f"  Error reading CSV: {e}")
            self._listbox_labels.append(None)
            return

        self.total_label.config(text=str(len(df)))
        counts = df["label"].value_counts().sort_index().to_dict()

        if not counts:
            self.letter_listbox.insert("end", "  No data collected yet.")
            self._listbox_labels.append(None)
            return

        max_count = max(counts.values())
        for label, count in sorted(counts.items()):
            bar_len = int(16 * count / max_count)
            bar = "█" * bar_len + "·" * (16 - bar_len)
            motion_tag = "  ✦" if label in ("J", "Z") else ""
            line = f"  {label:<6}{count:>5} samples   {bar}{motion_tag}"
            self.letter_listbox.insert("end", line)
            self._listbox_labels.append(label)

    def _on_listbox_select(self, event):
        """Double-click loads the letter into the input field."""
        sel = self.letter_listbox.curselection()
        if sel and hasattr(self, "_listbox_labels") and sel[0] < len(self._listbox_labels):
            self.letter_var.set(self._listbox_labels[sel[0]])

    def _select_all_letters(self):
        self.letter_listbox.select_set(0, "end")

    def clear_selected_letters(self):
        sel = self.letter_listbox.curselection()
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Click one or more letters in the list first.")
            return
        if not hasattr(self, "_listbox_labels"):
            return

        labels_to_remove = [self._listbox_labels[i] for i in sel
                             if i < len(self._listbox_labels)
                             and self._listbox_labels[i] is not None]
        if not labels_to_remove:
            messagebox.showinfo("Nothing selected",
                                "Select a letter row, not a section header.")
            return

        msg = ("Delete data for: " + ", ".join(labels_to_remove) +
               "?\n\nThis removes all samples for those letters from the CSV. "
               "This cannot be undone.")
        if not messagebox.askyesno("Clear selected?", msg):
            return

        try:
            df = pd.read_csv(DATA_PATH)
            before = len(df)
            df = df[~df["label"].isin(labels_to_remove)]
            df.to_csv(DATA_PATH, index=False)
            removed = before - len(df)
            self.refresh_status()
            self.app.set_status(f"● Removed {removed} samples", DANGER)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_collection(self):
        letter = self.letter_var.get().strip()
        if not letter:
            messagebox.showwarning("Missing input", "Enter a letter or gesture name.")
            return

        # Normalise casing
        label = letter.upper() if len(letter) == 1 else letter.lower()

        # Check if already collecting
        if self.collection_proc and self.collection_proc.poll() is None:
            messagebox.showinfo("Already running", "Collection is already in progress.")
            return

        # All letters route to collect_data.py — J/Z are handled by the
        # windowed feature pipeline now (no special motion_letters.py path).
        samples = self.samples_var.get()
        use_variation = self.variation_var.get()

        cmd = [sys.executable, os.path.join("src", "collect_data.py"),
               "--gesture", label, "--samples", str(samples)]
        if not use_variation:
            cmd.append("--no-variation")

        self.collect_btn.config(text=f"Recording {label} — see camera window",
                                 state="disabled")
        self.app.set_status(f"● Recording {label}", WARN)

        def worker():
            try:
                self.collection_proc = subprocess.Popen(cmd, cwd=BASE_DIR)
                self.collection_proc.wait()
            finally:
                self.app.frame.after(0, self._on_collection_done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_collection_done(self):
        self.collect_btn.config(text="📷  Open Camera & Record", state="normal")
        self.app.set_status("● Ready", ACCENT)
        self.refresh_status()

    def clear_data(self):
        if not os.path.exists(DATA_PATH):
            messagebox.showinfo("No data", "No dataset to clear.")
            return
        if messagebox.askyesno("Clear all data?",
                               "This will permanently delete all collected samples.\nContinue?"):
            try:
                os.remove(DATA_PATH)
                self.refresh_status()
                self.app.set_status("● Data cleared", DANGER)
            except Exception as e:
                messagebox.showerror("Error", str(e))




# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
class TrainingTab:
    def __init__(self, parent, app):
        self.app = app
        self.frame = tk.Frame(parent, bg=BG)

        # Header
        header = tk.Frame(self.frame, bg=BG, padx=20, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="Train Model",
                 font=("Helvetica", 18, "bold"),
                 bg=BG, fg=FG).pack(anchor="w")
        tk.Label(header,
                 text="Train a classifier on your collected data and view results.",
                 font=("Helvetica", 10), bg=BG, fg=FG_DIM).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(self.frame, bg=BG, padx=20, pady=10)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # ── LEFT: training controls ────────────────────────────────────────
        left = tk.Frame(body, bg=BG_CARD, padx=18, pady=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(left, text="MODEL CONFIGURATION",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(0, 12))

        # Model picker
        tk.Label(left, text="Algorithm",
                 font=("Helvetica", 10), bg=BG_CARD, fg=FG).pack(anchor="w")

        self.model_var = tk.StringVar(value="svm")
        for value, label, desc in [
            ("svm", "SVM (RBF kernel)", "Better for similar signs"),
            ("knn", "K-Nearest Neighbors", "Faster training"),
        ]:
            row = tk.Frame(left, bg=BG_CARD)
            row.pack(fill="x", pady=4)
            tk.Radiobutton(row, text=label, variable=self.model_var, value=value,
                            bg=BG_CARD, fg=FG, selectcolor="#404040",
                            activebackground=BG_CARD, activeforeground=FG,
                            font=("Helvetica", 10)).pack(anchor="w")
            tk.Label(row, text=f"   {desc}",
                     font=("Helvetica", 8), bg=BG_CARD, fg=FG_DIM).pack(anchor="w")

        # K param (for KNN)
        k_frame = tk.Frame(left, bg=BG_CARD)
        k_frame.pack(fill="x", pady=(12, 0))
        tk.Label(k_frame, text="KNN k value",
                 font=("Helvetica", 10), bg=BG_CARD, fg=FG).pack(anchor="w")
        self.k_var = tk.IntVar(value=5)
        tk.Scale(k_frame, from_=1, to=21, resolution=2,
                  orient="horizontal", variable=self.k_var,
                  bg=BG_CARD, fg=FG, troughcolor="#404040",
                  highlightthickness=0, sliderrelief="flat",
                  font=("Helvetica", 9)).pack(fill="x")

        # Train button
        self.train_btn = make_button(left, "▶  Start Training", self.start_training)
        self.train_btn.pack(fill="x", pady=(20, 0))

        # Progress bar
        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", pady=(12, 0))

        # ── RIGHT: training results ────────────────────────────────────────
        right = tk.Frame(body, bg=BG_CARD, padx=18, pady=18)
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(right, text="TRAINING RESULTS",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(0, 12))

        # Top metrics row
        metrics_row = tk.Frame(right, bg=BG_CARD)
        metrics_row.pack(fill="x", pady=(0, 12))

        self.metric_widgets = {}
        for key, label, color in [
            ("accuracy", "Accuracy",        ACCENT),
            ("cv_mean",  "5-Fold CV",       INFO),
            ("n_samples", "Samples",        FG),
            ("n_classes", "Classes",        FG),
        ]:
            card = tk.Frame(metrics_row, bg=BG_HIGHLIGHT, padx=16, pady=12)
            card.pack(side="left", expand=True, fill="x", padx=2)
            val_label = tk.Label(card, text="—",
                                   font=("Helvetica", 18, "bold"),
                                   bg=BG_HIGHLIGHT, fg=color)
            val_label.pack(anchor="w")
            tk.Label(card, text=label,
                     font=("Helvetica", 8), bg=BG_HIGHLIGHT, fg=FG_DIM).pack(anchor="w")
            self.metric_widgets[key] = val_label

        # Detailed report
        tk.Label(right, text="Per-class breakdown",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(12, 8))

        self.report_text = tk.Text(right,
                                     font=("Consolas", 9),
                                     bg="#1f1f1f", fg=FG,
                                     relief="flat", padx=12, pady=10,
                                     highlightthickness=0)
        self.report_text.pack(fill="both", expand=True)
        self.report_text.config(state="disabled")

        self.refresh_results()

    def refresh_results(self):
        if not os.path.exists(REPORT_PATH):
            # Only reset text if it currently says "No model trained yet"
            # so we don't wipe a training-in-progress or error message
            self.report_text.config(state="normal")
            current = self.report_text.get("1.0", "end").strip()
            if not current or current.startswith("No model"):
                self.report_text.delete("1.0", "end")
                self.report_text.insert("1.0", "No model trained yet.\n\nClick 'Start Training' once you have collected data.")
            self.report_text.config(state="disabled")
            return

        try:
            with open(REPORT_PATH) as f:
                report = json.load(f)
        except Exception as e:
            return

        # Update metric cards
        self.metric_widgets["accuracy"].config(text=f"{report['accuracy']*100:.1f}%")
        self.metric_widgets["cv_mean"].config(
            text=f"{report['cv_mean']*100:.1f}%")
        self.metric_widgets["n_samples"].config(text=str(report["n_samples"]))
        self.metric_widgets["n_classes"].config(text=str(report["n_classes"]))

        # Per-class report
        self.report_text.config(state="normal")
        self.report_text.delete("1.0", "end")

        per_class = report["per_class"]
        self.report_text.insert("end", f"  Model: {report['model_type'].upper()}    "
                                          f"Features: {report['n_features']}    "
                                          f"Train/Test: {report['n_train']}/{report['n_test']}\n\n")
        self.report_text.insert("end", f"  {'Class':<10}{'Precision':>11}{'Recall':>11}{'F1':>11}{'Support':>11}\n")
        self.report_text.insert("end", "  " + "-" * 55 + "\n")

        for cls in report["classes"]:
            stats = per_class.get(cls, {})
            p = stats.get("precision", 0) * 100
            r = stats.get("recall", 0) * 100
            f1 = stats.get("f1-score", 0) * 100
            sup = int(stats.get("support", 0))
            self.report_text.insert("end",
                f"  {cls:<10}{p:>10.1f}%{r:>10.1f}%{f1:>10.1f}%{sup:>11}\n")

        # Macro avg
        if "macro avg" in per_class:
            m = per_class["macro avg"]
            self.report_text.insert("end", "  " + "-" * 55 + "\n")
            self.report_text.insert("end",
                f"  {'macro avg':<10}{m['precision']*100:>10.1f}%"
                f"{m['recall']*100:>10.1f}%{m['f1-score']*100:>10.1f}%"
                f"{int(m['support']):>11}\n")

        self.report_text.config(state="disabled")

    def _check_data_format(self):
        """
        Verify the CSV matches the current pipeline's feature size.
        Returns (ok: bool, message: str).
        """
        if not os.path.exists(DATA_PATH):
            return False, "No dataset found. Collect data first."
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception as e:
            return False, f"Cannot read CSV: {e}"

        # Import lazily — features module path setup
        sys.path.insert(0, os.path.join(BASE_DIR, "src"))
        try:
            from features import get_feature_size
            expected_features = get_feature_size()
        except Exception:
            expected_features = 115   # current pipeline default

        expected_cols = expected_features + 1   # + label column
        if len(df.columns) != expected_cols:
            return False, (
                f"Dataset format mismatch.\n\n"
                f"Your CSV has {len(df.columns)} columns but the current pipeline "
                f"expects {expected_cols} ({expected_features} features + label).\n\n"
                f"This happens when data was collected with an older version of the code.\n\n"
                f"Fix: go to the Collect tab, clear all data, and re-collect your letters."
            )
        if len(df) < 10:
            return False, f"Only {len(df)} samples found. Collect more data before training."
        n_classes = df["label"].nunique()
        if n_classes < 2:
            only_label = df["label"].iloc[0]
            msg = ('Only 1 class found: ' + only_label +
                   '. You need at least 2 letters before training. '
                   'Go to the Collect tab and record at least one more letter.')
            return False, msg
        return True, "OK"

    def start_training(self):
        ok, msg = self._check_data_format()
        if not ok:
            messagebox.showerror("Cannot train", msg)
            return

        self.train_btn.config(state="disabled", text="Training...")
        self.progress.start(10)
        self.app.set_status("● Training...", WARN)

        # Show "training in progress" message immediately
        self.report_text.config(state="normal")
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", "Training in progress...\n\nThis may take 30-60 seconds.")
        self.report_text.config(state="disabled")

        def worker():
            cmd = [sys.executable, os.path.join("src", "train_model.py"),
                   "--model", self.model_var.get(),
                   "--k", str(self.k_var.get())]
            try:
                result = subprocess.run(cmd, cwd=BASE_DIR,
                                          capture_output=True, text=True, timeout=300)
                output = result.stdout + result.stderr
                success = result.returncode == 0
            except subprocess.TimeoutExpired:
                output = "Training timed out after 5 minutes."
                success = False
            except Exception as e:
                output = str(e)
                success = False

            self.app.frame.after(0, lambda: self._on_training_done(output, success))

        threading.Thread(target=worker, daemon=True).start()

    def _on_training_done(self, output, success=True):
        self.progress.stop()
        self.train_btn.config(state="normal", text="▶  Start Training")

        if not success or not os.path.exists(MODEL_PATH):
            # Show error output and keep it visible
            self.report_text.config(state="normal")
            self.report_text.delete("1.0", "end")
            error_msg = "Training failed.\n\n"
            error_msg += output if output.strip() else "No output captured."
            self.report_text.insert("1.0", error_msg)
            self.report_text.config(state="disabled")
            self.app.set_status("● Training failed", DANGER)
            messagebox.showerror("Training failed",
                                  "Model was not saved. Check the output below for details.")
            return

        # Training succeeded — load structured report
        self.app.set_status("● Training complete", ACCENT)
        self.refresh_results()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
class AnalysisTab:
    def __init__(self, parent, app):
        self.app = app
        self.frame = tk.Frame(parent, bg=BG)

        header = tk.Frame(self.frame, bg=BG, padx=20, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="Analysis",
                 font=("Helvetica", 18, "bold"),
                 bg=BG, fg=FG).pack(anchor="w")
        tk.Label(header,
                 text="Run model comparison, hyperparameter tuning, error analysis, and view plots.",
                 font=("Helvetica", 10), bg=BG, fg=FG_DIM).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(self.frame, bg=BG, padx=20, pady=10)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # LEFT: control panel
        left = tk.Frame(body, bg=BG_CARD, padx=18, pady=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(left, text="ANALYSIS PIPELINE",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(0, 12))

        items = [
            "✓  Baseline model comparison",
            "✓  KNN vs SVM evaluation",
            "✓  Hyperparameter tuning",
            "✓  Confusion matrix",
            "✓  Inference benchmark",
            "✓  Error analysis",
        ]
        for item in items:
            tk.Label(left, text=item,
                     font=("Helvetica", 10), bg=BG_CARD, fg=FG, anchor="w").pack(fill="x", pady=2)

        self.run_btn = make_button(left, "▶  Run Full Analysis", self.run_analysis)
        self.run_btn.pack(fill="x", pady=(20, 0))

        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", pady=(12, 0))

        tk.Label(left, text="RESULTS FILES",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(20, 8))

        self.files_listbox = tk.Listbox(left,
                                          font=("Consolas", 9),
                                          bg="#1f1f1f", fg=FG,
                                          selectbackground=ACCENT,
                                          relief="flat", borderwidth=0,
                                          highlightthickness=0)
        self.files_listbox.pack(fill="both", expand=True)
        self.files_listbox.bind("<<ListboxSelect>>", self.show_selected)

        make_button(left, "Open results folder", self.open_results,
                    color=BG_HIGHLIGHT, hover="#404040", fg=FG).pack(fill="x", pady=(8, 0))

        # RIGHT: image viewer
        right = tk.Frame(body, bg=BG_CARD, padx=18, pady=18)
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(right, text="VIEWER",
                 font=("Helvetica", 9, "bold"),
                 bg=BG_CARD, fg=FG_DIM).pack(anchor="w", pady=(0, 12))

        self.image_label = tk.Label(right, bg="#1f1f1f",
                                      text="Select a file from the list",
                                      fg=FG_DIM, font=("Helvetica", 11))
        self.image_label.pack(fill="both", expand=True)
        self.image_label.bind("<Configure>", lambda e: self._refresh_displayed_image())

        self._displayed_path = None
        self.refresh_files()

    def run_analysis(self):
        if not os.path.exists(DATA_PATH):
            messagebox.showwarning("No data", "Collect data first.")
            return

        self.run_btn.config(state="disabled", text="Running analysis...")
        self.progress.start(10)
        self.app.set_status("● Running analysis...", WARN)

        def worker():
            cmd = [sys.executable, os.path.join("src", "analysis.py")]
            try:
                subprocess.run(cmd, cwd=BASE_DIR, timeout=600,
                                 capture_output=True, text=True)
            except Exception:
                pass
            self.app.frame.after(0, self._on_done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self):
        self.progress.stop()
        self.run_btn.config(state="normal", text="▶  Run Full Analysis")
        self.app.set_status("● Analysis complete", ACCENT)
        self.refresh_files()

    def refresh_files(self):
        self.files_listbox.delete(0, "end")
        if os.path.exists(RESULTS_DIR):
            files = sorted(os.listdir(RESULTS_DIR))
            for f in files:
                self.files_listbox.insert("end", f)

    def show_selected(self, event):
        selection = self.files_listbox.curselection()
        if not selection:
            return
        filename = self.files_listbox.get(selection[0])
        path = os.path.join(RESULTS_DIR, filename)
        self._displayed_path = path

        if filename.endswith((".png", ".jpg", ".jpeg")):
            self._refresh_displayed_image()
        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(path)
                content = df.to_string(index=False)
            except Exception as e:
                content = str(e)
            self.image_label.config(image="", text=content,
                                      font=("Consolas", 9), justify="left", anchor="nw")
        else:
            self.image_label.config(image="", text=f"Cannot preview: {filename}",
                                      font=("Helvetica", 11))

    def _refresh_displayed_image(self):
        if not self._displayed_path or not self._displayed_path.endswith((".png", ".jpg", ".jpeg")):
            return
        try:
            img = Image.open(self._displayed_path)
            w = self.image_label.winfo_width()
            h = self.image_label.winfo_height()
            if w > 50 and h > 50:
                img.thumbnail((w - 20, h - 20), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            self.image_label.config(image="", text=str(e))

    def open_results(self):
        if not os.path.exists(RESULTS_DIR):
            messagebox.showinfo("No results", "Run analysis first.")
            return
        if sys.platform == "win32":
            os.startfile(RESULTS_DIR)
        elif sys.platform == "darwin":
            subprocess.run(["open", RESULTS_DIR])
        else:
            subprocess.run(["xdg-open", RESULTS_DIR])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class TrainingApp:
    def __init__(self, root):
        self.root = root
        root.title("ASL Training & Analysis Studio")
        root.configure(bg=BG)
        root.geometry("1200x800")
        root.minsize(1000, 700)

        # Header
        header = tk.Frame(root, bg=BG, padx=20, pady=10)
        header.pack(fill="x")
        tk.Label(header, text="ASL Training Studio",
                 font=("Helvetica", 14, "bold"),
                 bg=BG, fg=FG).pack(side="left")
        self.status_label = tk.Label(header, text="● Ready",
                                       font=("Helvetica", 10),
                                       bg=BG, fg=ACCENT)
        self.status_label.pack(side="right")

        # Notebook (tabs)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",
                          background=BG_PANEL, foreground=FG_DIM,
                          padding=[20, 10], font=("Helvetica", 10, "bold"),
                          borderwidth=0)
        style.map("TNotebook.Tab",
                    background=[("selected", BG_CARD)],
                    foreground=[("selected", FG)])

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # We'll use a wrapper frame that has a `.set_status` method
        self.frame = root  # used by tabs to call .after

        self.tab_collect  = CollectionTab(self.notebook, self)
        self.tab_train    = TrainingTab(self.notebook, self)
        self.tab_analysis = AnalysisTab(self.notebook, self)

        self.notebook.add(self.tab_collect.frame,  text="  📷 Collect  ")
        self.notebook.add(self.tab_train.frame,    text="  🧠 Train  ")
        self.notebook.add(self.tab_analysis.frame, text="  📊 Analyze  ")

        # Refresh dataset on tab switch
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _on_tab_change(self, event):
        idx = self.notebook.index("current")
        if idx == 0:
            self.tab_collect.refresh_status()
        elif idx == 1:
            self.tab_train.refresh_results()
        elif idx == 2:
            self.tab_analysis.refresh_files()

    def set_status(self, text, color):
        self.status_label.config(text=text, fg=color)


def _show_first_launch_dialog(root):
    """
    Show a one-time onboarding dialog on first launch.
    Returns True if user chose to load pre-collected data, False otherwise.
    If the dialog is dismissed without choosing, defaults to fresh start.
    """
    # Already seen this dialog — skip
    if os.path.exists(FIRST_LAUNCH_FLAG):
        return False

    # Pre-collected data must exist to offer this option
    if not os.path.exists(PRETRAINED_CSV):
        _mark_launched()
        return False

    result = {"choice": None}

    # Build the dialog window
    dialog = tk.Toplevel(root)
    dialog.title("Welcome to ASL Training Studio")
    dialog.configure(bg=BG)
    dialog.resizable(False, False)
    dialog.grab_set()  # modal

    # Center on screen
    dialog.update_idletasks()
    w, h = 540, 380
    x = (dialog.winfo_screenwidth()  - w) // 2
    y = (dialog.winfo_screenheight() - h) // 2
    dialog.geometry(f"{w}x{h}+{x}+{y}")

    # Header
    tk.Label(dialog, text="Welcome!",
             font=("Helvetica", 20, "bold"),
             bg=BG, fg=FG).pack(pady=(30, 6))
    tk.Label(dialog,
             text="Would you like to start with pre-collected training data,\nor collect your own from scratch?",
             font=("Helvetica", 11), bg=BG, fg=FG_DIM,
             justify="center").pack(pady=(0, 30))

    # Two option cards side by side
    cards = tk.Frame(dialog, bg=BG)
    cards.pack(padx=30, fill="x")

    def make_card(parent, title, subtitle, icon, command, accent_color):
        card = tk.Frame(parent, bg=BG_CARD, padx=20, pady=20,
                        highlightthickness=2,
                        highlightbackground=BG_HIGHLIGHT,
                        highlightcolor=accent_color)
        card.pack(side="left", expand=True, fill="both", padx=(0, 8))

        tk.Label(card, text=icon, font=("Helvetica", 28),
                 bg=BG_CARD, fg=accent_color).pack()
        tk.Label(card, text=title, font=("Helvetica", 12, "bold"),
                 bg=BG_CARD, fg=FG).pack(pady=(8, 4))
        tk.Label(card, text=subtitle, font=("Helvetica", 9),
                 bg=BG_CARD, fg=FG_DIM, wraplength=180,
                 justify="center").pack()

        btn = tk.Button(card, text="Choose this",
                        font=("Helvetica", 9, "bold"),
                        bg=accent_color, fg="white",
                        activebackground=accent_color,
                        relief="flat", borderwidth=0,
                        padx=12, pady=6, cursor="hand2",
                        command=command)
        btn.pack(pady=(14, 0))

        # Hover highlight on the card border
        def on_enter(e):
            card.config(highlightbackground=accent_color)
        def on_leave(e):
            card.config(highlightbackground=BG_HIGHLIGHT)
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        return card

    def choose_pretrained():
        result["choice"] = "pretrained"
        dialog.destroy()

    def choose_fresh():
        result["choice"] = "fresh"
        dialog.destroy()

    make_card(cards,
              "Use pre-collected data",
              "Load ~400 samples per letter recorded by the developer. You can still add your own data on top.",
              "📦", choose_pretrained, ACCENT)

    make_card(cards,
              "Start from scratch",
              "Collect your own data from your webcam. Takes ~90 min but is personalised to your hands.",
              "📷", choose_fresh, "#534AB7")

    # Fine print
    tk.Label(dialog,
             text="You can always clear data and re-collect later from the Collect tab.",
             font=("Helvetica", 8), bg=BG, fg=FG_DIM).pack(pady=(20, 0))

    root.wait_window(dialog)

    # Act on the choice
    chosen_pretrained = result["choice"] == "pretrained"
    if chosen_pretrained:
        _load_pretrained_data()

    _mark_launched()
    return chosen_pretrained


def _load_pretrained_data():
    """Copy the bundled gestures.csv into data/ if it exists."""
    if not os.path.exists(PRETRAINED_CSV):
        messagebox.showerror(
            "Pre-collected data not found",
            f"Expected file not found:\n{PRETRAINED_CSV}\n\n"
            "Starting with an empty dataset instead."
        )
        return

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # If user already has data, ask before overwriting
    if os.path.exists(DATA_PATH):
        try:
            existing = pd.read_csv(DATA_PATH)
            if len(existing) > 0:
                answer = messagebox.askyesno(
                    "Existing data found",
                    f"You already have {len(existing)} samples in your dataset.\n\n"
                    "Merge the pre-collected data on top of your existing data?\n\n"
                    "(Choose No to replace your data entirely.)"
                )
                if answer:
                    # Merge: append pretrained to existing
                    pretrained = pd.read_csv(PRETRAINED_CSV)
                    merged = pd.concat([existing, pretrained], ignore_index=True)
                    merged.to_csv(DATA_PATH, index=False)
                    messagebox.showinfo(
                        "Data merged",
                        f"Added {len(pretrained)} pre-collected samples to your dataset.\n"
                        f"Total samples: {len(merged)}"
                    )
                    return
                # else fall through to replace
        except Exception:
            pass  # Can't read existing — just overwrite

    # Copy pretrained CSV as the active dataset
    import shutil
    shutil.copy2(PRETRAINED_CSV, DATA_PATH)
    try:
        n = len(pd.read_csv(DATA_PATH))
        messagebox.showinfo(
            "Pre-collected data loaded",
            f"Loaded {n} samples ({n // 26} per letter average).\n\n"
            "You can add your own samples on top at any time from the Collect tab.\n"
            "Train the model when you're ready."
        )
    except Exception:
        pass


def _mark_launched():
    """Write the flag file so the dialog doesn't show again."""
    os.makedirs(os.path.dirname(FIRST_LAUNCH_FLAG), exist_ok=True)
    open(FIRST_LAUNCH_FLAG, "w").write("done")


def main():
    root = tk.Tk()
    root.withdraw()   # hide main window during dialog

    _show_first_launch_dialog(root)

    root.deiconify()  # show main window
    app = TrainingApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("ERROR: Pillow is not installed. Run: pip install Pillow")
        sys.exit(1)
    main()
