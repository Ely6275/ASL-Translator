# Attribution

## AI Tool Usage

This project used **Claude** (Anthropic) as a development assistant during implementation. The role of AI in this project was supportive — handling syntax, boilerplate, and front-end development — while all meaningful technical and design decisions were made by me.

---

### My contributions (architecture, design, and technical decisions)

**Project concept and problem framing:** I identified ASL finger-spelling as the application domain, motivated by the accessibility gap for deaf and hard-of-hearing users. The decision to target real-time letter-by-letter translation rather than full-word or phrase recognition was mine, based on what was technically feasible within the project timeline.

**Core technical insight — windowed motion features:** The fundamental approach of this project — encoding recent fingertip motion statistics alongside static hand pose geometry so a single classifier can distinguish motion letters (J, Z) from their static look-alikes (I, S) — was my idea. I arrived at this after observing that the per-frame classifier failed on I/J, reasoning that the distinguishing information is temporal rather than spatial, and proposing the windowed feature approach as a solution. This is the central ML contribution of the project.

**Feature engineering design:** I specified which geometric properties of the hand are meaningful for distinguishing ASL letters: curl angles per finger, pairwise fingertip distances, thumb position relative to the knuckle line, and palm orientation. These choices were based on my understanding of what makes letters like A/E/S, M/N/T, and I/J similar or different at the joint level.

**Motion feature specification:** I specified that motion features should be computed per-fingertip over a rolling 15-frame window, and that mean velocity, max velocity, path length, net displacement, and direction were the right quantities to capture — enough to characterize both the magnitude and shape of a motion path without being overfit to any particular execution speed.

**Model selection rationale:** I chose SVM over KNN as the primary classifier after reasoning that SVM's margin-maximizing decision boundary would generalize better to the tight, overlapping clusters formed by similar letter pairs. This was confirmed by the quantitative model comparison in `src/analysis.py`.

**System architecture decisions:** I designed the multi-module pipeline (collect → features → train → predict → GUI), specified that the `WindowedFeatureExtractor` should be a stateful class shared between training and inference so feature distributions match exactly, and specified that the letter acceptor should require temporal stability across frames rather than committing on a single high-confidence prediction.

**First-launch onboarding flow:** I designed the pre-collected data feature — the idea of bundling `data/pretrained/gestures.csv` in the repo and offering it to new users on first launch so they can skip data collection and go straight to training. I specified the dialog behavior (merge vs. replace logic, one-time flag file) and the `.gitignore` rules separating user-generated data from committed data.

**Debugging and iteration:** I identified the root causes of the major issues encountered during development:
- Recognizing that the 1fps stutter in the translator was caused by a double-scheduled Tkinter callback, not by classification latency
- Identifying that training was failing silently because the CSV was in the wrong column format after a pipeline update
- Recognizing that the I/J DTW approach was architecturally flawed (required hardcoded routing, did not generalize) and proposing the windowed feature replacement
- Diagnosing the missing `x-api-key` and `anthropic-version` headers as the cause of the AI suggestion feature returning blank results
- Identifying the uppercase/lowercase label mismatch (`a` vs `A`) caused by the data collection script

**Data collection:** All training data was recorded by me — approximately 400 samples per letter across all 26 letters of the ASL alphabet, with deliberate variation in hand position, angle, and distance as prompted by the collection script's four-phase variation system.

---

### What AI assisted with

**Syntax and implementation:** Once I specified what each module should do, Claude translated those specifications into working Python. This includes the NumPy operations inside `src/features.py` (curl angle computation, path length, DTW), the scikit-learn pipeline configuration in `src/train_model.py`, the MediaPipe Tasks API initialization in `src/predict.py`, and the OpenCV overlay drawing code.

**Front-end development:** The Tkinter GUIs (`translator_gui.py`, `training_gui.py`) were built by Claude based on my description of what each section should contain and how it should behave. GUI layout, widget configuration, theming, and the training GUI's three-tab structure were generated rather than hand-coded by me.

**Boilerplate and infrastructure:** File I/O, CSV handling, argparse setup, subprocess spawning in the training GUI, the `.gitignore`, and similar implementation-level plumbing were AI-generated based on my specifications.

**Debugging assistance:** When errors occurred, I diagnosed the root cause and described what needed to change. Claude then implemented the fix. Examples: MediaPipe API migration from `mp.solutions` to the Tasks API (I identified the breaking change, Claude rewrote the affected calls), the Tkinter frame loop restructure (I identified the double-scheduling, Claude rewrote the update loop without early returns), and the `api_config.py` module (I identified the missing headers, Claude wrote the key management module).

**Analysis pipeline:** `src/analysis.py` was generated by Claude based on my specification of which analyses to run — baseline comparison, KNN vs SVM comparison, hyperparameter sweeps, confusion matrices, inference benchmarking, and error analysis — and which plots to produce for the rubric.

---

### Summary

The research question, system design, feature engineering approach, model selection, and all significant technical decisions are mine. Claude accelerated the implementation of those decisions — handling the parts of software development that are time-consuming but not intellectually central to the ML work: GUI layout, boilerplate, syntax, and plot generation. Every piece of code Claude produced was written in response to a specification I gave, and every significant bug was diagnosed by me before being fixed.

---

## Third-Party Models and Libraries

| Resource | Use | License |
|---|---|---|
| **MediaPipe** (Google) | Hand landmark detection — 21-keypoint hand pose estimation, used as frozen feature extractor | Apache 2.0 |
| **scikit-learn** | SVM and KNN classifiers, StandardScaler, cross-validation, evaluation metrics | BSD-3 |
| **OpenCV** | Webcam capture, frame rendering, landmark overlay drawing | Apache 2.0 |
| **Anthropic Claude API** | AI sentence completion suggestions (`claude-sonnet-4-20250514`) | Commercial API |
| **Pillow** | OpenCV BGR frame → Tkinter-compatible image conversion | HPND |
| **Matplotlib / Seaborn** | Analysis and evaluation plots in `src/analysis.py` | BSD / BSD |
| **NumPy / Pandas** | Numerical computation and dataset management | BSD |

The MediaPipe `hand_landmarker.task` model file (~2MB) is downloaded automatically from Google's public model repository on first run and cached in `models/`.

## ASL Reference

ASL hand shape references were consulted from publicly available educational resources during data collection. No copyrighted imagery was used in the project. All training data was self-recorded by the project author.
