# Setup Instructions

## Requirements

- Python 3.10 or higher (tested on Python 3.14)
- A webcam connected to your computer
- ~500 MB of disk space (for training data and models)

## Step 1: Clone or download the repository

```bash
git clone <your-repo-url>
cd gesture_robot_controller
```

## Step 2: Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs: opencv-python, mediapipe, scikit-learn, numpy, pandas, matplotlib, seaborn, Pillow.

> **Note:** On first run, the system will automatically download the MediaPipe hand landmark model (~2MB) into `models/hand_landmarker.task`. An internet connection is required for this.

## Step 3: Collect training data

Launch the Training Studio GUI:

```bash
python training_gui.py
```

In the **Collect** tab:

1. Select a letter (A–Z) using the quick-pick grid or type it in the input field
2. Set the sample count to 400 (default)
3. Click **📷 Open Camera & Record**
4. A camera window will open with a 3-second countdown
5. Hold the ASL sign for the selected letter for the duration of the recording
   - **For static letters (A–I, K–Y):** Hold the pose but let your hand drift and vary naturally — do NOT freeze it
   - **For motion letters (J, Z):** Repeat the motion continuously throughout the recording (the system captures the motion pattern)
6. Press **Q** to save and close the camera window
7. Repeat for all 26 letters

Tip: Run `python src/check_data.py` to see how many samples you have per letter.

## Step 4: Train the model

In the **Train** tab of the Training Studio:

1. Select **SVM** (recommended) or KNN
2. Click **▶ Start Training**
3. Training takes 30–60 seconds depending on dataset size
4. Results appear automatically when complete

Or from the command line:
```bash
python src/train_model.py --model svm
```

## Step 5: Run the translator

```bash
python translator_gui.py
```

Or the CLI version:
```bash
python app.py
```

## Step 6 (Optional): Run the ML analysis

```bash
python src/analysis.py
```

This generates 7 plots in the `results/` folder covering baseline comparison, model comparison, hyperparameter tuning, confusion matrices, inference benchmarks, and error analysis.

## Troubleshooting

**"Classifier not found" error:** You need to train the model first (Step 4).

**Webcam not opening:** Try `python app.py --camera 1` to use a different camera index.

**Wrong feature size error:** Your existing data was collected with a previous version of the code. Go to the Collect tab, click **Clear ALL data**, and re-collect all letters.

**Package version conflicts:** The code has been tested with the latest available versions of all packages. Remove version pins from `requirements.txt` if you encounter conflicts.

## Fallback (v1 system)

If you have data collected from a previous version and don't want to re-collect, the original DTW-based system is available:

```bash
python translator_gui_v1.py
```

The v1 system uses separate files (`data/gestures_v1.csv`, `models/gesture_classifier_v1.pkl`) and does not conflict with v2.
