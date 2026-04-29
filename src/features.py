"""
features.py (v2)

Windowed feature extraction for ASL letter recognition with motion awareness.

Each sample combines:
  1. Static pose features from the current frame (85 features)
  2. Motion features computed over a rolling window of the last N frames
     (per-fingertip velocity statistics, path length, net displacement, direction)

This lets a single classifier distinguish motion letters (J, Z) from their
static look-alikes (I, S) without hardcoded rules — the model learns that
"high pinky velocity + curving path = J" on its own.

Total feature size: 85 (static) + 30 (motion) = 115 features per sample
"""

import numpy as np
from collections import deque

# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP, THUMB_CMC = 4, 3, 2, 1
INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP = 8, 7, 6, 5
MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP = 12, 11, 10, 9
RING_TIP, RING_DIP, RING_PIP, RING_MCP = 16, 15, 14, 13
PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP = 20, 19, 18, 17

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

FINGER_CHAINS = {
    "thumb":  [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
    "index":  [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
    "middle": [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
    "ring":   [RING_MCP, RING_PIP, RING_DIP, RING_TIP],
    "pinky":  [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
}

WINDOW_SIZE = 15            # frames in the motion window (~0.5s at 30fps)
STATIC_FEATURES = 85
MOTION_FEATURES = 30
TOTAL_FEATURES  = STATIC_FEATURES + MOTION_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# STATIC POSE FEATURES (same as v1)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_landmarks(coords):
    coords = coords.copy()
    coords -= coords[WRIST]
    scale = np.max(np.linalg.norm(coords, axis=1)) + 1e-6
    coords /= scale
    return coords


def _curl_angles(coords):
    angles = []
    for finger, chain in FINGER_CHAINS.items():
        total_angle = 0.0
        for i in range(1, len(chain) - 1):
            v1 = coords[chain[i - 1]] - coords[chain[i]]
            v2 = coords[chain[i + 1]] - coords[chain[i]]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            total_angle += np.pi - np.arccos(cos_angle)
        angles.append(total_angle)
    return np.array(angles)


def _fingertip_distances(coords):
    distances = []
    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            distances.append(np.linalg.norm(coords[FINGERTIPS[i]] - coords[FINGERTIPS[j]]))
    return np.array(distances)


def _thumb_features(coords):
    return np.array([
        np.linalg.norm(coords[THUMB_TIP] - coords[INDEX_PIP]),
        np.linalg.norm(coords[THUMB_TIP] - coords[PINKY_MCP]),
        coords[THUMB_TIP][1] - coords[INDEX_MCP][1],
        np.linalg.norm(coords[THUMB_TIP] - coords[WRIST]) /
            (np.linalg.norm(coords[THUMB_CMC] - coords[WRIST]) + 1e-6),
    ])


def _palm_orientation(coords):
    v1 = coords[INDEX_MCP] - coords[WRIST]
    v2 = coords[PINKY_MCP] - coords[WRIST]
    n = np.cross(v1, v2)
    return n / (np.linalg.norm(n) + 1e-6)


def extract_static_features(landmarks):
    """Returns 85-dim static pose feature vector."""
    if hasattr(landmarks[0], "x"):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    else:
        coords = np.array(landmarks)
    coords = _normalize_landmarks(coords)
    return np.concatenate([
        coords.flatten(),
        _curl_angles(coords),
        _fingertip_distances(coords),
        _thumb_features(coords),
        _palm_orientation(coords),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# MOTION FEATURES — computed from a rolling window of recent frames
# ══════════════════════════════════════════════════════════════════════════════

def extract_motion_features(landmark_history):
    """
    Compute 30-dim motion feature vector from a window of recent landmark frames.

    Args:
        landmark_history: list/deque of (21, 3) numpy arrays, each one a frame
                          of normalized landmarks. Must have at least 2 frames.

    For each fingertip (5 of them), we compute:
        - mean velocity magnitude over the window
        - max velocity magnitude over the window
        - path length (total distance traveled)
        - net displacement (start to end straight-line distance)
        - x-direction of net displacement
        - y-direction of net displacement

    5 fingertips * 6 features = 30 features.
    """
    history = list(landmark_history)
    if len(history) < 2:
        return np.zeros(MOTION_FEATURES, dtype=np.float32)

    # Stack into (T, 21, 3) array
    history_arr = np.stack(history)  # (T, 21, 3)

    motion_features = []
    for tip_idx in FINGERTIPS:
        # Position of this fingertip over time, wrist-relative (already normalized)
        path = history_arr[:, tip_idx, :2]  # (T, 2) — only x,y matter for motion

        # Frame-to-frame velocities
        diffs = np.diff(path, axis=0)
        speeds = np.linalg.norm(diffs, axis=1)  # (T-1,)

        mean_v = float(np.mean(speeds)) if len(speeds) else 0.0
        max_v  = float(np.max(speeds))  if len(speeds) else 0.0

        # Path length = sum of step distances
        path_length = float(np.sum(speeds))

        # Net displacement = straight line from start to end
        net_disp = path[-1] - path[0]
        net_disp_mag = float(np.linalg.norm(net_disp))

        # Direction (unit vector); fall back to (0,0) if barely moved
        if net_disp_mag > 1e-4:
            direction = net_disp / net_disp_mag
        else:
            direction = np.zeros(2, dtype=np.float32)

        motion_features.extend([
            mean_v, max_v, path_length, net_disp_mag,
            float(direction[0]), float(direction[1]),
        ])

    return np.array(motion_features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED EXTRACTOR — used at both training and inference time
# ══════════════════════════════════════════════════════════════════════════════

class WindowedFeatureExtractor:
    """
    Stateful extractor that maintains a rolling buffer of recent normalized
    landmark frames and produces combined (static + motion) feature vectors
    on each call.

    Use the same class at training time (in collect_data) and inference time
    (in predict) so the feature distributions match.
    """

    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def reset(self):
        self.history.clear()

    def is_warmed_up(self):
        """True when we have enough history to produce real motion features."""
        return len(self.history) >= self.window_size

    def update(self, landmarks):
        """
        Add a frame to the buffer and return a (115,) feature vector.

        Args:
            landmarks: list of MediaPipe landmark objects with .x, .y, .z

        Returns:
            (115,) numpy array — the combined feature vector for this frame.
            The motion portion will be zeros until the buffer is full.
        """
        if hasattr(landmarks[0], "x"):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        else:
            coords = np.array(landmarks)

        # Normalize coords once and store
        coords_norm = _normalize_landmarks(coords)
        self.history.append(coords_norm)

        # Static features from the current frame
        static = np.concatenate([
            coords_norm.flatten(),
            _curl_angles(coords_norm),
            _fingertip_distances(coords_norm),
            _thumb_features(coords_norm),
            _palm_orientation(coords_norm),
        ])

        # Motion features from the window
        motion = extract_motion_features(self.history)

        return np.concatenate([static, motion]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_size():
    return TOTAL_FEATURES


def get_window_size():
    return WINDOW_SIZE


def get_feature_names():
    names = []
    # Static block (85)
    for i in range(21):
        for axis in ["x", "y", "z"]:
            names.append(f"lm{i}_{axis}")
    for finger in FINGER_NAMES:
        names.append(f"curl_{finger}")
    fingers = ["T", "I", "M", "R", "P"]
    for i in range(len(fingers)):
        for j in range(i + 1, len(fingers)):
            names.append(f"dist_{fingers[i]}{fingers[j]}")
    names += ["thumb_to_index_pip", "thumb_to_pinky_mcp",
              "thumb_y_rel_index", "thumb_extension"]
    names += ["palm_nx", "palm_ny", "palm_nz"]
    # Motion block (30)
    for finger in FINGER_NAMES:
        names += [
            f"motion_{finger}_mean_vel",
            f"motion_{finger}_max_vel",
            f"motion_{finger}_path_len",
            f"motion_{finger}_net_disp",
            f"motion_{finger}_dir_x",
            f"motion_{finger}_dir_y",
        ]
    return names
