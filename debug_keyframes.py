"""Standalone debug tool for find_key_frames.

Usage:
    python debug_keyframes.py [swing_name]

Loads a saved swing video (swings/<swing_name>.mp4), re-runs the heavy pose
landmarker over it (same model used by analyzer.py's re-analysis pass), then
runs find_key_frames and prints + plots the wrist-y signal with the four
selected key frame indices marked -- so a change to find_key_frames can be
visually sanity-checked against real recorded swings before trusting the
resulting metrics.

If no swing_name is given, runs against every swing in swings/.
Landmark extraction is cached under swing_keys/ since the heavy model takes
a few seconds per swing.
"""
import os
import pickle
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from body_metrics import find_key_frames, tempo_ratio, top_hand_height
from pose_models import create_heavy_landmarker, reanalyze_with_heavy_model

SWINGS_DIR = "swings"
CACHE_DIR = "swing_keys"
PRE_SWING_LENGTH = 60  # must match analyzer.py


def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_landmarks(name, landmarker):
    cache_path = os.path.join(CACHE_DIR, f"{name}_landmarks_debug.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    frames = load_frames(os.path.join(SWINGS_DIR, f"{name}.mp4"))
    landmarks = reanalyze_with_heavy_model(frames, landmarker)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(landmarks, f)
    return landmarks


def debug_swing(name, landmarker):
    landmarks = get_landmarks(name, landmarker)
    n = len(landmarks)
    swing_start_idx = min(PRE_SWING_LENGTH - 1, n - 1)

    key_frames = find_key_frames(landmarks, swing_start_idx=swing_start_idx)
    wrist_y = [frame[15].y for frame in landmarks]

    tempo = tempo_ratio(landmarks, swing_start_idx=swing_start_idx)
    hand_height = top_hand_height(landmarks, swing_start_idx=swing_start_idx)

    print(f"\n=== {name} ===")
    print(f"frames: {n}, swing_start_idx (as passed by analyzer.py): {swing_start_idx}")
    print(f"key_frames: {key_frames}")
    print(f"tempo_ratio: {tempo:.2f}  top_hand_height: {hand_height:.1f}%")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(wrist_y, color="steelblue", linewidth=1, label="wrist-y (left wrist, landmark 15)")
    ax.axvline(swing_start_idx, color="gray", linestyle=":", label="swing_start_idx (raw)")

    markers = [
        ("address", key_frames["address"], "green"),
        ("top", key_frames["top"], "orange"),
        ("impact", key_frames["impact"], "red"),
        ("finish", key_frames["finish"], "purple"),
    ]
    for label, idx, color in markers:
        ax.axvline(idx, color=color, linestyle="--", linewidth=1.5)
        ax.annotate(label, (idx, wrist_y[idx]), textcoords="offset points",
                    xytext=(4, 8), color=color, fontweight="bold")

    ax.invert_yaxis()  # lower y = higher hand position
    ax.set_xlabel("frame index")
    ax.set_ylabel("wrist y (normalized, inverted)")
    ax.set_title(f"find_key_frames debug: {name}")
    ax.legend(loc="upper right", fontsize=8)

    out_path = os.path.join(CACHE_DIR, f"{name}_keyframes_debug.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"plot saved: {out_path}")


def main():
    landmarker = create_heavy_landmarker()
    try:
        if len(sys.argv) > 1:
            names = [sys.argv[1]]
        else:
            names = sorted(f[:-4] for f in os.listdir(SWINGS_DIR) if f.endswith(".mp4"))
        for name in names:
            debug_swing(name, landmarker)
    finally:
        landmarker.close()


if __name__ == "__main__":
    main()
