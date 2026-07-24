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
a few seconds per swing. Plots are saved to swing_graphs/, same place
analyzer.py writes them automatically for each newly recorded swing.
"""
import os
import pickle
import sys

import cv2

from body_metrics import find_key_frames, tempo_ratio, top_hand_height
from plotting import save_keyframes_plot
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

    tempo = tempo_ratio(landmarks, swing_start_idx=swing_start_idx)
    hand_height = top_hand_height(landmarks, swing_start_idx=swing_start_idx)

    print(f"\n=== {name} ===")
    print(f"frames: {n}, swing_start_idx (as passed by analyzer.py): {swing_start_idx}")
    print(f"key_frames: {key_frames}")
    print(f"tempo_ratio: {tempo:.2f}  top_hand_height: {hand_height:.1f}%")

    # video filenames are "swing_{timestamp}.mp4" -- strip the prefix back
    # off so save_keyframes_plot can re-add it consistently with the other
    # swing_keys/ and swings/ artifacts for this same timestamp.
    timestamp = name[len("swing_"):] if name.startswith("swing_") else name
    out_path = save_keyframes_plot(landmarks, key_frames, timestamp, swing_start_idx=swing_start_idx)
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
