import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode


LITE_MODEL_PATH = "pose_landmarker_lite.task"
LITE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

HEAVY_MODEL_PATH = "pose_landmarker_heavy.task"
HEAVY_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"


def ensure_model(path, url):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)
        print("Done.")


def create_lite_landmarker():
    ensure_model(LITE_MODEL_PATH, LITE_MODEL_URL)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=LITE_MODEL_PATH),
        running_mode=RunningMode.VIDEO
    )
    return PoseLandmarker.create_from_options(options)


def create_heavy_landmarker():
    ensure_model(HEAVY_MODEL_PATH, HEAVY_MODEL_URL)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HEAVY_MODEL_PATH),
        running_mode=RunningMode.IMAGE
    )
    return PoseLandmarker.create_from_options(options)


def reanalyze_with_heavy_model(frames, heavy_landmarker):
    results = []
    last_good = None

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = heavy_landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            last_good = landmarks
            results.append(landmarks)
        elif last_good is not None:
            results.append(last_good)
        else:
            results.append(None)

    first_good = next((r for r in results if r is not None), None)
    if first_good is not None:
        for i, r in enumerate(results):
            if r is None:
                results[i] = first_good
            else:
                break

    return results
