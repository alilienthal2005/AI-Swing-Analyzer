import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import urllib.request
import os

MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


if not os.path.exists(MODEL_PATH):
    print("Downloading pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (25, 27), (24, 26), (26, 28),             # legs
    (27, 29), (29, 31), (28, 30), (30, 32)              # feet
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO
)

cap = cv2.VideoCapture(0)
print("Starting Camera — press Q to quit")

with PoseLandmarker.create_from_options(options) as landmarker:
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((frame_num / 30) * 1000)
        frame_num += 1

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            h, w = frame.shape[:2]
            landmarks = result.pose_landmarks[0]

            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            for start, end in CONNECTIONS:
                x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Swing AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
    




