import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import urllib.request
import os


from detection import SwingDetector
from posture import spine_angle, arm_hang_angle, knee_flex
from buffer import FrameBuffer
from recording import save_swing

MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading pose model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


SPINE_LINES = [(11, 12), (11, 23), (12, 24), (23, 24)]
ARM_LINES = [(11, 13), (13, 15), (12, 14), (14, 16)]
LEG_LINES = [(23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32)]


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO
)

cap = cv2.VideoCapture(0)
print("Starting Camera — press Q to quit")

detector = SwingDetector()
buffer = FrameBuffer(max_frames=120) 

swing_in_progress = False
post_swing_frames = []
POST_SWING_LENGTH = 120

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

            spine = spine_angle(landmarks)
            arms = arm_hang_angle(landmarks)
            knees = knee_flex(landmarks)


            ## adjustable parameters based on personal swing

            spine_color = (0, 255, 0) if 5 <= spine <= 25 else (0, 0, 255)
            arms_color = (0, 255, 0) if 82 <= arms <= 90 else (0, 0, 255)
            legs_color = (0, 255, 0) if 145 <= knees <= 165 else (0, 0, 255)

            all_green =  5 <= spine <= 25 and 82 <= arms <= 90 and 145 <= knees <= 165

            event = detector.update(landmarks, all_green)
            if event == "swing_started":
                swing_in_progress = True 
                post_swing_frames = []
            if swing_in_progress:
                post_swing_frames.append(frame.copy())

                if len(post_swing_frames) >= POST_SWING_LENGTH:
                    all_frames = buffer.get_all() + post_swing_frames
                    filename = save_swing(all_frames, fps=60)
                    print(f"Saved: {filename}")

                    swing_in_progress = False
                    post_swing_frames = []
                    buffer.clear()

            


            state_text = detector.state 

            state_colors = {"IDLE": (255, 255, 255), "ARMED": (0, 255, 0),"SWINGING": (0, 255, 0),"COOLDOWN": (0, 0, 255)}
            color = state_colors.get(state_text , (255,255,255))

            cv2.putText(frame, f"State: {state_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


            if event == "armed":
                print(" ARMED")
            elif event == "swing_started":
                print("🏌️ SWING DETECTED")
            elif event == "disarmed":
                print(" Disarmed")

            for start, end in SPINE_LINES:
                x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), spine_color, 3)

            for start, end in ARM_LINES:
                x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), arms_color, 3)

            for start, end in LEG_LINES:
                x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), legs_color, 3)

            bar_height = 50
            cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)
            
            cv2.putText(frame, f"Spine: {spine:.0f}", (20, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, spine_color, 2)
            cv2.putText(frame, f"Arms: {arms:.0f}", (250, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, arms_color, 2)
            cv2.putText(frame, f"Knees: {knees:.0f}", (480, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, legs_color, 2)
            
        buffer.add(frame)
            
        cv2.imshow("Swing AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
    




