import cv2
import mediapipe as mp
import os
from datetime import datetime

from detection import SwingDetector
from posture import spine_angle, arm_hang_angle, knee_flex
from buffer import FrameBuffer, LandmarkBuffer
from recording import save_swing, save_key_frames
from body_metrics import tempo_ratio, head_movement, top_hand_height, spine_angle_maintenance, find_key_frames
from coach import analyze_session
from database import init_db, save_swing_to_db, get_last_n_swings
from pose_models import create_lite_landmarker, create_heavy_landmarker, reanalyze_with_heavy_model


SPINE_LINES = [(11, 12), (11, 23), (12, 24), (23, 24)]
ARM_LINES = [(11, 13), (13, 15), (12, 14), (14, 16)]
LEG_LINES = [(23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32)]

VISIBILITY_THRESHOLD = 0.6
PRE_SWING_LENGTH = 60
POST_SWING_LENGTH = 90

SPINE_MIN, SPINE_MAX = 5, 25
ARMS_MIN, ARMS_MAX = 82, 90
KNEES_MIN, KNEES_MAX = 135, 165

CLUB = input("Which club are you using? (e.g. 7iron, driver, pw): ").strip()

heavy_landmarker = create_heavy_landmarker()

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1:
    fps = 30
print(f"Actual FPS: {fps}")
print("Starting Camera — press Q to quit")

init_db()
os.makedirs("reports", exist_ok=True)

detector = SwingDetector()
buffer = FrameBuffer(max_frames=PRE_SWING_LENGTH)
landmark_buffer = LandmarkBuffer(max_frames=PRE_SWING_LENGTH)

swing_in_progress = False
pre_swing_frames = []
post_swing_frames = []

with create_lite_landmarker() as landmarker:
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((frame_num / fps) * 1000)
        frame_num += 1

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        event = "idle"
        landmarks = None

        if result.pose_landmarks:
            h, w = frame.shape[:2]
            landmarks = result.pose_landmarks[0]
            landmark_buffer.add(landmarks)

            spine = spine_angle(landmarks)
            arms = arm_hang_angle(landmarks)
            knees = knee_flex(landmarks)

            spine_color = (0, 255, 0) if SPINE_MIN <= spine <= SPINE_MAX else (0, 0, 255)
            arms_color = (0, 255, 0) if ARMS_MIN <= arms <= ARMS_MAX else (0, 0, 255)
            legs_color = (0, 255, 0) if KNEES_MIN <= knees <= KNEES_MAX else (0, 0, 255)

            all_green = (
                SPINE_MIN <= spine <= SPINE_MAX and
                ARMS_MIN <= arms <= ARMS_MAX and
                KNEES_MIN <= knees <= KNEES_MAX
            )

            event = detector.update(landmarks, all_green)

            for lm in landmarks:
                if lm.visibility >= VISIBILITY_THRESHOLD:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            for start, end in SPINE_LINES:
                if landmarks[start].visibility < VISIBILITY_THRESHOLD or landmarks[end].visibility < VISIBILITY_THRESHOLD:
                    continue
                x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), spine_color, 3)

            for start, end in ARM_LINES:
                if landmarks[start].visibility < VISIBILITY_THRESHOLD or landmarks[end].visibility < VISIBILITY_THRESHOLD:
                    continue
                x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), arms_color, 3)

            for start, end in LEG_LINES:
                if landmarks[start].visibility < VISIBILITY_THRESHOLD or landmarks[end].visibility < VISIBILITY_THRESHOLD:
                    continue
                x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
                x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), legs_color, 3)

            bar_height = 50
            cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, f"Spine: {spine:.0f}", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, spine_color, 2)
            cv2.putText(frame, f"Arms: {arms:.0f}", (250, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, arms_color, 2)
            cv2.putText(frame, f"Knees: {knees:.0f}", (480, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, legs_color, 2)

            state_text = detector.state
            state_colors = {"IDLE": (255, 255, 255), "ARMED": (0, 255, 0), "SWINGING": (0, 255, 0), "COOLDOWN": (0, 0, 255)}
            color = state_colors.get(state_text, (255, 255, 255))
            cv2.putText(frame, f"State: {state_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            if event == "armed":
                print("ARMED")
            elif event == "swing_started":
                print("SWING DETECTED")
            elif event == "disarmed":
                print("Disarmed")

        buffer.add(frame)

        if event == "swing_started" and not swing_in_progress:
            swing_in_progress = True
            pre_swing_frames = buffer.get_all()
            post_swing_frames = []

        if swing_in_progress:
            post_swing_frames.append(frame.copy())

            if len(post_swing_frames) >= POST_SWING_LENGTH:
                all_frames = pre_swing_frames + post_swing_frames
                swing_start_idx = max(0, len(pre_swing_frames) - 1)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                filename = save_swing(all_frames, fps=fps, timestamp=timestamp)
                print(f"Saved: {filename}")

                print("Re-analyzing with heavy model for accurate metrics...")
                all_landmarks = reanalyze_with_heavy_model(all_frames, heavy_landmarker)
                print(f"Heavy reanalysis produced {len(all_landmarks)} landmark frames (matches {len(all_frames)} video frames: {len(all_landmarks) == len(all_frames)})")

                if None in all_landmarks:
                    print("Pose detection failed on frames — skipping metrics")
                else:
                    key_frame_indices = find_key_frames(all_landmarks, swing_start_idx=swing_start_idx)
                    key_frame_paths = save_key_frames(all_frames, key_frame_indices, timestamp=timestamp)
                    print(f"Key frames saved: {key_frame_paths}")

                    tempo = tempo_ratio(all_landmarks, swing_start_idx=swing_start_idx)
                    head_move = head_movement(all_landmarks, swing_start_idx=swing_start_idx)
                    hand_height = top_hand_height(all_landmarks, swing_start_idx=swing_start_idx)
                    spine_maint = spine_angle_maintenance(all_landmarks, swing_start_idx=swing_start_idx)

                    print(f"Tempo Ratio: {tempo:.2f} (target 3.0)")
                    print(f"Head Movement: {head_move:.3f} (target <0.05)")
                    print(f"Top Hand Height: {hand_height:.1f}% of torso length")
                    print(f"Spine Maintenance: {spine_maint:.1f}° deviation")

                    save_swing_to_db(
                        timestamp=timestamp,
                        club=CLUB,
                        video_path=filename,
                        key_frame_paths=key_frame_paths,
                        tempo=tempo,
                        head_move=head_move,
                        hand_height=hand_height,
                        spine_maint=spine_maint
                    )
                    print("Saved to database")

                swing_in_progress = False
                pre_swing_frames = []
                post_swing_frames = []
                buffer.clear()
                landmark_buffer.clear()

        cv2.imshow("Swing AI", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            print("\nGenerating coaching report...")
            swings = get_last_n_swings(10)

            if not swings:
                print("No swings in database yet. Do some swings first.")
            else:
                feedback = analyze_session(swings)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"reports/report_{timestamp}.txt"

                with open(report_path, 'w') as f:
                    f.write(f"SESSION REPORT — {timestamp}\n")
                    f.write(f"{len(swings)} swings analyzed | {swings[0].get('club', 'unknown')}\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(feedback)

                print("\n" + "=" * 40)
                print("SWING AI — SESSION REPORT")
                print(f"{len(swings)} swings | {swings[0].get('club', 'unknown')}")
                print("=" * 40 + "\n")
                print(feedback)
                print("\n" + "=" * 40)
                print(f"Report saved: {report_path}")
                print("=" * 40 + "\n")

cap.release()
cv2.destroyAllWindows()
heavy_landmarker.close()



