import math

from posture import spine_angle


ADDRESS_LOOKBACK_FRAMES = 15
MAX_BACKSWING_FRAMES = 60
TOP_REVERSAL_THRESHOLD = 0.03
TOP_REVERSAL_FRAMES = 6
MIN_VISIBILITY_FOR_TOP = 0.3
MIN_FRAMES_TOP_TO_IMPACT = 3
IMPACT_DEBOUNCE_FRAMES = 2


def find_key_frames(landmarks_per_frame, swing_start_idx=0):
    n = len(landmarks_per_frame)

    wrist_y = [frame[15].y for frame in landmarks_per_frame]
    wrist_visibility = [frame[15].visibility for frame in landmarks_per_frame]
    hip_y = [(frame[23].y + frame[24].y) / 2 for frame in landmarks_per_frame]

    # swing_start_idx is the first frame the SwingDetector state machine
    # classifies as "swinging" -- it only fires once the wrist has already
    # moved past the armed baseline, so it can land a few frames into the
    # backswing. Rewind within a short window to the last frame before that
    # motion (highest wrist-y = hands at their lowest, i.e. still at address).
    lookback_start = max(0, swing_start_idx - ADDRESS_LOOKBACK_FRAMES)
    address_idx = max(range(lookback_start, swing_start_idx + 1), key=lambda i: wrist_y[i])

    # Top of backswing: track the running minimum wrist-y, but stop as soon
    # as the wrist clearly reverses direction and holds that reversal for
    # several frames. A plain global-minimum search (no bound, no reversal
    # check) can get dragged into the follow-through/rotation phase, where a
    # foreshortened or partially-occluded wrist can read as an even lower y
    # than the real top -- hijacking "top" away from the actual backswing peak.
    top_window_end = min(address_idx + MAX_BACKSWING_FRAMES, n)
    top_idx = address_idx
    min_y = wrist_y[address_idx]
    reversal_run = 0
    for i in range(address_idx + 1, top_window_end):
        if wrist_visibility[i] < MIN_VISIBILITY_FOR_TOP:
            continue
        if wrist_y[i] < min_y:
            min_y = wrist_y[i]
            top_idx = i
            reversal_run = 0
        elif wrist_y[i] > min_y + TOP_REVERSAL_THRESHOLD:
            reversal_run += 1
            if reversal_run >= TOP_REVERSAL_FRAMES:
                break
        else:
            reversal_run = 0

    # Impact: first frame (sustained for a couple of frames, to ignore
    # single-frame landmark noise) where the wrist drops back to/past hip
    # height, searched only within a plausible downswing window past a short
    # guard after top. The original heuristic scanned unbounded all the way
    # to the end of the clip using a single hip landmark with no guard, so it
    # could fire on unrelated post-swing motion (walking off, cooldown) --
    # producing wild backswing:downswing ratios (0, 5x, 67x).
    backswing_frames = max(top_idx - address_idx, 1)
    downswing_window_end = min(top_idx + max(2 * backswing_frames, 20), n - 1)

    impact_idx = downswing_window_end
    search_start = top_idx + MIN_FRAMES_TOP_TO_IMPACT
    for i in range(search_start, downswing_window_end - IMPACT_DEBOUNCE_FRAMES + 1):
        if all(wrist_y[i + k] >= hip_y[i + k] for k in range(IMPACT_DEBOUNCE_FRAMES)):
            impact_idx = i
            break

    finish_idx = min(impact_idx + 30, n - 1)

    return {
        "address": address_idx,
        "top": top_idx,
        "impact": impact_idx,
        "finish": finish_idx
    }

def tempo_ratio(landmarks_per_frame, swing_start_idx=0):
    key_frames = find_key_frames(landmarks_per_frame, swing_start_idx)

    backswing_frames = key_frames["top"] - key_frames["address"]
    downswing_frames = key_frames["impact"] - key_frames["top"]

    if downswing_frames == 0:
        return 0

    return backswing_frames / downswing_frames

def head_movement(landmarks_per_frame, swing_start_idx=0, post_impact_buffer=5):
    key_frames = find_key_frames(landmarks_per_frame, swing_start_idx)

    address_landmarks = landmarks_per_frame[key_frames["address"]]
    address_x = address_landmarks[0].x
    address_y = address_landmarks[0].y

    end_idx = min(key_frames["impact"] + post_impact_buffer, len(landmarks_per_frame) - 1)

    max_distance = 0

    for i in range(key_frames["address"], end_idx + 1):
        nose = landmarks_per_frame[i][0]
        dx = nose.x - address_x
        dy = nose.y - address_y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > max_distance:
            max_distance = distance

    return max_distance

def top_hand_height(landmarks_per_frame, swing_start_idx=0):
    key_frames = find_key_frames(landmarks_per_frame, swing_start_idx)
    address = landmarks_per_frame[key_frames["address"]]
    top = landmarks_per_frame[key_frames["top"]]

    address_wrist_y = address[15].y
    top_wrist_y = top[15].y
    raw_rise = address_wrist_y - top_wrist_y

    # Normalized against full body height (nose to ankle), not torso length.
    # A real backswing raises the hands from waist height to roughly head
    # height -- a vertical span comparable to torso+head, not just the
    # hip-to-shoulder segment. Dividing by torso length alone (~30% of
    # height) inflates this to 150-300%+ even for a correctly-measured
    # swing; normalizing by full body height keeps it in the expected
    # 60-100% range.
    nose_y = address[0].y
    ankle_y = (address[27].y + address[28].y) / 2
    body_height = abs(ankle_y - nose_y)

    if body_height == 0:
        return 0

    return (raw_rise / body_height) * 100

def spine_angle_maintenance(landmarks_per_frame, swing_start_idx=0, post_impact_buffer=5):
    key_frames = find_key_frames(landmarks_per_frame, swing_start_idx)

    address_landmarks = landmarks_per_frame[key_frames["address"]]
    address_angle = spine_angle(address_landmarks)

    end_idx = min(key_frames["impact"] + post_impact_buffer, len(landmarks_per_frame) - 1)

    max_deviation = 0

    for i in range(key_frames["address"], end_idx + 1):
        current_angle = spine_angle(landmarks_per_frame[i])
        deviation = abs(current_angle - address_angle)

        if deviation > max_deviation:
            max_deviation = deviation

    return max_deviation