import math

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from posture import spine_angle


ADDRESS_LOOKBACK_FRAMES = 15
MIN_WRIST_VISIBILITY = 0.3

# SwingDetector (detection.py) holds SWINGING for swing_duration_frames (60)
# before moving to COOLDOWN -- used here as a hard ceiling on how far past
# address we'll look for top/impact/finish, so a candidate can never be
# picked from the post-swing walk-off/cooldown footage.
SWING_DURATION_FRAMES = 60

SAVGOL_WINDOW = 9
SAVGOL_POLYORDER = 2

# Tuned against 5 real recorded swings (see debug_keyframes.py): the
# genuine top/impact/finish peaks all have prominence >= 0.19 once smoothed,
# while jitter-driven spurious peaks top out around 0.08. 0.12 sits cleanly
# between the two with margin on both sides. distance=8 (~0.27s at 30fps)
# stops a single physical peak from being split into two by noise.
PEAK_PROMINENCE = 0.12
PEAK_MIN_DISTANCE = 8


def find_key_frames(landmarks_per_frame, swing_start_idx=0):
    key_frames, _peaks, _troughs = _find_key_frames_with_diagnostics(landmarks_per_frame, swing_start_idx)
    return key_frames


def find_key_frames_with_diagnostics(landmarks_per_frame, swing_start_idx=0):
    """Same as find_key_frames, but also returns the raw peak/trough indices
    find_peaks detected (before the first-peak/first-trough/next-peak
    selection), for debug plotting -- see debug_keyframes.py."""
    return _find_key_frames_with_diagnostics(landmarks_per_frame, swing_start_idx)


def _find_key_frames_with_diagnostics(landmarks_per_frame, swing_start_idx=0):
    n = len(landmarks_per_frame)

    wrist_y = [frame[15].y for frame in landmarks_per_frame]
    wrist_visibility = [frame[15].visibility for frame in landmarks_per_frame]
    # Fall back to the right wrist on frames where the left wrist is barely
    # tracked (occlusion/motion blur) -- reduces single-frame spikes feeding
    # into the peak search below.
    for i in range(n):
        if wrist_visibility[i] < MIN_WRIST_VISIBILITY:
            wrist_y[i] = landmarks_per_frame[i][16].y

    # swing_start_idx is the first frame the SwingDetector state machine
    # classifies as "swinging" -- it only fires once the wrist has already
    # moved past the armed baseline, so it can land a few frames into the
    # backswing. Rewind within a short window to the last frame before that
    # motion (highest wrist-y = hands at their lowest, i.e. still at address).
    lookback_start = max(0, swing_start_idx - ADDRESS_LOOKBACK_FRAMES)
    address_idx = max(range(lookback_start, swing_start_idx + 1), key=lambda i: wrist_y[i])

    # Bound relative to swing_start_idx (when SwingDetector actually enters
    # SWINGING), not address_idx -- address_idx can rewind a few frames
    # earlier, and shortening the window by that same amount was enough to
    # truncate a real, wide follow-through peak before it fully registered.
    window_end = min(swing_start_idx + SWING_DURATION_FRAMES, n)
    segment = np.array(wrist_y[address_idx:window_end])

    window_length = min(SAVGOL_WINDOW, len(segment))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 3:
        smoothed = savgol_filter(segment, window_length=window_length, polyorder=min(SAVGOL_POLYORDER, window_length - 1))
    else:
        smoothed = segment

    # Every real swing shows a double-hump wrist-y curve: a first peak (true
    # top of backswing, hands raised), a dip through the downswing/impact,
    # then a second peak (follow-through, often taller/wider than the
    # first). A plain global-minimum search picks the tallest point in the
    # whole clip, which is usually the second hump -- hijacking "top" into
    # the follow-through. Detecting actual peaks/troughs and taking them in
    # sequence (first peak, first trough after it, next peak after that)
    # tracks the real swing phases instead.
    peaks, _ = find_peaks(-smoothed, prominence=PEAK_PROMINENCE, distance=PEAK_MIN_DISTANCE)
    troughs, _ = find_peaks(smoothed, prominence=PEAK_PROMINENCE, distance=PEAK_MIN_DISTANCE)
    peaks = peaks + address_idx
    troughs = troughs + address_idx

    if len(peaks) >= 1:
        top_idx = int(peaks[0])
    else:
        print(f"[find_key_frames] fallback: no peak found for top in window {address_idx}-{window_end}, using global min")
        top_idx = address_idx + int(np.argmin(smoothed))

    troughs_after_top = troughs[troughs > top_idx]
    if len(troughs_after_top) >= 1:
        impact_idx = int(troughs_after_top[0])
    else:
        print(f"[find_key_frames] fallback: no trough found for impact after top={top_idx}, using local max")
        tail = smoothed[top_idx - address_idx:]
        impact_idx = top_idx + int(np.argmax(tail)) if len(tail) else top_idx

    peaks_after_impact = peaks[peaks > impact_idx]
    if len(peaks_after_impact) >= 1:
        finish_idx = int(peaks_after_impact[0])
    else:
        print(f"[find_key_frames] fallback: no peak found for finish after impact={impact_idx}, using impact+30")
        finish_idx = min(impact_idx + 30, n - 1)

    key_frames = {
        "address": address_idx,
        "top": top_idx,
        "impact": impact_idx,
        "finish": finish_idx
    }
    return key_frames, peaks, troughs

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