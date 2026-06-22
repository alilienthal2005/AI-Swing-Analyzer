import math

from posture import spine_angle


def find_key_frames(landmarks_per_frame):
    wrist_y_values = [frame[16].y for frame in landmarks_per_frame]
    
    address_idx = 0
    top_idx = wrist_y_values.index(min(wrist_y_values))
    
    post_top_values = wrist_y_values[top_idx:]
    impact_offset = post_top_values.index(max(post_top_values))
    impact_idx = top_idx + impact_offset
    
    finish_idx = len(landmarks_per_frame) - 1
    
    return {
        "address": address_idx,
        "top": top_idx,
        "impact": impact_idx,
        "finish": finish_idx
    }

def tempo_ratio(landmarks_per_frame):
    key_frames = find_key_frames(landmarks_per_frame)
    
    backswing_frames = key_frames["top"] - key_frames["address"]
    downswing_frames = key_frames["impact"] - key_frames["top"]
    
    if downswing_frames == 0:
        return 0
    
    return backswing_frames / downswing_frames

def head_movement(landmarks_per_frame, post_impact_buffer=5):
    key_frames = find_key_frames(landmarks_per_frame)
    
    address_landmarks = landmarks_per_frame[0]
    address_x = address_landmarks[0].x
    address_y = address_landmarks[0].y
    
    end_idx = min(key_frames["impact"] + post_impact_buffer, len(landmarks_per_frame) - 1)
    
    max_distance = 0
    
    for i in range(0, end_idx + 1):
        nose = landmarks_per_frame[i][0]
        dx = nose.x - address_x
        dy = nose.y - address_y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > max_distance:
            max_distance = distance
    
    return max_distance

def top_hand_height(landmarks_per_frame):
    key_frames = find_key_frames(landmarks_per_frame)
    address = landmarks_per_frame[key_frames["address"]]
    top = landmarks_per_frame[key_frames["top"]]
    
    address_wrist_y = address[16].y
    top_wrist_y = top[16].y
    raw_rise = address_wrist_y - top_wrist_y
    
    shoulder_y = (address[11].y + address[12].y) / 2
    hip_y = (address[23].y + address[24].y) / 2
    torso_length = abs(hip_y - shoulder_y)
    
    if torso_length == 0:
        return 0
    
    return (raw_rise / torso_length) * 100

def spine_angle_maintenance(landmarks_per_frame, post_impact_buffer=5):
    key_frames = find_key_frames(landmarks_per_frame)
    
    address_landmarks = landmarks_per_frame[0]
    address_angle = spine_angle(address_landmarks)
    
    end_idx = min(key_frames["impact"] + post_impact_buffer, len(landmarks_per_frame) - 1)
    
    max_deviation = 0
    
    for i in range(0, end_idx + 1):
        current_angle = spine_angle(landmarks_per_frame[i])
        deviation = abs(current_angle - address_angle)
        
        if deviation > max_deviation:
            max_deviation = deviation
    
    return max_deviation