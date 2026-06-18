import math

def angle_between_three_points(a, b, c):
    ba_x = a.x - b.x
    ba_y = a.y - b.y
    
    bc_x = c.x - b.x
    bc_y = c.y - b.y
    
    dot = ba_x * bc_x + ba_y * bc_y
    
    mag_ba = math.sqrt(ba_x**2 + ba_y**2)
    mag_bc = math.sqrt(bc_x**2 + bc_y**2)
    
    if mag_ba == 0 or mag_bc == 0:
        return 0
    
    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    return math.degrees(math.acos(cos_angle))


def spine_angle(landmarks):
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    
    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_x = (left_hip.x + right_hip.x) / 2
    hip_y = (left_hip.y + right_hip.y) / 2
    
    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y
    
    return math.degrees(math.atan2(abs(dx), abs(dy)))

def arm_hang_angle(landmarks):
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    
    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    wrist_x = (left_wrist.x + right_wrist.x) / 2
    wrist_y = (left_wrist.y + right_wrist.y) / 2
    
    dx = wrist_x - shoulder_x
    dy = wrist_y - shoulder_y
    
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def knee_flex(landmarks):
    left_hip = landmarks[23]
    left_knee = landmarks[25]
    left_ankle = landmarks[27]
    
    right_hip = landmarks[24]
    right_knee = landmarks[26]
    right_ankle = landmarks[28]
    
    left_angle = angle_between_three_points(left_hip, left_knee, left_ankle)
    right_angle = angle_between_three_points(right_hip, right_knee, right_ankle)
    
    return (left_angle + right_angle) / 2



