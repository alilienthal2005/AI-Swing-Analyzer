import cv2
import os
from datetime import datetime

def save_swing(frames, output_dir="swings", fps=60, timestamp=None):
    if not frames:
        return None
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/swing_{timestamp}.mp4"
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    
    return filename


def save_key_frames(frames, key_frame_indices, output_dir="swing_keys", timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = {}
    
    for label, idx in key_frame_indices.items():
        filename = f"{output_dir}/swing_{timestamp}_{label}.jpg"
        cv2.imwrite(filename, frames[idx])
        saved_paths[label] = filename
    
    return saved_paths



    