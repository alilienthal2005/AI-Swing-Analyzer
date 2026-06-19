import cv2
import os
from datetime import datetime

def save_swing(frames, output_dir="swings", fps=60):
    if not frames:
        return None 
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/swing_{timestamp}.mp4"

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    
    return filename



    