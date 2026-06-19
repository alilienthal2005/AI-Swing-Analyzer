from collections import deque 

class FrameBuffer:
    def __init__(self, max_frames=120):
        self.frames = deque(maxlen=max_frames)

    def add(self, frame):
        self.frames.append(frame.copy())

    def get_all(self):
        return list(self.frames)
    
    def clear(self):
        self.frames.clear()

    def __len__(self):
        return len(self.frames)

    
