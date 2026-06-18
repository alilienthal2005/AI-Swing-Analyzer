class SwingDetector:


    def __init__(self):
        self.state = "IDLE"
        self.green_frames = 0
        self.cooldown_frames = 0 
        self.swing_hold_frames = 0 # temporary
        self.armed_wrist_y = 0


        self.frames_needed_to_arm = 60
        self.wrist_move_threshold = .04
        self.swing_duration_frames = 90
        self.cooldown_duration = 60


    def update(self, landmarks, all_green):

        if self.state == "IDLE":
            if all_green:
                self.green_frames +=1 
                if self.green_frames >= self.frames_needed_to_arm:
                    self.state = "ARMED"
                    self.armed_wrist_y = landmarks[16].y
                    return "armed"
            else:
                self.green_frames = 0
            return "idle"
        
        elif self.state == "ARMED":
            if not all_green:
                self.state = "IDLE"
                self.green_frames = 0 
                self.armed_wrist_y = None
                return "disarmed"
            
            current_wrist_y = landmarks[16].y
            wrist_movement = self.armed_wrist_y - current_wrist_y
            if wrist_movement > self.wrist_move_threshold:
                self.state = "SWINGING"
                self.swing_hold_frames = 0 
                return "swing started"
            return "armed "
        
        elif self.state == "SWINGING":
            self.swing_hold_frames +=1
            if self.swing_hold_frames >= self.swing_duration_frames:
                self.state = "COOLDOWN"
                self.cooldown_frames = 0 
            return "swinging"
        
        elif self.state == "COOLDOWN":
            self.cooldown_frames +=1
            if self.cooldown_frames >= self.cooldown_duration:
                self.state = "IDLE"
                self.green_frames = 0
                self.armed_wrist_y = None 
            return "cooldown"

        









