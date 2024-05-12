import cv2
from ultralytics import YOLO
import math
import time

class Shot:
    
    def __init__(self):
        self.model = YOLO("best.pt")
        self.class_names = ['ball', 'person', 'rim']
        #Change to 0 for built in webcam
        self.cap = cv2.VideoCapture(1)
        self.dots = []  # List to store dot positions
        self.run()
    
    def run(self):
        ball_position = None
        rim_position = None
        ball_above_rim = False

        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break

            results = self.model(self.frame, stream=True)
            current_frame_dots = []  # Temporary list to hold dots for the current frame

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the box

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    # Draw rectangle
                    if conf > .4:
                        if current_class == "ball":
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                            ball_position = (cx, cy)
                        
                        elif current_class == "person":
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        elif current_class == "rim":
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            rim_box = x1, y1, x2, y2
                            rim_position = (cx, cy)

                            nx1 = int(cx - (x1 - cx)/2)
                            nx2 = int(cx + (x1 - cx)/2)
                            ny2 = int(cy - (y1 - cy)/2)

                            #cv2.rectangle(self.frame, (nx1, cy), (nx2, ny2), (100, 200, 255), 2)
                            net_box = nx1, cy, nx2, ny2

            # Check if the ball is above the rim and manage the dots
            if ball_position and rim_position and ball_position[1] < rim_position[1]:
                if not self.dots or time.time() - self.dots[-1]['time'] >= 0.1:
                    self.dots.append({'position': ball_position, 'time': time.time()})
                current_frame_dots = [dot['position'] for dot in self.dots]  # Update current frame dots
            
            else:
                self.dots = []  # Clear dots when ball goes below rim

            # Draw all current dots
            for dot_position in current_frame_dots:
                cv2.circle(self.frame, dot_position, 5, (0, 255, 0), -1)

            try: 
                self.ball_in_box(ball_position, rim_box)
            except:
                print("No Rim")

            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def ball_in_box(self, center, box):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = box
        
        # Check if the center of the ball lies within the bounding box
        if x1 < center[0] < x2 and y1 < center[1] < y2:
            print("BUCKET")
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return True
            
        else:
            return False

if __name__ == "__main__":
    Shot()
