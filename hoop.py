import cv2
from ultralytics import YOLO
import math
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

class Shot:
    
    def __init__(self):
        self.model = YOLO("best.pt")
        self.class_names = ['ball', 'person', 'rim']
        self.cap = cv2.VideoCapture(0)
        self.dots = []  # List to store dot positions
        self.goal_count = 0
        self.ball_in_top_box = False
        self.ball_positions = []  # List to store ball positions for smoothing
        self.team2_ids = set()  # Set to store team 2 IDs
        self.team_colors = defaultdict(lambda: (255, 0, 0))  # Default color blue for team 1
        self.num_orange_buckets = 0  # Score for orange team
        self.num_blue_buckets = 0  # Score for blue team
        self.current_shooting_team = None  # Team currently shooting
        self.last_shooting_team = None  # Last known shooting team
        self.frame_skip = 2  # Process every nth frame
        self.tracker = DeepSort(max_age=30)  # Initialize DeepSORT
        self.run()
    
    def run(self):
        ball_position = None
        rim_position = None

        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', self.on_mouse_click)

        frame_count = 0

        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break

            frame_count += 1

            if frame_count % self.frame_skip != 0:
                continue

            frame_height, frame_width, _ = self.frame.shape

            results = self.model(self.frame, stream=True)
            current_frame_dots = []  # Temporary list to hold dots for the current frame
            detections = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the box

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    # Draw rectangle
                    if conf > 0.4:  # Adjust the confidence threshold
                        if current_class == "ball":
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                            ball_position = (cx, cy)
                        
                        elif current_class == "person":
                            detections.append(([x1, y1, x2-x1, y2-y1], conf, 2))  # Add detection in [x, y, w, h] format

                        elif current_class == "rim":
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            rim_position = (cx, cy)

            # Update DeepSORT with detections
            tracks = self.tracker.update_tracks(detections, frame=self.frame)

            # Check if the ball is in the shooting zone of any player
            self.current_shooting_team = None
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                x1, y1, x2, y2 = track.to_ltwh()  # Get the bounding box
                shooting_zone_height = y2 // 3
                shooting_zone = (x1, y1, x1 + x2, y1 + shooting_zone_height)

                # Default color for the shooting zone
                shooting_zone_color = (0, 255, 0)  # Green

                # Check if the ball is in the shooting zone
                if ball_position and shooting_zone[0] < ball_position[0] < shooting_zone[2] and shooting_zone[1] < ball_position[1] < shooting_zone[3]:
                    shooting_zone_color = (0, 0, 255)  # Red
                    self.current_shooting_team = self.team_colors[track_id]
                    self.last_shooting_team = self.current_shooting_team

                # Assign team colors
                color = self.team_colors[track_id]
                if track_id in self.team2_ids:
                    color = (0, 165, 255)  # Orange for team 2

                # Draw the shooting zone and the bounding box for each player
                cv2.rectangle(self.frame, (int(shooting_zone[0]), int(shooting_zone[1])), (int(shooting_zone[2]), int(shooting_zone[3])), shooting_zone_color, 2)
                cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x1 + x2), int(y1 + y2)), color, 2)
                #cv2.putText(self.frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if rim_position:
                # Define the top and bottom boxes relative to the rim position
                rim_x, rim_y = rim_position
                box_width, box_height = 50, 50  # Define the size of the boxes

                top_box = (rim_x - box_width // 2, rim_y - box_height, rim_x + box_width // 2, rim_y)
                bottom_box = (rim_x - box_width // 2, rim_y, rim_x + box_width // 2, rim_y + box_height)

                # Draw the top and bottom boxes
                cv2.rectangle(self.frame, (top_box[0], top_box[1]), (top_box[2], top_box[3]), (0, 255, 255), 2)
                cv2.rectangle(self.frame, (bottom_box[0], bottom_box[1]), (bottom_box[2], bottom_box[3]), (255, 0, 255), 2)

                # Check if the ball is in the top box
                if ball_position and top_box[0] < ball_position[0] < top_box[2] and top_box[1] < ball_position[1] < top_box[3]:
                    self.ball_in_top_box = True

                # Check if the ball is in the bottom box
                if self.ball_in_top_box and ball_position and bottom_box[0] < ball_position[0] < bottom_box[2] and bottom_box[1] < ball_position[1] < bottom_box[3]:
                    if self.last_shooting_team == (0, 165, 255):  # Orange
                        self.num_orange_buckets += 1
                    elif self.last_shooting_team == (255, 0, 0):  # Blue
                        self.num_blue_buckets += 1
                    self.ball_in_top_box = False  # Reset for the next goal

            # Display the scores for both teams
            cv2.putText(self.frame, "Orange: ", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.putText(self.frame, f"{self.num_orange_buckets}", (150, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(self.frame, "| Blue: ", (200, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(self.frame, f"{self.num_blue_buckets}", (350, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the shooting team permanently
            if self.last_shooting_team == (0, 165, 255):  # Orange
                cv2.putText(self.frame, "Shooting: Orange", (frame_width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            elif self.last_shooting_team == (255, 0, 0):  # Blue
                cv2.putText(self.frame, "Shooting: Blue", (frame_width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Check if the ball is above the rim and manage the dots
            if ball_position and rim_position and ball_position[1] < rim_y:
                if not self.dots or time.time() - self.dots[-1]['time'] >= 0.1:
                    self.dots.append({'position': ball_position, 'time': time.time()})
                current_frame_dots = [dot['position'] for dot in self.dots]  # Update current frame dots
            else:
                self.dots = []  # Clear dots when ball goes below rim

            # Draw all current dots
            for dot_position in current_frame_dots:
                cv2.circle(self.frame, dot_position, 5, (0, 255, 0), -1)

            # Smoothing ball positions using Gaussian filter
            if ball_position:
                self.ball_positions.append(ball_position)
                if len(self.ball_positions) > 10:  
                    self.ball_positions.pop(0)
                smoothed_positions = gaussian_filter(np.array(self.ball_positions), sigma=1)
                ball_position = tuple(smoothed_positions[-1].astype(int))

            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is near any tracked object
            for track in self.tracker.tracker.tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = track.to_ltwh()
                cx, cy = x1 + x2 // 2, y1 + y2 // 2
                if abs(cx - x) < 10 and abs(cy - y) < 10:
                    # Assign this track ID to team 2
                    if track.track_id not in self.team2_ids:
                        self.team2_ids.add(track.track_id)
                        self.team_colors[track.track_id] = (0, 165, 255)  # Orange
                    else:
                        self.team2_ids.remove(track.track_id)  # Allow toggling back to team 1
                        self.team_colors[track.track_id] = (255, 0, 0)  # Blue

if __name__ == "__main__":
    Shot()
