import cv2
from ultralytics import YOLO
import math
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from centroid import CentroidTracker

class Shot:
    
    def __init__(self):
        self.model = YOLO("best.pt")
        self.class_names = ['ball', 'person', 'rim']
        self.cap = cv2.VideoCapture(0)
        self.dots = []  # List to store dot positions
        self.ct = CentroidTracker(max_disappeared=100)  # Initialize CentroidTracker
        self.goal_count = 0
        self.ball_in_top_box = False
        self.ball_positions = []  # List to store ball positions for smoothing
        self.team2_centroids = []  # List to store team 2 centroids
        self.possession_id = None  # ID of the person who has possession of the ball
        self.run()
    
    def run(self):
        ball_position = None
        rim_position = None

        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', self.on_mouse_click)

        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break

            results = self.model(self.frame, stream=True)
            current_frame_dots = []  # Temporary list to hold dots for the current frame
            centroids = []  # List to store centroids of detected people

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
                    if conf > 0.4:  # Adjust the confidence threshold
                        if current_class == "ball":
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                            ball_position = (cx, cy)
                        
                        elif current_class == "person":
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            centroids.append((cx, cy))

                        elif current_class == "rim":
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            rim_position = (cx, cy)

            # Update CentroidTracker with detected centroids
            tracked_centroids = self.ct.update(centroids)

            # Determine possession of the ball
            self.possession_id = self.get_possession_id(tracked_centroids, ball_position)

            # Loop over tracked centroids and draw them on the frame with IDs
            for (object_id, centroid) in tracked_centroids.items():
                # Determine the color based on the team
                color = (0, 255, 0)  # Default to green (team 1)
                if centroid in self.team2_centroids:
                    color = (0, 165, 255)  # Orange for team 2

                # Draw centroid and ID on the frame
                cv2.circle(self.frame, centroid, 5, color, -1)
                cv2.putText(self.frame, f"ID: {object_id}", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
                    self.goal_count += 1
                    self.ball_in_top_box = False  # Reset for the next goal

            # Draw the goal count on the screen
            cv2.putText(self.frame, f"Buckets: {self.goal_count}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the possession ID on the top left corner
            if self.possession_id is not None:
                cv2.putText(self.frame, f"Possession: {self.possession_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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
            # Check if the click is near any centroid
            for (object_id, centroid) in self.ct.objects.items():
                cx, cy = centroid
                if abs(cx - x) < 10 and abs(cy - y) < 10:
                    # Assign this centroid to team 2
                    if centroid not in self.team2_centroids:
                        self.team2_centroids.append(centroid)
                    else:
                        self.team2_centroids.remove(centroid)  # Allow toggling back to team 1

    def get_possession_id(self, centroids, ball_position):
        if ball_position is None:
            return None
        closest_id = None
        closest_distance = float('inf')
        for object_id, centroid in centroids.items():
            distance = self.calculate_distance(centroid, ball_position)
            if distance < closest_distance:
                closest_distance = distance
                closest_id = object_id
        return closest_id

    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

if __name__ == "__main__":
    Shot()
