import cv2
from ultralytics import YOLO
import math
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from collections import defaultdict
import threading
import pyttsx3
from sort.sort import Sort  # Import SORT from the sort directory

from sklearn.metrics import pairwise_distances
import speech_recognition as sr  # Import the speech recognition library

class Shot:
    
    def __init__(self):
        self.model = YOLO("best.pt")
        self.class_names = ['ball', 'person', 'rim']
        self.cap = cv2.VideoCapture(0)
        self.dots = []  # List to store dot positions
        self.goal_count = 0
        self.ball_in_top_box = False
        self.ball_positions = []  # List to store ball positions for smoothing
        self.team2_centroids = []  # List to store team 2 centroids
        self.team_colors = defaultdict(lambda: (255, 0, 0))  # Default color blue for team 1
        self.num_orange_buckets = 0  # Score for orange team
        self.num_blue_buckets = 0  # Score for blue team
        self.current_shooting_team = None  # Team currently shooting
        self.last_shooting_team = None  # Last known shooting team
        self.engine = pyttsx3.init()
        self.centroids = {}  # To store centroids of detected persons with their IDs
        self.histograms = {}  # To store histograms of detected persons with their IDs
        self.last_seen = {}  # To store the last seen time of each ID
        self.tracker = Sort(max_age=30, min_hits=3)  # Adjust max_age and min_hits for better tracking
        self.removal_time_threshold = 4  # Time in seconds to remove unused IDs
        self.recognizer = sr.Recognizer()  # Initialize the recognizer
        self.microphone = sr.Microphone()  # Initialize the microphone
        self.voice_thread = threading.Thread(target=self.listen_to_voice_commands)
        self.voice_thread.start()
        self.run()
    
    def listen_to_voice_commands(self):
        while True:
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    print("Listening for voice commands...")
                    audio = self.recognizer.listen(source)
                
                command = self.recognizer.recognize_google(audio).lower()
                print(f"Voice command received: {command}")

                if "read score" in command:
                    self.announce_score()
                elif "remove last point" in command:
                    self.remove_last_point()
                elif "set score" in command:
                    self.set_score(command)
                elif "new game" in command:
                    self.new_game()
            except sr.UnknownValueError:
                print("Could not understand the audio")
            except sr.RequestError:
                print("Could not request results; check your network connection")
    
    def announce_score(self):
        score_message = f"Orange team: {self.num_orange_buckets}, Blue team: {self.num_blue_buckets}"
        self.engine.say(score_message)
        self.engine.runAndWait()
    
    def remove_last_point(self):
        if self.last_shooting_team == (0, 165, 255) and self.num_orange_buckets > 0:
            self.num_orange_buckets -= 1
        elif self.last_shooting_team == (255, 0, 0) and self.num_blue_buckets > 0:
            self.num_blue_buckets -= 1
    
    def set_score(self, command):
        try:
            scores = [int(s) for s in command.split() if s.isdigit()]
            if len(scores) == 2:
                self.num_orange_buckets, self.num_blue_buckets = scores
        except ValueError:
            print("Error setting score from voice command")
    
    def new_game(self):
        self.num_orange_buckets = 0
        self.num_blue_buckets = 0

def run(self):
    ball_position = None
    rim_position = None

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', self.on_mouse_click)

    while True:
        ret, self.frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_height, frame_width, _ = self.frame.shape

        results = self.model(self.frame, stream=True)
        detections = []  # List to store detections

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                if conf > 0.4:  # Adjust the confidence threshold
                    if current_class == "person":
                        detections.append([x1, y1, x2, y2, conf])

        # Convert detections to DeepSORT format
        detection_boxes = np.array([det[:4] for det in detections])
        detection_scores = np.array([det[4] for det in detections])

        # Update DeepSORT tracker with current frame detections
        if len(detections) > 0:
            detections_np = np.array(detections)  # Convert to numpy array
            tracked_objects = self.tracker.update(detections_np)  # Update tracker
        else:
            tracked_objects = []

        current_time = time.time()

        # Extract the updated centroid positions and IDs
        person_boxes = []  # Initialize person_boxes list
        for obj in tracked_objects:
            try:
                x1, y1, x2, y2 = obj.to_tlbr()  # Assuming obj has this method
                obj_id = obj.track_id
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Get the histogram of the person's clothing
                person_roi = self.frame[y1:y2, x1:x2]
                histogram = self.calculate_histogram(person_roi)

                if histogram is not None:
                    if pairwise_distances and obj_id in self.histograms:
                        best_match_id = self.get_best_match_id(histogram)
                        if best_match_id is not None:
                            obj_id = best_match_id

                    self.centroids[obj_id] = (cx, cy)
                    self.histograms[obj_id] = histogram
                    self.last_seen[obj_id] = current_time  # Update last seen time

                    # Draw person boxes with team colors
                    color = self.team_colors[obj_id % len(self.team_colors)]  # Adjust for team colors
                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(self.frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    person_boxes.append((x1, y1, x2, y2))
            except AttributeError as e:
                print(f"Error accessing attributes of tracked object: {e}")

        # Remove IDs not seen in the last 4 seconds
        self.remove_unused_ids(current_time)

        # Check if the ball is in the shooting zone of any player
        self.current_shooting_team = None
        for (x1, y1, x2, y2) in person_boxes:
            shooting_zone_height = (y2 - y1) // 3
            shooting_zone = (x1, y1, x2, y1 + shooting_zone_height)

            # Draw the shooting zone for each player
            cv2.rectangle(self.frame, (shooting_zone[0], shooting_zone[1]), (shooting_zone[2], shooting_zone[3]), (0, 255, 0), 2)

            # Check if the ball is in the shooting zone
            if ball_position and shooting_zone[0] < ball_position[0] < shooting_zone[2] and shooting_zone[1] < ball_position[1] < shooting_zone[3]:
                self.current_shooting_team = self.team_colors[self.get_closest_centroid(self.centroids, ball_position)]
                self.last_shooting_team = self.current_shooting_team
                break

        # Loop over tracked centroids and draw them on the frame with team colors
        for (object_id, centroid) in self.centroids.items():
            color = self.team_colors[object_id % len(self.team_colors)]
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
                if self.last_shooting_team == (0, 165, 255):  # Orange
                    self.num_orange_buckets += 1
                    threading.Thread(target=self.announce_score, args=("Orange",)).start()
                elif self.last_shooting_team == (255, 0, 0):  # Blue
                    self.num_blue_buckets += 1
                    threading.Thread(target=self.announce_score, args=("Blue",)).start()
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

        # Initialize current_frame_dots to an empty list
        current_frame_dots = []

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
            for object_id, centroid in self.centroids.items():
                cx, cy = centroid
                if abs(cx - x) < 10 and abs(cy - y) < 10:
                    # Assign this centroid to team 2
                    if centroid not in self.team2_centroids:
                        self.team2_centroids.append(centroid)
                        self.team_colors[object_id] = (0, 165, 255)  # Orange
                    else:
                        self.team2_centroids.remove(centroid)  # Allow toggling back to team 1
                        self.team_colors[object_id] = (255, 0, 0)  # Blue

    def update_centroids(self, centroids):
        # This method updates centroids based on the given detections
        updated_centroids = {}
        for i, centroid in enumerate(centroids):
            updated_centroids[i] = centroid
        self.centroids = updated_centroids
        return updated_centroids

    def get_closest_centroid(self, centroids, ball_position):
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

    def announce_score(self, team_color):
        self.engine.say(f"{team_color} scored a point")
        self.engine.runAndWait()

    def calculate_histogram(self, roi):
        if roi is None or roi.size == 0:
            return None
        # Convert the ROI to HSV color space
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Calculate the color histogram for the ROI
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [50, 60], [0, 180, 0, 256])
        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def get_best_match_id(self, current_hist):
        # Calculate distances between current histogram and stored histograms
        distances = pairwise_distances([current_hist], list(self.histograms.values()), metric='cosine')[0]
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.5:  # Threshold for considering a match
            return list(self.histograms.keys())[best_match_index]
        return None

    def remove_unused_ids(self, current_time):
        ids_to_remove = [obj_id for obj_id, last_seen_time in self.last_seen.items() if current_time - last_seen_time > self.removal_time_threshold]
        for obj_id in ids_to_remove:
            del self.centroids[obj_id]
            del self.histograms[obj_id]
            del self.last_seen[obj_id]
            if obj_id in self.team_colors:
                del self.team_colors[obj_id]

if __name__ == "__main__":
    Shot()