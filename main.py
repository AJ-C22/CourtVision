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
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # For Combobox
import tkinter.font as tkFont
from PIL import ImageFont, ImageDraw, Image, ImageTk

class Shot:
    
    def __init__(self, team1_color_name, team2_color_name):
        self.model = YOLO("best.pt")
        self.class_names = ['ball', 'person', 'rim']
        self.cap = None  # Initialize cap to None
        self.dots = []  # List to store dot positions
        self.goal_count = 0
        self.ball_in_top_box = False
        self.ball_positions = []  # List to store ball positions for smoothing
        self.team_colors = defaultdict(lambda: (128, 128, 128))  # Default color gray
        self.num_team1_buckets = 0  # Score for team 1
        self.num_team2_buckets = 0  # Score for team 2
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
        self.running = False  # Flag to indicate if the video processing is running

        # Set team colors
        self.color_name_to_bgr = {
            'purple': (128, 0, 128),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'orange': (0, 165, 255),
            'black': (0, 0, 0),
            'white': (255, 255, 255)
        }

        self.color_name_to_hsv = {
            'purple': (np.array([125, 50, 50]), np.array([155, 255, 255])),
            'red': (np.array([0, 70, 50]), np.array([10, 255, 255])),
            'red2': (np.array([170, 70, 50]), np.array([180, 255, 255])),  # For red wrapping hue
            'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
            'green': (np.array([40, 50, 50]), np.array([80, 255, 255])),
            'blue': (np.array([90, 50, 50]), np.array([130, 255, 255])),
            'orange': (np.array([10, 100, 100]), np.array([20, 255, 255])),
            'black': (np.array([0, 0, 0]), np.array([180, 255, 30])),
            'white': (np.array([0, 0, 231]), np.array([180, 25, 255]))
        }

        self.team1_color_name = team1_color_name.lower()
        self.team2_color_name = team2_color_name.lower()

        # Check for invalid color names
        if self.team1_color_name not in self.color_name_to_bgr or self.team2_color_name not in self.color_name_to_bgr:
            raise ValueError("Invalid color names provided for teams.")

        self.team1_bgr = self.color_name_to_bgr[self.team1_color_name]
        self.team2_bgr = self.color_name_to_bgr[self.team2_color_name]

        # For red, handle the hue wrap-around
        if self.team1_color_name == 'red':
            self.team1_hsv_ranges = [self.color_name_to_hsv['red'], self.color_name_to_hsv['red2']]
        else:
            self.team1_hsv_ranges = [self.color_name_to_hsv[self.team1_color_name]]

        if self.team2_color_name == 'red':
            self.team2_hsv_ranges = [self.color_name_to_hsv['red'], self.color_name_to_hsv['red2']]
        else:
            self.team2_hsv_ranges = [self.color_name_to_hsv[self.team2_color_name]]
    
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
        score_message = f"{self.team1_color_name.capitalize()} team: {self.num_team1_buckets}, {self.team2_color_name.capitalize()} team: {self.num_team2_buckets}"
        self.engine.say(score_message)
        self.engine.runAndWait()
    
    def remove_last_point(self):
        if self.last_shooting_team == self.team1_bgr and self.num_team1_buckets > 0:
            self.num_team1_buckets -= 1
        elif self.last_shooting_team == self.team2_bgr and self.num_team2_buckets > 0:
            self.num_team2_buckets -= 1

    def set_score(self, command):
        try:
            scores = [int(s) for s in command.split() if s.isdigit()]
            if len(scores) == 2:
                self.num_team1_buckets, self.num_team2_buckets = scores
        except ValueError:
            print("Error setting score from voice command")
    
    def new_game(self):
        self.num_team1_buckets = 0
        self.num_team2_buckets = 0

    def start_video_processing(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.run()

    def run(self):
        ball_position = None
        rim_position = None

        cv2.namedWindow('Frame')

        while self.running:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
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
                        elif current_class == "ball":
                            ball_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                            # Draw the ball detection box
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(self.frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        elif current_class == "rim":
                            rim_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                            # Draw the rim detection box
                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(self.frame, "Rim", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if len(detections) > 0:
                tracked_objects = self.tracker.update(np.array(detections))
            else:
                tracked_objects = []

            current_time = time.time()

            person_boxes = []  # Initialize person_boxes list
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = map(int, obj)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                person_roi = self.frame[y1:y2, x1:x2]
                color_at_center = self.get_average_color(person_roi)

                if color_at_center is not None:
                    # Assign to teams based on the color at the center of the person
                    if self.is_color(color_at_center, self.team1_hsv_ranges):
                        self.team_colors[obj_id] = self.team1_bgr  # Team 1
                    elif self.is_color(color_at_center, self.team2_hsv_ranges):
                        self.team_colors[obj_id] = self.team2_bgr  # Team 2
                    else:
                        self.team_colors[obj_id] = (128, 128, 128)  # Gray (default)
                else:
                    self.team_colors[obj_id] = (128, 128, 128)  # Gray (default)

                # Draw person boxes with team colors
                color = self.team_colors[obj_id]
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                person_boxes.append((x1, y1, x2, y2))
                self.centroids[obj_id] = (cx, cy)
                self.last_seen[obj_id] = current_time  # Update last seen time

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
                    closest_id = self.get_closest_centroid(self.centroids, ball_position)
                    if closest_id in self.team_colors:
                        self.current_shooting_team = self.team_colors[closest_id]
                        self.last_shooting_team = self.current_shooting_team
                    break

            for (object_id, centroid) in self.centroids.items():
                color = self.team_colors[object_id]
                cv2.circle(self.frame, centroid, 5, color, -1)
                cv2.putText(self.frame, f"ID: {object_id}", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if rim_position:
                rim_x, rim_y = rim_position
                box_width, box_height = 50, 50  # Define the size of the boxes

                top_box = (rim_x - box_width // 2, rim_y - box_height, rim_x + box_width // 2, rim_y)
                bottom_box = (rim_x - box_width // 2, rim_y, rim_x + box_width // 2, rim_y + box_height)

                cv2.rectangle(self.frame, (top_box[0], top_box[1]), (top_box[2], top_box[3]), (0, 255, 255), 2)
                cv2.rectangle(self.frame, (bottom_box[0], bottom_box[1]), (bottom_box[2], bottom_box[3]), (255, 0, 255), 2)

                if ball_position and top_box[0] < ball_position[0] < top_box[2] and top_box[1] < ball_position[1] < top_box[3]:
                    self.ball_in_top_box = True

                if self.ball_in_top_box and ball_position and bottom_box[0] < ball_position[0] < bottom_box[2] and bottom_box[1] < ball_position[1] < bottom_box[3]:
                    if self.last_shooting_team == self.team1_bgr:  # Team 1
                        self.num_team1_buckets += 1
                        threading.Thread(target=self.announce_scoring_team, args=(self.team1_color_name.capitalize(),)).start()
                    elif self.last_shooting_team == self.team2_bgr:  # Team 2
                        self.num_team2_buckets += 1
                        threading.Thread(target=self.announce_scoring_team, args=(self.team2_color_name.capitalize(),)).start()
                    self.ball_in_top_box = False  # Reset for the next goal

            # Display the scores for both teams
            cv2.putText(self.frame, f"{self.team1_color_name.capitalize()}: ", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, self.team1_bgr, 2)
            cv2.putText(self.frame, f"{self.num_team1_buckets}", (150, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(self.frame, f"| {self.team2_color_name.capitalize()}: ", (200, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, self.team2_bgr, 2)
            cv2.putText(self.frame, f"{self.num_team2_buckets}", (400, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the shooting team permanently
            if self.last_shooting_team == self.team1_bgr:  # Team 1
                cv2.putText(self.frame, f"Shooting: {self.team1_color_name.capitalize()}", (frame_width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.team1_bgr, 2)
            elif self.last_shooting_team == self.team2_bgr:  # Team 2
                cv2.putText(self.frame, f"Shooting: {self.team2_color_name.capitalize()}", (frame_width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.team2_bgr, 2)

            current_frame_dots = []

            if ball_position and rim_position and ball_position[1] < rim_position[1]:
                if not self.dots or time.time() - self.dots[-1]['time'] >= 0.1:
                    self.dots.append({'position': ball_position, 'time': time.time()})
                current_frame_dots = [dot['position'] for dot in self.dots]  # Update current frame dots
            else:
                self.dots = []  # Clear dots when ball goes below rim

            for dot_position in current_frame_dots:
                cv2.circle(self.frame, dot_position, 5, (0, 255, 0), -1)

            if ball_position:
                self.ball_positions.append(ball_position)
                if len(self.ball_positions) > 10:  
                    self.ball_positions.pop(0)
                smoothed_positions = gaussian_filter(np.array(self.ball_positions), sigma=1)
                ball_position = tuple(smoothed_positions[-1].astype(int))

            cv2.imshow('Frame', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_video_processing()
                break

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def stop_video_processing(self):
        self.running = False

    def get_average_color(self, roi):
        """Calculate the average color in the ROI."""
        if roi is None or roi.size == 0:
            return None
        average_color = np.mean(roi, axis=(0, 1)).astype(int)
        return average_color

    def is_color(self, color_bgr, hsv_ranges):
        """Check if the color is within the specified HSV ranges."""
        hsv_color = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        for (lower_hsv, upper_hsv) in hsv_ranges:
            if cv2.inRange(np.uint8([[hsv_color]]), lower_hsv, upper_hsv)[0][0] > 0:
                return True
        return False

    def remove_unused_ids(self, current_time):
        ids_to_remove = [obj_id for obj_id, last_seen_time in self.last_seen.items() if current_time - last_seen_time > self.removal_time_threshold]
        for obj_id in ids_to_remove:
            if obj_id in self.centroids:
                del self.centroids[obj_id]
            if obj_id in self.histograms:
                del self.histograms[obj_id]
            if obj_id in self.last_seen:
                del self.last_seen[obj_id]
            if obj_id in self.team_colors:
                del self.team_colors[obj_id]

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

    def announce_scoring_team(self, team_color_name):
        self.engine.say(f"{team_color_name} team scored a point")
        self.engine.runAndWait()

class CourtVisionApp:
    def __init__(self, root):
        font_path = "Bombing.ttf"
        font_size = 12
        custom_font = ImageFont.truetype(font_path, font_size)

        # Create an image with text using the custom font
        image = Image.new("RGBA", (400, 50), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "CourtVision", font=custom_font, fill=(0, 0, 0))

        # Convert the image to a format Tkinter can display
        photo = ImageTk.PhotoImage(image)

        # Display the image with text in a Tkinter label
        label = tk.Label(root, image=photo)
        label.image = photo  # Keep a reference to avoid garbage collection
        label.pack(pady=10)

        self.title_label2 = tk.Label(self.title_frame, text="Vision", font=("Helvetica", 36, "bold"), fg='white', bg='#2F323C')
        self.title_label2.pack(side='left')

        self.available_colors = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Orange', 'White', 'Black']

        self.team1_label = tk.Label(self.card_frame, text="Team 1 Color", font=("Helvetica", 12), fg='#E5E7EB', bg='#2F323C')
        self.team1_label.pack(pady=(10, 5), anchor='w', padx=40)
        self.team1_color = tk.StringVar()
        self.team1_color.set(self.available_colors[0])  # Default selection
        self.team1_dropdown = ttk.Combobox(self.card_frame, textvariable=self.team1_color, values=self.available_colors, state="readonly")
        self.team1_dropdown.pack(pady=5, padx=40, fill='x')

        self.team2_label = tk.Label(self.card_frame, text="Team 2 Color", font=("Helvetica", 12), fg='#E5E7EB', bg='#2F323C')
        self.team2_label.pack(pady=(10, 5), anchor='w', padx=40)
        self.team2_color = tk.StringVar()
        self.team2_color.set(self.available_colors[1])  # Default selection
        self.team2_dropdown = ttk.Combobox(self.card_frame, textvariable=self.team2_color, values=self.available_colors, state="readonly")
        self.team2_dropdown.pack(pady=5, padx=40, fill='x')

        self.start_button = tk.Button(self.card_frame, text="Start Counting", command=self.start, font=("Helvetica", 14), bg='#FF9800', fg='white', activebackground='#E65100', activeforeground='white')
        self.start_button.pack(pady=(30, 10), padx=40, fill='x')

        self.exit_button = tk.Button(self.card_frame, text="Exit", command=self.exit, font=("Helvetica", 14), bg='#1F2937', fg='#FF9800', activebackground='#FF9800', activeforeground='white', highlightthickness=1, highlightbackground='#FF9800')
        self.exit_button.pack(pady=(0, 10), padx=40, fill='x')

        style = ttk.Style()
        style.theme_use('default')
        style.configure('TCombobox', fieldbackground='#374151', background='#374151', foreground='white')
        style.map('TCombobox', fieldbackground=[('readonly', '#374151')])

    def start(self):
        team1 = self.team1_color.get()
        team2 = self.team2_color.get()

        if team1 == team2:
            messagebox.showerror("Error", "Team colors must be different.")
            return

        self.shot = Shot(team1, team2)
        threading.Thread(target=self.shot.start_video_processing).start()

    def exit(self):
        if hasattr(self, 'shot'):
            self.shot.stop_video_processing()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CourtVisionApp(root)
    root.mainloop()
