import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

# Initialize the webcam.
cap = cv2.VideoCapture(0)

# Dictionary to hold the colors for each landmark based on the nearest click.
landmark_colors = {}

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        landmark_colors[(x, y)] = (0, 0, 255)  # Red color for left click
    elif event == cv2.EVENT_RBUTTONDOWN:
        landmark_colors[(x, y)] = (255, 0, 0)  # Blue color for right click

cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection.
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw pose landmarks.
        landmark_points = []
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            landmark_points.append((x, y))

        # Determine the color for each landmark based on nearest click.
        for i, point in enumerate(landmark_points):
            min_dist = float('inf')
            assigned_color = (255, 255, 255)  # Default color is white
            for click_point, color in landmark_colors.items():
                dist = np.linalg.norm(np.array(point) - np.array(click_point))
                if dist < min_dist:
                    min_dist = dist
                    assigned_color = color
            
            # Draw the pose with the assigned color.
            drawing_spec.color = assigned_color
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, drawing_spec, drawing_spec)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows.
cap.release()
cv2.destroyAllWindows()
