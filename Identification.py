import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utilities.
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam.
cap = cv2.VideoCapture(0)

def get_average_color(image, keypoint, radius=10):
    """Get the average color around the keypoint."""
    x, y = int(keypoint.x * image.shape[1]), int(keypoint.y * image.shape[0])
    x_min, x_max = max(0, x - radius), min(image.shape[1], x + radius)
    y_min, y_max = max(0, y - radius), min(image.shape[0], y + radius)
    
    region = image[y_min:y_max, x_min:x_max]
    average_color = np.mean(region, axis=(0, 1))
    
    return average_color

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect poses.
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        # Draw the pose landmarks on the frame.
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract shoulder keypoints.
        left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Get average colors.
        avg_color_left_shoulder = get_average_color(frame, left_shoulder)
        avg_color_right_shoulder = get_average_color(frame, right_shoulder)

        # Calculate the overall average color from shoulders.
        avg_color = np.mean([avg_color_left_shoulder, avg_color_right_shoulder], axis=0)

        # Display the estimated shirt color.
        color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
        color_patch[:] = avg_color
        cv2.putText(frame, 'Shirt Color', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        frame[10:110, 10:110] = color_patch

    # Display the frame.
    cv2.imshow('MediaPipe Pose Estimation', frame)

    # Exit when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
cv2.destroyAllWindows()
