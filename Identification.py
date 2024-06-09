import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utilities.
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect poses.
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame.
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame.
    cv2.imshow('MediaPipe Pose Estimation', frame)

    # Exit when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
cv2.destroyAllWindows()
