import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import mediapipe as mp

def main():
    model = YOLO("last.pt")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    # Set 'show=False' to prevent the method from creating its own window
    for result in model.track(source=0, show=False, stream=True, agnostic_nms=True):
        
        frame = result.orig_img
        
        # MediaPipe Pose Detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0 , 0), thickness=3, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2)
            )

        detections = sv.Detections.from_ultralytics(result)
        
        new_detections = []
        for detection in detections:
            class_id = detection[3]  
            if class_id == 2:
                print("RIM")

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,  # Pass detections directly without enclosing it in another list
        )

        cv2.imshow("yolov8", frame)

        if cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()