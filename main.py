import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import mediapipe as mp

def box_intersection(box1, box2):
    # Determine the (x, y)-coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Check if there is an overlap
    if x_right < x_left or y_bottom < y_top:
        return False
    return True

def main():
    model = YOLO("last.pt")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    for result in model.track(source=0, show=False, stream=True, agnostic_nms=True):
        frame = result.orig_img
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

        detections = sv.Detections.from_ultralytics(result)
        rim_box = None
        ball_box = None

        for detection in detections:
            class_id = detection[3]  # Class ID for detected object
            bounding_box = detection[0]  # (x1, y1, x2, y2)

            if class_id == 2:  # Rim detected
                rim_box = bounding_box
                x1, y1, x2, y2 = bounding_box
                box_height = y2 - y1
                new_box = [x1, y1 - box_height, x2, y1]  # New box above the rim
                cv2.rectangle(frame, (int(new_box[0]), int(new_box[1])), (int(new_box[2]), int(new_box[3])), (0, 165, 255), 2)
            
            if class_id == 0:  # Ball detected
                ball_box = bounding_box

        # Check if ball is in both rim_box and new_box
        if rim_box and ball_box and box_intersection(ball_box, rim_box) and box_intersection(ball_box, new_box):
            print("Bucket man")

        frame = box_annotator.annotate(scene=frame, detections=detections)
        cv2.imshow("yolov8", frame)

        if cv2.waitKey(30) == 27:  # Break on ESC key
            break

    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
