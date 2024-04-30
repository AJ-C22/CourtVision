import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np


def main():
    model = YOLO("last.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    
    for result in model.track(source=0, show=True, stream=True, agnostic_nms=True):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)


        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
        )

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()