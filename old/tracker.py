import cv2
from ultralytics import YOLO
import math

class Shot:
    
    def __init__(self):
        self.model = YOLO("best.pt")
        self.class_names = ['ball', 'person', 'rim']
        #Change to 0 if laptop webcam
        self.cap = cv2.VideoCapture(1)
        self.frame = None
        self.frame_count = 0
        self.ball_pos = [] 
        self.person_pos = []
        self.rim_pos = []
    
    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break

            results = self.model(self.frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]
                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Draw rectangle
                    if current_class == "ball" and conf > .3:
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    
                    elif current_class == "person" and conf > .6:
                        self.person_pos.append((center, self.frame_count, w, h, conf))
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (106, 6, 17), 2)

                    elif current_class == "rim" and conf > .5:
                        self.rim_pos.append((center, self.frame_count, w, h, conf))
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        box_height = y2 - y1
                        new_box = [x1, y1 - box_height, x2, y1]  # New box above the rim
                        cv2.rectangle(self.frame, (int(new_box[0]), int(new_box[1])), (int(new_box[2]), int(new_box[3])), (0, 165, 255), 2)

            try:
                if self.ball_in_box(center, new_box):
                    print("Bucket man")
            except:
                pass

            #self.make_path()
            self.frame_count += 1
            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        self.cap.release()
        cv2.destroyAllWindows()
    
    def ball_in_box(self, center, box):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = box
        
        # Check if the center of the ball lies within the bounding box
        if x1 < center[0] < x2 and y1 < center[1] < y2:
            return True
        else:
            return False

    def make_path(self):
        try: 
            for i in range(0, len(self.ball_pos)):
                cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)
            cv2.circle(self.frame, self.rim_pos[-1][0], 2, (128, 128, 0), 2)
        except:
            pass

if __name__ == "__main__":
    Shot().run()
