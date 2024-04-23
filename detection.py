import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow('MediaPipe Pose')
cv2.setMouseCallback('MediaPipe Pose', mouse_callback)


# Initial Setup Loop
while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)


    if key == 32:  # Spacebar pressed
        break
    elif key == 27:  # ESC key to exit early
        cap.release()
        cv2.destroyAllWindows()
        exit()