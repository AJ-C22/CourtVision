
import cv2
import base64
import numpy as np
import requests

upload_url = "".join([
    "https://detect.roboflow.com/",
    "basketball-cv",
    "?access_token=",
    "PYvHbfgH1mq0ErEakP6j",
    "&format=image",
    "&stroke=5"
])

video = cv2.VideoCapture(0)

def infer():
    # Get the current image from the webcam
    ret, img = video.read()
    if not ret:
        print("Failed to grab frame")
        return None  # Return None if no frame is captured

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = 416 / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    data = bytearray(resp.read())
    if not data:
        print("No data received from API")
        return None

    image = np.asarray(data, dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        print("Failed to decode image")
        return None

    return image

while True:
    # On "q" keypress, exit
    if cv2.waitKey(1) == ord('q'):
        break

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    if image is None or image.size == 0:
        print("Received an empty or invalid image.")
        continue

    # And display the inference results
    cv2.imshow('image', image)

video.release()
cv2.destroyAllWindows()
