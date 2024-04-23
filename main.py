
import cv2
import base64
import numpy as np
import requests


# Set up the endpoint and access token for Roboflow
API_URL = "https://detect.roboflow.com/basketball-cv?access_token=PYvHbfgH1mq0ErEakP6j"

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode()

def infer(image):
    encoded_image = encode_image_to_base64(image)
    response = requests.post(API_URL, data=encoded_image, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })
    
    if response.status_code == 200:
        # Convert response image from JPEG to numpy array
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        print("Failed to get valid response from Roboflow.")
        return None

def main():
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        processed_image = infer(frame)
        if processed_image is not None:
            cv2.imshow('Processed Image', processed_image)
        else:
            cv2.imshow('Original Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

