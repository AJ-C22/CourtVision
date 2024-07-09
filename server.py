from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import torch
from ultralytics import YOLO

app = FastAPI()

model = YOLO("best.pt")

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    # Read the uploaded video file
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame using YOLO model
    results = model(frame)

    # Convert the frame back to bytes
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
