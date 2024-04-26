from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  

# Use the model
model.train(data="config.yaml", epochs=1) 
