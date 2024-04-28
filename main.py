from ultralytics import YOLO


# Load a model
model = YOLO("yolov8n.yaml")  

# Use the model for training
results = model.train(data="config.yaml", epochs=11)

