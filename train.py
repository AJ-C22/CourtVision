from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # Call freeze_support() to ensure proper handling of multiprocessing
    multiprocessing.freeze_support()

    # Load a model
    model = YOLO("yolov8n.yaml")  
    model.to('cuda')

    # Use the model for training
    results = model.train(data="config.yaml", epochs=200, batch=8, cache=False, workers=1)
