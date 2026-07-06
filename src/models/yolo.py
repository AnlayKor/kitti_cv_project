from ultralytics import YOLO

def get_yolo_model(model_name="yolov8n"):

    model = YOLO(model_name)
    return model
