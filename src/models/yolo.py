from ultralytics import YOLO

def get_yolo_model(model_name="yolov8n"):
    """
    Инициализация модели YOLO.
    Аргумент model_name может быть 'yolov5s.pt' или 'yolov8n.pt'
    """
    # Автоматически загрузит предобученную модель
    model = YOLO(model_name)
    return model