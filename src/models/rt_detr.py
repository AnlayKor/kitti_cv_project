from ultralytics import RTDETR

def get_rtdetr_model(weights_name="rtdetr-l.pt"):

    print(f"[Инициализация] Загрузка архитектуры RT-DETR ({weights_name})...")
    model = RTDETR(weights_name)
    return model
