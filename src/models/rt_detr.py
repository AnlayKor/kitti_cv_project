# src/models/rt_detr.py
from ultralytics import RTDETR

def get_rtdetr_model(weights_name="rtdetr-l.pt"):
    """
    Инициализация современной трансформерной архитектуры RT-DETR 
    на базе движка Ultralytics.
    """
    print(f"[Инициализация] Загрузка архитектуры RT-DETR ({weights_name})...")
    model = RTDETR(weights_name)
    return model