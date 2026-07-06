# src/models/faster_rcnn.py
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn_model(num_classes=8):
    """
    Инициализация оригинальной двухстадийной архитектуры Faster R-CNN 
    с бэкбоном ResNet-50-FPN из библиотеки torchvision.
    """
    print(f"[Инициализация] Загрузка оригинальной архитектуры Faster R-CNN (ResNet-50-FPN)...")
    
    # Загружаем предобученную модель
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Меняем голову классификатора под наше количество классов (в KITTI их 8)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model