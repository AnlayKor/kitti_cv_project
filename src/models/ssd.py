import torchvision
from torchvision.models.detection.ssd import SSDHead

def get_ssd_model(num_classes=8):
  
    print(f"[Инициализация] Загрузка архитектуры SSD300 (VGG16)...")
    
    weights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
    model = torchvision.models.detection.ssd300_vgg16(weights=weights)
    
    in_channels = [512, 1024, 512, 256, 256, 256]
    
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    model.head = SSDHead(
        in_channels=in_channels, 
        num_anchors=num_anchors, 
        num_classes=num_classes + 1
    )
    
    print(f"[Конфигурация] Выходные слои SSD успешно адаптированы под {num_classes} классов KITTI.")
    return model
