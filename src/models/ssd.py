# src/models/ssd.py
import torchvision
from torchvision.models.detection.ssd import SSDHead

def get_ssd_model(num_classes=8):
    """
    Инициализация оригинальной архитектуры SSD300 (backbone: VGG16) из Torchvision
    с адаптацией под количество классов датасета KITTI.
    """
    print(f"[Инициализация] Загрузка архитектуры SSD300 (VGG16)...")
    
    # 1. Загружаем эталонную модель с дефолтными предобученными весами
    weights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
    model = torchvision.models.detection.ssd300_vgg16(weights=weights)
    
    # 2. Стандартные выходные каналы для блоков мультимасштабных карт признаков SSD300 VGG16:
    # [512, 1024, 512, 256, 256, 256]
    in_channels = [512, 1024, 512, 256, 256, 256]
    
    # 3. Забираем количество анкоров (box-предикторов) на каждую позицию
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    # 4. Перестраиваем классификатор под наши классы (8 классов + 1 фоновый)
    model.head = SSDHead(
        in_channels=in_channels, 
        num_anchors=num_anchors, 
        num_classes=num_classes + 1  # +1 класс на фон
    )
    
    print(f"[Конфигурация] Выходные слои SSD успешно адаптированы под {num_classes} классов KITTI.")
    return model