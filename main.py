import argparse
from src.models.yolo import get_yolo_model


def main():
    parser = argparse.ArgumentParser(description="Проект МТУСИ — Детекция объектов на KITTI")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Название модели (yolov8n.pt или yolov5s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    args = parser.parse_args()

    print(f"\n=== Запуск эксперимента: {args.model} | Эпох: {args.epochs} ===\n")

    # Инициализируем модель из нашей папки src
    model = get_yolo_model(args.model)

    # Запуск полноценного цикла обучения
    model.train(
        data="configs/default.yaml",  # наш файл конфигурации
        epochs=args.epochs,  # 50 эпох по плану
        imgsz=640,  # нормализация картинок
        device=0,  # Использовать видеокарту (GPU). Если будет ошибка, замени на 'cpu'
        project="results",  # сохранять результаты в папку results по методичке
        name=args.model.split('.')[0]  # имя подпапки эксперимента (например, yolov8n)
    )

    print(f"\n=== Обучение модели {args.model} завершено успешно! ===")


if __name__ == "__main__":
    main()