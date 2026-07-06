import argparse
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Эксперименты МТУСИ — Запуск обучения YOLOv5")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--batch", type=int, default=16, help="Размер Batch Size")
    parser.add_argument("--lr", type=float, default=0.01, help="Стартовый Learning Rate (lr0)")
    args = parser.parse_args()

    current_device = 0 if torch.cuda.is_available() else 'cpu'

    print(f"\n=== ЗАПУСК ЭКСПЕРИМЕНТА YOLOv5 ===")
    print(f"Архитектура: YOLOv5n | Эпох: {args.epochs} | Batch Size: {args.batch} | LR: {args.lr}")
    print(f"Используемое устройство: {current_device}\n")

    model = YOLO("yolov5n.pt")

    experiment_name = f"yolov5n_lr_{args.lr}_batch_{args.batch}"

    model.train(
        data="configs/default.yaml",
        epochs=args.epochs,
        imgsz=640,
        device=current_device,
        project="results",
        name=experiment_name,
        lr0=args.lr,
        batch=args.batch
    )

    print(f"\n=== Эксперимент {experiment_name} успешно завершен! ===")


if __name__ == "__main__":
    main()
