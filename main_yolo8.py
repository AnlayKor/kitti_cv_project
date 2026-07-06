import argparse
import torch
from src.models.yolo import get_yolo_model

def main():
    parser = argparse.ArgumentParser(description="Эксперименты МТУСИ — Изменение гиперпараметров")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Название модели")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=0.01, help="Стартовый Learning Rate (по дефолту 0.01)")
    args = parser.parse_args()

    current_device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"\n=== Запуск эксперимента ===")
    print(f"Модель: {args.model} | Эпох: {args.epochs} | Learning Rate: {args.lr}")
    print(f"Используемое устройство: {current_device}\n")
    
    model = get_yolo_model(args.model)
    
    experiment_name = f"{args.model.split('.')[0]}_lr_{args.lr}"
    
    model.train(
        data="configs/default.yaml",
        epochs=args.epochs,
        imgsz=640,
        device=current_device,
        project="results",
        name=experiment_name,
        lr0=args.lr
    )
    
    print(f"\n=== Эксперимент {experiment_name} успешно завершен! ===")

if __name__ == "__main__":
    main()
