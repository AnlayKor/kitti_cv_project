import argparse
import torch
from src.models.rt_detr import get_rtdetr_model

def main():
    parser = argparse.ArgumentParser(description="Эксперименты МТУСИ — Модульное обучение RT-DETR")
    parser.add_argument("--epochs", type=int, default=30, help="Количество эпох")
    parser.add_argument("--batch", type=int, default=8, help="Размер Batch Size (согласно таблице отчета: 8)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Стартовый Learning Rate (согласно таблице: 0.0001)")
    args = parser.parse_args()

    current_device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"\n=== ЗАПУСК МОДУЛЬНОГО ЭКСПЕРИМЕНТА RT-DETR ===")
    print(f"Архитектура: RT-DETR-L | Эпох: {args.epochs} | Batch Size: {args.batch} | LR: {args.lr}")
    print(f"Используемое устройство: {current_device}\n")

    model = get_rtdetr_model("rtdetr-l.pt")
    
    experiment_name = f"rtdetr_lr_{args.lr}_batch_{args.batch}"
    
    model.train(
        data="configs/default.yaml",
        epochs=args.epochs,
        imgsz=320,
        device=current_device,
        project="results",
        name=experiment_name,
        lr0=args.lr,
        batch=args.batch,
        optimizer="AdamW"
    )
    
    print(f"\n=== Эксперимент {experiment_name} успешно завершен! ===")

if __name__ == "__main__":
    main()
