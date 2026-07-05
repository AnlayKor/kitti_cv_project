# main.py
import argparse
import torch
from src.models.yolo import get_yolo_model

def main():
    parser = argparse.ArgumentParser(description="Эксперименты МТУСИ — Изменение гиперпараметров")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Название модели")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=0.01, help="Стартовый Learning Rate (по дефолту 0.01)")
    args = parser.parse_args()

    # Автоматически проверяем: если на втором компе есть CUDA (видеокарта NVIDIA), включаем её
    current_device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"\n=== Запуск эксперимента ===")
    print(f"Модель: {args.model} | Эпох: {args.epochs} | Learning Rate: {args.lr}")
    print(f"Используемое устройство: {current_device}\n")
    
    # Инициализируем модель
    model = get_yolo_model(args.model)
    
    # Формируем красивое понятное имя папки, чтобы результаты не перезаписались!
    # Например: yolov8n_lr_0.1 или yolov8n_lr_0.001
    experiment_name = f"{args.model.split('.')[0]}_lr_{args.lr}"
    
    # Запуск обучения с кастомным параметром lr0
    model.train(
        data="configs/default.yaml",   # наш файл конфигурации датасета
        epochs=args.epochs,            # 50 эпох
        imgsz=640,                     # нормализация
        device=current_device,         # видеокарта или процессор автоматически
        project="results",             # папка для результатов
        name=experiment_name,          # имя подпапки для графиков данного теста
        lr0=args.lr                    # ТУТ МЫ МЕНЯЕМ СКОРОСТЬ ОБУЧЕНИЯ
    )
    
    print(f"\n=== Эксперимент {experiment_name} успешно завершен! ===")

if __name__ == "__main__":
    main()
