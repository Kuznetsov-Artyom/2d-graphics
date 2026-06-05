#!/usr/bin/env python3
"""
Скрипт 3: Валидация модели и вычисление метрик.

Использует:
- models/best_model.pth (обученная модель)
- data/val/ (2,000 изображений)

Создает:
- results/confusion_matrix.png (матрица ошибок)
- results/metrics.json (итоговые метрики)

Время выполнения: ~5 минут
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

from config import CONFIG
from dataset import get_data_loaders
from model import load_model
from utils import setup_logging, plot_confusion_matrix, compute_metrics, save_metrics


def main():
    """Основная функция валидации."""
    
    # Настройка логирования
    logger = setup_logging()
    logger.info("="*70)
    logger.info("ВАЛИДАЦИЯ МОДЕЛИ")
    logger.info("="*70)
    
    # Проверка GPU
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Загрузка модели
    model_path = os.path.join(CONFIG['models_dir'], 'best_model.pth')
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Модель не найдена: {model_path}")
        logger.error("Сначала запустите: python 02_train_model.py")
        sys.exit(1)
    
    logger.info(f"\nЗагрузка модели: {model_path}")
    model = load_model(model_path).to(device)
    model.eval()
    logger.info("✓ Модель загружена")
    
    # Загрузка данных
    logger.info("\nЗагрузка validation данных...")
    _, val_loader = get_data_loaders()
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Валидация
    logger.info("\n" + "="*70)
    logger.info("ВЫЧИСЛЕНИЕ МЕТРИК")
    logger.info("="*70)
    
    all_outputs = []
    all_targets = []
    all_counts = []
    
    with torch.no_grad():
        for images, targets, counts in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            outputs = model(images)
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets)
            all_counts.append(counts)
    
    # Объединение результатов
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_counts = torch.cat(all_counts, dim=0)
    
    # Вычисление метрик
    metrics = compute_metrics(all_outputs, all_targets, all_counts)
    
    logger.info("\n" + "="*70)
    logger.info("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
    logger.info("="*70)
    logger.info(f"Top-1 Accuracy: {metrics['top1_accuracy']*100:.2f}%")
    logger.info(f"Top-3 Accuracy: {metrics['top3_accuracy']*100:.2f}%")
    logger.info(f"MAE: {metrics['mae']:.2f} клеток")
    logger.info("="*70)
    
    # Сохранение метрик
    save_metrics(metrics)
    
    # Confusion matrix
    logger.info("\nПостроение confusion matrix...")
    pred_classes = torch.argmax(all_outputs, dim=1).numpy()
    true_classes = torch.argmax(all_targets, dim=1).numpy()
    
    # Преобразование в реальное количество клеток
    # Индекс 0 соответствует 20 клеткам, индекс 20 соответствует 40 клеткам
    pred_counts = pred_classes + 20
    true_counts = true_classes + 20
    
    plot_confusion_matrix(true_counts, pred_counts)
    
    logger.info("\n✓ Валидация завершена!")
    logger.info("  Следующий шаг: python 04_test_predictions.py\n")


if __name__ == "__main__":
    main()
