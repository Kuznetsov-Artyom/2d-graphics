#!/usr/bin/env python3
"""
Скрипт 4: Тестирование модели на test_images.

Использует:
- models/best_model.pth (обученная модель)
- update_generator/test_images/ (тестовые изображения)

Создает:
- results/predictions_examples.png (примеры предсказаний)

Время выполнения: ~2 минуты
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from torchvision import transforms

from config import CONFIG
from model import load_model
from dataset import get_val_transform
from utils import setup_logging, plot_predictions


def load_test_images(test_dir, num_samples=5):
    """
    Загружает тестовые изображения из test_images/.
    
    Args:
        test_dir: путь к test_images/
        num_samples: количество примеров
    
    Returns:
        test_images: список (image_tensor, true_count)
    """
    test_images = []
    
    # Ищем папки с изображениями
    subdirs = sorted([d for d in os.listdir(test_dir) 
                     if os.path.isdir(os.path.join(test_dir, d))])
    
    # Берем первые num_samples папок
    for subdir in subdirs[:num_samples]:
        subdir_path = os.path.join(test_dir, subdir)
        
        # Ищем params.json
        params_path = os.path.join(subdir_path, 'params.json')
        
        if os.path.exists(params_path):
            with open(params_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            
            # Берем первое изображение
            if 'images' in params and len(params['images']) > 0:
                img_info = params['images'][0]
                img_filename = img_info.get('filename') or img_info.get('clean_filename')
                
                if img_filename:
                    img_path = os.path.join(subdir_path, img_filename)
                    
                    if os.path.exists(img_path):
                        # Загрузка изображения
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Трансформация
                        transform = get_val_transform()
                        img_tensor = transform(img)
                        
                        # True count
                        true_count = img_info.get('actual_cells', 0)
                        
                        test_images.append((img_tensor, true_count))
                        
                        print(f"Загружено: {subdir}/{img_filename} (клеток: {true_count})")
    
    return test_images


def main():
    """Основная функция тестирования."""
    
    # Настройка логирования
    logger = setup_logging()
    logger.info("="*70)
    logger.info("ТЕСТИРОВАНИЕ НА TEST_IMAGES")
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
    
    # Загрузка тестовых изображений
    test_dir = CONFIG['test_images_dir']
    
    if not os.path.exists(test_dir):
        logger.error(f"❌ Test images не найдены: {test_dir}")
        sys.exit(1)
    
    logger.info(f"\nЗагрузка тестовых изображений из: {test_dir}")
    test_images = load_test_images(test_dir, num_samples=5)
    
    if len(test_images) == 0:
        logger.error("❌ Не удалось загрузить тестовые изображения")
        sys.exit(1)
    
    logger.info(f"✓ Загружено {len(test_images)} изображений")
    
    # Предсказания
    logger.info("\n" + "="*70)
    logger.info("ПРЕДСКАЗАНИЯ")
    logger.info("="*70)
    
    predictions = []
    
    with torch.no_grad():
        for i, (img_tensor, true_count) in enumerate(test_images):
            img_batch = img_tensor.unsqueeze(0).to(device)
            output = model(img_batch)
            
            pred_class = torch.argmax(output, dim=1).item()
            pred_count = pred_class + 20  # Индекс 0 соответствует 20 клеткам
            
            predictions.append({
                'image_id': i + 1,
                'true_count': true_count,
                'predicted_count': pred_count,
                'error': abs(pred_count - true_count)
            })
            
            logger.info(f"Image {i+1}: True={true_count}, Pred={pred_count}, Error={abs(pred_count - true_count)}")
    
    # Статистика
    errors = [p['error'] for p in predictions]
    logger.info("\n" + "="*70)
    logger.info("СТАТИСТИКА ПРЕДСКАЗАНИЙ")
    logger.info("="*70)
    logger.info(f"Средняя ошибка: {np.mean(errors):.2f} клеток")
    logger.info(f"Медианная ошибка: {np.median(errors):.2f} клеток")
    logger.info(f"Max ошибка: {max(errors)} клеток")
    logger.info("="*70)
    
    # Визуализация
    logger.info("\nВизуализация предсказаний...")
    plot_predictions(model, test_images, num_examples=5)
    
    logger.info("\n✓ Тестирование завершено!")
    logger.info("\n" + "="*70)
    logger.info("ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ")
    logger.info("="*70)
    logger.info("\nРезультаты:")
    logger.info(f"  - Модель: {model_path}")
    logger.info(f"  - Графики: {CONFIG['results_dir']}/training_curves.png")
    logger.info(f"  - Confusion matrix: {CONFIG['results_dir']}/confusion_matrix.png")
    logger.info(f"  - Метрики: {CONFIG['results_dir']}/metrics.json")
    logger.info(f"  - Предсказания: {CONFIG['results_dir']}/predictions_examples.png")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
