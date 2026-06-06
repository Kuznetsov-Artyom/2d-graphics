#!/usr/bin/env python3
"""
Скрипт 4: Тестирование модели на независимой тестовой выборке.

Использует:
- models/best_model.pth (обученная модель)
- test_images/ (1000 тестовых изображений)

Создает:
- results/test_metrics.json (метрики на тестовой выборке)
- results/test_confusion_matrix.png (confusion matrix)
- results/test_predictions_batch_XX.png (визуализация предсказаний)
- results/test_predictions.json (детальные предсказания)

Время выполнения: ~5-7 минут
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import CONFIG
from model import load_model
from dataset import get_val_transform
from utils import setup_logging, compute_metrics, save_metrics, plot_confusion_matrix


def load_test_images(test_dir):
    """
    Загружает все тестовые изображения из плоской структуры.
    
    Args:
        test_dir: путь к test_images/
    
    Returns:
        test_images: список (image_tensor, true_count)
    """
    test_images = []
    
    # Ищем все JSON файлы
    json_files = sorted([f for f in os.listdir(test_dir) 
                        if f.startswith('image_') and f.endswith('.json')])
    
    for json_file in json_files:
        json_path = os.path.join(test_dir, json_file)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        img_path = os.path.join(test_dir, metadata['filename'])
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            transform = get_val_transform()
            img_tensor = transform(img)
            
            test_images.append((img_tensor, metadata['cell_count']))
    
    return test_images


def plot_all_predictions(model, test_images, batch_size=50, save_dir=None):
    """
    Визуализирует все предсказания, разбивая на несколько файлов.
    
    Args:
        model: обученная модель
        test_images: список (image_tensor, true_count)
        batch_size: количество изображений на один файл
        save_dir: папка для сохранения
    """
    if save_dir is None:
        save_dir = CONFIG['results_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    
    num_batches = (len(test_images) + batch_size - 1) // batch_size
    
    print(f"\nВизуализация {len(test_images)} предсказаний в {num_batches} файлов...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_images))
        batch_images = test_images[start_idx:end_idx]
        
        # Создание сетки 5x10 (50 изображений)
        fig, axes = plt.subplots(5, 10, figsize=(25, 15))
        axes = axes.flatten()
        
        for i, (img, true_count) in enumerate(batch_images):
            with torch.no_grad():
                img_tensor = img.unsqueeze(0).to(CONFIG['device'])
                output = model(img_tensor)
                pred_count = 5 + torch.argmax(output, dim=1).item()
            
            # Денормализация
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            axes[i].imshow(img_np)
            axes[i].set_title(f'T:{true_count}\nP:{pred_count}', fontsize=8)
            axes[i].axis('off')
        
        # Скрытие неиспользуемых осей
        for i in range(len(batch_images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Сохранение
        save_path = os.path.join(save_dir, f'test_predictions_batch_{batch_idx+1:02d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Сохранено: test_predictions_batch_{batch_idx+1:02d}.png ({len(batch_images)} изображений)")


def main():
    """Основная функция тестирования."""
    
    # Настройка логирования
    logger = setup_logging()
    logger.info("="*70)
    logger.info("ТЕСТИРОВАНИЕ НА НЕЗАВИСИМОЙ ТЕСТОВОЙ ВЫБОРКЕ")
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
        logger.error("Сначала запустите: python generate_test_set.py")
        sys.exit(1)
    
    logger.info(f"\nЗагрузка тестовых изображений из: {test_dir}")
    test_images = load_test_images(test_dir)
    
    if len(test_images) == 0:
        logger.error("❌ Не удалось загрузить тестовые изображения")
        sys.exit(1)
    
    logger.info(f"✓ Загружено {len(test_images)} изображений")
    
    # Предсказания
    logger.info("\n" + "="*70)
    logger.info("ПРЕДСКАЗАНИЯ")
    logger.info("="*70)
    
    all_outputs = []
    all_targets = []
    all_counts = []
    predictions = []
    
    with torch.no_grad():
        for i, (img_tensor, true_count) in enumerate(tqdm(test_images, desc='Testing')):
            img_batch = img_tensor.unsqueeze(0).to(device)
            output = model(img_batch)
            
            pred_class = torch.argmax(output, dim=1).item()
            pred_count = pred_class + 5  # Индекс 0 соответствует 5 клеткам
            target_class = true_count - 5  # Индекс класса
            
            all_outputs.append(output.cpu())
            all_targets.append(torch.tensor([target_class]))
            all_counts.append(torch.tensor([true_count]))
            
            predictions.append({
                'image_id': i + 1,
                'true_count': true_count,
                'predicted_count': pred_count,
                'error': abs(pred_count - true_count)
            })
    
    # Объединение результатов
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_counts = torch.cat(all_counts, dim=0)
    
    # Вычисление метрик
    metrics = compute_metrics(all_outputs, all_targets, all_counts)
    
    logger.info("\n" + "="*70)
    logger.info("МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    logger.info("="*70)
    logger.info(f"Top-1 Accuracy: {metrics['top1_accuracy']*100:.2f}%")
    logger.info(f"Top-3 Accuracy: {metrics['top3_accuracy']*100:.2f}%")
    logger.info(f"MAE: {metrics['mae']:.2f} клеток")
    
    # Статистика ошибок
    errors = [p['error'] for p in predictions]
    logger.info(f"\nСредняя ошибка: {np.mean(errors):.2f} клеток")
    logger.info(f"Медианная ошибка: {np.median(errors):.2f} клеток")
    logger.info(f"Max ошибка: {max(errors)} клеток")
    logger.info("="*70)
    
    # Сохранение метрик
    metrics_path = os.path.join(CONFIG['results_dir'], 'test_metrics.json')
    save_metrics(metrics, save_path=metrics_path)
    
    # Confusion matrix для тестовой выборки
    logger.info("\nПостроение confusion matrix для тестовой выборки...")
    pred_classes = torch.argmax(all_outputs, dim=1).numpy()
    true_classes = all_targets.squeeze().numpy()
    
    pred_counts = pred_classes + 5
    true_counts = true_classes + 5
    
    cm_path = os.path.join(CONFIG['results_dir'], 'test_confusion_matrix.png')
    plot_confusion_matrix(true_counts, pred_counts, save_path=cm_path)
    
    # Визуализация всех предсказаний
    logger.info("\nВизуализация предсказаний...")
    plot_all_predictions(model, test_images, batch_size=50)
    
    # Сохранение детальных предсказаний
    predictions_data = {
        'total_images': len(predictions),
        'metrics': metrics,
        'statistics': {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'max_error': int(max(errors)),
            'min_error': int(min(errors))
        },
        'predictions': predictions
    }
    
    predictions_path = os.path.join(CONFIG['results_dir'], 'test_predictions.json')
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Детальные предсказания сохранены: {predictions_path}")
    
    logger.info("\n✓ Тестирование завершено!")
    logger.info("\n" + "="*70)
    logger.info("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    logger.info("="*70)
    logger.info(f"  - Модель: {model_path}")
    logger.info(f"  - Тестовая выборка: {test_dir} ({len(test_images)} изображений)")
    logger.info(f"  - Метрики: {CONFIG['results_dir']}/test_metrics.json")
    logger.info(f"  - Confusion matrix: {CONFIG['results_dir']}/test_confusion_matrix.png")
    logger.info(f"  - Предсказания: {CONFIG['results_dir']}/test_predictions_batch_*.png")
    logger.info(f"  - Детали: {CONFIG['results_dir']}/test_predictions.json")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
