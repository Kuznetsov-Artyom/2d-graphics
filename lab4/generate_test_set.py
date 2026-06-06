#!/usr/bin/env python3
"""
Скрипт для генерации независимой тестовой выборки из 1000 изображений.

Генерирует:
- 1000 изображений с случайным количеством клеток (5-100)
- JSON метаданные для каждого изображения
- Статистику датасета

Время выполнения: ~3-5 минут
"""

import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Добавляем путь к генератору
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'update_generator'))
from blood_cell_generator import BloodCellGenerator

# Импортируем конфигурацию
from config import CONFIG


def main():
    """Основная функция генерации тестовой выборки."""
    
    print("\n" + "="*70)
    print("ГЕНЕРАЦИЯ ТЕСТОВОЙ ВЫБОРКИ")
    print("="*70)
    
    # Папка для тестовых данных
    test_dir = './test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    # Проверка наличия патчей
    patches_dir = CONFIG['patches_dir']
    cells_dir = os.path.join(patches_dir, 'cells')
    bg_dir = os.path.join(patches_dir, 'background')
    
    if not os.path.exists(cells_dir):
        print(f"\n❌ ОШИБКА: Папка не найдена: {cells_dir}")
        sys.exit(1)
    
    if not os.path.exists(bg_dir):
        print(f"\n❌ ОШИБКА: Папка не найдена: {bg_dir}")
        sys.exit(1)
    
    print(f"Параметры генерации:")
    print(f"  Размер изображений: {CONFIG['image_size']}")
    print(f"  Диапазон клеток: 5-100")
    print(f"  Max размер клетки: {CONFIG['max_cell_size']}")
    print(f"  Min покрытие: {CONFIG['min_cell_coverage']}")
    print(f"  Вероятность искусственной клетки: {CONFIG['p_cell_artificial']}")
    print(f"  Количество изображений: 1000")
    print(f"  Папка сохранения: {test_dir}")
    print("="*70 + "\n")
    
    # Создание генератора
    generator = BloodCellGenerator(
        cell_dir=cells_dir,
        bg_dir=bg_dir,
        size=CONFIG['image_size'],
        min_cells=5,
        max_cells=100,
        max_cell_size=CONFIG['max_cell_size'],
        min_cell_coverage=CONFIG['min_cell_coverage'],
        p_cell_artificial=CONFIG['p_cell_artificial']
    )
    
    # Генерация изображений
    cells_counts = []
    start_time = datetime.now()
    
    for i in tqdm(range(1000), desc="Генерация test_images"):
        # Генерация изображения
        img, cell_count = generator.generate_image()
        
        # Сохранение изображения
        filename = f"image_{i+1:05d}.png"
        img_path = os.path.join(test_dir, filename)
        cv2.imwrite(img_path, img)
        
        # Сохранение метаданных
        metadata = {
            "image_id": i + 1,
            "filename": filename,
            "cell_count": cell_count,
            "density": "low",
            "density_range": [5, 100],
            "image_size": list(CONFIG['image_size']),
            "timestamp": datetime.now().isoformat()
        }
        
        json_filename = f"image_{i+1:05d}.json"
        json_path = os.path.join(test_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        cells_counts.append(cell_count)
    
    # Статистика
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print(f"СТАТИСТИКА ГЕНЕРАЦИИ")
    print(f"{'='*70}")
    print(f"Всего сгенерировано: {len(cells_counts)} изображений")
    print(f"Время выполнения: {elapsed_time:.1f} секунд ({elapsed_time/60:.1f} минут)")
    print(f"Средняя скорость: {len(cells_counts)/elapsed_time:.2f} изобр/сек")
    print(f"\nРаспределение количества клеток:")
    print(f"  Min: {min(cells_counts)}")
    print(f"  Max: {max(cells_counts)}")
    print(f"  Среднее: {np.mean(cells_counts):.2f}")
    print(f"  Медиана: {np.median(cells_counts):.2f}")
    print(f"  Std: {np.std(cells_counts):.2f}")
    print(f"{'='*70}\n")
    
    # Сохранение общей статистики
    stats = {
        "dataset_name": "test",
        "density": "low",
        "density_range": [5, 100],
        "num_samples": len(cells_counts),
        "generation_time_seconds": elapsed_time,
        "cells_statistics": {
            "min": int(min(cells_counts)),
            "max": int(max(cells_counts)),
            "mean": float(np.mean(cells_counts)),
            "median": float(np.median(cells_counts)),
            "std": float(np.std(cells_counts))
        },
        "config": {
            "image_size": CONFIG['image_size'],
            "min_cells": 5,
            "max_cells": 100,
            "max_cell_size": CONFIG['max_cell_size'],
            "min_cell_coverage": CONFIG['min_cell_coverage'],
            "p_cell_artificial": CONFIG['p_cell_artificial']
        },
        "timestamp": datetime.now().isoformat()
    }
    
    stats_path = os.path.join(test_dir, 'dataset_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Статистика сохранена: {stats_path}")
    
    print("\n" + "="*70)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    print("="*70)
    print(f"✓ Сгенерировано 1000 изображений в {test_dir}")
    print(f"✓ Готово к тестированию модели!")
    print("  Следующий шаг: python 04_test_predictions.py\n")


if __name__ == "__main__":
    main()
