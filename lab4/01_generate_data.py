#!/usr/bin/env python3
"""
Скрипт 1: Генерация train и validation данных для обучения модели.

Генерирует:
- 10,000 изображений для обучения (data/train/)
- 2,000 изображений для валидации (data/val/)

Каждое изображение сохраняется с JSON метаданными (количество клеток).

Время выполнения: ~30 минут
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


def generate_dataset(output_dir, num_samples, dataset_name, density_config, density_name, start_id=0):
    """
    Генерирует датасет с указанной плотностью и сохраняет изображения с метаданными.
    
    Args:
        output_dir: путь для сохранения (data/train/ или data/val/)
        num_samples: количество изображений
        dataset_name: название датасета (для логов)
        density_config: конфигурация плотности (min_cells, max_cells)
        density_name: название плотности (low, medium, high, very_high)
        start_id: начальный ID для нумерации файлов
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ГЕНЕРАЦИЯ {dataset_name.upper()} ДАТАСЕТА")
    print(f"{'='*70}")
    print(f"Плотность: {density_name} ({density_config['min_cells']}-{density_config['max_cells']} клеток)")
    print(f"Количество изображений: {num_samples}")
    print(f"Папка сохранения: {output_dir}")
    print(f"{'='*70}\n")
    
    # Создание генератора с параметрами плотности
    patches_dir = CONFIG['patches_dir']
    cells_dir = os.path.join(patches_dir, 'cells')
    bg_dir = os.path.join(patches_dir, 'background')
    
    generator = BloodCellGenerator(
        cell_dir=cells_dir,
        bg_dir=bg_dir,
        size=CONFIG['image_size'],
        min_cells=density_config['min_cells'],
        max_cells=density_config['max_cells'],
        max_cell_size=CONFIG['max_cell_size'],
        min_cell_coverage=CONFIG['min_cell_coverage'],
        p_cell_artificial=CONFIG['p_cell_artificial']
    )
    
    cells_counts = []
    start_time = datetime.now()
    
    for i in tqdm(range(num_samples), desc=f"Генерация {dataset_name}"):
        # Генерация изображения
        img, cell_count = generator.generate_image()
        
        # Использование start_id для нумерации
        global_id = start_id + i
        img_filename = f"image_{global_id+1:05d}.png"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, img)
        
        # Сохранение метаданных с меткой плотности
        metadata = {
            "image_id": global_id + 1,
            "filename": img_filename,
            "cell_count": cell_count,
            "density": density_name,
            "density_range": [density_config['min_cells'], density_config['max_cells']],
            "image_size": list(CONFIG['image_size']),
            "timestamp": datetime.now().isoformat()
        }
        
        json_filename = f"image_{global_id+1:05d}.json"
        json_path = os.path.join(output_dir, json_filename)
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
        "dataset_name": dataset_name,
        "density": density_name,
        "density_range": [density_config['min_cells'], density_config['max_cells']],
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
            "min_cells": density_config['min_cells'],
            "max_cells": density_config['max_cells'],
            "max_cell_size": CONFIG['max_cell_size'],
            "min_cell_coverage": CONFIG['min_cell_coverage'],
            "p_cell_artificial": CONFIG['p_cell_artificial']
        },
        "timestamp": datetime.now().isoformat()
    }
    
    stats_path = os.path.join(output_dir, f'dataset_stats_{density_name}.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Статистика сохранена: {stats_path}")
    
    return stats


def main():
    """Основная функция генерации датасета с разными плотностями."""
    
    print("\n" + "="*70)
    print("ГЕНЕРАЦИЯ ДАТАСЕТА С РАЗНЫМИ ПЛОТНОСТЯМИ")
    print("="*70)
    print(f"Конфигурации плотности:")
    for name, config in CONFIG['density_configs'].items():
        ratio = CONFIG['density_distribution'][name]
        print(f"  - {name}: {config['min_cells']}-{config['max_cells']} клеток ({ratio*100:.0f}%)")
    print("="*70)
    
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
    
    # Папки для данных
    train_dir = os.path.join(CONFIG['data_dir'], 'train')
    val_dir = os.path.join(CONFIG['data_dir'], 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Генерация для каждой плотности
    all_train_stats = []
    all_val_stats = []
    image_counter_train = 0
    image_counter_val = 0
    
    for density_name, density_config in CONFIG['density_configs'].items():
        ratio = CONFIG['density_distribution'][density_name]
        num_train = int(CONFIG['train_size'] * ratio)
        num_val = int(CONFIG['val_size'] * ratio)
        
        print(f"\n{'='*70}")
        print(f"ГЕНЕРАЦИЯ ПЛОТНОСТИ: {density_name.upper()}")
        print(f"{'='*70}")
        print(f"Диапазон: {density_config['min_cells']}-{density_config['max_cells']} клеток")
        print(f"Train: {num_train} изображений ({ratio*100:.0f}%)")
        print(f"Val: {num_val} изображений ({ratio*100:.0f}%)")
        print(f"{'='*70}")
        
        # Генерация train
        train_stats = generate_dataset(
            output_dir=train_dir,
            num_samples=num_train,
            dataset_name=f'train_{density_name}',
            density_config=density_config,
            density_name=density_name,
            start_id=image_counter_train
        )
        all_train_stats.append(train_stats)
        image_counter_train += num_train
        
        # Генерация val
        val_stats = generate_dataset(
            output_dir=val_dir,
            num_samples=num_val,
            dataset_name=f'val_{density_name}',
            density_config=density_config,
            density_name=density_name,
            start_id=image_counter_val
        )
        all_val_stats.append(val_stats)
        image_counter_val += num_val
    
    # Итоговая статистика
    print("\n" + "="*70)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    print("="*70)
    print(f"Train датасет: {train_dir}")
    print(f"  - Всего изображений: {image_counter_train}")
    for stats in all_train_stats:
        print(f"  - {stats['density']}: {stats['num_samples']} (avg: {stats['cells_statistics']['mean']:.1f})")
    
    print(f"\nValidation датасет: {val_dir}")
    print(f"  - Всего изображений: {image_counter_val}")
    for stats in all_val_stats:
        print(f"  - {stats['density']}: {stats['num_samples']} (avg: {stats['cells_statistics']['mean']:.1f})")
    
    print(f"\nОбщий размер: {image_counter_train + image_counter_val} изображений")
    print("="*70)
    print("\n✓ Готово к обучению модели!")
    print("  Следующий шаг: python 02_train_model.py\n")


if __name__ == "__main__":
    main()
