#!/usr/bin/env python3
"""
Тестовый скрипт: Генерация изображений с разной плотностью клеток.

Генерирует по 5 изображений для каждой конфигурации плотности:
- low: 20-40 клеток
- medium: 40-60 клеток
- high: 60-80 клеток
- very_high: 80-100 клеток

Итого: 20 изображений для визуальной проверки.
"""

import os
import sys
import json
import cv2
import numpy as np
from datetime import datetime

# Добавляем путь к генератору
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'update_generator'))
from blood_cell_generator import BloodCellGenerator

# Импортируем конфигурацию
from config import CONFIG

# Конфигурации плотности
DENSITY_CONFIGS = {
    'low': {
        'min_cells': 20,
        'max_cells': 40,
        'description': 'Низкая плотность'
    },
    'medium': {
        'min_cells': 40,
        'max_cells': 60,
        'description': 'Средняя плотность'
    },
    'high': {
        'min_cells': 60,
        'max_cells': 80,
        'description': 'Высокая плотность'
    },
    'very_high': {
        'min_cells': 80,
        'max_cells': 100,
        'description': 'Очень высокая плотность'
    }
}


def generate_test_images(density_name, density_config, output_dir, num_samples=5):
    """
    Генерирует тестовые изображения для указанной плотности.
    
    Args:
        density_name: название плотности (low, medium, high, very_high)
        density_config: конфигурация плотности
        output_dir: папка для сохранения
        num_samples: количество изображений
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ГЕНЕРАЦИЯ: {density_name.upper()} ({density_config['description']})")
    print(f"{'='*70}")
    print(f"Диапазон клеток: {density_config['min_cells']}-{density_config['max_cells']}")
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
    
    print(f"✓ Генератор создан")
    print(f"  Патчей клеток: {len(generator.cell_files)}")
    print(f"  Патчей фона: {len(generator.bg_files)}\n")
    
    # Генерация изображений
    cells_counts = []
    
    for i in range(num_samples):
        # Генерация изображения
        img, cell_count = generator.generate_image()
        
        # Сохранение изображения
        img_filename = f"{density_name}_image_{i+1:02d}.png"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, img)
        
        # Сохранение метаданных
        metadata = {
            "image_id": i + 1,
            "filename": img_filename,
            "cell_count": cell_count,
            "density": density_name,
            "density_range": [density_config['min_cells'], density_config['max_cells']],
            "image_size": list(CONFIG['image_size']),
            "timestamp": datetime.now().isoformat()
        }
        
        json_filename = f"{density_name}_image_{i+1:02d}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        cells_counts.append(cell_count)
        print(f"  ✓ {img_filename}: {cell_count} клеток")
    
    # Статистика
    print(f"\nСтатистика для {density_name}:")
    print(f"  Min: {min(cells_counts)}")
    print(f"  Max: {max(cells_counts)}")
    print(f"  Среднее: {np.mean(cells_counts):.1f}")
    print(f"  Медиана: {np.median(cells_counts):.1f}")
    
    return cells_counts


def main():
    """Основная функция генерации тестовых изображений."""
    
    print("\n" + "="*70)
    print("ТЕСТОВАЯ ГЕНЕРАЦИЯ: РАЗНАЯ ПЛОТНОСТЬ КЛЕТОК")
    print("="*70)
    print(f"Конфигурации плотности:")
    for name, config in DENSITY_CONFIGS.items():
        print(f"  - {name}: {config['min_cells']}-{config['max_cells']} клеток ({config['description']})")
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
    
    # Папка для тестовых изображений
    test_dir = os.path.join(CONFIG['data_dir'], 'test_density')
    os.makedirs(test_dir, exist_ok=True)
    
    # Генерация для каждой плотности
    all_stats = {}
    
    for density_name, density_config in DENSITY_CONFIGS.items():
        output_dir = os.path.join(test_dir, density_name)
        cells_counts = generate_test_images(
            density_name=density_name,
            density_config=density_config,
            output_dir=output_dir,
            num_samples=5
        )
        
        all_stats[density_name] = {
            'min': int(min(cells_counts)),
            'max': int(max(cells_counts)),
            'mean': float(np.mean(cells_counts)),
            'median': float(np.median(cells_counts)),
            'samples': cells_counts
        }
    
    # Сохранение общей статистики
    summary = {
        'test_name': 'density_configs',
        'num_samples_per_density': 5,
        'total_samples': 5 * len(DENSITY_CONFIGS),
        'density_configs': DENSITY_CONFIGS,
        'statistics': all_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(test_dir, 'test_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Итоговая сводка
    print("\n" + "="*70)
    print("ТЕСТОВАЯ ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    print("="*70)
    print(f"Всего сгенерировано: {5 * len(DENSITY_CONFIGS)} изображений")
    print(f"Папка: {test_dir}")
    print(f"\nСтруктура:")
    for density_name in DENSITY_CONFIGS.keys():
        density_dir = os.path.join(test_dir, density_name)
        num_files = len([f for f in os.listdir(density_dir) if f.endswith('.png')])
        print(f"  - {density_name}/: {num_files} изображений")
    
    print(f"\nСтатистика по плотностям:")
    for density_name, stats in all_stats.items():
        print(f"  {density_name:12s}: {stats['min']:2d}-{stats['max']:2d} клеток (avg: {stats['mean']:.1f})")
    
    print("="*70)
    print("\n✓ Готово! Проверьте изображения в папке test_density/")
    print("  Если результат устраивает, можно запускать полную генерацию.\n")


if __name__ == "__main__":
    main()
