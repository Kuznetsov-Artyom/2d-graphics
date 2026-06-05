#!/usr/bin/env python3
"""
Скрипт для генерации тестовых изображений с разными параметрами.
Создаёт 13 папок по 5 изображений в каждой.
"""

import os
import json
import sys
import time
import random
from datetime import datetime
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from blood_cell_generator import BloodCellGenerator, add_noise, apply_filters

# Конфигурации
CONFIGS = {
    "01_low_density_small_cells": {
        "min_cells": 10, "max_cells": 20,
        "max_cell_size": 40, "min_cell_coverage": 0.95,
        "p_cell_artificial": 0.5,
        "description": "Мало маленьких клеток"
    },
    "02_low_density_large_cells": {
        "min_cells": 10, "max_cells": 20,
        "max_cell_size": 90, "min_cell_coverage": 0.95,
        "p_cell_artificial": 0.5,
        "description": "Мало больших клеток"
    },
    "03_medium_density_strict": {
        "min_cells": 40, "max_cells": 60,
        "max_cell_size": 70, "min_cell_coverage": 0.95,
        "p_cell_artificial": 0.5,
        "description": "Средняя плотность, строгий контроль перекрытий"
    },
    "04_medium_density_relaxed": {
        "min_cells": 40, "max_cells": 60,
        "max_cell_size": 70, "min_cell_coverage": 0.70,
        "p_cell_artificial": 0.5,
        "description": "Средняя плотность, перекрытия разрешены"
    },
    "05_high_density_small_cells": {
        "min_cells": 80, "max_cells": 100,
        "max_cell_size": 50, "min_cell_coverage": 0.95,
        "p_cell_artificial": 0.5,
        "description": "Много маленьких клеток"
    },
    "06_high_density_medium_cells": {
        "min_cells": 80, "max_cells": 100,
        "max_cell_size": 70, "min_cell_coverage": 0.70,
        "p_cell_artificial": 0.5,
        "description": "Много средних клеток"
    },
    "07_with_gaussian_noise": {
        "min_cells": 40, "max_cells": 60,
        "max_cell_size": 70, "min_cell_coverage": 0.95,
        "p_cell_artificial": 0.5,
        "noise_type": "gaussian",
        "description": "С шумом Гаусса (std=20)"
    },
    "08_with_uniform_noise": {
        "min_cells": 40, "max_cells": 60,
        "max_cell_size": 70, "min_cell_coverage": 0.95,
        "p_cell_artificial": 0.5,
        "noise_type": "uniform",
        "description": "С равномерным шумом (±25)"
    },
    "09_denoised_median": {
        "min_cells": 40, "max_cells": 60,
        "max_cell_size": 70, "min_cell_coverage": 0.95,
        "p_cell_artificial": 0.5,
        "filter_method": "median",
        "description": "Зашумлённое → Медианный фильтр (k=5)"
    },
    "10_denoised_nlm": {
        "min_cells": 40, "max_cells": 60,
        "max_cell_size": 70, "min_cell_coverage": 0.95,
        "p_cell_artificial": 0.5,
        "filter_method": "nlm",
        "description": "Зашумлённое → NLM фильтр (p=10)"
    },
    "11_only_artificial_cells": {
        "min_cells": 40, "max_cells": 60,
        "max_cell_size": 70, "min_cell_coverage": 0.95,
        "p_cell_artificial": 1.0,
        "description": "Только искусственные клетки"
    },
    "12_only_real_cells": {
        "min_cells": 40, "max_cells": 60,
        "max_cell_size": 70, "min_cell_coverage": 0.95,
        "p_cell_artificial": 0.0,
        "description": "Только реальные клетки"
    },
    "13_hybrid_cells": {
        "min_cells": 40, "max_cells": 60,
        "max_cell_size": 70, "min_cell_coverage": 0.95,
        "p_cell_artificial": 0.5,
        "description": "Гибрид (50% искусственных, 50% реальных)"
    }
}


def generate_config_folder(config_name, params, output_dir, num_images=5):
    """Генерирует папку с изображениями для одной конфигурации."""
    # 1. Создание папки
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # 2. Генерация случайного seed
    seed = random.randint(1, 999999999)
    np.random.seed(seed)
    random.seed(seed)
    
    # 3. Создание генератора
    gen = BloodCellGenerator(
        cell_dir='update_generator/patches/cells',
        bg_dir='update_generator/patches/background',
        size=(512, 512),
        min_cells=params['min_cells'],
        max_cells=params['max_cells'],
        max_cell_size=params['max_cell_size'],
        min_cell_coverage=params['min_cell_coverage'],
        p_cell_artificial=params['p_cell_artificial']
    )
    
    # 4. Генерация изображений
    images_info = []
    total_time = 0
    cells_counts = []
    
    for i in range(num_images):
        start_time = time.time()
        img, actual_cells = gen.generate_image()
        gen_time = (time.time() - start_time) * 1000
        total_time += gen_time
        
        # Сохранение изображения
        filename = f"image_{i+1:02d}.png"
        filepath = os.path.join(config_dir, filename)
        cv2.imwrite(filepath, img)
        
        # Подсчёт покрытия
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        coverage = np.count_nonzero(thresh) / thresh.size * 100
        
        cells_counts.append(actual_cells)
        images_info.append({
            "filename": filename,
            "actual_cells": actual_cells,
            "coverage_percent": round(coverage, 2),
            "generation_time_ms": round(gen_time, 2)
        })
    
    # 5. Сохранение метаданных
    metadata = {
        "configuration": config_name,
        "parameters": params,
        "generation_info": {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "total_images": num_images,
            "python_version": sys.version.split()[0],
            "opencv_version": cv2.__version__
        },
        "statistics": {
            "avg_cells_per_image": round(np.mean(cells_counts), 2),
            "min_cells": min(cells_counts),
            "max_cells": max(cells_counts),
            "avg_coverage_percent": round(np.mean([img['coverage_percent'] for img in images_info]), 2),
            "total_generation_time_ms": round(total_time, 2)
        },
        "images": images_info
    }
    
    with open(os.path.join(config_dir, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata


def generate_noise_config(config_name, params, output_dir, noise_type, num_images=5):
    """Генерирует папку с зашумлёнными изображениями."""
    # 1. Создание папки
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # 2. Генерация случайного seed
    seed = random.randint(1, 999999999)
    np.random.seed(seed)
    random.seed(seed)
    
    # 3. Создание генератора
    gen = BloodCellGenerator(
        cell_dir='update_generator/patches/cells',
        bg_dir='update_generator/patches/background',
        size=(512, 512),
        min_cells=params['min_cells'],
        max_cells=params['max_cells'],
        max_cell_size=params['max_cell_size'],
        min_cell_coverage=params['min_cell_coverage'],
        p_cell_artificial=params['p_cell_artificial']
    )
    
    # 4. Генерация изображений
    images_info = []
    total_time = 0
    cells_counts = []
    
    for i in range(num_images):
        start_time = time.time()
        
        # Генерация чистого изображения
        clean_img, actual_cells = gen.generate_image()
        
        # Добавление шума
        noisy_img = add_noise(clean_img, noise_type=noise_type)
        
        gen_time = (time.time() - start_time) * 1000
        total_time += gen_time
        
        # Сохранение изображений
        clean_filename = f"image_{i+1:02d}_clean.png"
        noisy_filename = f"image_{i+1:02d}_noisy.png"
        
        cv2.imwrite(os.path.join(config_dir, clean_filename), clean_img)
        cv2.imwrite(os.path.join(config_dir, noisy_filename), noisy_img)
        
        # Подсчёт покрытия
        gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        coverage = np.count_nonzero(thresh) / thresh.size * 100
        
        cells_counts.append(actual_cells)
        images_info.append({
            "clean_filename": clean_filename,
            "noisy_filename": noisy_filename,
            "actual_cells": actual_cells,
            "coverage_percent": round(coverage, 2),
            "generation_time_ms": round(gen_time, 2)
        })
    
    # 5. Сохранение метаданных
    metadata = {
        "configuration": config_name,
        "parameters": params,
        "noise_type": noise_type,
        "generation_info": {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "total_images": num_images,
            "python_version": sys.version.split()[0],
            "opencv_version": cv2.__version__
        },
        "statistics": {
            "avg_cells_per_image": round(np.mean(cells_counts), 2),
            "min_cells": min(cells_counts),
            "max_cells": max(cells_counts),
            "avg_coverage_percent": round(np.mean([img['coverage_percent'] for img in images_info]), 2),
            "total_generation_time_ms": round(total_time, 2)
        },
        "images": images_info
    }
    
    with open(os.path.join(config_dir, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata


def generate_denoise_config(config_name, params, output_dir, filter_method, num_images=5):
    """Генерирует папку с отфильтрованными изображениями."""
    # 1. Создание папки
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # 2. Генерация случайного seed
    seed = random.randint(1, 999999999)
    np.random.seed(seed)
    random.seed(seed)
    
    # 3. Создание генератора
    gen = BloodCellGenerator(
        cell_dir='update_generator/patches/cells',
        bg_dir='update_generator/patches/background',
        size=(512, 512),
        min_cells=params['min_cells'],
        max_cells=params['max_cells'],
        max_cell_size=params['max_cell_size'],
        min_cell_coverage=params['min_cell_coverage'],
        p_cell_artificial=params['p_cell_artificial']
    )
    
    # 4. Генерация изображений
    images_info = []
    total_time = 0
    cells_counts = []
    
    for i in range(num_images):
        start_time = time.time()
        
        # Генерация чистого изображения
        clean_img, actual_cells = gen.generate_image()
        
        # Добавление шума (Gaussian по умолчанию)
        noisy_img = add_noise(clean_img, noise_type='gaussian')
        
        # Применение фильтра
        if filter_method == 'median':
            denoised_img = apply_filters(noisy_img, method='median', param=5)
        elif filter_method == 'nlm':
            denoised_img = apply_filters(noisy_img, method='nlm', param=10)
        else:
            denoised_img = noisy_img
        
        gen_time = (time.time() - start_time) * 1000
        total_time += gen_time
        
        # Сохранение изображений
        noisy_filename = f"image_{i+1:02d}_noisy.png"
        denoised_filename = f"image_{i+1:02d}_denoised.png"
        
        cv2.imwrite(os.path.join(config_dir, noisy_filename), noisy_img)
        cv2.imwrite(os.path.join(config_dir, denoised_filename), denoised_img)
        
        # Подсчёт покрытия
        gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        coverage = np.count_nonzero(thresh) / thresh.size * 100
        
        cells_counts.append(actual_cells)
        images_info.append({
            "noisy_filename": noisy_filename,
            "denoised_filename": denoised_filename,
            "actual_cells": actual_cells,
            "coverage_percent": round(coverage, 2),
            "generation_time_ms": round(gen_time, 2)
        })
    
    # 5. Сохранение метаданных
    metadata = {
        "configuration": config_name,
        "parameters": params,
        "filter_method": filter_method,
        "generation_info": {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "total_images": num_images,
            "python_version": sys.version.split()[0],
            "opencv_version": cv2.__version__
        },
        "statistics": {
            "avg_cells_per_image": round(np.mean(cells_counts), 2),
            "min_cells": min(cells_counts),
            "max_cells": max(cells_counts),
            "avg_coverage_percent": round(np.mean([img['coverage_percent'] for img in images_info]), 2),
            "total_generation_time_ms": round(total_time, 2)
        },
        "images": images_info
    }
    
    with open(os.path.join(config_dir, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata


def main():
    output_dir = 'update_generator/test_images'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("ГЕНЕРАЦИЯ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ")
    print("=" * 70)
    print(f"Размер изображений: 512x512")
    print(f"Количество конфигураций: {len(CONFIGS)}")
    print(f"Изображений на конфигурацию: 5")
    print(f"Всего изображений: {len(CONFIGS) * 5}")
    print("=" * 70)
    print()
    
    all_stats = []
    
    for config_name, params in CONFIGS.items():
        print(f"[{config_name}]")
        print(f"  Описание: {params.get('description', 'N/A')}")
        
        if "noise_type" in params:
            noise_type = params["noise_type"]
            metadata = generate_noise_config(config_name, params, output_dir, noise_type)
            print(f"  Тип: Зашумлённые изображения ({noise_type})")
        elif "filter_method" in params:
            filter_method = params["filter_method"]
            metadata = generate_denoise_config(config_name, params, output_dir, filter_method)
            print(f"  Тип: Отфильтрованные изображения ({filter_method})")
        else:
            metadata = generate_config_folder(config_name, params, output_dir)
            print(f"  Тип: Базовая конфигурация")
        
        all_stats.append(metadata)
        print(f"  ✓ Создано {metadata['generation_info']['total_images']} изображений")
        print(f"  ✓ Seed: {metadata['generation_info']['seed']}")
        print(f"  ✓ Среднее клеток: {metadata['statistics']['avg_cells_per_image']}")
        print(f"  ✓ Среднее покрытие: {metadata['statistics']['avg_coverage_percent']}%")
        print()
    
    # Сохранение общей статистики
    summary = {
        "total_configurations": len(CONFIGS),
        "total_images": len(CONFIGS) * 5,
        "image_size": [512, 512],
        "timestamp": datetime.now().isoformat(),
        "configurations": all_stats
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("=" * 70)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"Всего создано: {len(CONFIGS) * 5} изображений")
    print(f"Папка: {output_dir}")
    print(f"Общая статистика: {output_dir}/summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
