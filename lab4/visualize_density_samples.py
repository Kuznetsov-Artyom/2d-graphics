#!/usr/bin/env python3
"""
Визуализация тестовых изображений с разной плотностью клеток.

Создает сетку 2x2 с примерами из каждой категории плотности.
"""

import os
import cv2
import matplotlib.pyplot as plt
from config import CONFIG

def visualize_density_samples():
    """Визуализирует по одному примеру из каждой плотности."""
    
    # Пути к примерам
    examples = {
        'low (20-40)': 'data/test_density/low/low_image_01.png',
        'medium (40-60)': 'data/test_density/medium/medium_image_01.png',
        'high (60-80)': 'data/test_density/high/high_image_01.png',
        'very_high (80-100)': 'data/test_density/very_high/very_high_image_01.png'
    }
    
    # Создание фигуры 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle('Примеры изображений с разной плотностью клеток', fontsize=16, fontweight='bold')
    
    # Загрузка и отображение изображений
    for ax, (density_name, img_path) in zip(axes.flat, examples.items()):
        if os.path.exists(img_path):
            # Загрузка изображения (BGR -> RGB)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Отображение
            ax.imshow(img_rgb)
            ax.set_title(density_name, fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Добавление информации о количестве клеток
            json_path = img_path.replace('.png', '.json')
            if os.path.exists(json_path):
                import json
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                cell_count = metadata['cell_count']
                ax.text(10, 30, f'Клеток: {cell_count}', 
                       color='white', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        else:
            ax.text(0.5, 0.5, f'Файл не найден:\n{img_path}', 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Сохранение
    output_path = os.path.join(CONFIG['results_dir'], 'density_samples.png')
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Визуализация сохранена: {output_path}")
    return output_path


if __name__ == "__main__":
    output = visualize_density_samples()
    print(f"\nОткройте файл для просмотра: {output}")
