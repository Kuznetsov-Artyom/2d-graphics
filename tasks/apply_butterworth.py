import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task3 import ButterworthFilter, FilterAnalyzer


def process_single_image(image_path, output_dir):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image is None:
        return None
    
    img_name = os.path.basename(image_path)
    base_name = os.path.splitext(img_name)[0]
    
    print(f'Обработка: {img_name}')
    print(f'Размер: {image.shape}')
    
    # Optimized parameters for faster processing
    cutoffs = [40, 80]
    orders = [1, 2]
    
    fig, axes = plt.subplots(len(orders), len(cutoffs) + 1, 
                            figsize=(5 * (len(cutoffs) + 1), 5 * len(orders)))
    
    results = []
    
    for i, order in enumerate(orders):
        for j, cutoff in enumerate(cutoffs):
            low_pass = ButterworthFilter.low_pass_filter(image, cutoff, order)
            
            axes[i, j].imshow(low_pass, cmap='gray')
            axes[i, j].set_title(f'Low-Pass\nD0={cutoff}, n={order}', fontsize=10)
            axes[i, j].axis('off')
            
            mse = FilterAnalyzer.calculate_mse(image, low_pass)
            psnr = FilterAnalyzer.calculate_psnr(image, low_pass)
            axes[i, j].text(5, 20, f'PSNR: {psnr:.1f} dB\nMSE: {mse:.1f}', 
                           color='white', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            results.append({
                'image': img_name,
                'filter_type': 'Low-Pass',
                'cutoff': cutoff,
                'order': order,
                'psnr': psnr,
                'mse': mse
            })
        
        # High-pass filter with middle cutoff
        high_pass = ButterworthFilter.high_pass_filter(image, cutoffs[len(cutoffs)//2], order)
        axes[i, -1].imshow(high_pass, cmap='gray')
        axes[i, -1].set_title(f'High-Pass\nD0={cutoffs[len(cutoffs)//2]}, n={order}', fontsize=10)
        axes[i, -1].axis('off')
    
    plt.suptitle(f'Фильтры Баттерворта: {img_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{base_name}_butterworth_filters.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Сохранено: {base_name}_butterworth_filters.png')
    
    return results


def main():
    images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(output_dir, exist_ok=True)
    
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print('=== Применение фильтра Баттерворта на реальных изображениях ===\n')
    
    all_results = []
    
    for img_name in images:
        image_path = os.path.join(images_dir, img_name)
        results = process_single_image(image_path, output_dir)
        if results:
            all_results.extend(results)
        print()
    
    print('=== Обработка завершена ===')
    
    # Print summary table
    print('\n=== Сводная таблица результатов ===')
    print(f"{'Изображение':<15} {'Фильтр':<10} {'D0':<6} {'n':<4} {'PSNR':<8} {'MSE':<10}")
    print('-' * 65)
    
    for r in all_results:
        print(f"{r['image']:<15} {r['filter_type']:<10} {r['cutoff']:<6} {r['order']:<4} {r['psnr']:<8.2f} {r['mse']:<10.2f}")
    
    print('-' * 65)
    print(f'Всего обработано: {len(images)} изображений')
    print(f'Всего комбинаций: {len(all_results)}')


if __name__ == "__main__":
    main()
