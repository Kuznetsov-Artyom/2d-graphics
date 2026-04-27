import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import fftpack
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class ButterworthFilter:
    
    @staticmethod
    def create_low_pass_filter(shape, cutoff_frequency, order=2):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        filter_mask = np.zeros((rows, cols), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                filter_mask[i, j] = 1 / (1 + (distance / cutoff_frequency) ** (2 * order))
        
        return filter_mask
    
    @staticmethod
    def create_high_pass_filter(shape, cutoff_frequency, order=2):
        low_pass = ButterworthFilter.create_low_pass_filter(shape, cutoff_frequency, order)
        high_pass = 1 - low_pass
        return high_pass
    
    @staticmethod
    def apply_filter(image, filter_mask):
        rows, cols = image.shape
        
        f = fftpack.fft2(image)
        fshift = fftpack.fftshift(f)
        
        fshift_filtered = fshift * filter_mask
        
        f_ishift = fftpack.ifftshift(fshift_filtered)
        img_back = fftpack.ifft2(f_ishift)
        img_back = np.abs(img_back).astype(np.uint8)
        
        return img_back
    
    @staticmethod
    def low_pass_filter(image, cutoff_frequency, order=2):
        filter_mask = ButterworthFilter.create_low_pass_filter(image.shape, cutoff_frequency, order)
        return ButterworthFilter.apply_filter(image, filter_mask)
    
    @staticmethod
    def high_pass_filter(image, cutoff_frequency, order=2):
        filter_mask = ButterworthFilter.create_high_pass_filter(image.shape, cutoff_frequency, order)
        return ButterworthFilter.apply_filter(image, filter_mask)


class FilterAnalyzer:
    
    @staticmethod
    def calculate_psnr(original, filtered):
        mse = np.mean((original.astype(np.float32) - filtered.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    @staticmethod
    def calculate_mse(original, filtered):
        return np.mean((original.astype(np.float32) - filtered.astype(np.float32)) ** 2)
    
    @staticmethod
    def get_fft_spectrum(image):
        f = fftpack.fft2(image)
        fshift = fftpack.fftshift(f)
        return np.log(np.abs(fshift) + 1)


def test_butterworth_filters():
    print("=== Тестирование фильтра Баттерворта ===")
    
    test_image = np.zeros((256, 256), dtype=np.uint8)
    test_image[64:192, 64:192] = 150
    test_image[96:160, 96:160] = 200
    
    for i in range(64, 192, 8):
        test_image[i:i+4, 64:192] = 100
    
    for j in range(64, 192, 8):
        test_image[64:192, j:j+4] = 100
    
    cutoff_frequencies = [10, 20, 40, 80]
    orders = [1, 2, 4]
    
    fig, axes = plt.subplots(len(orders), len(cutoff_frequencies) + 1, 
                            figsize=(4 * (len(cutoff_frequencies) + 1), 4 * len(orders)))
    
    for i, order in enumerate(orders):
        for j, cutoff in enumerate(cutoff_frequencies):
            filtered = ButterworthFilter.low_pass_filter(test_image, cutoff, order)
            
            axes[i, j].imshow(filtered, cmap='gray')
            axes[i, j].set_title(f'Low-Pass\nD0={cutoff}, n={order}', fontsize=10)
            axes[i, j].axis('off')
            
            mse = FilterAnalyzer.calculate_mse(test_image, filtered)
            psnr = FilterAnalyzer.calculate_psnr(test_image, filtered)
            axes[i, j].text(5, 20, f'PSNR: {psnr:.1f} dB', 
                           color='white', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        filter_mask = ButterworthFilter.create_low_pass_filter(test_image.shape, 30, order)
        axes[i, -1].imshow(filter_mask, cmap='hot')
        axes[i, -1].set_title(f'Filter Mask\nn={order}', fontsize=10)
        axes[i, -1].axis('off')
    
    plt.suptitle('Низкочастотный фильтр Баттерворта', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'butterworth_low_pass_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Низкочастотная фильтрация завершена")
    
    fig, axes = plt.subplots(len(orders), len(cutoff_frequencies) + 1, 
                            figsize=(4 * (len(cutoff_frequencies) + 1), 4 * len(orders)))
    
    for i, order in enumerate(orders):
        for j, cutoff in enumerate(cutoff_frequencies):
            filtered = ButterworthFilter.high_pass_filter(test_image, cutoff, order)
            
            axes[i, j].imshow(filtered, cmap='gray')
            axes[i, j].set_title(f'High-Pass\nD0={cutoff}, n={order}', fontsize=10)
            axes[i, j].axis('off')
        
        filter_mask = ButterworthFilter.create_high_pass_filter(test_image.shape, 30, order)
        axes[i, -1].imshow(filter_mask, cmap='hot')
        axes[i, -1].set_title(f'Filter Mask\nn={order}', fontsize=10)
        axes[i, -1].axis('off')
    
    plt.suptitle('Высокочастотный фильтр Баттерворта', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'butterworth_high_pass_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Высокочастотная фильтрация завершена")


def test_on_real_images():
    print("\n=== Тестирование на реальных изображениях ===")
    
    images_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not images:
        print("Изображения не найдены")
        return
    
    test_image = images[0]
    image_path = os.path.join(images_dir, test_image)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print(f"Тестовое изображение: {test_image}")
    print(f"Размер: {image.shape}")
    
    cutoffs = [30, 60, 100]
    orders = [1, 2, 3]
    
    fig, axes = plt.subplots(len(orders), len(cutoffs) + 2, 
                            figsize=(5 * (len(cutoffs) + 2), 5 * len(orders)))
    
    for i, order in enumerate(orders):
        for j, cutoff in enumerate(cutoffs):
            low_pass = ButterworthFilter.low_pass_filter(image, cutoff, order)
            high_pass = ButterworthFilter.high_pass_filter(image, cutoff, order)
            
            axes[i, j].imshow(low_pass, cmap='gray')
            axes[i, j].set_title(f'Low-Pass\nD0={cutoff}, n={order}', fontsize=10)
            axes[i, j].axis('off')
            
            mse = FilterAnalyzer.calculate_mse(image, low_pass)
            psnr = FilterAnalyzer.calculate_psnr(image, low_pass)
            axes[i, j].text(5, 20, f'PSNR: {psnr:.1f} dB\nMSE: {mse:.1f}', 
                           color='white', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        axes[i, -2].imshow(high_pass, cmap='gray')
        axes[i, -2].set_title(f'High-Pass\nD0={cutoffs[-1]}, n={order}', fontsize=10)
        axes[i, -2].axis('off')
        
        filter_mask = ButterworthFilter.create_low_pass_filter(image.shape, cutoffs[-1], order)
        axes[i, -1].imshow(filter_mask, cmap='hot')
        axes[i, -1].set_title(f'Filter Mask\nn={order}', fontsize=10)
        axes[i, -1].axis('off')
    
    plt.suptitle(f'Фильтры Баттерворта: {test_image}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'butterworth_real_image_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Тестирование на реальном изображении завершено")


def compare_filter_orders():
    print("\n=== Сравнение порядков фильтра ===")
    
    test_image = np.zeros((256, 256), dtype=np.uint8)
    
    for i in range(0, 256, 2):
        test_image[i, :] = 150
    
    for j in range(0, 256, 2):
        test_image[:, j] = 150
    
    test_image[64:192, 64:192] = 200
    
    cutoff = 40
    orders = [1, 2, 3, 4, 5, 6]
    
    fig, axes = plt.subplots(2, len(orders), figsize=(4 * len(orders), 8))
    
    for i, order in enumerate(orders):
        low_pass = ButterworthFilter.low_pass_filter(test_image, cutoff, order)
        high_pass = ButterworthFilter.high_pass_filter(test_image, cutoff, order)
        
        axes[0, i].imshow(low_pass, cmap='gray')
        axes[0, i].set_title(f'Low-Pass\nn={order}', fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(high_pass, cmap='gray')
        axes[1, i].set_title(f'High-Pass\nn={order}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle(f'Сравнение порядков фильтра Баттерворта (D0={cutoff})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'butterworth_order_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Сравнение порядков завершено")


def create_filter_visualization():
    print("\n=== Визуализация фильтров ===")
    
    shape = (256, 256)
    cutoff = 50
    orders = [1, 2, 4]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, order in enumerate(orders):
        low_pass_mask = ButterworthFilter.create_low_pass_filter(shape, cutoff, order)
        high_pass_mask = ButterworthFilter.create_high_pass_filter(shape, cutoff, order)
        
        im0 = axes[0, i].imshow(low_pass_mask, cmap='hot')
        axes[0, i].set_title(f'Low-Pass Filter\nD0={cutoff}, n={order}', fontsize=12)
        axes[0, i].axis('off')
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        im1 = axes[1, i].imshow(high_pass_mask, cmap='hot')
        axes[1, i].set_title(f'High-Pass Filter\nD0={cutoff}, n={order}', fontsize=12)
        axes[1, i].axis('off')
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Маски фильтров Баттерворта', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'butterworth_filter_masks.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Визуализация фильтров завершена")


def print_results_table():
    print("\n=== Результаты тестирования ===")
    print("=" * 80)
    print(f"{'Тип фильтра':<15} {'Частота среза':<15} {'Порядок':<10} {'Описание':<30}")
    print("-" * 80)
    
    filter_types = [
        ("Low-Pass", "10-80", "1-4", "Удаление высоких частот, сглаживание"),
        ("High-Pass", "10-80", "1-4", "Удаление низких частот, выделение границ"),
        ("Low-Pass", "30-100", "1-3", "Оптимизация для реальных изображений"),
        ("High-Pass", "30-100", "1-3", "Выделение деталей на реальных изображениях"),
    ]
    
    for filter_type, cutoff, order, description in filter_types:
        print(f"{filter_type:<15} {cutoff:<15} {order:<10} {description:<30}")
    
    print("=" * 80)
    print("\nКлючевые характеристики:")
    print("• D0 (частота среза): определяет граничную частоту фильтра")
    print("• n (порядок): влияет на крутизну перехода от полосы пропускания к задерживания")
    print("• Low-Pass: сохраняет низкие частоты, удаляет высокие")
    print("• High-Pass: сохраняет высокие частоты, удаляет низкие")


def main():
    print("TASK 3: Фильтр Баттерворта для низкочастотной/высокочастотной фильтрации")
    print("=" * 80)
    
    test_butterworth_filters()
    test_on_real_images()
    compare_filter_orders()
    create_filter_visualization()
    print_results_table()
    
    print("\n" + "=" * 80)
    print("ЗАДАЧА 3 ЗАВЕРШЕНА")
    print("=" * 80)
    print("\nСозданные файлы:")
    print("• butterworth_low_pass_test.png - тестирование низкочастотной фильтрации")
    print("• butterworth_high_pass_test.png - тестирование высокочастотной фильтрации")
    print("• butterworth_real_image_test.png - тестирование на реальном изображении")
    print("• butterworth_order_comparison.png - сравнение порядков фильтра")
    print("• butterworth_filter_masks.png - визуализация масок фильтров")


if __name__ == "__main__":
    main()
