import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import fftpack
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from homework3 import FourierTransform


def create_demonstration():
    
    test_image = np.zeros((256, 256), dtype=np.uint8)
    test_image[64:192, 64:192] = 150
    test_image[80:176, 80:176] = 200
    test_image[96:160, 96:160] = 100
    
    for i in range(64, 192, 16):
        test_image[i:i+8, 64:192] = 180
    
    noise_combinations = [
        {
            'name': 'Слабый шум',
            'frequency': 12,
            'amplitude': 20,
            'threshold': 60,
            'description': 'Малая амплитуда, высокая частота'
        },
        {
            'name': 'Средний шум',
            'frequency': 8,
            'amplitude': 40,
            'threshold': 50,
            'description': 'Средняя амплитуда, средняя частота'
        },
        {
            'name': 'Сильный шум',
            'frequency': 6,
            'amplitude': 70,
            'threshold': 40,
            'description': 'Высокая амплитуда, низкая частота'
        },
        {
            'name': 'Очень сильный шум',
            'frequency': 5,
            'amplitude': 100,
            'threshold': 30,
            'description': 'Очень высокая амплитуда, низкая частота'
        },
        {
            'name': 'Частый мелкий шум',
            'frequency': 15,
            'amplitude': 60,
            'threshold': 70,
            'description': 'Высокая частота, высокая амплитуда'
        },
        {
            'name': 'Редкий крупный шум',
            'frequency': 4,
            'amplitude': 80,
            'threshold': 35,
            'description': 'Низкая частота, очень высокая амплитуда'
        }
    ]
    
    results = []
    
    for combo in noise_combinations:
        print(f"Обработка: {combo['name']}")
        
        noisy = FourierTransform.add_periodic_noise(
            test_image, 
            frequency=combo['frequency'], 
            amplitude=combo['amplitude']
        )
        
        denoised = FourierTransform.fft_denoise(
            noisy, 
            threshold=combo['threshold'], 
            noise_frequency=combo['frequency']
        )
        
        mse_noisy = np.mean((test_image.astype(np.float32) - noisy.astype(np.float32)) ** 2)
        mse_denoised = np.mean((test_image.astype(np.float32) - denoised.astype(np.float32)) ** 2)
        psnr = FourierTransform.calculate_psnr(test_image, denoised)
        improvement = mse_noisy - mse_denoised
        
        results.append({
            'combo': combo,
            'noisy': noisy,
            'denoised': denoised,
            'mse_noisy': mse_noisy,
            'mse_denoised': mse_denoised,
            'psnr': psnr,
            'improvement': improvement
        })
    
    create_comparison_visualization(test_image, results)
    create_detailed_analysis(test_image, results)
    print_results_table(results)


def create_comparison_visualization(test_image, results):
    
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 4 * len(results)))
    
    for i, result in enumerate(results):
        combo = result['combo']
        
        axes[i, 0].imshow(result['noisy'], cmap='gray')
        axes[i, 0].set_title(f"{combo['name']}\n(с шумом)", fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(result['denoised'], cmap='gray')
        axes[i, 1].set_title(f"После FFT\nPSNR: {result['psnr']:.2f} dB", fontsize=10)
        axes[i, 1].axis('off')
        
        error_map = np.abs(test_image.astype(np.float32) - result['denoised'].astype(np.float32))
        axes[i, 2].imshow(error_map, cmap='hot')
        axes[i, 2].set_title(f"Карта ошибок\nMSE: {result['mse_denoised']:.2f}", fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(SCRIPT_DIR, 'fft_noise_demonstration.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Визуализация сохранена: {output_path}")


def create_detailed_analysis(test_image, results):
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, result in enumerate(results):
        combo = result['combo']
        
        noisy_spectrum = get_fft_spectrum(result['noisy'])
        denoised_spectrum = get_fft_spectrum(result['denoised'])
        
        row = i // 3
        col = i % 3
        
        im = axes[row, col].imshow(noisy_spectrum - denoised_spectrum, cmap='RdBu', vmin=-5, vmax=5)
        axes[row, col].set_title(
            f"{combo['name']}\n"
            f"freq={combo['frequency']}, amp={combo['amplitude']}\n"
            f"PSNR: {result['psnr']:.2f} dB, улучшение: {result['improvement']:.2f}",
            fontsize=9
        )
        axes[row, col].axis('off')
    
    plt.suptitle('Разница FFT спектров (шумный - очищенный)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fft_spectrum_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Анализ спектров сохранен: {output_path}")


def get_fft_spectrum(image):
    f = fftpack.fft2(image)
    fshift = fftpack.fftshift(f)
    return np.log(np.abs(fshift) + 1)


def print_results_table(results):
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ РАЗЛИЧНЫХ ПАРАМЕТРОВ ШУМА")
    print("=" * 80)
    print(f"{'Тип шума':<20} {'Частота':<8} {'Амплитуда':<10} {'Порог':<8} {'MSE до':<10} {'MSE после':<12} {'PSNR':<8} {'Улучшение':<12}")
    print("-" * 100)
    
    for result in results:
        combo = result['combo']
        print(f"{combo['name']:<20} {combo['frequency']:<8} {combo['amplitude']:<10} "
              f"{combo['threshold']:<8} {result['mse_noisy']:<10.2f} {result['mse_denoised']:<12.2f} "
              f"{result['psnr']:<8.2f} {result['improvement']:<12.2f}")
    
    print("-" * 100)
    
    avg_improvement = np.mean([r['improvement'] for r in results])
    avg_psnr = np.mean([r['psnr'] for r in results])
    print(f"{'СРЕДНЕЕ':<20} {'-':<8} {'-':<10} {'-':<8} {'-':<10} {'-':<12} {avg_psnr:<8.2f} {avg_improvement:<12.2f}")
    print("=" * 100)


def test_on_real_image():
    
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images', 'image1.jpg')
    
    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        return
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return
    
    print(f"\nТестирование на реальном изображении: {os.path.basename(image_path)}")
    print(f"Размер: {image.shape}")
    
    noise_combinations = [
        {'name': 'Слабый', 'frequency': 12, 'amplitude': 20, 'threshold': 60},
        {'name': 'Средний', 'frequency': 8, 'amplitude': 40, 'threshold': 50},
        {'name': 'Сильный', 'frequency': 6, 'amplitude': 70, 'threshold': 40},
        {'name': 'Очень сильный', 'frequency': 5, 'amplitude': 100, 'threshold': 30}
    ]
    
    fig, axes = plt.subplots(len(noise_combinations), 4, figsize=(16, 4 * len(noise_combinations)))
    
    for i, combo in enumerate(noise_combinations):
        noisy = FourierTransform.add_periodic_noise(image, combo['frequency'], combo['amplitude'])
        denoised = FourierTransform.fft_denoise(noisy, combo['threshold'], combo['frequency'])
        
        mse_noisy = np.mean((image.astype(np.float32) - noisy.astype(np.float32)) ** 2)
        mse_denoised = np.mean((image.astype(np.float32) - denoised.astype(np.float32)) ** 2)
        psnr = FourierTransform.calculate_psnr(image, denoised)
        
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title('Оригинал', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(noisy, cmap='gray')
        axes[i, 1].set_title(f"{combo['name']} шум\nMSE: {mse_noisy:.2f}", fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(denoised, cmap='gray')
        axes[i, 2].set_title(f"После FFT\nPSNR: {psnr:.2f} dB", fontsize=10)
        axes[i, 2].axis('off')
        
        error = np.abs(image.astype(np.float32) - denoised.astype(np.float32))
        axes[i, 3].imshow(error, cmap='hot')
        axes[i, 3].set_title(f"Ошибка\nMSE: {mse_denoised:.2f}", fontsize=10)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_image_demonstration.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Результаты на реальном изображении сохранены: {output_path}")


if __name__ == "__main__":
    print("ДЕМОНСТРАЦИЯ FFT ШУМОПОДАВЛЕНИЯ С РАЗЛИЧНЫМИ ПАРАМЕТРАМИ")
    print("=" * 80)
    
    create_demonstration()
    test_on_real_image()
    
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)
