import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import fftpack
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class FourierTransform:
    
    @staticmethod
    def add_periodic_noise(image, frequency=10, amplitude=30):
        rows, cols = image.shape
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)
        
        noise = amplitude * np.sin(2 * np.pi * frequency * X / cols)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    @staticmethod
    def fft_denoise(image, threshold=50):
        f = fftpack.fft2(image)
        fshift = fftpack.fftshift(f)
        
        magnitude = np.abs(fshift)
        mask = magnitude > threshold
        fshift_filtered = fshift * mask
        
        f_ishift = fftpack.ifftshift(fshift_filtered)
        img_back = fftpack.ifft2(f_ishift)
        img_back = np.abs(img_back).astype(np.uint8)
        
        return img_back
    
    @staticmethod
    def calculate_psnr(original, denoised):
        mse = np.mean((original.astype(np.float32) - denoised.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    @staticmethod
    def calculate_mse(original, denoised):
        return np.mean((original.astype(np.float32) - denoised.astype(np.float32)) ** 2)


class HaarWavelet:
    
    @staticmethod
    def haar_transform_1d(signal):
        n = len(signal)
        if n == 1:
            return signal
        
        approximation = (signal[0::2] + signal[1::2]) / np.sqrt(2)
        detail = (signal[0::2] - signal[1::2]) / np.sqrt(2)
        
        return np.concatenate([HaarWavelet.haar_transform_1d(approximation), detail])
    
    @staticmethod
    def inverse_haar_transform_1d(coeffs):
        n = len(coeffs)
        if n == 1:
            return coeffs
        
        current = coeffs.copy().astype(np.float64)
        
        levels = int(np.log2(n))
        
        for level in range(levels):
            current_length = 2 ** (levels - level)
            half = current_length // 2
            
            approximation = current[:half]
            detail = current[half:current_length]
            
            reconstructed = np.zeros(current_length, dtype=np.float64)
            reconstructed[0::2] = (approximation + detail) / np.sqrt(2)
            reconstructed[1::2] = (approximation - detail) / np.sqrt(2)
            
            current[:current_length] = reconstructed
        
        return current[:n]
    
    @staticmethod
    def haar_transform_2d(image):
        rows, cols = image.shape
        
        transformed_rows = np.zeros_like(image, dtype=np.float64)
        for i in range(rows):
            transformed_rows[i, :] = HaarWavelet.haar_transform_1d(image[i, :].astype(np.float64))
        
        transformed = np.zeros_like(transformed_rows, dtype=np.float64)
        for j in range(cols):
            transformed[:, j] = HaarWavelet.haar_transform_1d(transformed_rows[:, j].astype(np.float64))
        
        return transformed
    
    @staticmethod
    def inverse_haar_transform_2d(coeffs):
        rows, cols = coeffs.shape
        
        reconstructed_cols = np.zeros_like(coeffs, dtype=np.float64)
        for j in range(cols):
            reconstructed_cols[:, j] = HaarWavelet.inverse_haar_transform_1d(coeffs[:, j].astype(np.float64))
        
        reconstructed = np.zeros_like(reconstructed_cols, dtype=np.float64)
        for i in range(rows):
            reconstructed[i, :] = HaarWavelet.inverse_haar_transform_1d(reconstructed_cols[i, :].astype(np.float64))
        
        return reconstructed
    
    @staticmethod
    def remove_low_high_frequency(coeffs, threshold=5.0):
        rows, cols = coeffs.shape
        filtered = coeffs.copy()
        
        for i in range(rows):
            for j in range(cols):
                if abs(filtered[i, j]) < threshold:
                    filtered[i, j] = 0
        
        return filtered


class RLE:
    
    @staticmethod
    def encode(data):
        if len(data) == 0:
            return []
        
        encoded = []
        zero_count = 0
        
        for value in data:
            if abs(value) < 0.01:
                zero_count += 1
            else:
                if zero_count > 0:
                    encoded.append((zero_count, 0))
                    zero_count = 0
                encoded.append((0, int(round(value))))
        
        if zero_count > 0:
            encoded.append((zero_count, 0))
        
        return encoded
    
    @staticmethod
    def decode(encoded):
        decoded = []
        
        for zero_count, value in encoded:
            if value == 0:
                decoded.extend([0] * zero_count)
            else:
                decoded.extend([0] * zero_count)
                decoded.append(value)
        
        return np.array(decoded)
    
    @staticmethod
    def save_to_file(encoded, filename):
        with open(filename, 'w') as f:
            for zero_count, value in encoded:
                f.write(f"{zero_count} {value}\n")
    
    @staticmethod
    def load_from_file(filename):
        encoded = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    zero_count, value = int(parts[0]), int(parts[1])
                    encoded.append((zero_count, value))
        return encoded


class ImageProcessor:
    
    @staticmethod
    def save_to_text_file(image, filename):
        with open(filename, 'w') as f:
            for row in image:
                for pixel in row:
                    f.write(f"{pixel}\n")
    
    @staticmethod
    def load_from_text_file(filename, shape):
        with open(filename, 'r') as f:
            data = [int(line.strip()) for line in f]
        return np.array(data).reshape(shape)
    
    @staticmethod
    def calculate_compression_ratio(original_size, compressed_size):
        return original_size / compressed_size


def test_fourier_transform():
    print("=== Testing Fourier Transform ===")
    
    test_image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    
    noisy_image = FourierTransform.add_periodic_noise(test_image, frequency=10, amplitude=30)
    
    denoised_image = FourierTransform.fft_denoise(noisy_image, threshold=50)
    
    psnr = FourierTransform.calculate_psnr(test_image, denoised_image)
    mse = FourierTransform.calculate_mse(test_image, denoised_image)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"MSE: {mse:.2f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(noisy_image, cmap='gray')
    axes[1].set_title('Noisy')
    axes[2].imshow(denoised_image, cmap='gray')
    axes[2].set_title('Denoised')
    plt.savefig(os.path.join(SCRIPT_DIR, 'fourier_test.png'))
    plt.close()
    
    print("Fourier transform test completed!")
    return psnr > 20


def test_haar_wavelet():
    print("\n=== Testing Haar Wavelet ===")
    
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    transformed = HaarWavelet.haar_transform_2d(test_image.astype(np.float64))
    
    filtered = HaarWavelet.remove_low_high_frequency(transformed, threshold=20.0)
    
    reconstructed = HaarWavelet.inverse_haar_transform_2d(filtered)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    mse = np.mean((test_image.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    
    print(f"Reconstruction MSE: {mse:.2f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(transformed, cmap='gray')
    axes[1].set_title('Transformed')
    axes[2].imshow(reconstructed, cmap='gray')
    axes[2].set_title('Reconstructed')
    plt.savefig(os.path.join(SCRIPT_DIR, 'wavelet_test.png'))
    plt.close()
    
    print("Haar wavelet test completed!")
    return True


def test_rle():
    print("\n=== Testing RLE ===")
    
    test_data = np.array([0, 0, 0, 5, 0, 0, 10, 15, 0, 0, 0, 0, 20])
    
    encoded = RLE.encode(test_data)
    print(f"Encoded: {encoded}")
    
    decoded = RLE.decode(encoded)
    print(f"Decoded: {decoded}")
    
    assert np.array_equal(test_data, decoded), "RLE encoding/decoding failed!"
    
    print("RLE test completed!")
    return True


def test_full_compression_pipeline():
    print("\n=== Testing Full Compression Pipeline ===")
    
    test_image = np.random.randint(0, 50, (64, 64), dtype=np.uint8)
    
    ImageProcessor.save_to_text_file(test_image, os.path.join(SCRIPT_DIR, 'original.txt'))
    original_size = os.path.getsize(os.path.join(SCRIPT_DIR, 'original.txt'))
    print(f"Original file size: {original_size} bytes")
    
    transformed = HaarWavelet.haar_transform_2d(test_image.astype(np.float64))
    filtered = HaarWavelet.remove_low_high_frequency(transformed, threshold=20.0)
    
    flattened = filtered.flatten().astype(np.int32)
    encoded = RLE.encode(flattened)
    
    RLE.save_to_file(encoded, os.path.join(SCRIPT_DIR, 'compressed.txt'))
    compressed_size = os.path.getsize(os.path.join(SCRIPT_DIR, 'compressed.txt'))
    print(f"Compressed file size: {compressed_size} bytes")
    
    compression_ratio = ImageProcessor.calculate_compression_ratio(original_size, compressed_size)
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    loaded_encoded = RLE.load_from_file(os.path.join(SCRIPT_DIR, 'compressed.txt'))
    decoded = RLE.decode(loaded_encoded)
    
    reconstructed_coeffs = decoded.reshape(filtered.shape).astype(np.float64)
    reconstructed = HaarWavelet.inverse_haar_transform_2d(reconstructed_coeffs)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    mse = np.mean((test_image.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    print(f"Final reconstruction MSE: {mse:.2f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title('Reconstructed')
    plt.savefig(os.path.join(SCRIPT_DIR, 'compression_test.png'))
    plt.close()
    
    print("Full compression pipeline test completed!")
    return compression_ratio > 1.0


def main():
    print("Starting Lab 3: Fourier Transform and Wavelet Compression")
    
    results = []
    
    results.append(("Fourier Transform", test_fourier_transform()))
    results.append(("Haar Wavelet", test_haar_wavelet()))
    results.append(("RLE", test_rle()))
    results.append(("Full Pipeline", test_full_compression_pipeline()))
    
    print("\n=== Test Results ===")
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    main()
