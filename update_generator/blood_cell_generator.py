import os
import cv2
import numpy as np
from glob import glob
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class BloodCellGenerator:
    def __init__(self, cell_dir, bg_dir, size=(256, 256), min_cells=40, max_cells=100, min_cell_coverage=0.95, max_cell_size=35, p_cell_artificial=0.5):
        self.cell_files = glob(os.path.join(cell_dir, '*.png'))
        self.bg_files = glob(os.path.join(bg_dir, '*.png'))
        self.size = size
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.min_cell_coverage = min_cell_coverage
        self.max_cell_size = max_cell_size
        self.p_cell_artificial = p_cell_artificial

    def generate_artificial_cell(self, x, y):
        """Генерация реалистичной искусственной клетки."""
        rad = np.random.randint(8, 15)
        
        # Создаем клетку с градиентом
        cell = np.zeros((rad*2, rad*2, 3), dtype=np.uint8)
        
        # Создаем градиент от центра к краям
        center = (rad, rad)
        for i in range(rad*2):
            for j in range(rad*2):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if dist <= rad:
                    # Градиент от центра к краям
                    intensity = 1.0 - (dist / rad) * 0.3
                    color = (random.randint(130, 170), 30, random.randint(110, 150))
                    cell[i, j] = [int(c * intensity) for c in color]
        
        # Добавляем ядро клетки (более темное в центре)
        nucleus_rad = rad // 3
        cv2.circle(cell, (rad, rad), nucleus_rad, (80, 20, 80), -1)
        
        # Добавляем мембрану
        cv2.circle(cell, (rad, rad), rad, (100, 20, 90), 2)
        
        return cell
    
    def create_background(self):
        """Создание реалистичного фона."""
        bg = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        
        # Используем патчи из датасета
        if self.bg_files:
            # Выбираем случайный патч фона
            bg_patch = cv2.imread(random.choice(self.bg_files))
            bg_patch = cv2.resize(bg_patch, self.size)
            bg = bg_patch
        else:
            # Создаем однотонный фон с вариативностью
            base_color = random.randint(200, 235)
            bg = np.full((self.size[1], self.size[0], 3), base_color, dtype=np.uint8)
            
            # Добавляем шум для реалистичности
            noise = np.random.normal(0, 5, bg.shape).astype(np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Применяем легкое размытие для реалистичности
        bg = cv2.GaussianBlur(bg, (5, 5), 0)
        
        return bg

    @staticmethod
    def overlay_image_center(background_img, foreground_img, center_x, center_y):
        h2, w2 = background_img.shape[:2]
        h1, w1 = foreground_img.shape[:2]

        x1_start = center_x - w1 // 2
        y1_start = center_y - h1 // 2

        x1_end = x1_start + w1
        y1_end = y1_start + h1

        x2_start = max(0, x1_start)
        y2_start = max(0, y1_start)
        x2_end = min(w2, x1_end)
        y2_end = min(h2, y1_end)

        x1_crop_start = max(0, -x1_start)
        y1_crop_start = max(0, -y1_start)
        x1_crop_end = x1_crop_start + (x2_end - x2_start)
        y1_crop_end = y1_crop_start + (y2_end - y2_start)

        result = background_img.copy()

        if x2_end > x2_start and y2_end > y2_start:
            foreground_img_region = foreground_img[y1_crop_start:y1_crop_end, x1_crop_start:x1_crop_end]
            result[y2_start:y2_end, x2_start:x2_end] = foreground_img_region

        return result

    @staticmethod
    def add_image_with_mask(a, b, mask):
        a1 = cv2.bitwise_and(a, a, mask=cv2.bitwise_not(mask))
        b1 = cv2.bitwise_and(b, b, mask=mask)
        return cv2.add(a1, b1)

    def generate_image(self):
        img = self.create_background()
        num_cells = random.randint(self.min_cells, self.max_cells)

        cells_mask = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        cells_img = np.zeros_like(img)
        temp_img = np.zeros_like(cells_img)
        temp_mask_history = []

        actual_cells_count = 0

        for i in range(num_cells):
            for _max_attempts in range(999):
                temp_mask = np.zeros_like(cells_mask)
                temp_img.fill(0)

                x = np.random.randint(cells_img.shape[1])
                y = np.random.randint(cells_img.shape[0])

                use_artificial_cell = bool(np.random.choice(2, p=[1-self.p_cell_artificial, self.p_cell_artificial]))

                if use_artificial_cell:
                    # Генерируем реалистичную искусственную клетку
                    cell = self.generate_artificial_cell(x, y)
                    h, w = cell.shape[:2]
                    
                    # Создаем маску для клетки
                    cell_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(cell_mask, (w//2, h//2), w//2, 255, -1)
                    
                    temp_mask = self.overlay_image_center(temp_mask, cell_mask, x, y)
                    temp_img = self.overlay_image_center(temp_img, cell, x, y)
                else:
                    if not self.cell_files:
                        continue
                    cell = cv2.imread(random.choice(self.cell_files))
                    if cell is None:
                        continue

                    # Уменьшаем клетку, если она слишком большая
                    h, w = cell.shape[:2]
                    max_dim = max(h, w)
                    if max_dim > self.max_cell_size:
                        scale = self.max_cell_size / max_dim
                        cell = cv2.resize(cell, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                    _, cell_mask = cv2.threshold(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY), 3, 255, cv2.THRESH_BINARY)
                    temp_mask = self.overlay_image_center(temp_mask, cell_mask, x, y)
                    temp_img = self.overlay_image_center(temp_img, cell, x, y)

                temp_mask_without = cv2.bitwise_and(temp_mask, cv2.bitwise_not(cells_mask))

                if np.count_nonzero(temp_mask) == 0:
                    continue

                coverage = np.count_nonzero(temp_mask_without) / np.count_nonzero(temp_mask)
                can_insert = coverage >= self.min_cell_coverage

                if can_insert:
                    for prev_temp_mask in temp_mask_history:
                        if np.count_nonzero(cv2.bitwise_and(temp_mask, prev_temp_mask)) > 0:
                            temp_mask_without_ind = cv2.bitwise_and(temp_mask, cv2.bitwise_not(prev_temp_mask))
                            prev_temp_mask_without = cv2.bitwise_and(prev_temp_mask, cv2.bitwise_not(temp_mask))

                            coverage = np.count_nonzero(temp_mask_without_ind) / np.count_nonzero(temp_mask)
                            prev_coverage = np.count_nonzero(prev_temp_mask_without) / np.count_nonzero(prev_temp_mask)

                            if coverage < self.min_cell_coverage or prev_coverage < self.min_cell_coverage:
                                can_insert = False
                                break

                if can_insert:
                    cells_img = self.add_image_with_mask(cells_img, temp_img, temp_mask_without)
                    cells_mask = self.add_image_with_mask(cells_mask, temp_mask, temp_mask_without)
                    temp_mask_history.append(temp_mask)
                    actual_cells_count += 1
                    break
            else:
                break

        # Накладываем клетки на фон
        img = self.add_image_with_mask(img, cells_img, cells_mask)

        return img, actual_cells_count


def add_noise(img, noise_type='gaussian'):
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 20, img.shape).astype(np.float32)
    else:
        noise = np.random.uniform(-25, 25, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_filters(img, method='median', param=5):
    if method == 'median':
        return cv2.medianBlur(img, param)
    elif method == 'gaussian':
        return cv2.GaussianBlur(img, (param, param), 0)
    elif method == 'bilateral':
        return cv2.bilateralFilter(img, 9, param*10, param*10)
    elif method == 'nlm':
        return cv2.fastNlMeansDenoisingColored(img, None, param, param, 7, 21)
    return img


if __name__ == '__main__':
    patches_dir = os.path.join(SCRIPT_DIR, 'patches')
    gen = BloodCellGenerator(
        os.path.join(patches_dir, 'cells'),
        os.path.join(patches_dir, 'background')
    )

    clean_img, count = gen.generate_image()

    output_path = os.path.join(SCRIPT_DIR, 'generated_sample.png')
    cv2.imwrite(output_path, clean_img)
    print(f'Сгенерировано клеток: {count}')
    print(f'Сохранено: {output_path}')
