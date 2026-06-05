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

    def create_background(self):
        bg = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        p_size = 64
        for i in range(0, self.size[1], p_size):
            for j in range(0, self.size[0], p_size):
                if random.random() > 0.4 and self.bg_files:
                    p = cv2.imread(random.choice(self.bg_files))
                    p = cv2.resize(p, (p_size, p_size))
                else:
                    color = random.randint(215, 235)
                    p = np.full((p_size, p_size, 3), color, dtype=np.uint8)
                
                # Вычисляем реальные размеры для вставки (учитываем края)
                h_insert = min(p_size, self.size[1] - i)
                w_insert = min(p_size, self.size[0] - j)
                bg[i:i+h_insert, j:j+w_insert] = p[:h_insert, :w_insert]
        
        return cv2.GaussianBlur(bg, (7, 7), 0)

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
                    rad = np.random.randint(15, 25)
                    cv2.circle(temp_mask, center=(x, y), radius=rad, color=255, thickness=-1)
                    color = (random.randint(130, 170), 30, random.randint(110, 150))
                    cv2.circle(temp_img, center=(x, y), radius=rad, color=color, thickness=-1)
                    stroke_color = tuple(255 - c for c in color)
                    cv2.circle(temp_img, center=(x, y), radius=rad, color=stroke_color, thickness=3)
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
