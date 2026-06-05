"""
Модуль для работы с данными: Dataset, augmentation, загрузка
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import CONFIG


def create_gaussian_target(cell_count, min_cells=20, max_cells=40, sigma=None):
    """
    Создает вероятностной вектор с гауссовым распределением.
    
    Args:
        cell_count: истинное количество клеток (20-40)
        min_cells: минимальное количество клеток (20)
        max_cells: максимальное количество клеток (40)
        sigma: стандартное отклонение гауссианы (по умолчанию из CONFIG)
    
    Returns:
        target: numpy array размера (num_classes,) = (21,)
    """
    if sigma is None:
        sigma = CONFIG['sigma']
    
    num_classes = max_cells - min_cells + 1  # 21 класс
    target = np.zeros(num_classes, dtype=np.float32)
    x = np.arange(num_classes, dtype=np.float32)
    class_idx = cell_count - min_cells
    
    # Гауссово распределение
    target = np.exp(-((x - class_idx) ** 2) / (2 * sigma ** 2))
    
    return target


def get_train_transform():
    """Возвращает трансформации для обучения (с augmentation)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transform():
    """Возвращает трансформации для валидации (без augmentation)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


class BloodCellDataset(Dataset):
    """
    PyTorch Dataset для изображений клеток крови.
    
    Загружает изображения из папки с JSON метаданными.
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: путь к папке с изображениями (data/train/ или data/val/)
            transform: torchvision transforms для аугментации
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Загрузка списка файлов
        self.samples = []
        
        # Ищем только JSON файлы изображений (image_*.json)
        json_files = sorted([f for f in os.listdir(data_dir) if f.startswith('image_') and f.endswith('.json')])
        
        for json_file in json_files:
            json_path = os.path.join(data_dir, json_file)
            
            # Загрузка метаданных
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Путь к изображению
            img_filename = metadata['filename']
            img_path = os.path.join(data_dir, img_filename)
            
            # Проверка существования изображения
            if os.path.exists(img_path):
                self.samples.append({
                    'img_path': img_path,
                    'cell_count': metadata['cell_count'],
                    'image_id': metadata['image_id']
                })
        
        print(f"Загружено {len(self.samples)} образцов из {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Загрузка изображения (BGR)
        img = cv2.imread(sample['img_path'])
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Применение трансформаций
        if self.transform:
            img = self.transform(img)
        
        # Создание target вектора
        cell_count = sample['cell_count']
        target = create_gaussian_target(cell_count)
        target = torch.FloatTensor(target)
        
        return img, target, cell_count


def get_data_loaders(batch_size=None, num_workers=None):
    """
    Создает DataLoader'ы для train и validation.
    
    Args:
        batch_size: размер батча (по умолчанию из CONFIG)
        num_workers: количество worker'ов (по умолчанию из CONFIG)
    
    Returns:
        train_loader, val_loader
    """
    if batch_size is None:
        batch_size = CONFIG['batch_size']
    if num_workers is None:
        num_workers = CONFIG['num_workers']
    
    # Пути к данным
    train_dir = os.path.join(CONFIG['data_dir'], 'train')
    val_dir = os.path.join(CONFIG['data_dir'], 'val')
    
    # Проверка наличия данных
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train data not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation data not found: {val_dir}")
    
    # Создание датасетов
    train_dataset = BloodCellDataset(train_dir, transform=get_train_transform())
    val_dataset = BloodCellDataset(val_dir, transform=get_val_transform())
    
    # Создание DataLoader'ов
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
