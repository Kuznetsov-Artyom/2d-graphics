"""
Модуль модели: CellCounter (ResNet50) и функции потерь
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import CONFIG


class SmoothBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss с label smoothing для гауссова распределения.
    
    Сглаживает target вектор для предотвращения переобучения.
    """
    
    def __init__(self, smoothing=None):
        """
        Args:
            smoothing: коэффициент сглаживания (по умолчанию из CONFIG)
        """
        super(SmoothBCELoss, self).__init__()
        
        if smoothing is None:
            smoothing = CONFIG['label_smoothing']
        
        self.smoothing = smoothing
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: предсказания модели [batch_size, num_classes]
            targets: target векторы [batch_size, num_classes]
        
        Returns:
            loss: scalar
        """
        # Label smoothing
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # BCE with logits (стабильнее чем BCELoss + Sigmoid)
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        
        return loss


class CellCounter(nn.Module):
    """
    Модель для подсчета клеток на основе ResNet.
    
    Архитектура:
    - ResNet18/ResNet50 (pretrained ImageNet)
    - BatchNorm + Dropout
    - FC: fc_in_features -> 512 -> num_classes
    """
    
    def __init__(self, num_classes=None, pretrained=None, dropout=None, model_name=None):
        """
        Args:
            num_classes: количество классов (по умолчанию из CONFIG)
            pretrained: использовать предобученные веса (по умолчанию из CONFIG)
            dropout: dropout rate (по умолчанию из CONFIG)
            model_name: название модели ('resnet18' или 'resnet50')
        """
        super(CellCounter, self).__init__()
        
        if num_classes is None:
            num_classes = CONFIG['num_classes']
        if pretrained is None:
            pretrained = CONFIG['pretrained']
        if dropout is None:
            dropout = CONFIG['dropout']
        if model_name is None:
            model_name = CONFIG['model_name']
        
        # Загрузка предобученной модели
        if model_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
            fc_in_features = 512
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
            fc_in_features = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'resnet18' or 'resnet50'")
        
        # Замена FC слоя
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(fc_in_features),
            nn.Dropout(dropout),
            nn.Linear(fc_in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),  # Меньший dropout
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: input tensor [batch_size, 3, 512, 512]
        
        Returns:
            logits: output tensor [batch_size, num_classes]
        """
        return self.resnet(x)


class SmoothBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss с label smoothing.
    
    Сглаживает target вектор для предотвращения переобучения.
    """
    
    def __init__(self, smoothing=None):
        """
        Args:
            smoothing: коэффициент сглаживания (по умолчанию из CONFIG)
        """
        super(SmoothBCELoss, self).__init__()
        
        if smoothing is None:
            smoothing = CONFIG['label_smoothing']
        
        self.smoothing = smoothing
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: предсказания модели [batch_size, num_classes]
            targets: target векторы [batch_size, num_classes]
        
        Returns:
            loss: scalar
        """
        # Label smoothing
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # BCE with logits (стабильнее чем BCELoss + Sigmoid)
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        
        return loss


def get_model(num_classes=None, model_name=None):
    """
    Создает и возвращает модель.
    
    Args:
        num_classes: количество классов (по умолчанию из CONFIG)
        model_name: название модели (по умолчанию из CONFIG)
    
    Returns:
        model: CellCounter instance
    """
    if num_classes is None:
        num_classes = CONFIG['num_classes']
    if model_name is None:
        model_name = CONFIG['model_name']
    
    model = CellCounter(num_classes=num_classes, model_name=model_name)
    
    return model


def load_model(model_path, num_classes=None, model_name=None):
    """
    Загружает модель из файла.
    
    Args:
        model_path: путь к файлу модели (.pth)
        num_classes: количество классов (по умолчанию из CONFIG)
        model_name: название модели (по умолчанию из CONFIG)
    
    Returns:
        model: CellCounter instance с загруженными весами
    """
    if num_classes is None:
        num_classes = CONFIG['num_classes']
    if model_name is None:
        model_name = CONFIG['model_name']
    
    model = CellCounter(num_classes=num_classes, pretrained=False, model_name=model_name)
    
    # Загрузка весов
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model
