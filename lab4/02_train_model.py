#!/usr/bin/env python3
"""
Скрипт 2: Обучение модели для подсчета клеток крови.

Использует:
- data/train/ (10,000 изображений)
- data/val/ (2,000 изображений)

Создает:
- models/best_model.pth (лучшая модель)
- results/training_curves.png (графики обучения)
- results/training.log (лог обучения)

Время выполнения: ~2-3 часа (зависит от модели и количества эпох)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import CONFIG
from dataset import get_data_loaders
from model import get_model
from utils import setup_logging, plot_training_curves


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accumulation_steps):
    """Обучение одну эпоху с gradient accumulation."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    
    for i, (images, targets, counts) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass с mixed precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # Для accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Optimizer step каждые accumulation_steps батчей
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Метрики
        total_loss += loss.item() * accumulation_steps
        
        pred_classes = torch.argmax(outputs, dim=1)
        true_classes = targets.squeeze().long()
        # Мягкая метрика (как в g2d-m1-labs): 1.0 - |pred - true| / num_classes
        is_correct = 1.0 - (torch.abs(pred_classes - true_classes)).float() / float(outputs.shape[1])
        correct += is_correct.sum().item()
        total += targets.size(0)
        
        # Обновление progress bar
        pbar.set_postfix({
            'loss': f'{total_loss / (i + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Валидация модели."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets, counts in tqdm(loader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            pred_classes = torch.argmax(outputs, dim=1)
            true_classes = targets.squeeze().long()
            # Мягкая метрика (как в g2d-m1-labs): 1.0 - |pred - true| / num_classes
            is_correct = 1.0 - (torch.abs(pred_classes - true_classes)).float() / float(outputs.shape[1])
            correct += is_correct.sum().item()
            total += targets.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    """Основная функция обучения."""
    
    # Настройка логирования
    logger = setup_logging()
    logger.info("="*70)
    logger.info(f"ОБУЧЕНИЕ МОДЕЛИ {CONFIG['model_name'].upper()}")
    logger.info("="*70)
    
    # Проверка GPU
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Загрузка данных
    logger.info("\nЗагрузка данных...")
    train_loader, val_loader = get_data_loaders()
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Создание модели
    logger.info("\nСоздание модели...")
    model = get_model().to(device)
    logger.info(f"Модель: {CONFIG['model_name']}")
    logger.info(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss, optimizer, scheduler (как в g2d-m1-labs + scheduler)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = GradScaler()
    
    # Training loop
    logger.info("\n" + "="*70)
    logger.info("НАЧАЛО ОБУЧЕНИЯ")
    logger.info("="*70)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-" * 70)
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            CONFIG['gradient_accumulation']
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Логирование
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        logger.info(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Сохранение метрик
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Сохранение лучшей модели
            model_path = os.path.join(CONFIG['models_dir'], 'best_model.pth')
            os.makedirs(CONFIG['models_dir'], exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"✓ Лучшая модель сохранена: {model_path}")
        else:
            patience_counter += 1
            logger.info(f"Early stopping: {patience_counter}/{CONFIG['early_stopping_patience']}")
            
            if patience_counter >= CONFIG['early_stopping_patience']:
                logger.info("\n⚠ Early stopping triggered!")
                break
    
    # Сохранение графиков
    logger.info("\n" + "="*70)
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    logger.info("="*70)
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    logger.info(f"\nЛучшая val loss: {best_val_loss:.4f}")
    logger.info(f"Финальная train acc: {train_accs[-1]:.2f}%")
    logger.info(f"Финальная val acc: {val_accs[-1]:.2f}%")
    
    logger.info("\n✓ Готово к валидации!")
    logger.info("  Следующий шаг: python 03_validate_model.py\n")


if __name__ == "__main__":
    main()
