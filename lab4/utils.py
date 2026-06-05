"""
Вспомогательные функции: визуализация, метрики, логирование
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from config import CONFIG


def setup_logging(log_file=None):
    """
    Настраивает логирование.
    
    Args:
        log_file: путь к файлу лога (по умолчанию results/training.log)
    """
    import logging
    
    if log_file is None:
        log_file = os.path.join(CONFIG['results_dir'], 'training.log')
    
    # Создание папки если нужно
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Настройка логгера
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Строит графики loss и accuracy.
    
    Args:
        train_losses: список train loss по эпохам
        val_losses: список val loss по эпохам
        train_accs: список train accuracy по эпохам
        val_accs: список val accuracy по эпохам
        save_path: путь для сохранения (по умолчанию results/training_curves.png)
    """
    if save_path is None:
        save_path = os.path.join(CONFIG['results_dir'], 'training_curves.png')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Графики сохранены: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Строит confusion matrix.
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        class_names: названия классов (по умолчанию 20-40)
        save_path: путь для сохранения (по умолчанию results/confusion_matrix.png)
    """
    if save_path is None:
        save_path = os.path.join(CONFIG['results_dir'], 'confusion_matrix.png')
    
    if class_names is None:
        class_names = [str(i) for i in range(20, 41)]  # 20-40 клеток
    
    # Вычисление confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Визуализация
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Сохранение
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix сохранена: {save_path}")


def plot_predictions(model, test_images, num_examples=5, save_path=None):
    """
    Визуализирует примеры предсказаний.
    
    Args:
        model: обученная модель
        test_images: список (image, true_count)
        num_examples: количество примеров
        save_path: путь для сохранения (по умолчанию results/predictions_examples.png)
    """
    if save_path is None:
        save_path = os.path.join(CONFIG['results_dir'], 'predictions_examples.png')
    
    model.eval()
    
    fig, axes = plt.subplots(1, num_examples, figsize=(20, 4))
    
    for i, (img, true_count) in enumerate(test_images[:num_examples]):
        # Предсказание
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(CONFIG['device'])
            output = model(img_tensor)
            pred_count = 20 + torch.argmax(output, dim=1).item()
        
        # Визуализация
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f'True: {true_count}\nPred: {pred_count}', fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Примеры предсказаний сохранены: {save_path}")


def compute_metrics(outputs, targets, counts):
    """
    Вычисляет метрики: top-1 accuracy, top-3 accuracy, MAE.
    
    Args:
        outputs: предсказания модели [batch_size, num_classes]
        targets: target векторы [batch_size, num_classes]
        counts: истинные количества клеток [batch_size]
    
    Returns:
        metrics: dict с метриками
    """
    # Top-1 accuracy
    pred_classes = torch.argmax(outputs, dim=1)
    true_classes = torch.argmax(targets, dim=1)
    top1_acc = (pred_classes == true_classes).float().mean().item()
    
    # Top-3 accuracy
    top3_preds = torch.topk(outputs, k=3, dim=1).indices
    top3_correct = 0
    for i in range(len(true_classes)):
        if true_classes[i] in top3_preds[i]:
            top3_correct += 1
    top3_acc = top3_correct / len(true_classes)
    
    # MAE
    pred_counts = 20 + pred_classes.cpu().numpy()
    true_counts = counts.cpu().numpy()
    mae = np.mean(np.abs(pred_counts - true_counts))
    
    return {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'mae': mae
    }


def save_metrics(metrics, save_path=None):
    """
    Сохраняет метрики в JSON файл.
    
    Args:
        metrics: dict с метриками
        save_path: путь для сохранения (по умолчанию results/metrics.json)
    """
    if save_path is None:
        save_path = os.path.join(CONFIG['results_dir'], 'metrics.json')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Метрики сохранены: {save_path}")
