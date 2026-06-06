#!/usr/bin/env python3
"""
Скрипт для построения кривых обучения с точками тестовой выборки.

Использует:
- results/training.log (лог обучения)
- results/test_metrics.json (метрики тестовой выборки)

Создает:
- results/training_curves_with_test.png (графики с тестовыми точками)

Время выполнения: ~10 секунд
"""

import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from config import CONFIG


def parse_training_log(log_path):
    """
    Извлекает данные обучения из лога.
    
    Args:
        log_path: путь к training.log
    
    Returns:
        dict с данными по эпохам
    """
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    learning_rates = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Разбиваем на блоки по эпохам
    epoch_blocks = re.split(r'Epoch \d+/\d+', content)[1:]  # Пропускаем первый пустой блок
    
    for i, block in enumerate(epoch_blocks):
        # Извлечение метрик
        train_loss_match = re.search(r'Train Loss: ([\d.]+)', block)
        train_acc_match = re.search(r'Train Acc: ([\d.]+)%', block)
        val_loss_match = re.search(r'Val Loss: ([\d.]+)', block)
        val_acc_match = re.search(r'Val Acc: ([\d.]+)%', block)
        lr_match = re.search(r'LR: ([\d.e+-]+)', block)
        
        if all([train_loss_match, train_acc_match, val_loss_match, val_acc_match, lr_match]):
            epochs.append(i + 1)
            train_losses.append(float(train_loss_match.group(1)))
            train_accs.append(float(train_acc_match.group(1)))
            val_losses.append(float(val_loss_match.group(1)))
            val_accs.append(float(val_acc_match.group(1)))
            learning_rates.append(float(lr_match.group(1)))
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'learning_rates': learning_rates
    }


def plot_learning_curves_with_test(training_data, test_metrics, save_path=None):
    """
    Строит графики обучения с точками тестовой выборки.
    
    Args:
        training_data: dict с данными обучения
        test_metrics: dict с тестовыми метриками
        save_path: путь для сохранения
    """
    if save_path is None:
        save_path = os.path.join(CONFIG['results_dir'], 'training_curves_with_test.png')
    
    epochs = training_data['epochs']
    train_losses = training_data['train_losses']
    train_accs = training_data['train_accs']
    val_losses = training_data['val_losses']
    val_accs = training_data['val_accs']
    learning_rates = training_data['learning_rates']
    
    # Тестовые метрики
    test_acc = test_metrics['top1_accuracy'] * 100
    test_top3_acc = test_metrics['top3_accuracy'] * 100
    test_mae = test_metrics['mae']
    
    # Создание фигуры с 4 графиками
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # График 1: Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2, color='blue')
    ax1.plot(epochs, val_losses, label='Val Loss', linewidth=2, color='orange')
    
    # Тестовая точка (используем последнюю val loss как приближение)
    ax1.scatter([epochs[-1]], [val_losses[-1]], 
                color='red', s=200, marker='*', 
                label='Test Loss (approx)', zorder=5, edgecolors='black', linewidth=1)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # График 2: Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_accs, label='Train Accuracy', linewidth=2, color='blue')
    ax2.plot(epochs, val_accs, label='Val Accuracy', linewidth=2, color='orange')
    
    # Тестовые точки
    ax2.scatter([epochs[-1]], [test_acc], 
                color='red', s=200, marker='*', 
                label=f'Test Top-1 Acc ({test_acc:.2f}%)', zorder=5, edgecolors='black', linewidth=1)
    ax2.scatter([epochs[-1]], [test_top3_acc], 
                color='green', s=200, marker='^', 
                label=f'Test Top-3 Acc ({test_top3_acc:.2f}%)', zorder=5, edgecolors='black', linewidth=1)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # График 3: Learning Rate
    ax3 = axes[1, 0]
    ax3.plot(epochs, learning_rates, label='Learning Rate', linewidth=2, color='purple')
    
    # Вертикальная линия на эпохе тестирования
    ax3.axvline(x=epochs[-1], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Test Epoch')
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # График 4: MAE (только тестовая точка)
    ax4 = axes[1, 1]
    
    # Создаем пустой график с тестовой точкой
    ax4.scatter([epochs[-1]], [test_mae], 
                color='red', s=300, marker='*', 
                label=f'Test MAE ({test_mae:.2f} cells)', zorder=5, edgecolors='black', linewidth=1.5)
    
    # Добавляем горизонтальную линию для ориентира
    ax4.axhline(y=test_mae, color='red', linestyle='--', linewidth=1, alpha=0.3)
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('MAE (cells)', fontsize=12)
    ax4.set_title('Mean Absolute Error (Test Only)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max(epochs) + 5)
    
    # Общий заголовок
    fig.suptitle('Learning Curves with Test Set Results', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Графики сохранены: {save_path}")


def main():
    """Основная функция."""
    
    print("\n" + "="*70)
    print("ПОСТРОЕНИЕ КРИВЫХ ОБУЧЕНИЯ С ТЕСТОВЫМИ ТОЧКАМИ")
    print("="*70)
    
    # Пути к файлам
    log_path = os.path.join(CONFIG['results_dir'], 'training.log')
    test_metrics_path = os.path.join(CONFIG['results_dir'], 'test_metrics.json')
    
    # Проверка наличия файлов
    if not os.path.exists(log_path):
        print(f"\n❌ ОШИБКА: Training log не найден: {log_path}")
        return
    
    if not os.path.exists(test_metrics_path):
        print(f"\n❌ ОШИБКА: Test metrics не найдены: {test_metrics_path}")
        return
    
    # Парсинг training log
    print(f"\nЗагрузка training log: {log_path}")
    training_data = parse_training_log(log_path)
    
    print(f"✓ Загружено {len(training_data['epochs'])} эпох")
    print(f"  Диапазон эпох: {min(training_data['epochs'])}-{max(training_data['epochs'])}")
    print(f"  Финальная Train Acc: {training_data['train_accs'][-1]:.2f}%")
    print(f"  Финальная Val Acc: {training_data['val_accs'][-1]:.2f}%")
    
    # Загрузка тестовых метрик
    print(f"\nЗагрузка тестовых метрик: {test_metrics_path}")
    with open(test_metrics_path, 'r', encoding='utf-8') as f:
        test_metrics = json.load(f)
    
    print(f"✓ Test Top-1 Accuracy: {test_metrics['top1_accuracy']*100:.2f}%")
    print(f"✓ Test Top-3 Accuracy: {test_metrics['top3_accuracy']*100:.2f}%")
    print(f"✓ Test MAE: {test_metrics['mae']:.2f} клеток")
    
    # Построение графиков
    print(f"\nПостроение графиков...")
    plot_learning_curves_with_test(training_data, test_metrics)
    
    print("\n" + "="*70)
    print("ГОТОВО!")
    print("="*70)
    print(f"Результат: {CONFIG['results_dir']}/training_curves_with_test.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
