"""
Конфигурация для Lab 4: Cell Counting with ResNet
"""

CONFIG = {
    # === Данные ===
    'image_size': (224, 224),           # Размер изображений (width, height)
    'max_cell_size': 35,                 # Максимальный размер клетки
    'min_cell_coverage': 0.95,           # Минимальное покрытие клетки
    'p_cell_artificial': 0.5,            # Вероятность искусственной клетки
    'sigma': 1.0,                        # Sigma для гауссова target
    
    # Конфигурации плотности
    'density_configs': {
        'low': {
            'min_cells': 20,
            'max_cells': 40,
            'description': 'Низкая плотность'
        }
    },
    
    # Распределение изображений по плотностям
    'density_distribution': {
        'low': 1.0          # 100% = 4000 train, 1000 val
    },
    
    # Общий диапазон для num_classes (20-40 = 21 класс)
    'num_classes': 21,
    
    'train_size': 4000,                  # Размер train датасета (уменьшено для скорости)
    'val_size': 1000,                    # Размер validation датасета (уменьшено для скорости)
    
    # === Модель ===
    'model_name': 'resnet18',            # Архитектура модели (resnet18 для среднего обучения)
    'pretrained': True,                  # Использовать предобученные веса
    'dropout': 0.3,                      # Dropout rate
    
    # === Обучение ===
    'batch_size': 32,                    # Размер батча
    'gradient_accumulation': 1,          # Шагов accumulation (эффективный batch = 32)
    'num_epochs': 20,                    # Максимум эпох
    'learning_rate': 0.001,              # Learning rate
    'weight_decay': 1e-4,                # Weight decay (L2 регуляризация)
    'label_smoothing': 0.1,              # Label smoothing для BCELoss
    'early_stopping_patience': 5,        # Patience для early stopping
    
    # === Пути ===
    'patches_dir': '../update_generator/patches',
    'test_images_dir': '../update_generator/test_images',
    'data_dir': './data',
    'models_dir': './models',
    'results_dir': './results',
    
    # === Device ===
    'device': 'cuda',                    # 'cuda' или 'cpu'
    'num_workers': 4,                    # Количество worker'ов для DataLoader
}
