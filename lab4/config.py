"""
Конфигурация для Lab 4: Cell Counting with ResNet
"""

CONFIG = {
    # === Данные ===
    'image_size': (224, 224),           # Размер изображений (width, height)
    'max_cell_size': 20,                 # Максимальный размер клетки (уменьшено для лучшего разделения)
    'min_cell_coverage': 0.7,            # Минимальное покрытие клетки (уменьшено для меньшего перекрытия)
    'p_cell_artificial': 0.5,            # Вероятность искусственной клетки
    
    # Конфигурации плотности
    'density_configs': {
        'low': {
            'min_cells': 5,
            'max_cells': 100,
            'description': 'Полный диапазон клеток'
        }
    },
    
    # Распределение изображений по плотностям
    'density_distribution': {
        'low': 1.0          # 100% = 20000 train, 4000 val
    },
    
    # Общий диапазон для num_classes (5-100 = 96 классов)
    'num_classes': 96,
    
    'train_size': 20000,                 # Размер train датасета (увеличено до 20000)
    'val_size': 4000,                    # Размер validation датасета (увеличено до 4000)
    
    # === Модель ===
    'model_name': 'resnet50',            # Архитектура модели (resnet50 для максимальной точности)
    'pretrained': True,                  # Использовать предобученные веса
    'dropout': 0.3,                      # Dropout rate
    
    # === Обучение ===
    'batch_size': 32,                    # Размер батча
    'gradient_accumulation': 1,          # Шагов accumulation (эффективный batch = 32)
    'num_epochs': 200,                   # Максимум эпох (увеличено до 200)
    'learning_rate': 5e-4,               # Learning rate (уменьшен для стабильности)
    'weight_decay': 0,                   # Weight decay (без регуляризации, как в g2d-m1-labs)
    'early_stopping_patience': 15,       # Patience для early stopping (увеличено до 15)
    
    # === Пути ===
    'patches_dir': '../update_generator/patches',
    'test_images_dir': './test_images',
    'data_dir': './data',
    'models_dir': './models',
    'results_dir': './results',
    
    # === Device ===
    'device': 'cuda',                    # 'cuda' или 'cpu'
    'num_workers': 4,                    # Количество worker'ов для DataLoader
}
