# Lab 4: Cell Counting with ResNet

## Задача

Обучить нейронную сеть ResNet50 для решения задачи подсчета количества клеток крови на изображениях.

**Математическая постановка:** Задача классификации, где номер класса = количество клеток на изображении.

**Функция потерь:** BCELoss с label smoothing

**Target:** Вероятностной вектор с гауссовым распределением (uncertainty)

Пример target-вектора для 7 клеток (max=10):
```
[0, 0, 0, 0, 0.1, 0.5, 1.0, 0.5, 0.1, 0]
```

## Установка

```bash
cd lab4
pip install -r requirements.txt
```

## Использование

### Шаг 1: Генерация данных

```bash
python 01_generate_data.py
```

**Время:** ~30 минут  
**Результат:** `data/train/` (10,000 изображений) + `data/val/` (2,000 изображений)

### Шаг 2: Обучение модели

```bash
python 02_train_model.py
```

**Время:** ~2-3 часа  
**Результат:** `models/best_model.pth` + `results/training_curves.png`

### Шаг 3: Валидация модели

```bash
python 03_validate_model.py
```

**Время:** ~5 минут  
**Результат:** `results/confusion_matrix.png` + `results/metrics.json`

### Шаг 4: Тестирование на test_images

```bash
python 04_test_predictions.py
```

**Время:** ~2 минуты  
**Результат:** `results/predictions_examples.png`

## Параметры

См. `config.py` для настройки параметров:

- **Размер изображений:** 512×512
- **Диапазон клеток:** 40-60 (21 класс)
- **Train set:** 10,000 изображений
- **Validation set:** 2,000 изображений
- **Модель:** ResNet50 (pretrained ImageNet)
- **Batch size:** 8 (с gradient accumulation ×4)
- **Эпохи:** 50 (с early stopping patience=10)

## Ожидаемые результаты

- **Validation accuracy (top-1):** 92-95%
- **Validation accuracy (top-3):** 98-99%
- **Mean absolute error:** 0.5-1 клетка

## Структура проекта

```
lab4/
├── config.py                      # Общие параметры
├── dataset.py                     # BloodCellDataset
├── model.py                       # CellCounter (ResNet50)
├── utils.py                       # Вспомогательные функции
├── 01_generate_data.py            # Генерация данных
├── 02_train_model.py              # Обучение модели
├── 03_validate_model.py           # Валидация
├── 04_test_predictions.py         # Тестирование
├── requirements.txt               # Зависимости
├── README.md                      # Документация
├── data/                          # Сгенерированные данные
│   ├── train/                     # 10,000 изображений
│   └── val/                       # 2,000 изображений
├── models/                        # Сохраненные модели
│   └── best_model.pth
└── results/                       # Результаты обучения
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── predictions_examples.png
    └── metrics.json
```

## Требования к системе

- **GPU:** NVIDIA RTX 4050 (6 GB VRAM) или лучше
- **RAM:** 8-10 GB
- **Диск:** ~2 GB для данных + ~100 MB для модели

## Автор

Домашняя работа 4 по курсу 2D Graphics
