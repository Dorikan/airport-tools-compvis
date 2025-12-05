# 🚀 ML Detection Service

> Этот проект — сервис для автоматического распознавания инструментов на изображениях.\
> В основе лежит YOLO для общей детекции и дополнительная нейросеть EfficientNet, выступающая экспертом по распознаванию отверток.
>
> Весь пайплайн собран в FastAPI-сервис, что позволяет использовать модель как готовый REST API.\
> Мы уделили внимание точности и стабильности работы, сохранив при этом удобство развертывания.

## ✨ Возможности

* ⚡ **Быстрый REST API** на базе FastAPI

* 🧠 **YOLO-детектор** для выделения объектов

* 🔧 **EfficientNet-эксперт**  и эмбеддер для «трудных» кейсов

* 📦 **Docker-сборка** для лёгкого деплоя

---

## 🏃 Быстрый старт

<details>
<summary>Установка PyTorch</summary>

Обратите внимание. В файле requirements.txt нет необходимых для работы проекта библиотек torch и torchvision. Это связано с тем, что для разных конфигураций устройств следует использовать разные билды библиотек. Получить index-url для своей сборки вы можете на [официальном сайте PyTorch](https://pytorch.org/get-started/locally/) через их виджет. В коде будут приведены примеры для некоторых конфигураций.

</details>

### 🔹 Запуск локально

#### Linux/MacOS

```bash
cd ML-Service

# Создать виртуальное окружение (venv)
python3 -m venv venv

# Активировать окружение
source venv/bin/activate

# Установка PyTorch. Пример для CPU.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt

# Запустить проект
python main.py
```

#### Windows

```bash
cd ML-Service

# Создать виртуальное окружение (venv)
python3 -m venv venv

# Активировать окружение
venv\Scripts\activate

# Установка PyTorch. Пример для CPU.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt

# Запустить проект
python main.py
```

## 🔹 Запуск в Docker

Вариант конфигурации для PyTorch можно указать явно. Если его не указывать, установка произойдет из репозитория PyPI.

#### CPU

```bash
docker build -t ml_service --build-arg TORCH_INDEX=сpu .
docker run -p 8000:8000 ml_service
```

#### CUDA 12.8

```bash
docker build -t ml_service --build-arg TORCH_INDEX=cu128 .
docker run -p 8000:8000 ml_service
```

---

## 📡 Api

Обращение к эндпоинту предикта для изображения:

```http
GET http://{HOST}:{PORT}/api/v1/predict/?image_id={image_id, str}&url={image_url}&thresh={threshold, float}
```

При локальном обращении по 127.0.0.1 в полу image\_url можно указать абсолютный путь до изображения.

Ответ предоставляется в формате JSON:

```json
{
"instruments": [
  {
    "bbox": [
      3787.427734375,
      1443.26416015625,
      4833.0146484375,
      2827.8193359375
    ],
    "class": 1,
    "confidence": 0.9444190859794617,
    "embedding": []
  },
  {
    "bbox": [
      3142.17724609375,
      780.5576171875,
      4768.83740234375,
      1782.2603759765625
    ],
    "class": 10,
    "confidence": 0.9379329681396484,
    "embedding": []
  },
  ...
],
"debug_image": BASE64_IMAGE,
"image_id": "abc123"
}
```

## ⚙️ Конфигурация

Конфигурация проекта осуществляется через файл [config.py](ML-Service/utils/config.py).


|Model|FLOPs|cpu (ms)|T4 (ms)|
|--|--|--|--|
|YOLOv11l|86\.9 B|800 ± 200|6\.2  ±  0.1|
|EfficientNet-b0 (Embedder)|0\.39 G|20 ± 5|0\.00008 ±  0.00016|
|EfficientNet-b0 (Screwdriver clasifier)|0\.39 G|20 ± 5|0\.00008 ±  0.00016|



## 🐒 Авторство

* **Dorikan** — ML и FastApi сервис
