FROM python:3.12-slim

# Аргументы: версия и тип устройства (cpu или cu121 и т.п.)
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG DEVICE=cpu

WORKDIR /app

# Системные зависимости (например, для OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Устанавливаем PyTorch с нужным билдом (cpu или cuda)
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+${DEVICE} \
    torchvision==${TORCHVISION_VERSION}+${DEVICE} \
    --index-url https://download.pytorch.org/whl/${DEVICE}

# Копируем зависимости проекта
COPY ML-Service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY ML-Service/ .

EXPOSE 8000

CMD ["python", "main.py"]