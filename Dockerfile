FROM python:3.12-slim

WORKDIR /app

# Системные зависимости (например, для OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX=none

ENV TORCH_INDEX=${TORCH_INDEX}

# Установка PyTorch + torchvision
RUN if [ "$TORCH_INDEX" = "none" ]; then \
        pip install torch torchvision; \
    else \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_INDEX}; \
    fi

# Копируем зависимости проекта
COPY ML-Service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY ML-Service/ .

EXPOSE 8000

CMD ["python", "main.py"]