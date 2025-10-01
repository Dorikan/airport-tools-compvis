FROM python:3.12-slim

ARG PYTORCH_URL="https://download.pytorch.org/whl/cpu"

WORKDIR /app

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Устанавливаем PyTorch и torchvision из аргумента 
RUN pip install --no-cache-dir torch torchvision --index-url ${PYTORCH_URL}
 

# Копируем зависимости
COPY ML-Service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код ML-проекта
COPY ML-Service/ .

EXPOSE 8000

CMD ["python", "main.py"]
