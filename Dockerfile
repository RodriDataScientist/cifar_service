FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .

# 1. Instalar dependencias de PyPI normalmente
RUN pip install --no-cache-dir fastapi uvicorn pillow python-multipart

# 2. Instalar PyTorch CPU desde el index correcto
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY app ./

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
