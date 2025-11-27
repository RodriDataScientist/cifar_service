FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
  fastapi uvicorn pillow python-multipart \
  torch --index-url https://download.pytorch.org/whl/cpu \
  torchvision --index-url https://download.pytorch.org/whl/cpu

# Copia de la app y del modelo real (ya descargado por lfs pull)
COPY app ./

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
