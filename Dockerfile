FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs && \
    git lfs install && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pillow python-multipart

RUN pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cpu \
  torchvision --index-url https://download.pytorch.org/whl/cpu

# Copia de la app
COPY app ./app

# Copia del repo git para que LFS pueda descargar el modelo real
COPY .git ./.git
RUN git lfs pull

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
