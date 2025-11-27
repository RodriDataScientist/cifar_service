FROM python:3.11-slim

WORKDIR /app

# Instalar git y git-lfs (necesario para descargar el .pth real)
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs && \
    git lfs install && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pillow python-multipart

RUN pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cpu \
  torchvision --index-url https://download.pytorch.org/whl/cpu

COPY app ./app
COPY .git ./.git  # ← Necesario para que LFS pueda descargar el archivo
RUN git lfs pull  # ← Aquí descarga el .pth real

RUN rm -rf /root/.cache/torchvision

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
