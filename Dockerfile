# Usa imagen base ligera con Python 3.11
FROM python:3.11-slim

# Evita que Python cree archivos pyc y buffers de logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo en el contenedor
WORKDIR /app

# Copiar requirements primero para aprovechar cache
COPY requirements.txt .

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu

# Copia todo el código incluyendo el modelo
COPY app ./app
COPY app/main.py ./app/main.py

# Verifica que el modelo está presente y su tamaño
RUN if [ ! -f "app/best_model.pth" ]; then echo "ERROR: best_model.pth no encontrado"; exit 1; fi \
    && ls -lh app/best_model.pth

# Expone el puerto que usará FastAPI
EXPOSE 8000

# Comando para arrancar el servidor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]