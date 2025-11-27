FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copia el requirements
COPY requirements.txt .

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir requests  

# Copia todo el código
COPY app ./app

# Expone el puerto que usará FastAPI
EXPOSE 8000

# Comando para arrancar el servidor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
