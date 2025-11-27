FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu

COPY app ./app

# Verifica que el modelo está presente
RUN if [ ! -f "app/best_model.pth" ]; then echo "ERROR: best_model.pth no encontrado"; exit 1; fi \
    && ls -lh app/best_model.pth

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]