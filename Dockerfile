FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pillow python-multipart python-multipart

RUN pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cpu \
  torchvision --index-url https://download.pytorch.org/whl/cpu

COPY app ./app

RUN rm -rf /root/.cache/torchvision

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
