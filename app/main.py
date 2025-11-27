from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

from .model import predict_image

app = FastAPI()

# Servir frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def home():
    return FileResponse("app/static/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = predict_image(image)

    return {
        "filename": file.filename,
        "prediction": result

    }
