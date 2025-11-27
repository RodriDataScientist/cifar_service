import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import os
import requests

# ======================
# CONFIGURACIÓN
# ======================
device = "cpu"

CIFAR10_CLASSES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

# ======================
# ARCHIVO DE PESOS
# ======================

weights_path = os.path.join(os.path.dirname(__file__), "cifar10_model.pth")
# Reemplaza TU_ID_DEL_ARCHIVO con el ID de Google Drive
drive_file_id = "19IKfrv3P2DscAFAv8FaSYr53cz87ElFn"
drive_url = f"https://drive.google.com/uc?export=download&id={drive_file_id}"

# Descargar modelo desde Drive si no existe
if not os.path.exists(weights_path):
    print("Modelo no encontrado, descargando desde Google Drive...")
    response = requests.get(drive_url)
    with open(weights_path, "wb") as f:
        f.write(response.content)
    print("Modelo descargado correctamente!")

# ======================
# CARGAR MODELO
# ======================
model = models.resnet18(weights=None)  # NO usar pretrained=True
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 = 10 clases

checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)

# ======================
# TRANSFORMACIONES CIFAR-10
# ======================
transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

# ======================
# FUNCIÓN DE PREDICCIÓN
# ======================
def predict_image(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        label_idx = torch.argmax(probs).item()
    return {
        "label": CIFAR10_CLASSES[label_idx],
        "label_idx": int(label_idx),
        "confidence": float(probs[label_idx]),
        "all_probs": probs.tolist()
    }
