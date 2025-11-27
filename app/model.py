import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import os

# ======================
# LOAD TRAINED MODEL (CIFAR-10 → 10 classes)
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

# Debe ser exactamente igual al modelo con el que entrenaste
model = models.resnet18(weights=None)  # NO usar pretrained=True
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 = 10 clases

# Cargar pesos entrenados
weights_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)


# ======================
# CIFAR-10 TRANSFORMS (DEBEN SER IGUALES)
# ======================
transform = T.Compose([
    T.Resize((32, 32)),  # CIFAR-10 imágenes 32×32
    T.ToTensor(),
    T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


# ======================
# INFERENCE FUNCTION
# ======================
def predict_image(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        label_idx = torch.argmax(probs).item()

    return {
        "label": CIFAR10_CLASSES[label_idx],   # ← etiqueta real
        "label_idx": int(label_idx),           # ← índice por si lo quieres
        "confidence": float(probs[label_idx]),
        "all_probs": probs.tolist()
    }