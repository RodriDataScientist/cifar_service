"""
train_cifar10_resnet.py
Entrena ResNet18 (fine-tuning) en CIFAR-10 usando CUDA si está disponible.
Muestra gráficas de loss, accuracy y F1 (macro).
"""

import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms, datasets, models

from sklearn.metrics import f1_score, accuracy_score

# ---------------------------
# Configuración / hiperparámetros
# ---------------------------
SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
VALID_SPLIT = 0.1
NUM_WORKERS = 2
PIN_MEMORY = True

# ---------------------------
# Reproducibilidad
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------------------------
# Device (GPU si hay)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando device:", device)

# ---------------------------
# Transforms (augmentación + normalización)
# ---------------------------
# Estadísticas de CIFAR-10 (mean/std)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ---------------------------
# Dataset y DataLoaders
# ---------------------------
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transforms)
num_train = len(full_train)
num_val = int(VALID_SPLIT * num_train)
num_train = num_train - num_val

train_ds, val_ds = random_split(full_train, [num_train, num_val], generator=torch.Generator().manual_seed(SEED))
val_ds.dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=val_transforms)

test_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ---------------------------
# Modelo (ResNet18 fine-tuning)
# ---------------------------
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device)

# ---------------------------
# Loss, optimizer, scheduler
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ---------------------------
# Entrenamiento / Validación
# ---------------------------
scaler = torch.amp.GradScaler()  # mixed precision

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    loop = tqdm(loader, desc="train", leave=False)
    for inputs, targets in loop:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')

    return epoch_loss, epoch_acc, epoch_f1

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        loop = tqdm(loader, desc="eval", leave=False)
        for inputs, targets in loop:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')

    return epoch_loss, epoch_acc, epoch_f1

# ---------------------------
# Loop principal
# ---------------------------
best_val_loss = float('inf')
history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
           'val_loss': [], 'val_acc': [], 'val_f1': []}

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

    train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
    val_loss, val_acc, val_f1 = eval_model(model, val_loader, criterion, device)

    print(f" Train loss: {train_loss:.4f} | acc: {train_acc:.4f} | f1_macro: {train_f1:.4f}")
    print(f" Val   loss: {val_loss:.4f} | acc: {val_acc:.4f} | f1_macro: {val_f1:.4f}")

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1'].append(train_f1)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    # scheduler usa la métrica de validación
    scheduler.step(val_loss)

    # guardar mejor modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, "best_model.pth")
        print(" --> Guardado best_model.pth")

# ---------------------------
# Evaluación en test set usando el mejor checkpoint
# ---------------------------
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc, test_f1 = eval_model(model, test_loader, criterion, device)
print("\nResultado final en TEST:")
print(f" Test loss: {test_loss:.4f} | acc: {test_acc:.4f} | f1_macro: {test_f1:.4f}")

# ---------------------------
# Plots: loss, accuracy, f1
# ---------------------------
epochs = np.arange(1, NUM_EPOCHS + 1)

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.plot(epochs, history['train_loss'], label='train_loss', marker='o')
plt.plot(epochs, history['val_loss'], label='val_loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss por epoch')
plt.legend()
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(epochs, history['train_acc'], label='train_acc', marker='o')
plt.plot(epochs, history['val_acc'], label='val_acc', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy por epoch')
plt.legend()
plt.grid(True)

plt.subplot(1,3,3)
plt.plot(epochs, history['train_f1'], label='train_f1_macro', marker='o')
plt.plot(epochs, history['val_f1'], label='val_f1_macro', marker='o')
plt.xlabel('Epoch')
plt.ylabel('F1 (macro)')
plt.title('F1 (macro) por epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_plots.png", dpi=150)
print("Guardadas gráficas en training_plots.png")
plt.show()
