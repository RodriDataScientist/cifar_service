# 🚀 Clasificador CIFAR-10 con ResNet18 | FastAPI + Docker + Railway

Este proyecto implementa un clasificador de imágenes basado en **ResNet18** con *fine-tuning* utilizando el dataset **CIFAR-10**.  
El modelo se integra en un servicio web construido con **FastAPI**, acompañado de un **frontend web** y listo para despliegue mediante **Docker** en Railway.

---

## 📌 Características Principales

- 🧠 **Modelo ResNet18 Fine-Tuned** con PyTorch  
- ⚙️ Entrenamiento con:
  - Data augmentation
  - Mixed precision
  - AdamW + ReduceLROnPlateau
- 🌐 **Backend con FastAPI**
  - Endpoint `/predict` para inferencia
  - Servido con Uvicorn
- 🎨 **Frontend Web**
  - Carga y previsualización de imágenes
  - Consumo del API REST
  - Respuesta con clase y probabilidad
- 📦 **Contenedorización con Docker**
  - Imagen ligera basada en Python 3.11-slim
  - Descarga automática de pesos desde Google Drive
- ☁️ **Despliegue en Railway**
  - Dockerfile autodetectado
  - Configuración automática del servicio
  - Modelo funcionando 24/7

---

## 📁 Estructura del Proyecto

```

cifar_service/
│── app/
│   ├── main.py                # API principal (FastAPI)
│   ├── model.py               # Carga del modelo y predicción
│   ├── static/
│   │   ├── index.html         # Frontend
│   │   ├── styles.css
│   │   └── script.js
│   └── weights/
│       └── (se descargan automáticamente)
│
│── train_model.py              # Script de entrenamiento
│
│── requirements.txt
│── Dockerfile
│── README.md

````

---

## 🛠️ Instalación local

### 1️⃣ Clona el repositorio

```bash
git clone https://github.com/RodriDataScientist/cifar_service
cd cifar_service
````

### 2️⃣ Crea un entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3️⃣ Instala dependencias

```bash
pip install -r requirements.txt
```

### 4️⃣ Ejecuta la API

```bash
uvicorn app.main:app --reload
```

La aplicación estará disponible en:

👉 `http://localhost:8000`

---

## 🐳 Ejecución con Docker

### Construir la imagen

```bash
docker build -t cifar-service .
```

### Ejecutar el contenedor

```bash
docker run -p 8000:8000 cifar-service
```

---

## 🌐 Despliegue en Railway

1. Conecta el repositorio a Railway
2. Railway detecta automáticamente el `Dockerfile`
3. Expone el puerto `8000`
4. El contenedor descarga los pesos desde Google Drive
5. ¡Listo! Tu API estará disponible con un dominio público

---

## 🖼️ Uso del Endpoint `/predict`

El endpoint espera una imagen en formato **JPEG/PNG**:

### Ejemplo con `curl`

```bash
curl -X POST -F "file=@imagen.png" https://tu-servicio.up.railway.app/predict
```

### Respuesta

```json
{
  "class": "airplane",
  "probability": 0.87
}
```

---

## 📚 Entrenamiento del Modelo

El entrenamiento se realizó con:

* 30 épocas
* Data augmentation (RandomCrop, HorizontalFlip, ColorJitter)
* Mixed precision (`torch.cuda.amp`)
* Optimizer AdamW
* Scheduler ReduceLROnPlateau
* Mejores pesos guardados por validación

---

## 👨‍💻 Autor

**Rodrigo Fabián Cervantes Martínez**
Ingeniería en Datos e Inteligencia Artificial — Universidad de Guanajuato
📧 [rf.cervantesmartinez@ugto.mx](mailto:rf.cervantesmartinez@ugto.mx)
📎 GitHub: [https://github.com/RodriDataScientist](https://github.com/RodriDataScientist)

---
