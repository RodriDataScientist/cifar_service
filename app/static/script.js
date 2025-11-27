const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("fileElem");
const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");
const result = document.getElementById("result");
const predictBtn = document.getElementById("predictBtn");

let currentFile = null;

dropArea.addEventListener("click", () => fileInput.click());

// Drag & Drop
dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.borderColor = "white";
});

dropArea.addEventListener("dragleave", () => {
    dropArea.style.borderColor = "rgba(255,255,255,0.6)";
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
    handleFile(fileInput.files[0]);
});

function handleFile(file) {
    currentFile = file;
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    placeholder.style.display = "none";
    predictBtn.disabled = false;
    result.innerHTML = "";
}

predictBtn.addEventListener("click", () => {
    if (currentFile) predict(currentFile);
});

async function predict(file) {
    result.innerHTML = "Procesando...";

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    result.innerHTML = `
        <b>Predicción:</b> ${data.prediction.label}<br>
        <b>Confianza:</b> ${(data.prediction.confidence * 100).toFixed(2)}%
    `;
}