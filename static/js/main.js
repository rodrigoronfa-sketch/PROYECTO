const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const promptText = document.querySelector('.drop-zone__prompt');
const uploadForm = document.getElementById('uploadForm');
const resultContainer = document.getElementById('resultContainer');
const debugContainer = document.getElementById('debugContainer');

// --- ACTUALIZAR HISTORIAL DE LA LISTA ENLAZADA ---
function actualizarHistorial() {
    fetch('/historial')
        .then(response => response.json())
        .then(data => {
            const listaUL = document.getElementById('lista-diagnosticos');
            if (!listaUL) return;
            
            listaUL.innerHTML = ''; 

            if (!data || data.length === 0) {
                listaUL.innerHTML = '<li class="empty-history">Esperando procesamiento de imágenes médicas para inicializar nodos...</li>';
                return;
            }

            data.forEach(diag => {
                const li = document.createElement('li');
                const esSano = diag.resultado.toLowerCase().includes('normal') || diag.resultado.toLowerCase().includes('sano');
                const badgeColor = esSano ? 'style="background: #064e3b; color: #a7f3d0;"' : 'style="background: #7f1d1d; color: #fca5a5;"';

                li.innerHTML = `
                    <div>
                        <strong>Archivo:</strong> <span style="color: var(--text-muted); font-family: monospace;">${diag.archivo}</span><br>
                        <span ${badgeColor} class="badge-confianza">${diag.resultado}</span>
                    </div>
                    <span class="badge-confianza">${diag.confianza} Certeza</span>
                `;
                listaUL.appendChild(li);
            });
        })
        .catch(err => console.error("Error al obtener estructura de datos:", err));
}

document.addEventListener('DOMContentLoaded', actualizarHistorial);

// --- CONTROL DEL ÁRBOL DE DECISIÓN ---
function actualizarArbolVisual(fase, resultado = '') {
    const nodeRoot = document.getElementById('node-root');
    const nodeProcess = document.getElementById('node-process');
    const nodeBenign = document.getElementById('node-benign');
    const nodeMalign = document.getElementById('node-malign');
    const edge1 = document.getElementById('edge-1');
    const edge2 = document.getElementById('edge-2');

    if (fase === 'inicio') {
        [nodeRoot, nodeProcess, nodeBenign, nodeMalign, edge1, edge2].forEach(el => {
            if (el) {
                el.classList.remove('active');
                if (el.classList.contains('leaf')) {
                    el.style.backgroundColor = '#111827';
                    el.style.borderColor = 'var(--border)';
                    el.style.color = 'var(--text-muted)';
                }
            }
        });
        if (nodeRoot) nodeRoot.classList.add('active');
    } 
    else if (fase === 'procesando') {
        if (edge1) edge1.className = 'edge active';
        if (nodeProcess) nodeProcess.classList.add('active');
    } 
    else if (fase === 'final') {
        if (edge2) edge2.className = 'edge active';
        
        const resLimpio = resultado.toLowerCase();
        if (resLimpio.includes('lesión') || resLimpio.includes('lesion') || resLimpio.includes('anomalía') || resLimpio.includes('anomalia')) {
            if (nodeMalign) {
                nodeMalign.classList.add('active');
                nodeMalign.style.backgroundColor = 'var(--danger-bg)';
                nodeMalign.style.borderColor = 'var(--danger)';
                nodeMalign.style.color = '#fca5a5';
            }
        } else {
            if (nodeBenign) {
                nodeBenign.classList.add('active');
                nodeBenign.style.backgroundColor = 'var(--success-bg)';
                nodeBenign.style.borderColor = 'var(--success)';
                nodeBenign.style.color = '#a7f3d0';
            }
        }
    }
}

// --- LÓGICA DE INTERFAZ Y VISTA PREVIA ---
if (dropZone && imageInput) {
    dropZone.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', () => {
        if (imageInput.files && imageInput.files.length > 0) {
            // CORREGIDO: Pasamos el primer archivo [0], no la lista completa
            updatePreview(imageInput.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => { 
        e.preventDefault(); 
        dropZone.classList.add('drop-zone--over'); 
    });

    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => dropZone.classList.remove('drop-zone--over'));
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drop-zone--over');

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const dataTransferContainer = new DataTransfer();
            dataTransferContainer.items.add(e.dataTransfer.files[0]);
            
            imageInput.files = dataTransferContainer.files;
            // CORREGIDO: Pasamos el primer archivo [0] de la transferencia
            updatePreview(e.dataTransfer.files[0]);
        }
    });
}

function updatePreview(archivoUnico) {
    // Si no llega un archivo válido, detenemos el proceso para evitar colapsos
    if (!archivoUnico) return;

    const reader = new FileReader();
    reader.readAsDataURL(archivoUnico);
    reader.onload = () => {
        if (previewImage) {
            previewImage.src = reader.result;
            previewImage.style.display = 'block';
        }
        if (promptText) promptText.style.display = 'none';
        actualizarArbolVisual('inicio');
    };
}

// --- CONEXIÓN DE ENVÍO CON EL BACKEND FLASK ---
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const resEstado = document.getElementById('resEstado');
    const resConfianza = document.getElementById('resConfianza');

    resultContainer.style.display = 'none';
    debugContainer.style.display = 'none';
    debugContainer.innerText = '';

    if (!imageInput.files || imageInput.files.length === 0) {
        alert('Por favor selecciona una imagen.');
        return;
    }

    const formData = new FormData();
    // CORREGIDO: Enviamos el archivo real [0] esperado por el backend de Flask
    formData.append('file', imageInput.files[0]);

    submitBtn.innerText = 'Procesando en la IA...';
    submitBtn.disabled = true;

    actualizarArbolVisual('inicio');
    actualizarArbolVisual('procesando');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            resEstado.innerText = data.resultado;
            resConfianza.innerText = data.confianza;
            resultContainer.style.display = 'block';
            actualizarArbolVisual('final', data.resultado);
            actualizarHistorial();
        } else {
            debugContainer.innerText = `Error del Servidor (Status ${response.status}):\n${data.error || JSON.stringify(data)}`;
            debugContainer.style.display = 'block';
        }
    } catch (error) {
        debugContainer.innerText = `Fallo de conexión o red:\n${error.message}`;
        debugContainer.style.display = 'block';
    } finally {
        submitBtn.innerText = 'Analizar Tejido';
        submitBtn.disabled = false;
    }
});
