import os
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

class NodoDiagnostico:
    def __init__(self, archivo, resultado, confianza_num, confianza_str):
        self.archivo = archivo
        self.resultado = resultado
        self.confianza_num = confianza_num  # Usado para ordenar (float)
        self.confianza_str = confianza_str  # Formato texto "85.30%"
        self.siguiente = None

class ListaHistorialClinico:
    def __init__(self):
        self.cabeza = None

    def insertar_ordenado(self, archivo, resultado, confianza_num, confianza_str):
        """Inserta los diagnósticos ordenados de mayor a menor confianza numérica"""
        nuevo_nodo = NodoDiagnostico(archivo, resultado, confianza_num, confianza_str)
        
        # Caso 1: Lista vacía o el nuevo nodo tiene mayor confianza que la cabeza
        if self.cabeza is None or nuevo_nodo.confianza_num > self.cabeza.confianza_num:
            nuevo_nodo.siguiente = self.cabeza
            self.cabeza = nuevo_nodo
            return

        # Caso 2: Buscar la posición correcta en medio o al final de la lista
        actual = self.cabeza
        while actual.siguiente is not None and actual.siguiente.confianza_num >= nuevo_nodo.confianza_num:
            actual = actual.siguiente
            
        nuevo_nodo.siguiente = actual.siguiente
        actual.siguiente = nuevo_nodo

    def transformar_a_lista(self):
        """Convierte los nodos a una lista de diccionarios estándar para JSON"""
        lista_plana = []
        actual = self.cabeza
        while actual:
            lista_plana.append({
                'archivo': actual.archivo,
                'resultado': actual.resultado,
                'confianza': actual.confianza_str
            })
            actual = actual.siguiente
        return lista_plana

# Instanciamos la lista enlazada global en memoria
historial_clinico = ListaHistorialClinico()
# =====================================================================


# 1. RECONSTRUIR LA ARQUITECTURA EXACTA DEL MODELO
def cargar_modelo():
    model = models.mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),  
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    ruta_modelo = 'modelo_esofago.pt'
    if os.path.exists(ruta_modelo):
        model.load_state_dict(torch.load(ruta_modelo, map_location=torch.device('cpu')))
        print("¡Modelo clínico de PyTorch cargado correctamente en memoria!")
    else:
        print(f"ADVERTENCIA CRÍTICA: No se encontró el archivo '{ruta_modelo}' en la raíz.")
        
    model.eval() 
    return model

ia_model = cargar_modelo()

# 2. TRANSFORMACIONES REQUERIDAS PARA EL DIAGNÓSTICO
transformacion_imagen = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')


# Ruta API: Recibe la imagen y guarda el resultado en la Lista Enlazada
@app.route('/predict', methods=['POST'])
def predict():
    clave_archivo = None
    if 'file' in request.files:
        clave_archivo = 'file'
    elif 'myFile' in request.files:
        clave_archivo = 'myFile'
        
    if not clave_archivo:
        return jsonify({'error': 'No se recibió ninguna imagen en el servidor médico.'}), 400
        
    file = request.files[clave_archivo]
    if file.filename == '':
        return jsonify({'error': 'El archivo seleccionado está vacío.'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img_tensor = transformacion_imagen(img).unsqueeze(0) 

        with torch.no_grad():
            salida = ia_model(img_tensor)
            probabilidad = salida.item() 

        if probabilidad < 0.5:
            resultado = "Posible lesión o anomalía detectada en el esófago. Se sugiere revisión médica."
            confianza_valor = (1 - probabilidad) * 100
            confianza = f"{confianza_valor:.2f}%"
        else:
            resultado = "Tejido esofágico normal (Sano)."
            confianza_valor = probabilidad * 100
            confianza = f"{confianza_valor:.2f}%"

        # --- SECCIÓN ACADÉMICA: Guardamos el diagnóstico en la Lista Enlazada ---
        historial_clinico.insertar_ordenado(
            archivo=file.filename,
            resultado=resultado,
            confianza_num=confianza_valor,
            confianza_str=confianza
        )

        return jsonify({
            'resultado': resultado,
            'confianza': confianza,
            'archivo_procesado': file.filename
        })

    except Exception as e:
        return jsonify({'error': f'Fallo en el motor de inferencia de PyTorch: {str(e)}'}), 500


# --- NUEVA RUTA API: Retorna el historial de la lista enlazada en formato JSON ---
@app.route('/historial', methods=['GET'])
def obtener_historial():
    datos_historial = historial_clinico.transformar_a_lista()
    return jsonify(datos_historial)


if __name__ == '__main__':
    app.run(debug=True)
