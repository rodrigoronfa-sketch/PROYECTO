import os
import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'dataset', 'validation')

transformaciones_entreno = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Tolera cambios de luz de otras cámaras
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transformaciones_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    dataset_entreno = datasets.ImageFolder(TRAIN_DIR, transform=transformaciones_entreno)
    dataset_val = datasets.ImageFolder(VAL_DIR, transform=transformaciones_val)

    # Usamos un batch_size estándar automático
    train_loader = DataLoader(dataset_entreno, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=4, shuffle=False)

    print(f"🏷️ Clases mapeadas en orden alfabético: {dataset_entreno.class_to_idx}")

    # Cargar MobileNetV2 de forma segura
    pesos = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=pesos)

    # Congelar el extractor de características para evitar que explote la memoria
    for param in model.parameters():
        param.requires_grad = False

    # ARQUITECTURA ANTISOBREAJUSTE (Agregamos Dropout seguro en el clasificador externo)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),  # Apaga la mitad de las neuronas al azar para obligar a generalizar imágenes nuevas
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    EPOCHS = 20
    print(f"\n🚀 Entrenando IA Generalizada durante {EPOCHS} épocas...")
    
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for imagenes, etiquetas in train_loader:
            etiquetas = etiquetas.float().unsqueeze(1)
            optimizer.zero_grad()
            salidas = model(imagenes)
            loss = criterion(salidas, etiquetas)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"  ✨ Época {epoch+1}/{EPOCHS} -> Pérdida Promedio: {running_loss / len(train_loader):.4f}")

    # Guardar los pesos definitivos
    torch.save(model.state_dict(), 'modelo_esofago.pt')
    print("\n🎉 ¡ENTRENAMIENTO COMPLETADO CON ÉXITO! 'modelo_esofago.pt' generado.")

except Exception as e:
    print(f"\n❌ El entrenamiento falló por el siguiente motivo técnico:\n{str(e)}")
