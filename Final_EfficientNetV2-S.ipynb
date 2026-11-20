import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
import seaborn as sns
import gc

torch.cuda.empty_cache()
gc.collect()

from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_dir = r'C:\Users\juanj\OneDrive\Documentos\ITESO\Cuarto Semestre\IDI IV\IDI III\idi_class_repo\data\processed\archive\Dataset_Binario'
train_path = os.path.join(base_dir, 'Train')
val_path = os.path.join(base_dir, 'Validation')
test_path = os.path.join(base_dir, 'Test')

best_model_path = r'C:\Users\juanj\OneDrive\Documentos\ITESO\Cuarto Semestre\IDI IV\IDI III\idi_class_repo\models\skin_cancer_efficientnet_binaria_final_2.pth'

train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=test_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Clases encontradas ({num_classes}):", class_names)
print("Mapeo clases->idx:", train_dataset.class_to_idx)

class_counts = Counter(train_dataset.targets)
total_samples = sum(class_counts.values())
weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
print(f"Pesos para la función de pérdida (Cancer/No_Cancer): {weights}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Total imágenes entrenamiento: {len(train_dataset)}")
print(f"Total imágenes validación: {len(val_dataset)}")
print(f"Total imágenes test: {len(test_dataset)}")

efficientnet = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

efficientnet.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(1280, 4096),
    nn.ReLU(True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Linear(4096, num_classes)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
efficientnet.to(device)

class_weights_tensor = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.Adam(efficientnet.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch} (Train)", leave=False)
    for data, target in progress:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(1)
        correct += pred.eq(target).sum().item()

        progress.set_postfix(loss=f"{total_loss/len(train_loader):.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / len(train_loader.dataset)
    return avg_loss, accuracy

def evaluate(model, device, loader, criterion, desc="Evaluation"):
    model.eval()
    total_loss = 0
    correct = 0
    preds = []
    targets_list = []
    probs_list = []

    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False)
        for data, target in progress:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = output.argmax(1)
            correct += pred.eq(target).sum().item()
            
            probs = torch.softmax(output, dim=1)

            preds.extend(pred.cpu().numpy())
            targets_list.extend(target.cpu().numpy())
            probs_list.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / len(loader.dataset)
    return avg_loss, accuracy, preds, targets_list, probs_list

num_epochs = 30
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

patience = 5
best_loss = float("inf")
counter = 0

print("\nIniciando entrenamiento (EfficientNetV2-S)...")

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(efficientnet, device, train_loader, criterion, optimizer, epoch)
    val_loss, val_acc, _, _, _ = evaluate(efficientnet, device, val_loader, criterion, desc="Validation")

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    scheduler.step(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        torch.save(efficientnet.state_dict(), best_model_path)
        print("  --> Nuevo mejor modelo guardado (Val Loss mejoró).")
    else:
        counter += 1
        if counter >= patience:
            print(f"\nEarly Stopping activado en la época {epoch}: no hubo mejora en Val Loss.")
            break

print("\nEntrenamiento terminado.")

print("\nIniciando evaluación clínica en el conjunto de Prueba (Test)...")
efficientnet.load_state_dict(torch.load(best_model_path, weights_only=True))
test_loss, test_acc, preds, targets, probs = evaluate(efficientnet, device, test_loader, criterion, desc="Testing")

probs_array = np.array(probs)
prob_class_1 = probs_array[:, 1]
roc_auc = roc_auc_score(targets, prob_class_1)
prec_macro = precision_score(targets, preds, average='macro')
rec_macro = recall_score(targets, preds, average='macro')
f1_macro = f1_score(targets, preds, average='macro')

epochs_range = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history["train_loss"], 'b-o', label="Train Loss")
plt.plot(epochs_range, history["val_loss"], 'r-o', label="Validation Loss")
plt.title("Loss (EfficientNetV2-S)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history["train_acc"], 'b-o', label="Train Acc")
plt.plot(epochs_range, history["val_acc"], 'r-o', label="Validation Acc")
plt.title("Accuracy (EfficientNetV2-S)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
loss_plot_path = os.path.join(os.path.dirname(best_model_path), 'loss_acc_efficientnet_binaria.png')
plt.savefig(loss_plot_path)
plt.show()

print("\n" + "="*50)
print("MÉTRICAS CLÍNICAS DEFINITIVAS (EFFICIENTNETV2-S)")
print("="*50)
print(f"Exactitud Global (Accuracy) : {test_acc:.2f}%")
print(f"Precisión (Macro)           : {prec_macro:.4f}")
print(f"Sensibilidad/Recall (Macro) : {rec_macro:.4f}")
print(f"Puntuación F1 (Macro)       : {f1_macro:.4f}")
print(f"Área Bajo la Curva (ROC-AUC): {roc_auc:.4f}")
print("="*50 + "\n")

print("--- Reporte de Clasificación Binaria ---")
print(classification_report(targets, preds, target_names=class_names))

cm = confusion_matrix(targets, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de Confusión EfficientNetV2-S: Cáncer vs No Cáncer")
plt.ylabel("Etiqueta Real")
plt.xlabel("Predicción del Modelo")
plt.tight_layout()

cm_plot_path = os.path.join(os.path.dirname(best_model_path), 'cm_efficientnet_binaria.png')
plt.savefig(cm_plot_path)
plt.show()
