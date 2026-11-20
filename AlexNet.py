import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import AlexNet_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

save_model_path = r'C:\Users\juanj\OneDrive\Documentos\ITESO\Cuarto Semestre\IDI IV\IDI III\idi_class_repo\models\skin_cancer_alexnet_binaria.pth'

train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=test_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Clases detectadas ({num_classes}):", class_names)
print("Mapeo de clases a índices:", train_dataset.class_to_idx)

class_counts = Counter(train_dataset.targets)
total_samples = sum(class_counts.values())
weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
print(f"Pesos para la función de pérdida (Cancer/No_Cancer): {weights}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)

for param in alexnet.features.parameters():
    param.requires_grad = False

alexnet.classifier = nn.Sequential(
    nn.Dropout(p=0.5), 
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, num_classes)
)

alexnet.to(device)
class_weights_tensor = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, alexnet.parameters()), lr=0.0001, weight_decay=1e-4) 
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0 
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        progress_bar.set_postfix(loss=train_loss/len(train_loader))

    avg_loss = train_loss / len(train_loader)
    avg_acc = 100. * correct / len(train_loader.dataset) 
    return avg_loss, avg_acc

def test(model, device, test_loader, criterion, desc="Validation"):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=desc, leave=False)
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            probs = torch.softmax(output, dim=1)
            
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) 

    avg_loss = test_loss / len(test_loader)
    avg_acc = 100. * correct / len(test_loader.dataset)
    return avg_loss, avg_acc, all_preds, all_targets, all_probs

num_epochs = 20
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

best_val_loss = float('inf')
early_stopping_patience = 5
patience_counter = 0

print("Iniciando entrenamiento (AlexNet Transfer Learning)...")

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(alexnet, device, train_loader, criterion, optimizer, epoch)
    val_loss, val_acc, _, _, _ = test(alexnet, device, val_loader, criterion, desc="Validation")
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save(alexnet.state_dict(), save_model_path)
        print("  --> Nuevo mejor modelo guardado.")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly Stopping activado en la época {epoch}.")
            break

print("\nEntrenamiento finalizado.")

print("\nIniciando evaluación clínica en el conjunto de Prueba (Test)...")
alexnet.load_state_dict(torch.load(save_model_path, weights_only=True))
test_loss, test_acc, preds, targets, probs = test(alexnet, device, test_loader, criterion, desc="Testing")

probs_array = np.array(probs)
prob_class_1 = probs_array[:, 1]

prec_macro = precision_score(targets, preds, average='macro')
rec_macro = recall_score(targets, preds, average='macro')
f1_macro = f1_score(targets, preds, average='macro')
roc_auc = roc_auc_score(targets, prob_class_1)

print("\n" + "="*45)
print("       MÉTRICAS CLÍNICAS DEFINITIVAS")
print("="*45)
print(f"Exactitud Global (Accuracy) : {test_acc:.2f}%")
print(f"Precisión (Macro)           : {prec_macro:.4f}")
print(f"Sensibilidad/Recall (Macro) : {rec_macro:.4f}")
print(f"Puntuación F1 (Macro)       : {f1_macro:.4f}")
print(f"Área Bajo la Curva (ROC-AUC): {roc_auc:.4f}")
print("="*45 + "\n")

print("--- Reporte de Clasificación (Desglose por Clase) ---")
print(classification_report(targets, preds, target_names=class_names))

epochs_range = range(1, len(history['train_loss']) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
plt.plot(epochs_range, history['val_loss'], 'r-o', label='Validation Loss')
plt.title('Training and Validation Loss - AlexNet (Transfer Learning)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
loss_plot_path = os.path.join(os.path.dirname(save_model_path), 'loss_alexnet_binaria.png')
plt.savefig(loss_plot_path)
plt.show()

cm = confusion_matrix(targets, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión AlexNet: Cáncer vs No Cáncer')
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
cm_plot_path = os.path.join(os.path.dirname(save_model_path), 'cm_alexnet_binaria.png')
plt.savefig(cm_plot_path)
plt.show()
