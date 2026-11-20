import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import os 
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 28 * 28)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3) 
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

base_dir = r'C:\Users\juanj\OneDrive\Documentos\ITESO\Cuarto Semestre\IDI IV\IDI III\idi_class_repo\data\processed\archive\Dataset_Binario'
train_path = os.path.join(base_dir, 'Train')
validation_path = os.path.join(base_dir, 'Validation')
test_path = os.path.join(base_dir, 'Test')

save_model_path = r'C:\Users\juanj\OneDrive\Documentos\ITESO\Cuarto Semestre\IDI IV\IDI III\idi_class_repo\models\skin_cancer_simplecnn_binario.pth'

train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(root=validation_path, transform=test_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Clases detectadas automáticamente ({num_classes}): {class_names}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class_counts = Counter(train_dataset.targets)
total_samples = sum(class_counts.values())
weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
class_weights_tensor = torch.FloatTensor(weights).to(device)
print(f"Pesos asignados por clase (Cancer/No_Cancer): {weights}")

model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
best_val_loss = float('inf')
os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

print("\nIniciando entrenamiento y validación...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        train_bar.set_postfix(loss=running_loss / (train_bar.n + 1))
    
    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_model_path)
        print(f"  --> Mejor modelo guardado (Mejora en Val Loss).")

print("\nIniciando evaluación en el conjunto de Prueba (Test)...")
model.load_state_dict(torch.load(save_model_path)) 
model.eval()

correct = 0
total = 0
all_preds = []
all_labels = []

test_bar = tqdm(test_loader, desc="Evaluating", leave=False)

with torch.no_grad():
    for images, labels in test_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        test_bar.set_postfix(accuracy=100 * correct / total)

test_accuracy = 100 * correct / total
print(f"\nExactitud general (Accuracy) en Test: {test_accuracy:.2f}%")

print("\n--- Reporte de Clasificación Binaria ---")
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

print("Generando matriz de confusión...")
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión SimpleCNN: Cáncer vs No Cáncer')
plt.ylabel('Etiqueta Real (Clínica)')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()

cm_save_path = os.path.join(os.path.dirname(save_model_path), 'matriz_confusion_simplecnn_binaria.png')
plt.savefig(cm_save_path)
print(f"Matriz de confusión guardada en: {cm_save_path}")
plt.show()
