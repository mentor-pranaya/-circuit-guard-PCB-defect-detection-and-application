# ========================================
# 1. Mount Google Drive
# ========================================
from google.colab import drive
drive.mount("/content/drive")

# Paths
TRAIN_CSV = "/content/drive/MyDrive/train1.csv"
VAL_CSV   = "/content/drive/MyDrive/val1.csv"
DATA_DIR  = "/content/drive/MyDrive/PCB_images"

# ========================================
# 2. Import libraries
# ========================================
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#from efficientnet_pytorch import EfficientNet
import torchvision.models as models

# ========================================
# 3. Dataset Class
# ========================================

class PCB_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {
            "missinghole": 0,
            "mousebites": 1,
            "short": 2,
            "spur": 3,
            "spuriouscopper": 4,
            "opencircuit": 5
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]  # full path is already in CSV

    # Normalize: lowercase + remove underscores + strip spaces
        label_str = str(self.data.iloc[idx, 1]).strip().lower().replace("_", "").replace(" ", "")

        if label_str not in self.label_map:
            raise ValueError(f"Unknown label found: {label_str}")
        label = self.label_map[label_str]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label



# ========================================
# 4. Data Transforms
# ========================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========================================
# 5. Load Data
# ========================================
train_dataset = PCB_Dataset(TRAIN_CSV, transform=train_transform)
val_dataset   = PCB_Dataset(VAL_CSV, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

# ========================================
# 6. Model Setup (EfficientNet-B4)
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Using torchvision's EfficientNet-B4
model = models.efficientnet_b4(weights="IMAGENET1K_V1")
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 6)  # 6 defect classes
model = model.to(device)


# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ========================================
# 7. Training Function
# ========================================
def train_model(model, train_loader, val_loader, device, epochs=10):
    best_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    save_path = "/content/drive/MyDrive/pcb_model.pth"  # unified path

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100 * correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model at epoch {epoch+1} with Val Acc: {val_acc:.2f}%")

    # Plot curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend(); plt.title("Accuracy")
    plt.show()

    print(f"Best Validation Accuracy: {best_acc:.2f}%")

# ========================================
# 8. Run Training
# ========================================
train_model(model, train_loader, val_loader, device, epochs=20)

# ========================================
# 9. Load Saved Best Model
# ========================================
model.load_state_dict(torch.load("/content/drive/MyDrive/pcb_model.pth", map_location=device))
model.eval()
print(" Model loaded and ready for testing/inference")

