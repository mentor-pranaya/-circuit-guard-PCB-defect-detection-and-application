import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from efficientnet_pytorch import EfficientNet

# =========================
# Dataset
# =========================
class PCB_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(self.data['label'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        label = self.class_to_idx[self.data.iloc[idx, 1]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# =========================
# Mixup
# =========================
def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =========================
# Model
# =========================
def build_model(num_classes):
    model = EfficientNet.from_pretrained("efficientnet-b4")
    in_features = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(p=0.4),   # stronger dropout
        nn.Linear(in_features, num_classes)
    )
    return model

# =========================
# Training
# =========================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=40, patience=7):
    best_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []
    patience_counter = 0

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (lam * preds.eq(targets_a).sum().item() +
                        (1 - lam) * preds.eq(targets_b).sum().item())
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # ---- Validation ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Logging
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")

        # Scheduler
        scheduler.step()

        # Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return train_losses, val_losses, val_accs, best_acc

# =========================
# Main
# =========================
if __name__ == "__main__":
    csv_file = r"C:\Users\Dell\Downloads\PCB_DATASET\PCB_DATASET\labels1.csv"
    img_dir = r"C:\Users\Dell\Downloads\PCB_DATASET\PCB_DATASET\roi_images1"
    save_dir = r"C:\Users\Dell\Downloads\PCB_DATASET\PCB_DATASET\results"
    os.makedirs(save_dir, exist_ok=True)

    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Datasets
    dataset = PCB_Dataset(csv_file, img_dir, transform=transform_train)
    num_classes = len(dataset.classes)
    val_split = 0.2
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = build_model(num_classes).to(device)

    # Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Train
    train_losses, val_losses, val_accs, best_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device
    )

    print(f"Best Accuracy: {best_acc*100:.2f}%")

    # Plot curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss Curve"); plt.savefig(os.path.join(save_dir, "loss_curve.png"))

    plt.figure()
    plt.plot(val_accs, label="Val Accuracy")
    plt.legend(); plt.title("Accuracy Curve"); plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))

    # Confusion Matrix on validation set
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
