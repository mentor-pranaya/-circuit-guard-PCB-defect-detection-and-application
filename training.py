import os
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm  # For pretrained models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==========================
# Reproducibility
# ==========================
def set_seed(seed=42):
    """
    Ensures reproducibility of results by fixing random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# ==========================
# Custom Dataset Class
# ==========================
class PCBDataset(Dataset):
    """
    Custom dataset for PCB defect detection.
    Expects a CSV file with columns: 'filename', 'label'
    Loads images from the given root directory and applies transforms.
    """
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = self.root_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================
# Data Transforms
# ==========================
def build_transforms(img_size=128):
    """
    Creates image transformations for training and validation.
    Includes normalization and basic augmentations for training.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Training augmentations for robustness
    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Validation transforms (no heavy augmentation)
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tfms, val_tfms


# ==========================
# Model Builder
# ==========================
def build_model(num_classes, dropout=0.6):
    """
    Builds an EfficientNet-B4 model from timm with a custom classifier head.
    """
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    return model


# ==========================
# Training Loop (One Epoch)
# ==========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one training epoch.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)  # Forward pass
        loss = criterion(logits, y)  # Loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return running_loss / len(loader), 100 * correct / total


# ==========================
# Validation Loop
# ==========================
@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Runs validation epoch and collects predictions for metrics.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_targets, all_preds = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item()
        preds = logits.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

        # Store predictions for classification report
        all_targets.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return running_loss / len(loader), 100 * correct / total, np.array(all_targets), np.array(all_preds)


# ==========================
# Main Training Function
# ==========================
def main():
    # Paths
    ROOT_DIR = "/content/ROI_again"
    CSV_FILE = "/content/ROI_again/ROI_LABELS.csv"
    OUT_DIR = Path("/content/results")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CSV (contains image filenames and labels)
    df = pd.read_csv(CSV_FILE)

    # Train/validation split (stratified to balance classes)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    # Build transforms
    train_tfms, val_tfms = build_transforms(img_size=128)

    # Datasets and loaders
    train_dataset = PCBDataset(train_df, ROOT_DIR, transform=train_tfms)
    val_dataset = PCBDataset(val_df, ROOT_DIR, transform=val_tfms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model setup
    num_classes = df["label"].nunique()
    model = build_model(num_classes).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)

    # Training setup
    EPOCHS = 50
    best_val_acc = 0.0
    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Training step
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation step
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)

        # Scheduler update based on validation loss
        scheduler.step(val_loss)

        # Save history
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        # Print progress
        print(f"[Epoch {epoch:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"Time: {time.time()-t0:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR / "efficientnet_b4_best.pth")

        # Optional early stopping
        if train_acc >= 94.0:
            print(f"\n✅ Early stopping: Validation accuracy reached {val_acc:.2f}% at epoch {epoch}")
            break

    # ==========================
    # Plot training curves
    # ==========================
    plt.figure(figsize=(12,5))

    # Loss curves
    plt.subplot(1,2,1)
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.legend(); plt.title("Loss")

    # Accuracy curves
    plt.subplot(1,2,2)
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Val Acc")
    plt.legend(); plt.title("Accuracy")
    plt.show()

    # ==========================
    # Final Evaluation
    # ==========================
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(df["label"].unique()))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.show()


    torch.save(model.state_dict(), "efficientnet_b4_best.pth")

    from google.colab import files
    files.download("efficientnet_b4_best.pth")

# Entry point
if __name__ == "__main__":
    main()
