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
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from google.colab import files
import zipfile

# ==========================
# 1. UPLOAD AND EXTRACT DATA
# ==========================
print("Upload your dataset ZIP file (images + CSV)...")
uploaded = files.upload()

# Assuming you uploaded a ZIP
for fn in uploaded.keys():
    if fn.endswith(".zip"):
        print(f"Extracting {fn}...")
        with zipfile.ZipFile(fn, "r") as zip_ref:
            zip_ref.extractall("/content/r")  # extracted to /content/r

# ==========================
# 1. CONFIGURE YOUR PATHS (Colab)
# ==========================
ROOT_DIR = Path("/content/r/r")  # path to the folder containing images
CSV_FILE = Path("/content/r/r/roi_data.csv")  # path to the CSV file
OUT_DIR = Path("/content/train")  # output folder for models and plots
OUT_DIR.mkdir(parents=True, exist_ok=True)
  # create the folder if it doesn't exist


# ==========================
# 2. Reproducibility
# ==========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ==========================
# 3. Dataset
# ==========================
class PCBDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = self.root_dir / row["roi_filename"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["category"])
        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================
# 4. Transforms
# ==========================
def build_transforms(img_size=128):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tfms, val_tfms

# ==========================
# 5. Model
# ==========================
def build_model(num_classes, dropout=0.6):
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    return model

# ==========================
# 6. Training / Validation
# ==========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / len(loader), 100 * correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
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
        all_targets.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    return running_loss / len(loader), 100 * correct / total, np.array(all_targets), np.array(all_preds)

# ==========================
# 7. Main Function
# ==========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CSV
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()  # remove spaces

    # Fix Windows paths if present
    if "\\" in df.iloc[0, 0]:
        df["roi_filename"] = df["roi_filename"].apply(lambda x: x.split("\\")[-1])

    train_df, val_df = train_test_split(df, test_size=0.2,
                                        stratify=df["category"], random_state=42)

    train_tfms, val_tfms = build_transforms()
    train_loader = DataLoader(PCBDataset(train_df, ROOT_DIR, train_tfms),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(PCBDataset(val_df, ROOT_DIR, val_tfms),
                            batch_size=32, shuffle=False)

    num_classes = df["category"].nunique()
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=3)

    EPOCHS = 50
    best_val_acc = 0.0
    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        print(f"[Epoch {epoch:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"Time: {time.time() - t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR / "efficientnet_b4_best.pth")

        if train_acc >= 97.0 and val_acc >= 97.0:
            print(f"\n✅ Early stopping at epoch {epoch}")
            break

    # Save Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.legend(); plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Val Acc")
    plt.legend(); plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curves.png")
    plt.close()
    print("✅ Saved training curves to training_curves.png")

    # Final Evaluation
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=sorted(df["category"].unique()))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png")
    plt.close()
    print("✅ Saved confusion matrix to confusion_matrix.png")

    # Save Final Model
    torch.save(model.state_dict(), OUT_DIR / "efficientnet_b4_final.pth")
    print(f"✅ Model saved to {OUT_DIR / 'efficientnet_b4_final.pth'}")

if __name__ == "__main__":
    main()
