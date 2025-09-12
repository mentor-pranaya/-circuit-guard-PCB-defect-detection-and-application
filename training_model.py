import os, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import timm

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------
# Seed for reproducibility
# -------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------------
# Device
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 0  # safer for Windows

# -------------------------------
# Dataset
# -------------------------------
class ROIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, class_to_idx=None):
        self.transform = transform

        # Build class mapping if not provided
        if class_to_idx is None:
            classes = sorted(df["label"].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        # Build absolute paths list
        abs_paths, labels = [], []
        for _, row in df.iterrows():
            fname = str(row["filename"])
            label = str(row["label"])
            if os.path.exists(fname):
                abs_paths.append(fname)
                labels.append(label)

        if len(abs_paths) == 0:
            raise ValueError("No images found! Check your CSV paths.")

        self.df = pd.DataFrame({"abs_path": abs_paths, "label": labels}).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["abs_path"]).convert("RGB")
        label = self.class_to_idx[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------------------
# Transforms
# -------------------------------
def build_transforms(img_size=128):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tfms, val_tfms

# -------------------------------
# Dataloaders
# -------------------------------
def make_loaders(train_csv, val_csv, batch_size=32):
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    train_tfms, val_tfms = build_transforms()

    train_ds = ROIDataset(train_df, transform=train_tfms)
    val_ds   = ROIDataset(val_df,   transform=val_tfms, class_to_idx=train_ds.class_to_idx)

    # Weighted sampler to handle class imbalance
    class_counts = train_df["label"].value_counts().to_dict()
    weights = train_df["label"].map(lambda c: 1.0 / class_counts[c]).values
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights),
                                    num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return train_loader, val_loader, train_ds.class_to_idx

# -------------------------------
# Model
# -------------------------------
def build_model(num_classes):
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
    return model

# -------------------------------
# Train/Validate
# -------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_targets, all_preds = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_targets.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_targets), np.array(all_preds)

# -------------------------------
# Plotting utilities
# -------------------------------
def plot_curves(history, out_png):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7,5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Curves"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_confusion(cm, class_names, out_png, normalize=True):
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# -------------------------------
# Main
# -------------------------------
def main():
    ROOT = Path(__file__).resolve().parent
    OUT_DIR = ROOT / "results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_CSV = ROOT / "ftrain.csv"
    VAL_CSV   = ROOT / "fval.csv"

    print("Reading CSVs...")
    train_loader, val_loader, class_to_idx = make_loaders(TRAIN_CSV, VAL_CSV)
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    with open(OUT_DIR / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    model = build_model(num_classes=len(class_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_acc = 0.0
    best_ckpt_path = OUT_DIR / "efficientnet_b4_best.pth"
    PATIENCE = 5
    patience_ct = 0

    for epoch in range(1, 31):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ct = 0
            torch.save({"model_state": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "epoch": epoch}, best_ckpt_path)
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save curves
    plot_curves(history, OUT_DIR / "training_curves.png")

    # Load best model
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # Final validation
    val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, DEVICE)
    print(f"Best Val Acc: {best_val_acc:.4f} | Final Val Acc: {val_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(idx_to_class))))
    plot_confusion(cm, [idx_to_class[i] for i in range(len(idx_to_class))],
                   OUT_DIR / "confusion_matrix.png")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=list(idx_to_class.values()), digits=4)
    with open(OUT_DIR / "metrics_report.txt", "w") as f:
        f.write(f"Final Val Acc: {val_acc:.4f}\n\n")
        f.write(report)
    print("Training complete. Results saved in 'results/' folder.")

if __name__ == "__main__":
    main()
