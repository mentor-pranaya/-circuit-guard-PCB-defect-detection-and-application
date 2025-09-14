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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


THIS_DIR = os.path.dirname(__file__)
ROIS_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "output", "rois"))
LABELS_CSV = os.path.abspath(os.path.join(THIS_DIR, "..", "output", "roi_labels.csv"))
OUT_DIR    = os.path.abspath(os.path.join(THIS_DIR, "checkpoints"))
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 4  # early stopping

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 0  # safer on Windows/CPU


class ROIDataset(Dataset):

    def __init__(self, df: pd.DataFrame, rois_root: str, transform=None, class_to_idx=None):
        self.rois_root = Path(rois_root)
        self.transform = transform

        df = df.copy().reset_index(drop=True)

        # build class mapping if not provided
        if class_to_idx is None:
            classes = sorted(df["label"].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        # resolve absolute image paths
        abs_paths, labels = [], []
        for _, row in df.iterrows():
            fname = str(row["filename"])
            label = str(row["label"])
            if os.path.isabs(fname) and os.path.exists(fname):
                path = fname
            else:
                path = str(self.rois_root / label / fname)
            if os.path.exists(path):
                abs_paths.append(path)
                labels.append(label)

        self.df = pd.DataFrame({"abs_path": abs_paths, "label": labels}).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["abs_path"]
        label = row["label"]
        y = self.class_to_idx[label]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y

# -------------------
# Utils
# -------------------
def build_transforms(img_size=IMG_SIZE):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.10, contrast=0.10),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tfms, val_tfms

def make_loaders(df, rois_root, batch_size=BATCH_SIZE, class_to_idx=None):
    """
    Split df into train/val/test. 
    Returns train_loader, val_loader, test_loader, class_to_idx
    """
    # first split train+val vs test
    trainval_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    # now split train vs val
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.176, random_state=42, stratify=trainval_df["label"]
    )  # 0.176 of 85% ≈ 15% val

    train_tfms, val_tfms = build_transforms(IMG_SIZE)

    train_ds = ROIDataset(train_df, rois_root, transform=train_tfms, class_to_idx=class_to_idx)
    val_ds   = ROIDataset(val_df,   rois_root, transform=val_tfms,   class_to_idx=train_ds.class_to_idx)
    test_ds  = ROIDataset(test_df,  rois_root, transform=val_tfms,   class_to_idx=train_ds.class_to_idx)

    # imbalance handling for train only
    class_counts = train_df["label"].value_counts().to_dict()
    weights = train_df["label"].map(lambda c: 1.0 / class_counts[c]).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx

def build_model(num_classes):
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
    return model

# -------------------
# Train / Validate
# -------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch=None, total_epochs=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
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

        # 👇 DEBUG PRINT
        if i % 10 == 0:  # print every 10 batches
            print(f"[Epoch {epoch}/{total_epochs}] Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")

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
        all_targets.extend(y.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
    return running_loss / total, correct / total, np.array(all_targets), np.array(all_preds)

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

def main():
    print(f"Using device: {DEVICE}")
    print("ROIS_ROOT:", ROIS_ROOT)
    print("LABELS_CSV:", LABELS_CSV)

    df = pd.read_csv(LABELS_CSV)
    assert {"filename","label"}.issubset(df.columns), "roi_labels.csv must have columns: filename,label"

    # build loaders
    train_loader, val_loader, test_loader, class_to_idx = make_loaders(df, ROIS_ROOT, BATCH_SIZE)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with open(os.path.join(OUT_DIR, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    model = build_model(num_classes=len(class_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_epoch = -1
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    patience_ct = 0
    best_ckpt_path = os.path.join(OUT_DIR, "efficientnet_b4_best.pth")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_ct = 0
            torch.save({"model_state": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "epoch": epoch}, best_ckpt_path)
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch} acc={best_val_acc:.4f})")
                break

    plot_curves(history, os.path.join(OUT_DIR, "training_curves.png"))

    # reload best model for final evaluation
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # final validation metrics
    val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, DEVICE)
    print(f"Best Val Acc: {best_val_acc:.4f} (epoch {best_epoch}) | Final Val Acc: {val_acc:.4f}")

    # test metrics 🔥
    test_loss, test_acc, y_true_test, y_pred_test = validate(model, test_loader, criterion, DEVICE)
    print(f"\n🔥 TEST Accuracy: {test_acc:.4f}")

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print("\nClassification Report (Test):")
    print(classification_report(y_true_test, y_pred_test, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true_test, y_pred_test, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, os.path.join(OUT_DIR, "confusion_matrix_test.png"), normalize=True)

    # also save a text report
    report = classification_report(y_true_test, y_pred_test, target_names=class_names, digits=4)
    with open(os.path.join(OUT_DIR, "metrics_report_test.txt"), "w") as f:
        f.write(f"Test Acc: {test_acc:.4f}\n\n")
        f.write(report)

if __name__ == "__main__":
    main()