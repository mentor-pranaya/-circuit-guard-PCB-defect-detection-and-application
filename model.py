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


# =====================================================
# SEED
# =====================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# =====================================================
# PATHS (CHANGE THESE TO YOUR DIRECTORIES)
# =====================================================
ROIS_ROOT = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\data\roi_images"
LABELS_CSV = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\data\roi_labels.csv"
OUT_DIR    = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\checkpoints"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# =====================================================
# SETTINGS
# =====================================================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 0  # safer for Windows


# =====================================================
# DATASET
# =====================================================
class ROIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, rois_root: str, transform=None, class_to_idx=None):
        self.rois_root = Path(rois_root)
        self.transform = transform

        df = df.copy().reset_index(drop=True)

        if class_to_idx is None:
            classes = sorted(df["label"].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        abs_paths, labels = [], []
        for _, row in df.iterrows():
            fname = str(row["filename"])
            label = str(row["label"])
            path = str(self.rois_root / label / fname)
            if os.path.exists(path):
                abs_paths.append(path)
                labels.append(label)
            else:
                print(f"[WARN] Missing file skipped: {path}")

        self.df = pd.DataFrame({"abs_path": abs_paths, "label": labels}).reset_index(drop=True)
        print(f"[DATA] Dataset initialized with {len(self.df)} samples across {len(self.class_to_idx)} classes.")

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


# =====================================================
# UTILS
# =====================================================
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
    print("\n[INFO] Splitting dataset into train/val/test...")
    trainval_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.176, random_state=42, stratify=trainval_df["label"]
    )

    print(f"[DATA] Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

    train_tfms, val_tfms = build_transforms(IMG_SIZE)

    train_ds = ROIDataset(train_df, rois_root, transform=train_tfms, class_to_idx=class_to_idx)
    val_ds   = ROIDataset(val_df,   rois_root, transform=val_tfms,   class_to_idx=train_ds.class_to_idx)
    test_ds  = ROIDataset(test_df,  rois_root, transform=val_tfms,   class_to_idx=train_ds.class_to_idx)

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

    print("[INFO] DataLoaders ready!")
    return train_loader, val_loader, test_loader, train_ds.class_to_idx


def build_model(num_classes):
    print(f"[INFO] Building EfficientNet-B4 model with {num_classes} output classes...")
    return timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)


# =====================================================
# TRAINING
# =====================================================
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

        if i % 10 == 0:
            print(f"   [TRAIN] Epoch {epoch}/{total_epochs} | Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, phase="VAL"):
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

    acc = correct / total
    print(f"   [{phase}] Loss: {running_loss/total:.4f} | Acc: {acc:.4f}")
    return running_loss / total, acc, np.array(all_targets), np.array(all_preds)


# =====================================================
# MAIN
# =====================================================
def main():
    print("[START] Training pipeline initiated...\n")
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] ROI Root: {ROIS_ROOT}")
    print(f"[INFO] Labels CSV: {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)
    assert {"filename","label"}.issubset(df.columns), "roi_labels.csv must have columns: filename,label"
    print(f"[DATA] Loaded {len(df)} ROI entries from CSV.")

    train_loader, val_loader, test_loader, class_to_idx = make_loaders(df, ROIS_ROOT, BATCH_SIZE)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with open(os.path.join(OUT_DIR, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print("[SAVE] Class mapping saved to class_to_idx.json")

    model = build_model(num_classes=len(class_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_epoch = -1
    patience_ct = 0
    best_ckpt_path = os.path.join(OUT_DIR, "efficientnet_b4_best.pth")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[INFO] Epoch {epoch}/{EPOCHS} started...")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE, phase="VAL")
        scheduler.step()

        if val_acc > best_val_acc:
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "epoch": epoch,
                "val_acc": val_acc
            }, best_ckpt_path)
            best_val_acc = val_acc
            best_epoch = epoch
            patience_ct = 0
            print(f"[SAVE] New best model saved at epoch {epoch} with acc {val_acc:.4f}")
        else:
            patience_ct += 1
            print(f"[INFO] No improvement. Patience counter: {patience_ct}/{PATIENCE}")
            if patience_ct >= PATIENCE:
                print("[EARLY STOP] Training stopped due to no improvement.")
                break

    print(f"\n[DONE] Training completed. Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}")
    print(f"[MODEL] Best checkpoint saved at {best_ckpt_path}")


if __name__ == "__main__":
    main()
