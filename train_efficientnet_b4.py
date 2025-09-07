# train_efficientnet_b4.py
import os
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------
# Train / Eval helpers
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += imgs.size(0)
        pbar.set_postfix(loss=running_loss / running_total, acc=100.0*running_correct/running_total)

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval ", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += imgs.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            pbar.set_postfix(loss=running_loss / running_total, acc=100.0*running_correct/running_total)

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="outputs_module2/processed128", help="path to processed128 (train/val/test)")
    parser.add_argument("--model_name", type=str, default="efficientnet_b4", help="timm model name")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs_training")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")

    # -----------------------
    # Transforms
    # -----------------------
    train_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    # -----------------------
    # Datasets and loaders
    # -----------------------
    data_dir = Path(args.data_dir)
    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transforms)
    val_dataset   = datasets.ImageFolder(data_dir / "val", transform=val_test_transforms)
    test_dataset  = datasets.ImageFolder(data_dir / "test", transform=val_test_transforms)

    num_classes = len(train_dataset.classes)
    print(f"📂 Found classes: {train_dataset.classes}")
    print(f"✅ Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # -----------------------
    # Model
    # -----------------------
    print(f"⚡ Initializing {args.model_name} ...")
    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # -----------------------
    # Training loop
    # -----------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = {"epoch":[],"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}

    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"📉 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"📊 Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc
        }, out_dir / "efficientnet_b4_last.pth")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "efficientnet_b4_best.pth")
            print("✅ New best model saved.")

    # -----------------------
    # Save training history (csv) and plot
    # -----------------------
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)

    plt.figure(figsize=(8,5))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig(out_dir / "training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history["epoch"], history["train_acc"], label="train_acc")
    plt.plot(history["epoch"], history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(out_dir / "training_acc.png", dpi=150)
    plt.close()

    # -----------------------
    # Final evaluation on test set (load best model)
    # -----------------------
    print("\n📌 Evaluating on test set with best model...")
    best_state = torch.load(out_dir / "efficientnet_b4_best.pth", map_location=device)
    model.load_state_dict(best_state)
    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    print(f"✅ Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    fig, ax = plt.subplots(figsize=(10,8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Save metrics summary
    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_val_acc": best_val_acc
    }
    pd.Series(metrics).to_csv(out_dir / "metrics_summary.csv")

    # Save final predictions CSV
    preds_df = pd.DataFrame({"label_idx": all_labels, "pred_idx": all_preds})
    preds_df["label_name"] = preds_df["label_idx"].apply(lambda x: train_dataset.classes[x])
    preds_df["pred_name"] = preds_df["pred_idx"].apply(lambda x: train_dataset.classes[x])
    preds_df.to_csv(out_dir / "predictions_test.csv", index=False)

    print("🎉 Training & evaluation complete. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
