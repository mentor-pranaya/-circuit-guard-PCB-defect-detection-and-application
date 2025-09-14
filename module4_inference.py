# module4_inference.py
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -----------------------
# Dataset wrapper
# -----------------------
class TestROIDataset(Dataset):
    def __init__(self, root, transform):
        # use ImageFolder to get class order
        self.root = root
        self.transform = transform
        base = datasets.ImageFolder(root)  # just to read class order and samples
        self.classes = base.classes
        self.samples = base.samples  # list of (path, class_idx)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)
        return img_t, label, path

# -----------------------
# Utilities
# -----------------------
def load_model(model_path, num_classes, device, model_name="efficientnet_b4"):
    # create model architecture
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    sd = torch.load(model_path, map_location=device)
    # handle saved statecases
    if isinstance(sd, dict):
        # if saved as full checkpoint with "model_state" key
        if "model_state" in sd:
            sd = sd["model_state"]
        # if user accidentally saved optimizer etc, try to find keys for model state
    try:
        model.load_state_dict(sd)
    except Exception as e:
        # try to be permissive: if state dict contains "state_dict" key
        if "state_dict" in sd:
            model.load_state_dict(sd["state_dict"])
        else:
            raise e
    model.to(device)
    model.eval()
    return model

def annotate_and_save(image_path, out_path, text):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)  # works on newer Pillow
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([(0, 0), (tw + 8, th + 8)], fill=(0, 0, 0))
        draw.text((4, 4), text, fill=(255, 255, 255), font=font)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
        print(f"✅ Annotated saved: {out_path}")
    except Exception as e:
        print(f"⚠️ Failed to annotate {image_path}: {e}")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="outputs_module2/processed128/test", help="test folder (class subfolders)")
    parser.add_argument("--model", type=str, default="outputs_training/efficientnet_b4_best.pth", help="path to model .pth")
    parser.add_argument("--output_dir", type=str, default="outputs_module4", help="where outputs go")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # prepare transforms (same normalization as training)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    # dataset
    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Test folder not found: {args.data_dir}")
    dataset = TestROIDataset(args.data_dir, transform=transform)
    classes = dataset.classes
    print("Classes detected:", classes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # model
    device = torch.device(args.device)
    model = load_model(args.model, num_classes=len(classes), device=device)
    print("Model loaded to", device)

    # prepare outputs
    os.makedirs(args.output_dir, exist_ok=True)
    ann_dir = os.path.join(args.output_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)

    results = []
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels, paths in tqdm(loader, desc="Inference"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)
            for pth, gt, pr, cf in zip(paths, labels.numpy(), preds, confs):
                true_label = classes[int(gt)]
                pred_label = classes[int(pr)]
                results.append([os.path.basename(pth), true_label, pred_label, float(cf)])
                y_true.append(true_label)
                y_pred.append(pred_label)
                # annotate original image (not resized)
                txt = f"{pred_label} {cf:.2f}"
                out_img_path = os.path.join(ann_dir, os.path.basename(pth))
                annotate_and_save(pth, out_img_path, txt)

    # save CSV
    df = pd.DataFrame(results, columns=["filename","true_label","pred_label","confidence"])
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)
    print("Saved predictions CSV:", csv_path)

    # metrics
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open(os.path.join(args.output_dir, "metrics_report.txt"), "w") as f:
        f.write(report)
    print("\nClassification report:\n", report)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved confusion matrix:", cm_path)

    print("All outputs saved in:", args.output_dir)

if __name__ == "__main__":
    main()
