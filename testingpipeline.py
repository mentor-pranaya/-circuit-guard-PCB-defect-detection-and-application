import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import timm

# ==========================
# 1. CONFIGURATION
# ==========================
ROOT_DIR   = Path(r"C:\Users\phaneendra ybs\Downloads\r")            # test images folder
TEST_CSV   = Path(r"C:\Users\phaneendra ybs\Downloads\roi_data.csv")  # new unseen CSV
OUT_DIR    = Path(r"C:\Users\phaneendra ybs\Downloads\training")     # same training output
RESULTS_DIR = OUT_DIR / r"C:\Users\phaneendra ybs\Downloads\testing 25"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / r"C:\Users\phaneendra ybs\Downloads\training\efficientnet_b4_final.pth"   # best model from training

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================
# 2. DATASET CLASS
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
# 3. TRANSFORMS (same as validation)
# ==========================
def build_transforms(img_size=128):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return val_tfms

# ==========================
# 4. MODEL BUILDER
# ==========================
def build_model(num_classes, dropout=0.6):
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    return model

# ==========================
# 5. LOAD TEST DATA
# ==========================
df_test = pd.read_csv(TEST_CSV)
val_tfms = build_transforms()
test_loader = DataLoader(
    PCBDataset(df_test, ROOT_DIR, val_tfms),
    batch_size=32, shuffle=False
)

num_classes = df_test["category"].nunique()

# ==========================
# 6. LOAD TRAINED MODEL
# ==========================
model = build_model(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded from:", MODEL_PATH)

# ==========================
# 7. INFERENCE & METRICS
# ==========================
all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.numpy())

# ---- Final Evaluation ----
print("\nClassification Report:\n")
print(classification_report(all_targets, all_preds, digits=4))

cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(df_test["category"].unique()))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix_test.png")
plt.close()
print("✅ Saved confusion matrix to:", RESULTS_DIR / "confusion_matrix_test.png")

# ==========================
# 8. ANNOTATED OUTPUT IMAGES
# ==========================
print("Saving annotated test images...")
font = ImageFont.load_default()

for idx, row in df_test.iterrows():
    img_path = ROOT_DIR / row["roi_filename"]
    img = Image.open(img_path).convert("RGB")

    x = val_tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()

    draw = ImageDraw.Draw(img)
    gt = row["category"]
    text = f"Pred: {pred} | GT: {gt}"
    color = "green" if pred == gt else "red"
    draw.text((10, 10), text, fill=color, font=font)

    img.save(RESULTS_DIR / f"annotated_{idx}.jpg")

print("✅ Annotated images saved to:", RESULTS_DIR)

# ==========================
# 9. SUMMARY
# ==========================
from sklearn.metrics import accuracy_score
acc = accuracy_score(all_targets, all_preds) * 100
print(f"\nFinal Prediction Match Rate (Accuracy): {acc:.2f}%")
print("Low false positive/negative rates can be observed in the confusion matrix.")
