import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==========================
# Dataset
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
        img_path = self.root_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================
# Transforms
# ==========================
def build_transforms(img_size=128):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return test_tfms

# ==========================
# Model
# ==========================
def build_model(num_classes, dropout=0.6):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model

# ==========================
# Main
# ==========================
def main():
    ROOT_DIR = "/content/ROI_again"
    CSV_DIR = Path("/content/results/Dataset")
    MODEL_PATH = "/content/results/efficientnet_b4_best.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load test CSV
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    test_tfms = build_transforms(img_size=128)
    test_dataset = PCBDataset(test_df, ROOT_DIR, transform=test_tfms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = test_df["label"].nunique()
    model = build_model(num_classes).to(device)

    # Load trained weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(" Model loaded successfully")

    # Evaluate
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Classification report
    print("\nClassification Report:\n")
    print(classification_report(all_targets, all_preds, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    class_names = sorted(test_df["label"].unique())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.show()

if __name__ == "__main__":
    main()
