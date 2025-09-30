import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==========================
# Dataset class (same as training)
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
# Transform for test images
# ==========================
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
test_tfms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

# ==========================
# Load test CSV
# ==========================
ROOT_DIR = "/content/ROI_again"  # Path where your images are stored
TEST_CSV = "/content/results/Dataset/test.csv"  # Path to your test CSV

test_df = pd.read_csv(TEST_CSV)
test_dataset = PCBDataset(test_df, ROOT_DIR, transform=test_tfms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==========================
# Evaluate on test set
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()

all_targets, all_preds = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        all_targets.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# ==========================
# Metrics and Confusion Matrix
# ==========================
all_targets = all_targets
all_preds = all_preds

print("\nClassification Report:\n")
print(classification_report(all_targets, all_preds, digits=4))

cm = confusion_matrix(all_targets, all_preds)
all_labels = sorted(test_df["label"].unique())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.show()
