import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
# Custom Dataset class
class RoiCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, roi_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.roi_dir = roi_dir
        self.transform = transform
        self.classes = sorted(self.data['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label_name = self.data.iloc[idx, 1]
        label_idx = self.class_to_idx[label_name]
        img_path = os.path.join(self.roi_dir, label_name, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx
# Paths
roi_dir = '/content/drive/MyDrive/CircuitGuard-PCB/data/roi_images'
test_csv = '/content/drive/MyDrive/CircuitGuard-PCB/exports/test_roi_labels.csv'
model_path = '/content/drive/MyDrive/CircuitGuard-PCB/exports/efficientnet_b4_best.pth'

# Transforms
test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
# Load test data
test_dataset = RoiCSVDataset(test_csv, roi_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Load model
num_classes = len(test_dataset.classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b4(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"Test Accuracy: {test_acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
plt.title("Test Confusion Matrix")
plt.show()

# Detailed classification report
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
