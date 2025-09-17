import torch
from torch.utils.data import DataLoader

# 1. Load test dataset (use your TEST_CSV if you have one)
test_dataset = PCB_Dataset(TEST_CSV, transform=val_transform)  # use val_transform for test
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# 2. Put model in evaluation mode
model.eval()
model.to(device)

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():  # disable gradient calculation for faster inference
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
