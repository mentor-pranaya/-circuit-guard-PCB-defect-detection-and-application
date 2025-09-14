import os, json
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# import the dataset, transforms, model builder, validate function, and plot_confusion from your train file
from train_efficientnet import (
    ROIDataset, build_transforms, build_model,
    make_loaders, validate, plot_confusion,
    ROIS_ROOT, LABELS_CSV, OUT_DIR, DEVICE, BATCH_SIZE
)

# ==== Load your CSV ====
df = pd.read_csv(LABELS_CSV)

# ==== Rebuild loaders (train, val, test) exactly as before ====
train_loader, val_loader, test_loader, class_to_idx = make_loaders(df, ROIS_ROOT, BATCH_SIZE)
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# ==== Load best checkpoint ====
best_ckpt_path = os.path.join(OUT_DIR, "efficientnet_b4_best.pth")
ckpt = torch.load(best_ckpt_path, map_location=DEVICE)

model = build_model(num_classes=len(class_to_idx)).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

criterion = torch.nn.CrossEntropyLoss()

# ==== Evaluate Train, Val, Test ====
print("Evaluating on Train set...")
train_loss, train_acc, y_true_train, y_pred_train = validate(model, train_loader, criterion, DEVICE)
print(f"Train Acc: {train_acc:.4f}")

print("Evaluating on Val set...")
val_loss, val_acc, y_true_val, y_pred_val = validate(model, val_loader, criterion, DEVICE)
print(f"Val Acc: {val_acc:.4f}")

print("Evaluating on Test set...")
test_loss, test_acc, y_true_test, y_pred_test = validate(model, test_loader, criterion, DEVICE)
print(f"Test Acc: {test_acc:.4f}")

# ==== Save reports ====
# Train metrics
report_train = classification_report(y_true_train, y_pred_train, target_names=class_names, digits=4)
with open(os.path.join(OUT_DIR, "metrics_report_train1.txt"), "w") as f:
    f.write(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n\n")
    f.write(report_train)

# Val metrics
report_val = classification_report(y_true_val, y_pred_val, target_names=class_names, digits=4)
with open(os.path.join(OUT_DIR, "metrics_report_val1.txt"), "w") as f:
    f.write(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n\n")
    f.write(report_val)

# Test metrics
report_test = classification_report(y_true_test, y_pred_test, target_names=class_names, digits=4)
with open(os.path.join(OUT_DIR, "metrics_report_tes1t.txt"), "w") as f:
    f.write(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n\n")
    f.write(report_test)

# Confusion matrix for test
cm = confusion_matrix(y_true_test, y_pred_test, labels=list(range(len(class_names))))
plot_confusion(cm, class_names, os.path.join(OUT_DIR, "confusion_matrix_test1.png"), normalize=True)

print("\n✅ All metrics saved in your output folder.")