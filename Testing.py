import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
model_path = r"C:\Users\Dell\Downloads\PCB_DATASET\best_efficientnet.pth"   # trained model
test_images_path = r"C:\Users\Dell\Downloads\PCB_DATASET\images"            # defect test images
pcb_ref_path = r"C:\Users\Dell\Downloads\PCB_DATASET\PCB_USED"              # reference PCBs
output_path = r"C:\Users\Dell\Downloads\PCB_DATASET\Annotated_Results"      # annotated results

os.makedirs(output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL LOADING
# =========================
labels = sorted(os.listdir(test_images_path))  # defect class names
num_classes = len(labels)

model = EfficientNet.from_name("efficientnet-b4", num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================
# HELPERS
# =========================
def preprocess_and_predict(img_roi):
    """Preprocess ROI and predict label index."""
    from PIL import Image
    roi_pil = Image.fromarray(cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB))
    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(roi_tensor)
        _, pred = torch.max(outputs, 1)
    return pred.item()

# =========================
# TESTING LOOP
# =========================
all_preds, all_labels = [], []

for defect_class in labels:
    class_folder = os.path.join(test_images_path, defect_class)
    ref_img_path = os.path.join(pcb_ref_path, f"{defect_class}.jpg")
    
    if not os.path.exists(ref_img_path):
        print(f"⚠️ Reference image missing for {defect_class}, skipping...")
        continue

    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)

    # output folder per class
    save_class_folder = os.path.join(output_path, defect_class)
    os.makedirs(save_class_folder, exist_ok=True)

    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        test_img = cv2.imread(img_path)
        if test_img is None:
            continue

        gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        # subtraction
        diff = cv2.absdiff(gray_test, ref_img)
        _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

        # contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = test_img.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h < 100:  # ignore small noise
                continue
            roi = test_img[y:y+h, x:x+w]
            pred_idx = preprocess_and_predict(roi)
            pred_label = labels[pred_idx]

            # draw bounding box and label
            cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(annotated, pred_label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            all_preds.append(pred_label)
            all_labels.append(defect_class)

        # save annotated image in its class folder
        save_path = os.path.join(save_class_folder, f"{os.path.splitext(img_name)[0]}_annotated.jpg")
        cv2.imwrite(save_path, annotated)

# =========================
# EVALUATION
# =========================
report = classification_report(all_labels, all_preds, target_names=labels, zero_division=0)
cm = confusion_matrix(all_labels, all_preds, labels=labels)

# Save evaluation report
report_path = os.path.join(output_path, "evaluation_report.txt")
with open(report_path, "w") as f:
    f.write("====== PCB Defect Detection Evaluation Report ======\n\n")
    f.write(report + "\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("✅ Testing complete.")
print(f"Annotated images saved in: {output_path}")
print(f"Evaluation report saved at: {report_path}")
