
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import io
# ==== Paths ====
BASE_FOLDER = "PCB_DATASET"
MODEL_PATH = os.path.join(BASE_FOLDER, "output/efficientnet_b4_best.pth")
ROI_BASE = os.path.join(BASE_FOLDER, "Pipeline_ROIs")
SUBTRACTED_SAVE = os.path.join(BASE_FOLDER, "subtracted_images")
os.makedirs(ROI_BASE, exist_ok=True)
os.makedirs(SUBTRACTED_SAVE, exist_ok=True)

# ==========================================
# 1️⃣ Load Model
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint['classes']
num_classes = len(classes)

model = models.efficientnet_b4(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# ==========================================
# 2️⃣ Image Subtraction
# ==========================================
def subtract_images(defect_img_path, ref_img_path, save_mask_path=None):
    defect_gray = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)
    defect_color = cv2.imread(defect_img_path)
    ref_gray = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)

    if defect_gray is None or defect_color is None:
        raise ValueError(f"❌ Could not load defect image: {defect_img_path}")
    if ref_gray is None:
        raise ValueError(f"❌ Could not load reference image: {ref_img_path}")

    # Resize reference to match defect
    if ref_gray.shape != defect_gray.shape:
        ref_gray = cv2.resize(ref_gray, (defect_gray.shape[1], defect_gray.shape[0]))

    # Blur
    defect_blur = cv2.GaussianBlur(defect_gray, (5,5), 0)
    ref_blur = cv2.GaussianBlur(ref_gray, (5,5), 0)

    # Subtract
    subtracted = cv2.absdiff(defect_blur, ref_blur)

    # Threshold
    binary = cv2.adaptiveThreshold(subtracted, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 35, -5)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)

    if save_mask_path:
        cv2.imwrite(save_mask_path, cleaned)

    return defect_color, cleaned

# ==========================================
# 3️⃣ ROI Extraction
# ==========================================
def extract_rois_and_save(orig_img, mask, save_base_folder, base_name,
                          min_area=200, min_w=10, min_h=10):
    os.makedirs(save_base_folder, exist_ok=True)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    roi_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_area or w < min_w or h < min_h:
            continue
        roi = orig_img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        roi_filename = f"{os.path.splitext(base_name)[0]}_roi{roi_count}.png"
        roi_save_path = os.path.join(save_base_folder, roi_filename)
        cv2.imwrite(roi_save_path, roi)
        results.append((roi_save_path, (x, y, x+w, y+h)))
        roi_count += 1
    return results

# ==========================================
# 4️⃣ Prediction
# ==========================================
def predict_roi(roi_path, model, class_names, transform, device):
    img = Image.open(roi_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        top3_idx = probs.argsort()[-3:][::-1]

    top3 = [(class_names[i], float(probs[i])) for i in top3_idx]
    label, conf = top3[0]
    return label, conf, top3

# ==========================================
# 5️⃣ Full Pipeline
# ==========================================
# ==========================================
# 5️⃣ Full Pipeline (Only return predictions, no ROI display)
# ==========================================
def process_images(defect_img_path, ref_img_path):
    base_name = os.path.basename(defect_img_path)
    mask_save_path = os.path.join(SUBTRACTED_SAVE, f"{os.path.splitext(base_name)[0]}_mask.png")

    defect_color, mask = subtract_images(defect_img_path, ref_img_path, save_mask_path=mask_save_path)

    roi_results = extract_rois_and_save(defect_color, mask, ROI_BASE, base_name)
    annotated_img = defect_color.copy()

    predictions = []
    for roi_path, (x1, y1, x2, y2) in roi_results:
        label, conf, _ = predict_roi(roi_path, model, classes, transform, device)
        predictions.append((label, conf))
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 🔹 Bigger font size & thickness for defect labels
        cv2.putText(annotated_img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    annotated_save_path = os.path.join(ROI_BASE, f"{os.path.splitext(base_name)[0]}_annotated.png")
    cv2.imwrite(annotated_save_path, annotated_img)

    return annotated_img, predictions, annotated_save_path

# ==========================================
# 🎛 Streamlit UI (Only annotated image)
# ==========================================
st.title("🔍 PCB Defect Detection")
st.write("Upload a **reference PCB** (golden) and a **defected PCB** to detect issues.")

col1, col2 = st.columns(2)
with col1:
    ref_file = st.file_uploader("Upload Reference PCB", type=["jpg","jpeg","png"], key="ref")
with col2:
    defect_file = st.file_uploader("Upload Defected PCB", type=["jpg","jpeg","png"], key="defect")

if ref_file and defect_file:
    ref_path = "temp_ref.png"
    defect_path = "temp_defect.png"
    with open(ref_path, "wb") as f:
        f.write(ref_file.getbuffer())
    with open(defect_path, "wb") as f:
        f.write(defect_file.getbuffer())

    st.image(ref_path, caption="Reference PCB", use_container_width=True)
    st.image(defect_path, caption="Defected PCB", use_container_width=True)

    if st.button("Run Detection"):
        annotated_img, preds, annotated_save_path = process_images(defect_path, ref_path)

        st.subheader("📌 Annotated PCB with Defects")
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        # 🔽 Download button for annotated image
        with open(annotated_save_path, "rb") as f:
            st.download_button(
                "⬇️ Download Annotated Image",
                f,
                file_name=os.path.basename(annotated_save_path),
                mime="image/png"
            )
