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

# =========================
# ==== Paths and Folders ====
# =========================
BASE_FOLDER = "PCB_DATASET"
MODEL_PATH = r"C:\Users\Dell\Downloads\PCB_DATASET\PCB_DATASET\output\efficientnet_b4_best.pth"
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
# 2️⃣ Image Subtraction Function
# ==========================================
def subtract_images(defect_img_path, ref_img_path, save_mask_path=None):
    defect_gray = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)
    defect_color = cv2.imread(defect_img_path)
    ref_gray = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)

    if defect_gray is None or defect_color is None:
        raise ValueError(f"❌ Could not load defect image: {defect_img_path}")
    if ref_gray is None:
        raise ValueError(f"❌ Could not load reference image: {ref_img_path}")

    if ref_gray.shape != defect_gray.shape:
        ref_gray = cv2.resize(ref_gray, (defect_gray.shape[1], defect_gray.shape[0]))

    defect_blur = cv2.GaussianBlur(defect_gray, (5,5), 0)
    ref_blur = cv2.GaussianBlur(ref_gray, (5,5), 0)

    subtracted = cv2.absdiff(defect_blur, ref_blur)

    binary = cv2.adaptiveThreshold(subtracted, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 35, -5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)

    if save_mask_path:
        cv2.imwrite(save_mask_path, cleaned)

    return defect_color, cleaned

# ==========================================
# 3️⃣ ROI Extraction Function
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
# 4️⃣ Prediction Function
# ==========================================
def predict_roi(roi_path, model, class_names, transform, device):
    img = Image.open(roi_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        top_idx = probs.argmax()

    label = class_names[top_idx]
    conf = float(probs[top_idx])
    return label, conf

# ==========================================
# 5️⃣ Full Pipeline
# ==========================================
def process_images(defect_img_path, ref_img_path):
    base_name = os.path.basename(defect_img_path)
    mask_save_path = os.path.join(SUBTRACTED_SAVE, f"{os.path.splitext(base_name)[0]}_mask.png")

    defect_color, mask = subtract_images(defect_img_path, ref_img_path, save_mask_path=mask_save_path)

    roi_results = extract_rois_and_save(defect_color, mask, ROI_BASE, base_name)
    annotated_img = defect_color.copy()

    for roi_path, (x1, y1, x2, y2) in roi_results:
        label, conf = predict_roi(roi_path, model, classes, transform, device)

        # Rectangle
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 🔹 Clearer labels (bold yellow text on black background)
        text = f"{label} ({conf:.2f})"
        font_scale = 1.2
        font_thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(annotated_img, (x1, y1 - th - baseline - 5), (x1 + tw, y1), (0, 0, 0), -1)
        cv2.putText(annotated_img, text, (x1, y1 - 7),
                    font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)

    annotated_save_path = os.path.join(ROI_BASE, f"{os.path.splitext(base_name)[0]}_annotated.png")
    cv2.imwrite(annotated_save_path, annotated_img)

    return annotated_img, annotated_save_path

# ==========================================
# 🎛 Streamlit UI
# ==========================================
st.set_page_config(page_title="PCB Defect Detection", layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"] { background-color: #f0f8ff; }
[data-testid="stSidebar"] { background-color: #e6f2ff; }
.css-ocqkz7, .st-emotion-cache-1l269bu { justify-content: center !important; }
div[data-testid="stFileUploader"] { border: 2px dashed #4a90e2; border-radius: 12px; padding: 20px; background-color: #ffffff; text-align: center; min-height: 160px; }
div[data-testid="stFileUploader"] section div div div { justify-content: center !important; }
div[data-testid="stFileUploader"] button { background-color: #4a90e2 !important; color: white !important; border-radius: 8px !important; padding: 6px 14px !important; font-size: 14px !important; font-weight: 600 !important; }
div[data-testid="stFileUploader"] button:hover { background-color: #357abd !important; }
div.stButton > button, div.stDownloadButton > button { display: block; margin: 12px auto; background-color: #4a90e2 !important; color: white !important; border-radius: 8px !important; padding: 8px 16px !important; font-size: 15px !important; font-weight: 600 !important; }
div.stButton > button:hover, div.stDownloadButton > button:hover { background-color: #357abd !important; }
h1 { color: #003366 !important; font-weight: 800; text-align: center; }
h2, h3 { color: #004080 !important; font-weight: 700; text-align: center; }
p, span, label { color: #222222 !important; font-size: 16px !important; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("🔍 PCB Defect Detection")
st.subheader("Upload a *reference PCB* (golden) and a *defected PCB* to detect issues.")

col1, col2 = st.columns([1,1], gap="large")

with col1:
    st.markdown("### 📌 Upload Reference PCB")
    ref_file = st.file_uploader("Upload Reference PCB", type=["jpg","jpeg","png"], key="ref")
    if ref_file:
        ref_path = "temp_ref.png"
        with open(ref_path, "wb") as f:
            f.write(ref_file.getbuffer())
        st.write(f"📂 File uploaded: *{ref_file.name}*")
        st.image(ref_path, caption="Reference PCB", use_container_width=True)

with col2:
    st.markdown("### ⚠ Upload Defected PCB")
    defect_file = st.file_uploader("Upload Defected PCB", type=["jpg","jpeg","png"], key="defect")
    if defect_file:
        defect_path = "temp_defect.png"
        with open(defect_path, "wb") as f:
            f.write(defect_file.getbuffer())
        st.write(f"📂 File uploaded: *{defect_file.name}*")
        st.image(defect_path, caption="Defected PCB", use_container_width=True)

# ==========================================
# 🚀 Run Detection (with PCB number check)
# ==========================================
if 'ref_file' in locals() and 'defect_file' in locals() and ref_file and defect_file:

    def get_pcb_number(filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        pcb_no = name.split("_")[0]
        return pcb_no

    ref_pcb_no = get_pcb_number(ref_file.name)
    defect_pcb_no = get_pcb_number(defect_file.name)

    if ref_pcb_no != defect_pcb_no:
        st.warning(f"⚠ PCB number mismatch! Reference: {ref_pcb_no}, Defected: {defect_pcb_no}. Detection not run.")
    else:
        if st.button("🚀 Run Detection"):
            annotated_img, annotated_save_path = process_images(defect_path, ref_path)

            st.markdown("---")
            st.subheader("📌 Annotated PCB with Defects")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            with open(annotated_save_path, "rb") as f:
                st.download_button(
                    "⬇ Download Annotated Image",
                    f,
                    file_name=os.path.basename(annotated_save_path),
                    mime="image/png"
                )
