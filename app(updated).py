import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from skimage.metrics import structural_similarity as ssim
import streamlit as st

# =============================
# 🔧 Folder Configuration
# =============================
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "PCB_DATASET")

MODEL_FILE = os.path.join(DATA_DIR, "output", "efficientnet_b4_best.pth")
GOLDEN_DIR = os.path.join(DATA_DIR, "PCB_USED")
ROI_DIR = os.path.join(DATA_DIR, "Pipeline_ROIs")
MASK_DIR = os.path.join(DATA_DIR, "subtracted_images")

for d in [ROI_DIR, MASK_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================
# 🧠 Load Trained Model
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
class_labels = checkpoint["classes"]

net = models.efficientnet_b4(pretrained=False)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(class_labels))
net.load_state_dict(checkpoint["model_state_dict"])
net = net.to(DEVICE).eval()

image_preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# =============================
# 🔍 Helper Functions
# =============================

def match_reference(defected_path):
    """Find the most similar golden PCB using SSIM."""
    defect_gray = cv2.imread(defected_path, cv2.IMREAD_GRAYSCALE)
    best_ref, highest_ssim = None, -1

    for ref_file in os.listdir(GOLDEN_DIR):
        ref_path = os.path.join(GOLDEN_DIR, ref_file)
        ref_gray = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref_gray is None:
            continue

        ref_gray = cv2.resize(ref_gray, (defect_gray.shape[1], defect_gray.shape[0]))
        score, _ = ssim(defect_gray, ref_gray, full=True)
        if score > highest_ssim:
            best_ref, highest_ssim = ref_path, score

    return best_ref, highest_ssim


def subtract_images(defected_path, reference_path, save_path=None):
    """Perform subtraction and generate a binary mask."""
    def_gray = cv2.imread(defected_path, cv2.IMREAD_GRAYSCALE)
    ref_gray = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    def_color = cv2.imread(defected_path)

    if def_gray is None or ref_gray is None:
        raise ValueError("❌ Unable to read one of the images")

    if def_gray.shape != ref_gray.shape:
        ref_gray = cv2.resize(ref_gray, (def_gray.shape[1], def_gray.shape[0]))

    diff = cv2.absdiff(
        cv2.GaussianBlur(def_gray, (5, 5), 0),
        cv2.GaussianBlur(ref_gray, (5, 5), 0)
    )

    thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 35, -5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)

    if save_path:
        cv2.imwrite(save_path, cleaned)

    return def_color, cleaned


def extract_rois(image, mask, save_folder, base_name):
    """Extract ROIs from mask and save."""
    os.makedirs(save_folder, exist_ok=True)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10 or cv2.contourArea(c) < 200:
            continue

        roi = image[y:y+h, x:x+w]
        roi_file = os.path.join(save_folder, f"{os.path.splitext(base_name)[0]}_roi{i}.png")
        cv2.imwrite(roi_file, roi)
        results.append((roi_file, (x, y, x+w, y+h)))
    return results


def classify_image(img_path):
    """Predict defect type for a given ROI."""
    img = Image.open(img_path).convert("RGB")
    tensor = image_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = net(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    idx = probs.argmax()
    label = class_labels[idx]
    confidence = float(probs[idx])
    return label, confidence


# =============================
# 🧩 Complete Processing
# =============================
def run_pipeline(defected_path):
    file_name = os.path.basename(defected_path)

    ref_path, score = match_reference(defected_path)
    if ref_path is None:
        raise ValueError("⚠️ No matching reference PCB found.")

    st.write(f"✅ Reference: **{os.path.basename(ref_path)}** (SSIM: {score:.4f})")

    mask_output = os.path.join(MASK_DIR, f"{os.path.splitext(file_name)[0]}_mask.png")
    color_img, mask = subtract_images(defected_path, ref_path, save_path=mask_output)

    rois = extract_rois(color_img, mask, ROI_DIR, file_name)
    annotated = color_img.copy()

    predictions = []
    for roi_file, (x1, y1, x2, y2) in rois:
        label, conf = classify_image(roi_file)
        predictions.append((label, conf))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    annotated_file = os.path.join(ROI_DIR, f"{os.path.splitext(file_name)[0]}_annotated.png")
    cv2.imwrite(annotated_file, annotated)
    return annotated, predictions, annotated_file

# =============================
# 🎨 Streamlit UI
# =============================
st.set_page_config(page_title="PCB Defect Detector", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #eef7ff; }
[data-testid="stSidebar"] { background-color: #d9ecff; }
h1 { text-align: center; color: #003366; font-weight: 800; }
.stButton>button { background-color: #4a90e2; color: white; font-weight: 600; border-radius: 8px; }
.stButton>button:hover { background-color: #357abd; }
</style>
""", unsafe_allow_html=True)

st.title("🔎 PCB Defect Detection System")
st.write("Upload a **defected PCB** to automatically detect and classify faulty regions.")

uploaded = st.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png"])

if uploaded:
    temp_file = os.path.join(DATA_DIR, "temp_upload.png")
    with open(temp_file, "wb") as f:
        f.write(uploaded.getbuffer())

    st.image(temp_file, caption="Uploaded PCB", use_container_width=True)

    if st.button("Run Detection"):
        annotated_img, results, saved_path = run_pipeline(temp_file)

        st.subheader("🖼 Annotated Result")
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with open(saved_path, "rb") as f:
            st.download_button("⬇️ Download Annotated Image", f, file_name=os.path.basename(saved_path), mime="image/png")
