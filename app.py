# app.py (Module 6 Web UI + Save Outputs to Folder + Logging + Show Uploaded Images)
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import glob
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from typing import Tuple, List
from datetime import datetime

st.set_page_config(page_title="AI-CircuitGuard — ROI Inference UI", layout="wide")

# -------------------------
# Config
# -------------------------
DEFAULT_MODEL_PATH = "outputs_training/efficientnet_b4_best.pth"
TEMPLATES_DIR = "outputs_pairs/templates"
OUTPUT_DIR = "outputs_module6"
CLASSES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
IMG_SIZE = 128

# Ensure output folders exist
for sub in ["annotated", "rois", "predictions", "log"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

# -------------------------
# Logging utility
# -------------------------
def log_run(template_name, test_name, num_rois, labels):
    """Append run info to log file"""
    log_path = os.path.join(OUTPUT_DIR, "log", "inference_log.csv")
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "template": template_name,
        "test": test_name,
        "num_rois": num_rois,
        "labels": ";".join(labels)
    }
    df = pd.DataFrame([row])
    if not os.path.exists(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode="a", header=False, index=False)

# -------------------------
# Utilities
# -------------------------
@st.cache_resource
def load_classifier(model_path: str, device: str = "cpu"):
    device = torch.device(device)
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            sd = ckpt["model_state"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    new_state = {}
    for k, v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model, device

def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

# -------------------------
# Defect Mask Function (fixed)
# -------------------------
def make_defect_mask(template_bgr: np.ndarray,
                     test_bgr: np.ndarray,
                     morph_iter: int = 1,
                     sensitivity: float = 1.0,
                     min_area: int = 30,
                     max_area: int = 1000,
                     max_rois: int = 10) -> Tuple[np.ndarray, List[tuple]]:
    """
    Returns (binary_mask, list_of_boxes)
    - binary_mask: black background, white blobs where defects are
    - list_of_boxes: list of (x,y,w,h) for detected blobs
    """

    h, w = template_bgr.shape[:2]

    # --- Step 1: blur + grayscale subtraction ---
    g1 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    g1_blur = cv2.GaussianBlur(g1, (7,7), 0)
    g2_blur = cv2.GaussianBlur(g2, (7,7), 0)

    diff = cv2.absdiff(g1_blur, g2_blur)

    # --- Step 2: threshold using Otsu ---
    retval, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_val = max(5, int(retval * sensitivity))
    _, mask = cv2.threshold(diff, thr_val, 255, cv2.THRESH_BINARY)

    # --- Step 3: morphology cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

    # --- Step 4: connected components filtering ---
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, num_labels):
        x, y, w_box, h_box, area = stats[i]
        if min_area <= area <= max_area:
            comps.append((i, x, y, w_box, h_box, area))

    # --- Step 5: keep only top-N blobs ---
    comps = sorted(comps, key=lambda c: c[5], reverse=True)[:max_rois]

    # --- Step 6: rebuild filtered mask ---
    filtered_mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []
    for (lbl, x, y, w_box, h_box, area) in comps:
        filtered_mask[labels_im == lbl] = 255
        boxes.append((x, y, w_box, h_box))

    return filtered_mask, boxes

# -------------------------
# Classification & Annotation
# -------------------------
clf_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def classify_roi_pil(img_pil: Image.Image, model, device) -> Tuple[str, float, list]:
    x = clf_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).cpu().numpy()[0]
    top_idx = probs.argmax()
    top_conf = float(probs[top_idx])
    top3 = [(CLASSES[i], float(probs[i])) for i in probs.argsort()[-3:][::-1]]
    return CLASSES[top_idx], top_conf, top3

def annotate_bboxes_on_image(image_bgr: np.ndarray, boxes, labels, scores) -> np.ndarray:
    out = image_bgr.copy()
    for (x,y,w,h), label, score in zip(boxes, labels, scores):
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        txt = f"{label} {score:.2f}"
        ((tw, th), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, max(0,y-20)), (x+tw+6,y), (0,0,0), -1)
        cv2.putText(out, txt, (x+2,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return out

# -------------------------
# Streamlit UI
# -------------------------
st.title("AI-CircuitGuard")

st.sidebar.header("Inputs & settings")

# ---- Template selection
st.sidebar.subheader("Template Image")
template_source = st.sidebar.radio("Select template source",
                                   ["From templates folder", "Upload from computer", "Enter custom path"], index=0)

tpl_pil = None
template_name = None
if template_source == "From templates folder":
    template_files = sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*.*")))
    template_choice = st.sidebar.selectbox("Choose template", ["-- choose --"] + [os.path.basename(f) for f in template_files])
    if template_choice != "-- choose --":
        tpl_pil = Image.open(os.path.join(TEMPLATES_DIR, template_choice)).convert("RGB")
        template_name = template_choice
elif template_source == "Upload from computer":
    template_upload = st.sidebar.file_uploader("Upload template image", type=["jpg","jpeg","png"])
    if template_upload:
        tpl_pil = Image.open(template_upload).convert("RGB")
        template_name = template_upload.name
elif template_source == "Enter custom path":
    custom_path = st.sidebar.text_input("Enter template path")
    if custom_path and os.path.exists(custom_path):
        tpl_pil = Image.open(custom_path).convert("RGB")
        template_name = os.path.basename(custom_path)

# ---- Test selection
st.sidebar.subheader("Test Image")
test_source = st.sidebar.radio("Select test source", ["Upload from computer", "Enter custom path"], index=0)

test_pil = None
test_name = None
if test_source == "Upload from computer":
    test_upload = st.sidebar.file_uploader("Upload test image", type=["jpg","jpeg","png"])
    if test_upload:
        test_pil = Image.open(test_upload).convert("RGB")
        test_name = test_upload.name
elif test_source == "Enter custom path":
    custom_test_path = st.sidebar.text_input("Enter test image path")
    if custom_test_path and os.path.exists(custom_test_path):
        test_pil = Image.open(custom_test_path).convert("RGB")
        test_name = os.path.basename(custom_test_path)

# -------------------------
# Show Inputs immediately
# -------------------------
col1, col2 = st.columns(2)
with col1:
    st.header("Template Image")
    if tpl_pil:
        st.image(tpl_pil, caption="Uploaded Template", use_container_width=True)
    else:
        st.info("Please provide a Template Image.")

with col2:
    st.header("Test Image")
    if test_pil:
        st.image(test_pil, caption="Uploaded Test", use_container_width=True)
    else:
        st.info("Please provide a Test Image.")

# ---- Model settings
st.sidebar.write("Model / device")
model_path = st.sidebar.text_input("Model path", DEFAULT_MODEL_PATH)
device_opt = st.sidebar.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
morph_iter = st.sidebar.slider("Morph iterations", 0, 5, 1)
run_button = st.sidebar.button("Run pipeline")

# Load model
model_obj = None
if os.path.exists(model_path):
    try:
        model_obj, device = load_classifier(model_path, device_opt)
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
else:
    st.sidebar.warning(f"Model not found at {model_path}")

# -------------------------
# Run pipeline
# -------------------------
if run_button:
    if tpl_pil is None:
        st.error("Template missing.")
    elif test_pil is None:
        st.error("Test missing.")
    elif model_obj is None:
        st.error("Model not loaded.")
    else:
        tpl_cv = pil_to_cv2(tpl_pil)
        test_cv = pil_to_cv2(test_pil)

        st.info("Generating defect mask (black background, white blobs for defects)...")
        mask, boxes = make_defect_mask(tpl_cv, test_cv, morph_iter=morph_iter)

        st.image(mask, caption="Defect Mask", use_container_width=True)
        st.write(f"Found {len(boxes)} ROIs")

        if boxes:
            labels, scores, rois_pil, top3s = [], [], [], []
            for i, (x,y,w,h) in enumerate(boxes):
                roi_pil = cv2_to_pil(test_cv[y:y+h, x:x+w])
                rois_pil.append(roi_pil)
                lab, sc, t3 = classify_roi_pil(roi_pil, model_obj, device)
                labels.append(lab)
                scores.append(sc)
                top3s.append(t3)

                # Save ROI
                roi_path = os.path.join(OUTPUT_DIR, "rois", f"roi_{i}_{lab}.png")
                roi_pil.save(roi_path)

            # Annotated image
            annotated_cv = annotate_bboxes_on_image(test_cv, boxes, labels, scores)
            annotated_pil = cv2_to_pil(annotated_cv)
            st.image(annotated_pil, caption="Annotated Test", use_container_width=True)

            # Save annotated image
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            annotated_path = os.path.join(OUTPUT_DIR, "annotated", f"annotated_{run_id}.png")
            annotated_pil.save(annotated_path)

            # Save Predictions CSV
            rows = []
            for i, (box, lab, sc, t3) in enumerate(zip(boxes, labels, scores, top3s)):
                x,y,w,h = box
                rows.append({
                    "roi_id": i, "x": x, "y": y, "w": w, "h": h,
                    "pred_label": lab, "confidence": sc,
                    "top3": "; ".join([f"{n}:{p:.3f}" for n,p in t3])
                })
            df = pd.DataFrame(rows)
            csv_path = os.path.join(OUTPUT_DIR, "predictions", f"predictions_{run_id}.csv")
            df.to_csv(csv_path, index=False)

            # Log run
            log_run(template_name, test_name, len(labels), labels)
            st.success("Outputs saved to outputs_module6/ (annotated, rois, predictions, log)")

            # Download buttons
            st.download_button("Download ROI predictions CSV",
                               df.to_csv(index=False).encode("utf-8"),
                               "roi_predictions.csv", "text/csv")

            st.download_button("Download annotated image",
                               open(annotated_path,"rb").read(),
                               "annotated_result.png", "image/png")
