# app.py (Module 5 Web UI with flexible template/test selection)
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import os
import glob
import pandas as pd
import torch


import torch.nn.functional as F
import timm
from torchvision import transforms
from typing import Tuple, List

st.set_page_config(page_title="AI-CircuitGuard — ROI Inference UI", layout="wide")

# -------------------------
# Config
# -------------------------
DEFAULT_MODEL_PATH = "outputs_training/efficientnet_b4_best.pth"
TEMPLATES_DIR = "outputs_pairs/templates"
CLASSES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
IMG_SIZE = 128  # classifier input size

# -------------------------
# Utilities
# -------------------------
@st.cache_resource
def load_classifier(model_path: str, device: str = "cpu"):
    device = torch.device(device)
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(model_path, map_location=device)

    # handle checkpoint keys
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
    if isinstance(sd, dict):
        for k, v in sd.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            new_state[nk] = v
    else:
        new_state = sd

    try:
        model.load_state_dict(new_state)
    except Exception:
        model.load_state_dict(sd)

    model.to(device)
    model.eval()
    return model, device

def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def align_images_orb(template_bgr: np.ndarray, test_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    try:
        gray1 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
            return test_bgr, False

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 8:
            return test_bgr, False

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return test_bgr, False

        h, w = template_bgr.shape[:2]
        aligned = cv2.warpPerspective(test_bgr, H, (w, h), flags=cv2.INTER_LINEAR)
        return aligned, True
    except Exception:
        return test_bgr, False

def subtract_and_mask(template_bgr: np.ndarray, test_aligned_bgr: np.ndarray, morph_iter: int = 1) -> np.ndarray:
    g1 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(test_aligned_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    diff = cv2.GaussianBlur(diff, (5,5), 0)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    return mask

def extract_bboxes_from_mask(mask: np.ndarray, min_area: int = 50) -> List[tuple]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h >= min_area:
            boxes.append((x,y,w,h))
    return sorted(boxes, key=lambda b:(b[1], b[0]))

clf_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def classify_roi_pil(img_pil: Image.Image, model, device) -> Tuple[str, float, list]:
    x = clf_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = probs.argmax()
    top_conf = float(probs[top_idx])
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(CLASSES[int(i)], float(probs[int(i)])) for i in top3_idx]
    return CLASSES[int(top_idx)], top_conf, top3

def annotate_bboxes_on_image(image_bgr: np.ndarray, boxes, labels, scores) -> np.ndarray:
    out = image_bgr.copy()
    for (x,y,w,h), label, score in zip(boxes, labels, scores):
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        txt = f"{label} {score:.2f}"
        ((tw, th), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, max(0, y-20)), (x+tw+6, y), (0,0,0), -1)
        cv2.putText(out, txt, (x+2, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return out

# -------------------------
# Streamlit UI
# -------------------------
st.title("AI-CircuitGuard")

st.sidebar.header("Inputs & settings")

# ---- Template selection ----
st.sidebar.subheader("Template Image")
template_source = st.sidebar.radio(
    "Select template source",
    ["From templates folder", "Upload from computer", "Enter custom path"],
    index=0
)

tpl_pil = None
if template_source == "From templates folder":
    template_files = sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*.*")))
    template_choice = st.sidebar.selectbox("Choose template", ["-- choose --"] + [os.path.basename(f) for f in template_files])
    if template_choice != "-- choose --":
        tpl_pil = Image.open(os.path.join(TEMPLATES_DIR, template_choice)).convert("RGB")

elif template_source == "Upload from computer":
    template_upload = st.sidebar.file_uploader("Upload template image", type=["jpg","jpeg","png"])
    if template_upload:
        tpl_pil = Image.open(template_upload).convert("RGB")

elif template_source == "Enter custom path":
    custom_path = st.sidebar.text_input("Enter template path")
    if custom_path and os.path.exists(custom_path):
        tpl_pil = Image.open(custom_path).convert("RGB")

# ---- Test selection ----
st.sidebar.subheader("Test Image")
test_source = st.sidebar.radio(
    "Select test source",
    ["Upload from computer", "Enter custom path"],
    index=0
)

test_pil = None
if test_source == "Upload from computer":
    test_upload = st.sidebar.file_uploader("Upload test image", type=["jpg","jpeg","png"])
    if test_upload:
        test_pil = Image.open(test_upload).convert("RGB")
elif test_source == "Enter custom path":
    custom_test_path = st.sidebar.text_input("Enter test image path")
    if custom_test_path and os.path.exists(custom_test_path):
        test_pil = Image.open(custom_test_path).convert("RGB")

# ---- Model settings ----
st.sidebar.write("Model / device")
model_path = st.sidebar.text_input("Model path", DEFAULT_MODEL_PATH)
device_opt = st.sidebar.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
min_area = st.sidebar.slider("Min ROI area (pixels)", 20, 2000, 80)
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
# Show Inputs
# -------------------------
col1, col2 = st.columns(2)
with col1:
    st.header("Template Image")
    if tpl_pil:
        st.image(tpl_pil, caption="Template Image", use_container_width=True)
    else:
        st.info("Please provide a Template Image.")

with col2:
    st.header("Test Image")
    if test_pil:
        st.image(test_pil, caption="Test Image", use_container_width=True)
    else:
        st.info("Please provide a Test Image.")

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

        st.info("Aligning test image to template...")
        aligned_cv, ok_align = align_images_orb(tpl_cv, test_cv)
        st.image(cv2_to_pil(aligned_cv), caption="Aligned Test", use_container_width=True)

        st.info("Running subtraction + masking...")
        mask = subtract_and_mask(tpl_cv, aligned_cv, morph_iter=morph_iter)
        st.image(mask, caption="Defect Mask", use_container_width=True)

        boxes = extract_bboxes_from_mask(mask, min_area=min_area)
        st.write(f"Found {len(boxes)} ROIs")

        if boxes:
            labels, scores, rois_pil, top3s = [], [], [], []
            for (x,y,w,h) in boxes:
                roi_pil = cv2_to_pil(aligned_cv[y:y+h, x:x+w])
                rois_pil.append(roi_pil)
                lab, sc, t3 = classify_roi_pil(roi_pil, model_obj, device)
                labels.append(lab)
                scores.append(sc)
                top3s.append(t3)

            annotated_cv = annotate_bboxes_on_image(aligned_cv, boxes, labels, scores)
            st.image(cv2_to_pil(annotated_cv), caption="Annotated", use_container_width=True)

            st.subheader("Cropped ROIs")
            cols = st.columns(4)
            for i, roi in enumerate(rois_pil):
                with cols[i % 4]:
                    st.image(roi, caption=f"{labels[i]} {scores[i]:.2f}", use_container_width=True)

                        # Download CSV
            rows = []
            for i, (box, lab, sc, t3) in enumerate(zip(boxes, labels, scores, top3s)):
                x,y,w,h = box
                rows.append({
                    "roi_id": i,
                    "x": x, "y": y, "w": w, "h": h,
                    "pred_label": lab,
                    "confidence": sc,
                    "top3": ";".join([f"{n}:{p:.3f}" for n,p in t3])
                })
            df = pd.DataFrame(rows)
            st.download_button(
                "Download ROI predictions CSV",
                df.to_csv(index=False).encode("utf-8"),
                "roi_predictions.csv",
                "text/csv"
            )

            # Download annotated image
            annotated_pil = cv2_to_pil(annotated_cv)
            annotated_buf = io.BytesIO()
            annotated_pil.save(annotated_buf, format="PNG")
            st.download_button(
                "Download annotated image",
                annotated_buf.getvalue(),
                "annotated_result.png",
                "image/png"
            )


