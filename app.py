# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import io
import time
import base64
from PIL import Image

# --------------------------
# Config / paths
# --------------------------
MODEL_FILENAME = "my_trained_model.h5"   # place your model here
ROI_CSV = "roi_labels.csv"               # present in your project (roi_filename, defect_type)
IMG_SIZE = (256, 256)                    # model input size (as used in training)
# --------------------------

# --------------------------
# Utility functions (your existing logic adapted)
# --------------------------
def preprocess_and_subtract_from_arrays(ref_arr, defect_arr, border_thickness=10):
    # assume ref_arr and defect_arr are BGR or grayscale arrays read via cv2
    if len(ref_arr.shape) == 3:
        ref_gray = cv2.cvtColor(ref_arr, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_arr.copy()
    if len(defect_arr.shape) == 3:
        defect_gray = cv2.cvtColor(defect_arr, cv2.COLOR_BGR2GRAY)
    else:
        defect_gray = defect_arr.copy()

    if defect_gray.shape != ref_gray.shape:
        defect_gray = cv2.resize(defect_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    ref_blur = cv2.GaussianBlur(ref_gray, (3, 3), 0)
    defect_blur = cv2.GaussianBlur(defect_gray, (3, 3), 0)
    diff = cv2.absdiff(defect_blur, ref_blur)
    _, defect_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    h, w = defect_mask.shape
    border_mask = np.zeros_like(defect_mask, dtype=np.uint8)
    cv2.rectangle(border_mask, (border_thickness, border_thickness),
                  (w - border_thickness, h - border_thickness), 255, -1)
    defect_mask = cv2.bitwise_and(defect_mask, border_mask)
    return defect_mask

def extract_bboxes_from_mask(mask, min_area=20, padding=3):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = mask.shape
    max_area = h * w // 2
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, ww, hh = cv2.boundingRect(cnt)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w - 1, x + ww + padding)
            y2 = min(h - 1, y + hh + padding)
            boxes.append((x1, y1, x2, y2))
    return boxes

def annotate_image_with_labels(img_bgr, predictions):
    # predictions: list of dict {box:(x1,y1,x2,y2), label:str, score:float}
    out = img_bgr.copy()
    for p in predictions:
        x1, y1, x2, y2 = p["box"]
        label = p.get("label", "Unknown")
        score = p.get("score", None)
        color = (0, 0, 255)  # red boxes
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label}" + (f" ({score:.2f})" if score is not None else "")
        # put text background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 0, 255), -1)
        cv2.putText(out, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out

def convert_bgr_to_bytes(img_bgr):
    _, im_png = cv2.imencode(".png", img_bgr)
    return im_png.tobytes()

# --------------------------
# UI decoration (small animation via CSS)
# --------------------------
st.set_page_config(page_title="CircuitGuard — Animated UI", layout="wide")
st.markdown(
    """
    <style>
    .big-title{
      font-size:34px; font-weight:700; background: linear-gradient(90deg,#00f,#0ff,#0f0,#ff0,#f0f);
      -webkit-background-clip: text; color: transparent; animation: slidebg 4s linear infinite;
    }
    @keyframes slidebg { 0%{background-position:0%} 100%{background-position:100%} }
    .center { display:flex; align-items:center; gap:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="center"><div class="big-title">🛡️ CircuitGuard — PCB Defect Detection</div></div>', unsafe_allow_html=True)
st.write("Upload a **Reference (original)** PCB and a **Defected** PCB. The app will highlight defect regions and label them using the trained model.")

# --------------------------
# Load label mapping (from roi_labels.csv)
# --------------------------
label_names = None
if os.path.exists(ROI_CSV):
    try:
        df_labels = pd.read_csv(ROI_CSV)
        unique_labels = list(df_labels['defect_type'].unique())
        # We will use sorted if you prefer deterministic order; using unique preserves dataset order
        label_names = unique_labels
    except Exception as e:
        st.warning(f"Could not read {ROI_CSV}: {e}")
else:
    st.info("roi_labels.csv not found in project folder. Labels will display as Class indices.")

# --------------------------
# Load model
# --------------------------
model = None
if os.path.exists(MODEL_FILENAME):
    try:
        with st.spinner("Loading trained model..."):
            model = load_model(MODEL_FILENAME)
        st.success("Loaded trained model.")
    except Exception as e:
        st.error(f"Failed to load model '{MODEL_FILENAME}': {e}")
        model = None
else:
    st.warning(f"Model file '{MODEL_FILENAME}' not found. Place your Keras model in the project folder to enable classification.")

# --------------------------
# Upload widgets
# --------------------------
col1, col2, col3 = st.columns([1,1,0.6])
with col1:
    ref_file = st.file_uploader("📥 Upload Reference PCB (original)", type=["jpg","png","jpeg"])
with col2:
    defect_file = st.file_uploader("📥 Upload Defected PCB", type=["jpg","png","jpeg"])
with col3:
    run_button = st.button("🔍 Run Detection", use_container_width=True)

# Extra options
min_area = st.sidebar.slider("Min defect area (px)", 10, 500, 20)
border_thickness = st.sidebar.slider("Ignore border thickness (px)", 0, 100, 10)
show_prob = st.sidebar.checkbox("Show prediction probability", True)

# --------------------------
# Run pipeline
# --------------------------
if run_button:
    if not (ref_file and defect_file):
        st.warning("Please upload both reference and defect images.")
    else:
        # Save uploaded files to temp files (cv2 needs filenames or byte arrays)
        ref_bytes = ref_file.read()
        defect_bytes = defect_file.read()
        ref_np = np.frombuffer(ref_bytes, np.uint8)
        defect_np = np.frombuffer(defect_bytes, np.uint8)
        ref_img = cv2.imdecode(ref_np, cv2.IMREAD_COLOR)
        defect_img = cv2.imdecode(defect_np, cv2.IMREAD_COLOR)
        if ref_img is None or defect_img is None:
            st.error("Failed to read uploaded images. Ensure they're valid image files.")
        else:
            # Animated progress
            progress = st.progress(0)
            status_text = st.empty()
            status_text.info("Step 1/4 — Subtraction & mask generation...")
            time.sleep(0.3)
            progress.progress(10)

            mask = preprocess_and_subtract_from_arrays(ref_img, defect_img, border_thickness=border_thickness)
            progress.progress(35)
            status_text.info("Step 2/4 — Finding defect regions (contours)...")
            time.sleep(0.3)
            boxes = extract_bboxes_from_mask(mask, min_area=min_area, padding=4)
            progress.progress(60)

            predictions = []
            if len(boxes) == 0:
                status_text.warning("No defect regions detected. Try lowering min area or border thickness.")
            else:
                status_text.info(f"Step 3/4 — Classifying {len(boxes)} ROIs (if model available)...")
                time.sleep(0.2)
                # classify each ROI if model exists
                for i, box in enumerate(boxes):
                    x1,y1,x2,y2 = box
                    roi = defect_img[y1:y2, x1:x2]
                    pred_label = None
                    pred_score = None
                    if model is not None:
                        try:
                            roi_resized = cv2.resize(roi, IMG_SIZE)
                            roi_array = img_to_array(roi_resized)
                            roi_array = np.expand_dims(roi_array, axis=0)
                            # Use VGG16 preprocess if trained with VGG16 (your training used VGG16)
                            roi_array = preprocess_input(roi_array)
                            pred = model.predict(roi_array)
                            score = float(np.max(pred))
                            class_idx = int(np.argmax(pred, axis=1)[0])
                            pred_score = score
                            if label_names:
                                # if label_names length matches classes, map; else fallback to Class index
                                if class_idx < len(label_names):
                                    pred_label = label_names[class_idx]
                                else:
                                    pred_label = f"Class_{class_idx}"
                            else:
                                pred_label = f"Class_{class_idx}"
                        except Exception as e:
                            pred_label = "ModelError"
                            pred_score = None
                    else:
                        pred_label = "NoModel"
                    predictions.append({"box": box, "label": pred_label, "score": pred_score})
                    progress.progress(60 + int((i+1)/len(boxes) * 30))

            status_text.info("Step 4/4 — Annotating image and preparing outputs...")
            annotated = annotate_image_with_labels(defect_img, predictions)
            progress.progress(100)
            status_text.success("Done ✅")

            # Display side-by-side original defect and annotated
            st.write("### Results")
            c1, c2 = st.columns([1,1])
            with c1:
                st.image(cv2.cvtColor(defect_img, cv2.COLOR_BGR2RGB), caption="Defected PCB (input)", use_column_width=True)
            with c2:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated output", use_column_width=True)

            # Create downloadable image
            img_bytes = convert_bgr_to_bytes(annotated)
            st.download_button("⬇️ Download Annotated Image", data=img_bytes, file_name="annotated_result.png", mime="image/png")

            # Create downloadable CSV of ROI predictions
            rows = []
            for i, p in enumerate(predictions):
                x1,y1,x2,y2 = p["box"]
                rows.append({
                    "roi_index": i,
                    "box": f"{x1},{y1},{x2},{y2}",
                    "label": p.get("label"),
                    "score": p.get("score")
                })
            df_out = pd.DataFrame(rows)
            csv_bytes = df_out.to_csv(index=False).encode()
            st.download_button("⬇️ Download Prediction CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

            # Show small table
            if len(rows) > 0:
                st.write("#### Detected ROIs")
                st.dataframe(df_out)

# --------------------------
# Footer / tips
# --------------------------
st.markdown("---")
st.write("Tips: If labels look wrong, ensure the model and roi_labels.csv used during training are consistent with the files in this folder.")

