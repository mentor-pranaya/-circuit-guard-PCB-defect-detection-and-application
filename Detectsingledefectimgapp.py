import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st
import numpy as np
import io
import time
import re

# ==========================================
# 1️⃣ Load Model and Configuration
# ==========================================
classes = ["missing_hole", "mousebite", "open_circuit", "short_circuit", "spurious_copper", "short"]

def predict_roi(roi_img, model, class_names):
    # Placeholder: replace with your actual model inference
    label = np.random.choice(class_names)
    conf = np.random.uniform(0.7, 0.99)
    return label, conf

# ==========================================
# 2️⃣ ROI Extraction Function (with subtraction)
# ==========================================
def extract_rois_from_diff(diff_img, min_area=200, min_w=10, min_h=10):
    gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_area or w < min_w or h < min_h:
            continue
        roi = diff_img[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        results.append({'roi_img': roi, 'bbox': (x, y, x + w, y + h)})
    return results

def get_reference_filename(defect_filename):
    # Extract the leading number (e.g., '04' from '04_mouse_bite_02.jpg')
    match = re.match(r"(\d+)", defect_filename)
    if match:
        ref_num = match.group(1)
        return f"{ref_num}.jpg"
    else:
        raise ValueError("Could not extract reference number from filename.")

# ==========================================
# 3️⃣ Full Pipeline
# ==========================================
def process_defect_image_with_reference(defect_bytes, defect_filename, reference_folder=r"C:\Users\phaneendra ybs\Downloads\PCB_DATASET\PCB_DATASET\PCB_USED"):
    # Decode defected image
    defect_np = np.frombuffer(defect_bytes, np.uint8)
    defect_color = cv2.imdecode(defect_np, cv2.IMREAD_COLOR)
    if defect_color is None:
        raise ValueError("❌ Could not load image from uploaded file.")

    # Get reference filename by extracting number
    reference_filename = get_reference_filename(defect_filename)
    reference_path = os.path.join(reference_folder, reference_filename)
    if not os.path.exists(reference_path):
        raise ValueError(f"Reference image not found: {reference_path}")

    reference_color = cv2.imread(reference_path)
    if reference_color is None:
        raise ValueError("❌ Could not load reference image.")

    # Resize reference to match defect image if needed
    if defect_color.shape != reference_color.shape:
        reference_color = cv2.resize(reference_color, (defect_color.shape[1], defect_color.shape[0]))

    # Subtract reference from defect image
    diff_img = cv2.absdiff(defect_color, reference_color)

    roi_results = extract_rois_from_diff(diff_img)
    annotated_img = defect_color.copy()
    detected_defects = []
    for roi in roi_results:
        label, conf = predict_roi(roi['roi_img'], None, classes)
        x1, y1, x2, y2 = roi['bbox']
        detected_defects.append({"label": label, "box": (x1, y1, x2, y2), "confidence": conf})
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        text = f"{label} ({conf:.2f})"
        font_scale = 1.2
        font_thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(annotated_img, (x1, y1 - th - baseline - 5), (x1 + tw, y1), (0, 0, 0), -1)
        cv2.putText(annotated_img, text, (x1, y1 - 7),
                    font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
    return annotated_img, detected_defects

def main():
    st.set_page_config(page_title="AI Circuit Guard - PCB Defect Detection and Classification", layout="wide")
    st.title("AI Circuit Guard - PCB Defect Detection and Classification")
    st.markdown(
        """
        <div class="container">
            <h2 style='text-align: center; color: #E0E0E0;'>Upload Defected PCB Image</h2>
            <p style='text-align: center; color: #A0A0A0;'>Please upload a defected PCB image. The system will use the corresponding reference image internally.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    defect_file = st.file_uploader("Upload Defected PCB Image", type=["jpg", "jpeg", "png"])
    if defect_file:
        st.image(defect_file, caption="Defected Image", use_column_width=True)
        st.divider()
        if st.button("Detect Defects", use_container_width=True):
            with st.spinner("Detecting defects... Please wait."):
                try:
                    time.sleep(2)
                    # Save uploaded file temporarily to get its name
                    temp_path = os.path.join("temp_upload", defect_file.name)
                    os.makedirs("temp_upload", exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(defect_file.getbuffer())
                    with open(temp_path, "rb") as f:
                        annotated_img, defects_list = process_defect_image_with_reference(f.read(), defect_file.name)
                    st.success("Defect analysis complete!")
                    st.header("Defect Analysis Output")
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Defected Image with Labels", use_column_width=True)
                    is_success, buffer = cv2.imencode(".png", annotated_img)
                    if is_success:
                        st.download_button(
                            label="Download Output Image",
                            data=io.BytesIO(buffer),
                            file_name="Aicircuit_guard_output.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    st.subheader("Summary of Defects")
                    if defects_list:
                        for i, defect in enumerate(defects_list):
                            st.markdown(
                                f"**Defect {i+1}:** {defect['label'].replace('_', ' ').title()} "
                                f"(Confidence: {defect['confidence']:.2f})"
                            )
                    else:
                        st.info("No major defects were detected.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
    else:
        st.info("Please upload a defected image to enable the 'Detect Defects' button.")

if __name__ == "__main__":
    main()


