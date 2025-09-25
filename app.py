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

# ==========================================
# 1️⃣ Load Model and Configuration
# ==========================================
# For this example, we'll use a placeholder function for prediction since
# the actual model file is not available in this environment.
#
# To use a real model, uncomment the code below and replace the
# placeholder functions with your own trained model's loading and inference logic.
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = "path/to/your/efficientnet_b4_best.pth"
#
# try:
#     checkpoint = torch.load(MODEL_PATH, map_location=device)
#     classes = checkpoint['classes']
#     num_classes = len(classes)
#
#     model = models.efficientnet_b4(pretrained=False)
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()
#
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
#     model_loaded = True
# except Exception as e:
#     st.error(f"❌ Error loading the model. Please check the path and file: {e}")
#     model_loaded = False
#
# Define the classes (can be hardcoded if the model is not loaded)
classes = ["missing_hole", "mousebite", "open_circuit", "short_circuit", "spurious_copper", "short"]

def predict_roi(roi_img, model, class_names):
    """
    Predicts the defect class for a given ROI image.
    
    This is a placeholder function that simulates a model's output.
    To use a real model, replace this with your actual inference logic.
    """
    # Placeholder logic: randomly pick a defect and a confidence score
    label = np.random.choice(class_names)
    conf = np.random.uniform(0.7, 0.99)
    return label, conf

# ==========================================
# 2️⃣ Image Subtraction Function
# ==========================================
def subtract_images(defect_bytes, ref_bytes):
    """
    Performs image subtraction on in-memory images.
    Returns the original defect image (as BGR) and the cleaned mask.
    """
    defect_np = np.frombuffer(defect_bytes, np.uint8)
    ref_np = np.frombuffer(ref_bytes, np.uint8)
    
    defect_color = cv2.imdecode(defect_np, cv2.IMREAD_COLOR)
    ref_color = cv2.imdecode(ref_np, cv2.IMREAD_COLOR)

    if defect_color is None or ref_color is None:
        raise ValueError("❌ Could not load images from uploaded files.")

    defect_gray = cv2.cvtColor(defect_color, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)

    if ref_gray.shape != defect_gray.shape:
        ref_gray = cv2.resize(ref_gray, (defect_gray.shape[1], defect_gray.shape[0]))

    defect_blur = cv2.GaussianBlur(defect_gray, (5, 5), 0)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)

    subtracted = cv2.absdiff(defect_blur, ref_blur)

    binary = cv2.adaptiveThreshold(subtracted, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 35, -5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)

    return defect_color, cleaned

# ==========================================
# 3️⃣ ROI Extraction Function
# ==========================================
def extract_rois(orig_img, mask, min_area=200, min_w=10, min_h=10):
    """
    Extracts Regions of Interest (ROIs) from an image based on a mask.
    Returns a list of dictionaries with 'roi_img' and 'bbox'.
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_area or w < min_w or h < min_h:
            continue
        
        roi = orig_img[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        
        results.append({'roi_img': roi, 'bbox': (x, y, x + w, y + h)})
        
    return results

# ==========================================
# 4️⃣ Full Pipeline
# ==========================================
def process_images(defect_bytes, ref_bytes):
    """
    Executes the full defect detection pipeline.
    """
    defect_color, mask = subtract_images(defect_bytes, ref_bytes)
    roi_results = extract_rois(defect_color, mask)
    annotated_img = defect_color.copy()

    detected_defects = []
    for roi in roi_results:
        # Use the prediction function to get the label and confidence
        label, conf = predict_roi(roi['roi_img'], None, classes)
        
        x1, y1, x2, y2 = roi['bbox']

        # Add the detected defect to the list
        detected_defects.append({"label": label, "box": (x1, y1, x2, y2), "confidence": conf})

        # Draw the bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Create the text label with a black background for clarity
        text = f"Defect ({conf:.2f})"
        font_scale = 1.2
        font_thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(annotated_img, (x1, y1 - th - baseline - 5), (x1 + tw, y1), (0, 0, 0), -1)
        cv2.putText(annotated_img, text, (x1, y1 - 7),
                    font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)

    return annotated_img, detected_defects

def main():
    """Main Streamlit app function with the new UI."""
    st.set_page_config(page_title="AI Circuit Guard", layout="wide")
    st.title("AI Circuit Guard")
    st.markdown(
        """
        <style>
        .stButton>button {
            border-radius: 20px;
            color: white;
            background-color: #6366f1;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #4f46e5;
        }
        .stFileUploader>div {
            border-radius: 12px;
            border: 2px dashed #424242;
            padding: 2rem;
            text-align: center;
        }
        .stFileUploader>div p {
            color: #9e9e9e;
        }
        .container {
            background-color: #1e1e1e;
            border-radius: 1.5rem;
            padding: 2.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="container">
            <h2 style='text-align: center; color: #E0E0E0;'>Upload PCB Images</h2>
            <p style='text-align: center; color: #A0A0A0;'>Please upload a golden (reference) image and a defected image to begin the analysis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    ref_file, defect_file = None, None

    with col1:
        st.header("Upload Reference Image")
        ref_file = st.file_uploader("Golden PCB", type=["jpg", "jpeg", "png"], key="ref_uploader")
        if ref_file:
            st.image(ref_file, caption="Reference Image", use_column_width=True)

    with col2:
        st.header("Upload Defected Image")
        defect_file = st.file_uploader("PCB to Test", type=["jpg", "jpeg", "png"], key="defected_uploader")
        if defect_file:
            st.image(defect_file, caption="Defected Image", use_column_width=True)

    if ref_file and defect_file:
        st.divider()
        if st.button("Detect Defects", use_container_width=True):
            with st.spinner("Detecting defects... Please wait."):
                try:
                    # Simulate a delay for the processing
                    time.sleep(2)
                    
                    annotated_img, defects_list = process_images(defect_file.getvalue(), ref_file.getvalue())

                    st.success("Defect analysis complete!")
                    st.header("Defect Analysis Output")
                    
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Defected Image with Labels", use_column_width=True)
                    
                    # Convert the output image to bytes for download
                    is_success, buffer = cv2.imencode(".png", annotated_img)
                    if is_success:
                        st.download_button(
                            label="Download Output Image",
                            data=io.BytesIO(buffer),
                            file_name="aicircuit_guard_output.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    # Display a summary of detected defects
                    st.subheader("Summary of Defects")
                    if defects_list:
                        for i, defect in enumerate(defects_list):
                            st.markdown(f"**Defect {i+1}:** {defect['label'].replace('_', ' ').title()}")
                    else:
                        st.info("No major defects were detected.")
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
    else:
        st.info("Please upload both images to enable the 'Detect Defects' button.")

if __name__ == "__main__":
    main()
