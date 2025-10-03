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
# ... (all your existing imports)

# ==========================================
# 1️⃣ Load Model and Configuration (Existing code)
# ==========================================
# ... (Your existing model loading and predict_roi function) ...

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
# 2️⃣ Image Subtraction Function (Existing code)
# ==========================================
def subtract_images(defect_bytes, ref_bytes):
    # ... (Your existing subtract_images function) ...
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
# 3️⃣ ROI Extraction Function (Existing code)
# ==========================================
def extract_rois(orig_img, mask, min_area=200, min_w=10, min_h=10):
    # ... (Your existing extract_rois function) ...
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
# 4️⃣ Full Pipeline (Existing code)
# ==========================================
def process_images(defect_bytes, ref_bytes):
    # ... (Your existing process_images function) ...
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

# ==========================================
# 5️⃣ NEW: Default Image Loading Function
# ==========================================

def load_default_ref_image(file_path):
    """
    Loads a local image file and returns its content as a bytes stream.
    This mimics the output of st.file_uploader.getvalue().
    """
    if not os.path.exists(file_path):
        st.error(f"❌ Default reference image file not found at: {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        return image_bytes
    except Exception as e:
        st.error(f"❌ Error loading default image: {e}")
        return None


def main():
    """Main Streamlit app function with the new UI."""
    st.set_page_config(page_title="AI Circuit Guard", layout="wide")
    st.title("AI Circuit Guard")
    # ... (Your existing CSS and container markdown) ...
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
            <h2 style='text-align: center; color: #E0E0E0;'>Select Reference and Upload Defected PCB Image</h2>
            <p style='text-align: center; color: #A0A0A0;'>Choose a golden (reference) image and upload a defected image to begin the analysis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Configuration for Default Reference Images
    # NOTE: You must ensure this path is accessible by the Streamlit application.
    REF_IMAGE_DIR = r"C:\Users\harsh\OneDrive\Documents\pythonvs\PCB_DATA\PCB_USED"
    
    # Get list of image files (e.g., .jpg, .png)
    try:
        ref_image_files = [f for f in os.listdir(REF_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not ref_image_files:
             st.error(f"No image files found in the directory: {REF_IMAGE_DIR}")
             return
    except FileNotFoundError:
        st.error(f"❌ Reference image directory not found: {REF_IMAGE_DIR}")
        return
    except Exception as e:
        st.error(f"An error occurred while listing files: {e}")
        return

    col1, col2 = st.columns(2)
    selected_ref_file = None
    ref_bytes = None
    defect_file = None

    with col1:
        st.header("Select Reference Image (Default)")
        
        # 2. Use st.selectbox for reference image selection
        selected_file_name = st.selectbox(
            "Choose a Golden PCB Reference Image", 
            options=ref_image_files, 
            index=0, # Default to the first image
            key="ref_selector"
        )
        
        # 3. Load the selected image and its bytes
        if selected_file_name:
            selected_ref_path = os.path.join(REF_IMAGE_DIR, selected_file_name)
            ref_bytes = load_default_ref_image(selected_ref_path)
            
            # Display the selected default image
            if ref_bytes:
                st.image(ref_bytes, caption=f"Reference Image: {selected_file_name}", use_column_width=True)

    with col2:
        st.header("Upload Defected Image")
        # Keep the file uploader for the defected image
        defect_file = st.file_uploader("PCB to Test", type=["jpg", "jpeg", "png"], key="defected_uploader")
        if defect_file:
            st.image(defect_file, caption="Defected Image", use_column_width=True)

    # 4. Update the processing logic to use the loaded ref_bytes
    # Check that both the default reference image and the uploaded defect image are available
    if ref_bytes and defect_file:
        st.divider()
        if st.button("Detect Defects", use_container_width=True):
            with st.spinner("Detecting defects... Please wait."):
                try:
                    # Simulate a delay for the processing
                    time.sleep(2)
                    
                    # Use the pre-loaded ref_bytes and the uploaded defect_file bytes
                    annotated_img, defects_list = process_images(defect_file.getvalue(), ref_bytes)

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
        st.info("Please select a reference image and upload a defected image to enable the 'Detect Defects' button.")

if __name__ == "__main__":
    main()
