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
from skimage.metrics import structural_similarity as ssim

# ==========================================
# 1️⃣ Load Model and Configuration (Existing code)
# ==========================================
# ... (Your existing model loading and predict_roi function) ...

classes = ["missing_hole", "mousebite", "open_circuit", "short_circuit", "spurious_copper", "short"]

def predict_roi(roi_img, model, class_names):
    """
    Placeholder: Predicts the defect class for a given ROI image.
    """
    label = np.random.choice(class_names)
    conf = np.random.uniform(0.7, 0.99)
    return label, conf

# ==========================================
# 2️⃣ Image Subtraction Function (Existing code)
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

    # Ensure reference image is the same size as the defect image for subtraction
    if ref_gray.shape != defect_gray.shape:
        ref_gray = cv2.resize(ref_gray, (defect_gray.shape[1], defect_gray.shape[0]))

    defect_blur = cv2.GaussianBlur(defect_gray, (5, 5), 0)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)

    subtracted = cv2.absdiff(defect_blur, ref_blur)

    # ... (rest of your existing thresholding and morphology) ...
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
# 4️⃣ NEW: SSIM and Auto-Matching Functions
# ==========================================

def calculate_ssim(img1, img2):
    """Calculates SSIM between two grayscale images of potentially different sizes."""
    # Resize image 2 to match image 1's size for SSIM comparison
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Compute SSIM
    score, diff = ssim(img1, img2_resized, full=True)
    return score

@st.cache_data
def find_best_match_ssim(defect_bytes, ref_dir, ref_files):
    """
    Finds the reference image in ref_dir that has the highest SSIM score
    with the uploaded defect image.
    """
    # Convert defect image bytes to OpenCV image (grayscale for SSIM)
    defect_np = np.frombuffer(defect_bytes, np.uint8)
    defect_color = cv2.imdecode(defect_np, cv2.IMREAD_COLOR)
    defect_gray = cv2.cvtColor(defect_color, cv2.COLOR_BGR2GRAY)

    best_score = -1
    best_match_name = None
    best_match_bytes = None

    for ref_name in ref_files:
        ref_path = os.path.join(ref_dir, ref_name)
        
        # Load reference image (grayscale)
        ref_image_cv = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        
        if ref_image_cv is None:
            continue # Skip if file cannot be read
            
        # Calculate SSIM (defect_gray is the reference size)
        score = calculate_ssim(defect_gray, ref_image_cv)

        if score > best_score:
            best_score = score
            best_match_name = ref_name
            # Read the bytes of the best match
            with open(ref_path, "rb") as f:
                best_match_bytes = f.read()

    return best_match_name, best_match_bytes, best_score

# ==========================================
# 5️⃣ Full Pipeline (Modified for Auto-Match)
# ==========================================
def process_images(defect_bytes, ref_bytes):
    """
    Executes the full defect detection pipeline.
    """
    # ... (Your existing process_images logic) ...
    defect_color, mask = subtract_images(defect_bytes, ref_bytes)
    roi_results = extract_rois(defect_color, mask)
    annotated_img = defect_color.copy()

    detected_defects = []
    for roi in roi_results:
        label, conf = predict_roi(roi['roi_img'], None, classes)
        
        x1, y1, x2, y2 = roi['bbox']

        detected_defects.append({"label": label, "box": (x1, y1, x2, y2), "confidence": conf})

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

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
# 6️⃣ Main Streamlit App (Simplified UI)
# ==========================================

def main():
    """Main Streamlit app function with the simplified UI and auto-matching."""
    st.set_page_config(page_title="AI Circuit Guard - Auto SSIM Match", layout="wide")
    st.title("AI Circuit Guard")
    # ... (Your existing CSS) ...
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
            <h2 style='text-align: center; color: #E0E0E0;'>Upload Defected PCB Image for Auto-Analysis</h2>
            <p style='text-align: center; color: #A0A0A0;'>The system will automatically find the best golden reference image from the database using Structural Similarity (SSIM).</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Configuration for Default Reference Images
    REF_IMAGE_DIR = r"C:\Users\harsh\OneDrive\Documents\pythonvs\PCB_DATA\PCB_USED"
    
    # Get list of image files
    try:
        ref_image_files = [f for f in os.listdir(REF_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not ref_image_files:
             st.error(f"No reference image files found in the directory: {REF_IMAGE_DIR}. Please check the path.")
             return
    except FileNotFoundError:
        st.error(f"❌ Reference image directory not found: **{REF_IMAGE_DIR}**. Please check the path.")
        return
    except Exception as e:
        st.error(f"An error occurred while listing files: {e}")
        return

    col1, col2 = st.columns([1, 1])
    defect_file = None
    
    # Simplified UI: Only defected image upload
    with col1:
        st.header("Upload Defected Image")
        defect_file = st.file_uploader("Upload PCB to Test", type=["jpg", "jpeg", "png"], key="defected_uploader")
        if defect_file:
            st.image(defect_file, caption="Uploaded Defected Image", use_column_width=True)
            
    # Placeholder for displaying the automatically selected reference image
    with col2:
        st.header("Automatically Selected Reference")
        ref_image_placeholder = st.empty()

    if defect_file:
        st.divider()
        if st.button("Detect Defects (Auto-Match Reference)", use_container_width=True):
            with st.spinner("1/2. Auto-matching reference image using SSIM..."):
                try:
                    # Auto-Match the reference image
                    defect_bytes = defect_file.getvalue()
                    
                    best_match_name, ref_bytes, ssim_score = find_best_match_ssim(
                        defect_bytes, REF_IMAGE_DIR, ref_image_files
                    )
                    
                    if ref_bytes is None:
                        st.error("❌ Failed to find or load a matching reference image.")
                        return

                    st.success(f"✅ Best Reference Found: **{best_match_name}** (SSIM: {ssim_score:.4f})")
                    
                    # Display the automatically selected reference image
                    ref_image_placeholder.image(io.BytesIO(ref_bytes), caption=f"Auto-Matched Reference: {best_match_name}", use_column_width=True)

                except Exception as e:
                    st.error(f"An error occurred during SSIM matching: {e}")
                    return # Stop processing if matching failed

            # Start defect analysis
            with st.spinner("2/2. Detecting defects... Please wait."):
                try:
                    time.sleep(1) # Simulated delay for processing
                    
                    # Process using the uploaded defect image and the auto-matched reference image
                    annotated_img, defects_list = process_images(defect_bytes, ref_bytes)

                    st.success("Defect analysis complete!")
                    st.header("Defect Analysis Output")
                    
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Defected Image with Labels", use_column_width=True)
                    
                    is_success, buffer = cv2.imencode(".png", annotated_img)
                    if is_success:
                        st.download_button(
                            label="Download Output Image",
                            data=io.BytesIO(buffer),
                            file_name="aicircuit_guard_output.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    st.subheader("Summary of Defects")
                    if defects_list:
                        for i, defect in enumerate(defects_list):
                            st.markdown(f"**Defect {i+1}:** {defect['label'].replace('_', ' ').title()}")
                    else:
                        st.info("No major defects were detected.")
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
    else:
        st.info("Please upload a defected image to begin the automatic reference matching and defect analysis.")

if __name__ == "__main__":
    main()
