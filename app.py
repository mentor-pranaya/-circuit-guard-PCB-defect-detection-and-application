import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import json
import os
from io import BytesIO
from PIL import Image


# ==================== IMPORT FROM YOUR EXISTING FILES ====================
from subtraction import subtract_images
from train_efficientnet import build_model, build_transforms

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="PCB Defect Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

st.markdown("""
<h1><i class="fas fa-microchip"></i> PCB Defect Detection</h1>
""", unsafe_allow_html=True)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main {
        background-color: #FFEDF3;
    }

    /* App background */
    .stApp {
        background-color: #FFEDF3;
    }

    /* Upload Box */
    .upload-box {
        border: 2px dashed #0ABAB5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: white;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background-color: #0ABAB5;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #56DFCF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Result Box */
    .result-box {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }

    /* Metric Card */
    .metric-card {
        background: linear-gradient(135deg, #0ABAB5 0%, #56DFCF 50%, #ADEED9 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
    }

    .metric-label {
        font-size: 14px;
        opacity: 0.95;
    }

    /* Headings */
    h1 {
        color: #0ABAB5;
        text-align: center;
        padding: 20px 0;
    }

    h3 {
        color: #16a085;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model(checkpoint_path="checkpoints/efficientnet_b4_best.pth", 
               class_mapping_path="checkpoints/class_to_idx.json"):
    """Load the trained EfficientNet model using your existing build_model function"""
    try:
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            class_to_idx = json.load(f)
        
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        
        # Use YOUR build_model function from train_efficientnet.py
        model = build_model(num_classes=num_classes)
        
        # Load weights
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        
        return model, idx_to_class, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# ==================== HELPER FUNCTIONS (Only UI-specific) ====================
def extract_rois_and_draw_contours(defect_img, mask):
    """Extract ROIs and draw bounding boxes - adapted from your contour_visualization.py"""
    if isinstance(defect_img, Image.Image):
        defect_img = np.array(defect_img)
    
    annotated = defect_img.copy()
    rois = []
    
    # Use same contour logic as your contour_visualization.py
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Same threshold as your code
        if w * h < 50:
            continue
        
        # Extract ROI (same as roi_extraction.py)
        roi = defect_img[y:y+h, x:x+w]
        rois.append(roi)
        
        # Draw bounding box (same as contour_visualization.py)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(annotated, f"Defect {i+1}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0,0), 2,lineType=cv2.LINE_AA  )
    
    return annotated, rois

def draw_contours_with_labels(defect_img, mask, defect_label, confidence):
    """Draw contours with classified label - from your contour_visualization.py logic"""
    if isinstance(defect_img, Image.Image):
        defect_img = np.array(defect_img)
    
    annotated = defect_img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w * h < 50:
            continue
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255,0, 0), 2)
        
        # Add classified label instead of generic "defect_type"
        label_text = f"{defect_label} ({confidence:.1f}%)"
        cv2.putText(annotated, label_text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2,lineType=cv2.LINE_AA  )
    
    return annotated

def classify_defect(model, rois, idx_to_class, device, img_size=300):
    """Classify defect using YOUR trained model and transforms"""
    if not rois or model is None:
        return "Unknown", 0.0
    
    # Use YOUR transform from train_efficientnet.py
    _, val_tfms = build_transforms(img_size)
    
    # Process all ROIs and aggregate predictions (like roi_extraction.py extracts multiple ROIs)
    all_probs = []
    
    with torch.no_grad():
        for roi in rois:
            # Convert ROI to PIL Image
            if len(roi.shape) == 2:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            elif roi.shape[2] == 4:
                roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)
            
            roi_pil = Image.fromarray(roi)
            img_tensor = val_tfms(roi_pil).unsqueeze(0).to(device)
            
            # Get prediction
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])
    
    # Average probabilities across all ROIs
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = np.argmax(avg_probs)
    confidence = avg_probs[pred_idx] * 100
    
    defect_class = idx_to_class[pred_idx]
    
    return defect_class, confidence

def pil_to_bytes(img, format='PNG'):
    """Convert image to bytes for download"""
    buf = BytesIO()
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(buf, format=format)
    buf.seek(0)
    return buf

# ==================== MAIN APP ====================
def main():

    
    # Load model
    with st.spinner("Loading model..."):
        model, idx_to_class, device = load_model()
    
    if model is None:
        st.error("⚠️ Failed to load model. Please check if the checkpoint files exist.")
        st.info("Required files: `checkpoints/efficientnet_b4_best.pth` and `checkpoints/class_to_idx.json`")
        return
    
  
    
    # Create two columns for image upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        st.markdown("### <i class='fas fa-camera'></i> Upload Golden Reference Image", unsafe_allow_html=True)
        st.markdown("*Upload a defect-free reference PCB image*")
        golden_file = st.file_uploader("Choose reference image", type=['jpg', 'jpeg', 'png'], key='golden')
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        st.markdown("### <i class='fas fa-bullseye'></i> Upload Defected Image", unsafe_allow_html=True)
        st.markdown("*Upload the PCB image to inspect for defects*")
        defect_file = st.file_uploader("Choose defected image", type=['jpg', 'jpeg', 'png'], key='defect')
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display uploaded images
    if golden_file and defect_file:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        golden_img = Image.open(golden_file).convert('RGB')
        defect_img = Image.open(defect_file).convert('RGB')
        
        with col1:
            st.image(golden_img, caption="Reference Image", use_container_width=True)
        
        with col2:
            st.image(defect_img, caption="Defected Image", use_container_width=True)
        
        # Process button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(" Run Defect Detection Pipeline", key='process'):
                process_images(golden_img, defect_img, model, idx_to_class, device)
        st.info("👆 Please upload both reference and defected images to begin analysis")

def process_images(golden_img, defect_img, model, idx_to_class, device):
    """Process images through the entire pipeline using YOUR existing functions"""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Image Subtraction - USING YOUR subtraction.py
    status_text.text("Step 1/4: Performing image subtraction...")
    progress_bar.progress(25)
    
    # Convert PIL to numpy for your subtract_images function
    golden_np = cv2.cvtColor(np.array(golden_img), cv2.COLOR_RGB2BGR)
    defect_np = cv2.cvtColor(np.array(defect_img), cv2.COLOR_RGB2BGR)
    
    # Use YOUR subtract_images function from subtraction.py
    mask = subtract_images(golden_np, defect_np)
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # Step 2: ROI Extraction - USING YOUR roi_extraction.py logic
    status_text.text("Step 2/4: Performing ROI Extraction...")
    progress_bar.progress(50)
    
    annotated_rois, rois = extract_rois_and_draw_contours(np.array(defect_img), mask)
    
    # Step 3: Classification - USING YOUR trained model
    status_text.text("Step 3/4: Performing image classification...")
    progress_bar.progress(75)
    
    if len(rois) > 0:
        defect_class, confidence = classify_defect(model, rois, idx_to_class, device)
    else:
        defect_class, confidence = "No Defect", 100.0
    
    # Step 4: Final Visualization - USING YOUR contour_visualization.py logic
    status_text.text("Step 4/4: Performing Final Visualization..")
    progress_bar.progress(90)
    
    final_annotated = draw_contours_with_labels(np.array(defect_img), mask, defect_class, confidence)
    
    progress_bar.progress(100)
    status_text.text("✅ Processing complete!")
    
    # Display Results
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'><i class='fas fa-chart-bar'></i> Detection Results</h2>", unsafe_allow_html=True)
    
    # Classification Result Card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>DETECTED DEFECT TYPE</div>
                <div class='metric-value'>{defect_class}</div>
                <div class='metric-label'>Confidence: {confidence:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display processed images
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("####  Subtracted Mask")
        st.image(mask_display, use_container_width=True)
        st.download_button(
            label="⬇️ Download Mask",
            data=pil_to_bytes(mask_display),
            file_name="subtracted_mask.png",
            mime="image/png",
            key='download_mask'
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='result-box'>",unsafe_allow_html=True)
        st.markdown("#### ROI Detection")
        st.image(annotated_rois, use_container_width=True)
        st.download_button(
            label="⬇️ Download ROI",
            data=pil_to_bytes(annotated_rois),
            file_name="roi_detection.png",
            mime="image/png",
            key='download_roi'
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("#### Final Classification")
        st.image(final_annotated, use_container_width=True)
        st.download_button(
            label="⬇️ Download Result",
            data=pil_to_bytes(final_annotated),
            file_name="classified_result.png",
            mime="image/png",
            key='download_final'
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    

if __name__ == "__main__":
    main()