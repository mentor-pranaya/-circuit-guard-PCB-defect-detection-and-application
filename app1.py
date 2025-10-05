import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import json
import os
from io import BytesIO
from pathlib import Path

# ==================== IMPORT ALL FROM YOUR EXISTING FILES ====================
try:
    from subtraction import run, subtract_images
except ImportError:
    # If run() doesn't exist in subtraction.py, use subtract_images
    from subtraction import subtract_images
    run = None

from contour_visualization import draw_contours
from train_efficientnet import (
    build_model, 
    build_transforms, 
    ROIDataset,
    DEVICE
)

# ==================== PATHS ====================
# Get the script directory (CODE folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to INFOSYSPROJECT, then into PCB_DATASET/PCB_USED
REFERENCE_IMAGES_DIR = os.path.join(SCRIPT_DIR, "..", "PCB_DATASET", "PCB_USED")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: white;
    }
    .result-box {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
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
        opacity: 0.9;
    }
    .match-info {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
        color:black;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px 0;
    }
    h3 {
        color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model(checkpoint_path="checkpoints/efficientnet_b4_best.pth", 
               class_mapping_path="checkpoints/class_to_idx.json"):
    """Load the trained model using YOUR build_model function"""
    try:
        with open(class_mapping_path, 'r') as f:
            class_to_idx = json.load(f)
        
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        
        model = build_model(num_classes=num_classes)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        
        return model, idx_to_class, device, class_to_idx
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

# ==================== REFERENCE IMAGE MATCHING ====================
def find_best_reference_image(defect_img, reference_dir=None):
    """
    Find the best matching reference image based on similarity score.
    Uses SSIM (Structural Similarity Index) or feature matching.
    """
    if reference_dir is None:
        reference_dir = REFERENCE_IMAGES_DIR
    
    # Check if directory exists
    if not os.path.exists(reference_dir):
        st.error(f"❌ Reference directory not found: `{reference_dir}`")
        st.info(f"💡 Please ensure the folder exists at: `{os.path.abspath(reference_dir)}`")
        return None, None, 0.0
    
    # Convert defect image to grayscale
    if isinstance(defect_img, Image.Image):
        defect_img = np.array(defect_img)
    defect_gray = cv2.cvtColor(defect_img, cv2.COLOR_RGB2GRAY)
    
    best_match = None
    best_score = 0.0
    best_ref_img = None
    best_ref_name = None
    
    # Get all reference images
    ref_files = [f for f in os.listdir(reference_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not ref_files:
        st.error(f"❌ No reference images found in `{reference_dir}`")
        st.info(f"📁 Found directory but it's empty. Please add reference PCB images (.jpg, .jpeg, .png)")
        return None, None, 0.0
    
    for ref_file in ref_files:
        ref_path = os.path.join(reference_dir, ref_file)
        ref_img = cv2.imread(ref_path)
        
        if ref_img is None:
            continue
        
        # Resize defect image to match reference size
        defect_resized = cv2.resize(defect_gray, (ref_img.shape[1], ref_img.shape[0]))
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Template Matching Score
        # Calculate correlation coefficient
        result = cv2.matchTemplate(defect_resized, ref_gray, cv2.TM_CCOEFF_NORMED)
        score = np.max(result)
        
        # Method 2: Histogram Comparison (additional validation)
        hist_defect = cv2.calcHist([defect_resized], [0], None, [256], [0, 256])
        hist_ref = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
        hist_score = cv2.compareHist(hist_defect, hist_ref, cv2.HISTCMP_CORREL)
        
        # Combined score
        combined_score = (score * 0.7 + hist_score * 0.3)
        
        if combined_score > best_score:
            best_score = combined_score
            best_match = ref_path
            best_ref_img = ref_img
            best_ref_name = ref_file
    
    # Convert best reference to RGB
    if best_ref_img is not None:
        best_ref_img = cv2.cvtColor(best_ref_img, cv2.COLOR_BGR2RGB)
    
    return best_ref_img, best_ref_name, best_score

# ==================== PROCESSING FUNCTIONS ====================
def extract_rois_from_mask(defect_img, mask):
    """Extract ROIs from mask - same logic as your roi_extraction.py"""
    if isinstance(defect_img, Image.Image):
        defect_img = np.array(defect_img)
    
    rois = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 50:
            continue
        roi = defect_img[y:y+h, x:x+w]
        rois.append(roi)
    
    return rois

def classify_defect(model, rois, idx_to_class, device, img_size=300):
    """Classify using YOUR model and transforms"""
    if not rois or model is None:
        return "Unknown", 0.0
    
    _, val_tfms = build_transforms(img_size)
    
    all_probs = []
    
    with torch.no_grad():
        for roi in rois:
            if len(roi.shape) == 2:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            elif roi.shape[2] == 4:
                roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)
            
            roi_pil = Image.fromarray(roi)
            img_tensor = val_tfms(roi_pil).unsqueeze(0).to(device)
            
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])
    
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
    st.markdown("<h1>🔍 PCB Defect Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Upload a PCB image to automatically detect and classify defects</p>", unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model, idx_to_class, device, class_to_idx = load_model()
    
    if model is None:
        st.error("⚠️ Failed to load model. Please check if the checkpoint files exist.")
        st.info("Required files: `checkpoints/efficientnet_b4_best.pth` and `checkpoints/class_to_idx.json`")
        return
    
    
    # Single image upload
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        st.markdown("### 📤 Upload PCB Image for Inspection")
        st.markdown("*The system will automatically find the best matching reference image*")
        defect_file = st.file_uploader("Choose PCB image", type=['jpg', 'jpeg', 'png'], key='defect')
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process uploaded image
    if defect_file:
        st.markdown("---")
        
        defect_img = Image.open(defect_file).convert('RGB')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(defect_img, caption="Uploaded PCB Image", use_container_width=True)
        
        # Process button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Defect Detection & Classification", key='process'):
                process_with_auto_reference(defect_img, model, idx_to_class, device)
    else:
        st.info("👆 Please upload a PCB image to begin automated defect detection")

def process_with_auto_reference(defect_img, model, idx_to_class, device):
    """Process image with automatic reference matching"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Find best reference image
    status_text.text("⏳ Step 1/5: Finding best matching reference image...")
    progress_bar.progress(20)
    
    reference_img, ref_name, similarity_score = find_best_reference_image(defect_img)
    
    if reference_img is None:
        st.error("❌ Could not find a suitable reference image. Please check your reference directory.")
        return
    
    # Display matched reference
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>✅ Best Match Found</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(reference_img, caption=f"Reference: {ref_name}", use_container_width=True)
    with col2:
        st.image(defect_img, caption="Test PCB Image", use_container_width=True)
    
    st.markdown(f"""
        <div class='match-info'>
            <strong>📊 Match Details:</strong><br>
            Reference Image: <code>{ref_name}</code><br>
            Similarity Score: <strong>{similarity_score*100:.2f}%</strong>
        </div>
    """, unsafe_allow_html=True)
    
    # Step 2: Subtraction
    status_text.text("⏳ Step 2/5: Performing image subtraction...")
    progress_bar.progress(40)
    
    # Convert to PIL for your run() function
    ref_pil = Image.fromarray(reference_img)
    mask = run(ref_pil, defect_img)
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # Step 3: ROI Extraction
    status_text.text("⏳ Step 3/5: Extracting regions of interest...")
    progress_bar.progress(60)
    
    rois = extract_rois_from_mask(np.array(defect_img), mask)
    
    # Use YOUR draw_contours from contour_visualization.py
    defect_img_np = cv2.cvtColor(np.array(defect_img), cv2.COLOR_RGB2BGR)
    annotated_rois = draw_contours(defect_img_np, mask, "Defect Region")
    annotated_rois = cv2.cvtColor(annotated_rois, cv2.COLOR_BGR2RGB)
    
    # Step 4: Classification
    status_text.text("⏳ Step 4/5: Classifying defects using trained model...")
    progress_bar.progress(80)
    
    if len(rois) > 0:
        defect_class, confidence = classify_defect(model, rois, idx_to_class, device)
    else:
        defect_class, confidence = "No Defect", 100.0
    
    # Step 5: Final Visualization
    status_text.text("⏳ Step 5/5: Generating final visualization...")
    progress_bar.progress(95)
    
    final_annotated = draw_contours(defect_img_np, mask, f"{defect_class} ({confidence:.1f}%)")
    final_annotated = cv2.cvtColor(final_annotated, cv2.COLOR_BGR2RGB)
    
    progress_bar.progress(100)
    status_text.text("✅ Processing complete!")
    
    # Display Results
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>📊 Detection & Classification Results</h2>", unsafe_allow_html=True)
    
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
        st.markdown("#### 🎯 Subtracted Mask")
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
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("#### 📦 ROI Detection")
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
        st.markdown("#### 🏷️ Final Classification")
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