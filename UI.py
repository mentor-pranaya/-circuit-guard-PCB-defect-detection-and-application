import streamlit as st
from pipeline import load_model, run_pipeline
from PIL import Image
import io
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="⚡",
    layout="wide"
)

# =========================
# -------------------------
# COLOR CUSTOMIZATION AREA
# -------------------------
HEADER_BG = "#5BD0E5"
HEADER_TEXT = "#051676"
SUBTEXT = "#051676"

DEFECT_CARD_BG =  "#5BD0E5"
DEFECT_CARD_TEXT ="#051676"
BBOX_COLOR =  "#5BD0E5"  # Bounding box
LABEL_COLOR = "#051676" # Label text

DETECTED_DEFECTS_COLOR = "#5BD0E5"  # Color for "Detected Defects" heading

# Image overlay (bounding box and text)
BBOX_COLOR = "#841B12"       # Bounding box color
LABEL_COLOR =  "#BA362A"       # Text label color
LABEL_FONT_SCALE = 1.0         # Text size on image
LABEL_THICKNESS = 4           # Text thickness
BBOX_THICKNESS = 3             # Rectangle thickness

# =========================
# Pastel UI Header
# =========================
st.markdown(f"""
    <div style="background-color:{HEADER_BG};padding:20px;border-radius:10px">
    <h1 style="color:{HEADER_TEXT};text-align:center;">⚡ PCB Defect Detection ⚡</h1>
    <p style="color:{SUBTEXT};text-align:center;font-size:18px;">
    Upload your PCB image to detect defects in real-time!
    </p>
    </div>
""", unsafe_allow_html=True)

# =========================
# Custom CSS for hover
# =========================
st.markdown("""
    <style>
    /* Hover effect for defect cards */
    div[data-testid="stMarkdownContainer"] div:hover {
        transform: scale(1.04);
        transition: all 0.3s ease-in-out;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Custom CSS for image hover
# =========================
st.markdown("""
    <style>
    /* Hover effect for images */
    img {
        transition: transform 0.3s ease;
        border-radius: 10px;
    }
    img:hover {
        transform: scale(1.025);
        box-shadow: 0px 6px 20px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.write("---")

# =========================
# Initialize session state
# =========================
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "defect_info" not in st.session_state:
    st.session_state.defect_info = None
if "input_image" not in st.session_state:
    st.session_state.input_image = None

MODEL_PATH = r"C:\Users\kavya\Downloads\efficientnet_b4_best .pth"
if "model" not in st.session_state:
    st.session_state.model, st.session_state.classes, st.session_state.device = load_model(MODEL_PATH)

# =========================
# Upload PCB Images (Reference & Defected)
# =========================
st.subheader("📤 Upload PCB Images")

col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader("Upload Reference Image (Good PCB)", type=["png", "jpg", "jpeg"], key="ref")
    if ref_file:
        st.session_state.ref_image = Image.open(ref_file).convert("RGB")
        st.image(st.session_state.ref_image, caption="Reference Image", width='stretch')

with col2:
    defect_file = st.file_uploader("Upload Defected Image (PCB to Test)", type=["png", "jpg", "jpeg"], key="defect")
    if defect_file:
        st.session_state.defect_image = Image.open(defect_file).convert("RGB")
        st.image(st.session_state.defect_image, caption="Defected Image", width='stretch')

# =========================
# Process button
# =========================
if "ref_image" in st.session_state and "defect_image" in st.session_state:
    if st.button("🛠️ Detect Defects"):

        defect_img = st.session_state.defect_image

        # Choose colors (HTML-like hex)
        BBOX_COLOR = "#73170F" 
        LABEL_COLOR = "#B82012"

        # Run pipeline with color options
        output_pil, detected_defects = run_pipeline(
            st.session_state.ref_image,
            st.session_state.defect_image,
            st.session_state.model,
            st.session_state.classes,
            st.session_state.device,
            bbox_color=BBOX_COLOR,
            label_color=LABEL_COLOR
        )



        # Save results in session
        st.session_state.processed_image = output_pil
        st.session_state.defect_info = detected_defects


# =========================
# Show results if available
# =========================
if st.session_state.processed_image:
    if st.session_state.defect_info:
        st.markdown(f"<h2 style='color:{DETECTED_DEFECTS_COLOR};'>⚠️ Detected Defects</h2>", unsafe_allow_html=True)
        st.image(st.session_state.processed_image, caption="Processed Image", width='stretch')

        st.write("### Defect Details")
        cols = st.columns(len(st.session_state.defect_info))
        for i, defect in enumerate(st.session_state.defect_info):
            with cols[i]:
                st.markdown(f"""
                    <div style="background-color:{DEFECT_CARD_BG};padding:15px;border-radius:10px">
                    <h4 style="color:{DEFECT_CARD_TEXT};text-align:center;">{defect['type']}</h4>
                    <p style="color:{DEFECT_CARD_TEXT};text-align:center;">BBox: {defect['bbox']}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:{DETECTED_DEFECTS_COLOR};'>✅ No defects detected!</h2>", unsafe_allow_html=True)
        st.image(st.session_state.processed_image, caption="Uploaded PCB Image", width='stretch')


    # =========================
    # Download button
    # =========================
    buf = io.BytesIO()
    st.session_state.processed_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Processed Image",
        data=byte_im,
        file_name="pcb_defects.png",
        mime="image/png"
    )
