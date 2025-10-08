import streamlit as st
from pipeline import load_model, run_pipeline
from PIL import Image
import io
import warnings
import base64

warnings.filterwarnings("ignore")

st.set_page_config(page_title="PCB Defect Detection", page_icon="⚡", layout="wide")

# =====================
# Set background image
# =====================
def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-color: rgba(255,255,255,0.3); 
            background-blend-mode: lighten;  
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_local(r"path_to_background_image")  

# =========================
# Colors & styles
# =========================
HEADER_GRADIENT = "linear-gradient(to bottom right, #B8F3EE, #66C0C8, #208498);"  # header gradient
HEADER_TEXT = "#FFFFFF"
SUBTEXT = "#FFFFFF"
TEXT_COLOR = "#097D72"

DEFECT_CARD_GRADIENT = "linear-gradient(to bottom right, #B8F3EE, #66C0C8, #208498);"  # same as header
DEFECT_CARD_TEXT = "#FFFFFF"
DETECTED_DEFECTS_COLOR = "#097D72"
BBOX_COLOR = "#841B12"
LABEL_COLOR = "#BA362A"

# =========================
# Gradient Header
# =========================
st.markdown(f"""
<div style="
    background: {HEADER_GRADIENT};
    padding:20px;
    border-radius:10px;
">
<h1 style="color:{HEADER_TEXT};text-align:center;">⚡ PCB Defect Detection ⚡</h1>
<p style="color:{SUBTEXT};text-align:center;font-size:18px;">
Upload your PCB image to detect defects!
</p>
</div>
""", unsafe_allow_html=True)

st.write("---")

# =========================
# Load model
# =========================
MODEL_PATH = r"C:\Users\kavya\PCB DEFECT DETECTION\bestest_model_full (2).pth"
REFERENCE_FOLDER = r"C:\Users\kavya\PCB DEFECT DETECTION\references"

if "model" not in st.session_state:
    st.session_state.model, st.session_state.classes, st.session_state.device = load_model(MODEL_PATH)

# =========================
# Initialize session state
# =========================
if "defect_image" not in st.session_state:
    st.session_state.defect_image = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "defect_info" not in st.session_state:
    st.session_state.defect_info = None

# =========================
# Upload PCB Image
# =========================
st.markdown(f'<h2 style="color:{TEXT_COLOR}; font-size:40px; font-weight:bold;">📤 Upload PCB Image</h2>', unsafe_allow_html=True)
defect_file = st.file_uploader("Upload your PCB here", type=["png","jpg","jpeg"])
if defect_file:
    st.session_state.defect_image = Image.open(defect_file).convert("RGB")
    
    st.markdown(f'<p style="color:{TEXT_COLOR}; font-size:20px;">Uploaded PCB Image</p>', unsafe_allow_html=True)
    st.image(st.session_state.defect_image, use_container_width=True)

# =========================
# Detect Defects
# =========================
if st.session_state.defect_image:
    if st.button("🛠️ Detect Defects"):
        annotated, detected_defects, save_path = run_pipeline(
            st.session_state.defect_image,
            st.session_state.model,
            st.session_state.classes,
            st.session_state.device,
            reference_folder=REFERENCE_FOLDER,
            bbox_color=BBOX_COLOR,
            label_color=LABEL_COLOR,
            min_area=50
        )
        st.session_state.processed_image = annotated
        st.session_state.defect_info = detected_defects

# =========================
# Show Results
# =========================
if st.session_state.defect_info is not None:
    if st.session_state.defect_info:
        st.markdown(f'<h2 style="color:{TEXT_COLOR}; font-size:40px; font-weight:bold;">⚠️ Detected Defects</h2>', unsafe_allow_html=True)
        st.image(st.session_state.processed_image, use_container_width=True)
        st.markdown(f'<h3 style="color:{TEXT_COLOR}; font-size:40px; font-weight:bold;">Defect Details</h3>', unsafe_allow_html=True)
        
        cols = st.columns(len(st.session_state.defect_info))
        for i, defect in enumerate(st.session_state.defect_info):
            with cols[i]:
                st.markdown(f"""
                <div style="
                    background: {DEFECT_CARD_GRADIENT};
                    padding:15px;
                    border-radius:10px;
                ">
                <h4 style="color:{DEFECT_CARD_TEXT};text-align:center;">{defect['type']}</h4>
                <p style="color:{DEFECT_CARD_TEXT};text-align:center;">BBox: {defect['bbox']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Download button
        st.write("---")
        st.write("")
        buf = io.BytesIO()
        st.session_state.processed_image.save(buf, format="PNG")
        st.download_button("📥 Download Processed Image", data=buf.getvalue(), file_name="pcb_defects.png", mime="image/png")
    else:
        st.warning("✅ No defects detected!")
