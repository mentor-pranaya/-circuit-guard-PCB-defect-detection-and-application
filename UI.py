import streamlit as st
from pipeline import load_model, run_pipeline
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="PCB Defect Detection", page_icon="⚡", layout="wide")

# Colors
HEADER_BG = "#5BD0E5"; HEADER_TEXT = "#051676"; SUBTEXT = "#051676"
DEFECT_CARD_BG = "#5BD0E5"; DEFECT_CARD_TEXT ="#051676"
DETECTED_DEFECTS_COLOR = "#5BD0E5"
BBOX_COLOR = "#841B12"; LABEL_COLOR = "#BA362A"

# Header
st.markdown(f"""
<div style="background-color:{HEADER_BG};padding:20px;border-radius:10px">
<h1 style="color:{HEADER_TEXT};text-align:center;">⚡ PCB Defect Detection ⚡</h1>
<p style="color:{SUBTEXT};text-align:center;font-size:18px;">
Upload your PCB image to detect defects in real-time!
</p>
</div>
""", unsafe_allow_html=True)
st.write("---")

# Load model
MODEL_PATH = r"C:\Users\kavya\PCB DEFECT DETECTION\best_model_full.pth"
REFERENCE_FOLDER = r"C:\Users\kavya\PCB DEFECT DETECTION\references"

if "model" not in st.session_state:
    st.session_state.model, st.session_state.classes, st.session_state.device = load_model(MODEL_PATH)

# Upload
st.subheader("📤 Upload PCB Image")
defect_file = st.file_uploader("Upload PCB", type=["png","jpg","jpeg"])

if defect_file:
    st.session_state.defect_image = Image.open(defect_file).convert("RGB")
    st.image(st.session_state.defect_image, caption="Uploaded PCB Image", width = 'stretch')

# Detect
if "defect_image" in st.session_state:
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

# Show results
if "defect_info" in st.session_state:
    if st.session_state.defect_info:
        st.markdown(f"<h2 style='color:{DETECTED_DEFECTS_COLOR};'>⚠️ Detected Defects</h2>", unsafe_allow_html=True)
        st.image(st.session_state.processed_image, caption="Processed Image", width = 'stretch')
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
        # Download button
        buf = io.BytesIO()
        st.session_state.processed_image.save(buf, format="PNG")
        st.download_button("📥 Download Processed Image", data=buf.getvalue(), file_name="pcb_defects.png", mime="image/png")
    else:
        st.warning("✅ No defects detected!")
