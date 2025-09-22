import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

# ================================
# Streamlit Page Configuration
# ================================
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="⚡",
    layout="wide"
)

# ================================
# Color and Style Configuration
# ================================
HEADER_BG = "#5BD0E5"
HEADER_TEXT = "#051676"
SUBTEXT = "#051676"

DEFECT_CARD_BG = "#5BD0E5"
DEFECT_CARD_TEXT = "#051676"

# Default bounding box and label colors
BBOX_COLOR = "#5BD0E5"
LABEL_COLOR = "#051676"

# Heading color for detected defects
DETECTED_DEFECTS_COLOR = "#5BD0E5"

# Override bounding box and label styles
BBOX_COLOR = "#BD2315"
LABEL_COLOR = "#650C04"
LABEL_FONT_SCALE = 1.0
LABEL_THICKNESS = 4
BBOX_THICKNESS = 3

# ================================
# Custom Header UI
# ================================
st.markdown(f"""
    <div style="background-color:{HEADER_BG};padding:20px;border-radius:10px">
    <h1 style="color:{HEADER_TEXT};text-align:center;">⚡ PCB Defect Detection ⚡</h1>
    <p style="color:{SUBTEXT};text-align:center;font-size:18px;">
    Upload your PCB image to detect defects in real-time!
    </p>
    </div>
""", unsafe_allow_html=True)

st.write("---")

# ================================
# Initialize Session State
# ================================
# Used to persist data across interactions
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "defect_info" not in st.session_state:
    st.session_state.defect_info = None
if "input_image" not in st.session_state:
    st.session_state.input_image = None

# ================================
# File Uploader for PCB Images
# ================================
uploaded_file = st.file_uploader("Choose a PCB image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load and store input image
    st.session_state.input_image = Image.open(uploaded_file).convert("RGB")
    st.image(st.session_state.input_image, caption="Uploaded Image", width='stretch')

# ================================
# Process Image Button
# ================================
if st.session_state.input_image and st.button("🛠️ Detect Defects"):

    input_image = st.session_state.input_image
    # Convert PIL image to OpenCV (BGR format)
    img_cv = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    # Example defects (replace with actual ML model outputs)
    detected_defects = [
        {"bbox": (50, 50, 150, 150), "type": "Scratch"},
        {"bbox": (200, 100, 300, 200), "type": "Mouse Bite"}
    ]

    # Convert hex colors to BGR for OpenCV
    bbox_bgr = tuple(int(BBOX_COLOR.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))[::-1]
    label_bgr = tuple(int(LABEL_COLOR.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))[::-1]

    # Draw bounding boxes and labels on image
    output_img = img_cv.copy()
    for defect in detected_defects:
        x1, y1, x2, y2 = defect["bbox"]
        cv2.rectangle(output_img, (x1, y1), (x2, y2), bbox_bgr, BBOX_THICKNESS)
        cv2.putText(output_img, defect["type"], (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, label_bgr, LABEL_THICKNESS)

    # Convert processed OpenCV image back to PIL
    output_pil = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))

    # Store results in session state
    st.session_state.processed_image = output_pil
    st.session_state.defect_info = detected_defects

# ================================
# Display Results if Available
# ================================
if st.session_state.processed_image:

    # Show heading
    st.markdown(f"<h2 style='color:{DETECTED_DEFECTS_COLOR};'>✅ Detected Defects</h2>", unsafe_allow_html=True)
    st.image(st.session_state.processed_image, caption="Processed Image", width='stretch')

    # Show defect details in columns
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

    # Add spacing before download button
    st.markdown("<div style='margin-top:30px'></div>", unsafe_allow_html=True)

    # Prepare processed image for download
    buf = io.BytesIO()
    st.session_state.processed_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    # Download button
    st.download_button(
        label="📥 Download Processed Image",
        data=byte_im,
        file_name="pcb_defects.png",
        mime="image/png"
    )
