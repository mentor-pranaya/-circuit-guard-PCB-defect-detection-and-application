# app.py
import streamlit as st
from PIL import Image, ImageChops, ImageDraw, ImageFont
from backend import load_model, run_inference

# ==========================
# Load model once
# ==========================
@st.cache_resource
def get_model():
    return load_model()

model, device = get_model()

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="PCB Defect Detection", page_icon="🔍", layout="centered")
st.title("🔍 PCB Defect Prediction with Template Comparison")

st.write("Upload a **template image** and a **test PCB image** to detect defects in real-time.")

# Upload Template Image
template_file = st.file_uploader("Upload Template Image", type=["jpg", "jpeg", "png"], key="template")

# Upload Test Image
test_file = st.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png"], key="test")

def highlight_differences(template: Image.Image, test: Image.Image):
    """
    Returns an image highlighting differences between template and test image.
    """
    # Make sure images are same size
    test_resized = test.resize(template.size)

    # Compute difference
    diff = ImageChops.difference(template, test_resized)

    # Convert to RGBA to draw red highlights
    diff_highlight = test_resized.convert("RGBA")
    red_overlay = Image.new("RGBA", diff_highlight.size, (255, 0, 0, 100))
    
    # Create mask where differences exist
    mask = diff.convert("L").point(lambda x: 255 if x > 30 else 0)
    diff_highlight.paste(red_overlay, (0, 0), mask)

    return diff_highlight

if template_file and test_file:
    template_img = Image.open(template_file).convert("RGB")
    test_img = Image.open(test_file).convert("RGB")

    st.image(template_img, caption="Template Image", use_container_width=True)
    st.image(test_img, caption="Test Image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Analyzing..."):
            # Run backend prediction on test image
            annotated_img, logs = run_inference(model, test_img, device)

            # Highlight differences
            diff_img = highlight_differences(template_img, test_img)

        st.success(logs)
        st.image(annotated_img, caption="Annotated Test Image", use_container_width=True)
        st.image(diff_img, caption="Differences Highlighted (Red)", use_container_width=True)

st.markdown("---")
st.caption("EfficientNet-B4 Model • Streamlit Frontend • Template Comparison")
