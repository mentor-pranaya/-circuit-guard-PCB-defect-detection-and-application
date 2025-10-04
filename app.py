# app.py (Streamlit Web UI)
import streamlit as st
from PIL import Image
from datetime import datetime
import os, glob
import inference_backend as ib

st.set_page_config(page_title="AI-CircuitGuard — Inference UI", layout="wide")

# -------------------------
# Config
# -------------------------
DEFAULT_MODEL_PATH = "outputs_training/efficientnet_b4_best.pth"
TEMPLATES_DIR = "outputs_pairs/templates"
OUTPUT_ROOT = "outputs_module6"

# -------------------------
# Sidebar: Inputs
# -------------------------
st.sidebar.header("Inputs & Settings")

# ---- Template selection
st.sidebar.subheader("Template Image")
template_source = st.sidebar.radio(
    "Select template source",
    ["From templates folder", "Upload from computer", "Enter custom path"], index=0
)

tpl_pil, template_name = None, None
if template_source == "From templates folder":
    template_files = sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*.*")))
    template_choice = st.sidebar.selectbox("Choose template", ["-- choose --"] + [os.path.basename(f) for f in template_files])
    if template_choice != "-- choose --":
        tpl_pil = Image.open(os.path.join(TEMPLATES_DIR, template_choice)).convert("RGB")
        template_name = template_choice
elif template_source == "Upload from computer":
    template_upload = st.sidebar.file_uploader("Upload template image", type=["jpg","jpeg","png"])
    if template_upload:
        tpl_pil = Image.open(template_upload).convert("RGB")
        template_name = template_upload.name
elif template_source == "Enter custom path":
    custom_path = st.sidebar.text_input("Enter template path")
    if custom_path and os.path.exists(custom_path):
        tpl_pil = Image.open(custom_path).convert("RGB")
        template_name = os.path.basename(custom_path)

# ---- Test selection
st.sidebar.subheader("Test Image")
test_source = st.sidebar.radio("Select test source", ["Upload from computer", "Enter custom path"], index=0)

test_pil, test_name = None, None
if test_source == "Upload from computer":
    test_upload = st.sidebar.file_uploader("Upload test image", type=["jpg","jpeg","png"])
    if test_upload:
        test_pil = Image.open(test_upload).convert("RGB")
        test_name = test_upload.name
elif test_source == "Enter custom path":
    custom_test_path = st.sidebar.text_input("Enter test image path")
    if custom_test_path and os.path.exists(custom_test_path):
        test_pil = Image.open(custom_test_path).convert("RGB")
        test_name = os.path.basename(custom_test_path)

# -------------------------
# Show Inputs immediately
# -------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Template Image")
    if tpl_pil:
        st.image(tpl_pil, caption=f"Template: {template_name}", use_container_width=True)
    else:
        st.info("Please provide a Template Image.")

with col2:
    st.subheader("Test Image")
    if test_pil:
        st.image(test_pil, caption=f"Test: {test_name}", use_container_width=True)
    else:
        st.info("Please provide a Test Image.")

# ---- Model settings
st.sidebar.write("Model / device")
model_path = st.sidebar.text_input("Model path", DEFAULT_MODEL_PATH)
device_opt = st.sidebar.selectbox("Device", ["cpu", "cuda"] if ib.torch.cuda.is_available() else ["cpu"])
morph_iter = st.sidebar.slider("Morph iterations", 0, 5, 1)
sensitivity = st.sidebar.slider("Sensitivity", 0.5, 2.0, 1.0, step=0.1)
run_button = st.sidebar.button("Run pipeline")

# -------------------------
# Run pipeline
# -------------------------
if run_button:
    if tpl_pil is None:
        st.error("Template missing.")
    elif test_pil is None:
        st.error("Test missing.")
    elif not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
    else:
        st.info("🔎 Running inference pipeline...")

        # Load model
        model, device = ib.load_model(model_path, device_opt)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = ib.run_single_pair(
            tpl_pil, test_pil, model, device,
            output_root=OUTPUT_ROOT,
            run_id=run_id,
            morph_iter=morph_iter,
            sensitivity=sensitivity
        )

        st.success(f"✅ Run {run_id} completed")

        # Show outputs
        st.image(Image.open(result["saved_paths"]["mask"]), caption="Defect Mask", use_container_width=True)
        st.image(Image.open(result["saved_paths"]["annotated"]), caption="Annotated Test", use_container_width=True)

        # Download buttons
        st.download_button("Download Predictions CSV",
                           open(result["saved_paths"]["predictions_csv"], "rb").read(),
                           file_name=f"predictions_{run_id}.csv",
                           mime="text/csv")

        st.download_button("Download Annotated Image",
                           open(result["saved_paths"]["annotated"], "rb").read(),
                           file_name=f"annotated_{run_id}.png",
                           mime="image/png")

        st.download_button("Download Full Run (ZIP)",
                           open(result["saved_paths"]["zip"], "rb").read(),
                           file_name=f"run_{run_id}.zip",
                           mime="application/zip")
