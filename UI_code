import streamlit as st
from PIL import Image
import time

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------
st.set_page_config(
    page_title="CircuitGuard — PCB Defect Detection",
    layout="wide",
    page_icon="🛡️"
)

# ------------------------------------------
# CUSTOM CSS FOR ANIMATION & STYLING
# ------------------------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            background: linear-gradient(90deg, #0072ff, #00c6ff, #00ff87, #ff6a00, #ff00c8);
            -webkit-background-clip: text;
            color: transparent;
            animation: glow 4s infinite linear;
        }
        @keyframes glow {
            0% { text-shadow: 0 0 5px #00c6ff; }
            50% { text-shadow: 0 0 20px #ff6a00; }
            100% { text-shadow: 0 0 5px #00c6ff; }
        }
        .sub-title {
            font-size: 18px;
            text-align: center;
            color: #ccc;
        }
        .footer {
            text-align: center;
            color: #aaa;
            font-size: 14px;
            margin-top: 40px;
        }
        .uploaded-img {
            border-radius: 12px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.4);
        }
        div[data-testid="stFileUploader"] > label {
            font-weight: bold;
            color: #00c6ff;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------
# HEADER
# ------------------------------------------
st.markdown('<div class="main-title">🧠 CircuitGuard — PCB Defect Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload your Original & Defected PCB images to detect and classify defects</div>', unsafe_allow_html=True)
st.write("")

# ------------------------------------------
# IMAGE UPLOAD SECTION
# ------------------------------------------
col1, col2 = st.columns(2)
with col1:
    ref_image = st.file_uploader("📥 Upload Original (Reference) PCB", type=["jpg", "png", "jpeg"])
with col2:
    defect_image = st.file_uploader("⚙️ Upload Defected PCB", type=["jpg", "png", "jpeg"])

# ------------------------------------------
# BUTTONS
# ------------------------------------------
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    detect_btn = st.button("🔍 Run Defect Detection", use_container_width=True)
with col_btn2:
    download_btn = st.button("⬇️ Download Results", use_container_width=True)

# ------------------------------------------
# DISPLAY UPLOADED IMAGES
# ------------------------------------------
if ref_image and defect_image:
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(ref_image, caption="Reference PCB", use_column_width=True)
    with col_b:
        st.image(defect_image, caption="Defected PCB", use_column_width=True)

# ------------------------------------------
# SIMULATED DETECTION (UI Animation Only)
# ------------------------------------------
if detect_btn:
    st.info("🧩 Running Defect Detection Pipeline...")
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress.progress(i + 1)
    st.success("✅ Detection Completed! Results Ready Below.")

    # Placeholder output image
    st.image("https://cdn-icons-png.flaticon.com/512/3940/3940045.png",
             caption="Defected PCB with highlighted defects (sample preview)",
             use_column_width=True)

    # Fake defect summary
    st.write("### 🔎 Defect Summary")
    st.markdown("""
    | Defect Type | Count | Severity |
    |--------------|--------|-----------|
    | Mouse Bite   | 3 | ⚠️ Medium |
    | Missing Hole | 1 | 🔴 High |
    | Open Circuit | 2 | ⚠️ Medium |
    """)

# ------------------------------------------
# DOWNLOAD SECTION
# ------------------------------------------
if download_btn:
    st.success("📁 Your processed file is ready for download (simulation).")

# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit  |  © 2025 CircuitGuard AI Team
</div>
""", unsafe_allow_html=True)


