import streamlit as st
import numpy as np
import cv2
import torch
import timm
import json
from PIL import Image
from io import BytesIO

from subtraction import run as subtract_mask  # your refactored subtraction
from contour_visualization import draw_contours_and_labels  # must return annotated cv2 BGR image

# -----------------------------
# Load trained model (whole-image classification)
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints/efficientnet_b4_best.pth"
CLASS_JSON = "checkpoints/class_to_idx.json"

with open(CLASS_JSON, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v:k for k,v in class_to_idx.items()}

# build same model you trained
backbone = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0, global_pool="avg")
classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(backbone.num_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, len(class_to_idx))
)
model = torch.nn.Sequential(backbone, classifier)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

import torchvision.transforms as T
IMG_SIZE = 300
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
val_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("PCB Defect Detection Demo")

col1, col2 = st.columns(2)
with col1:
    golden_file = st.file_uploader("Upload Golden (reference) PCB image", type=["jpg","png"])
with col2:
    defect_file = st.file_uploader("Upload Defective PCB image", type=["jpg","png"])

if golden_file and defect_file:
    golden_img = Image.open(golden_file).convert("RGB")
    defect_img = Image.open(defect_file).convert("RGB")
    st.subheader("Uploaded Images")
    st.image([golden_img, defect_img], caption=["Golden","Defect"], width=250)

    # A "Done" button to run pipeline
    if st.button("✅ Done (Run Full Pipeline)"):
        # 1. subtraction mask
        mask_np = subtract_mask(golden_img, defect_img)
        st.subheader("Difference Mask")
        st.image(mask_np, caption="Difference Mask", channels="GRAY")
        # download mask
        success, encoded = cv2.imencode('.png', mask_np)
        st.download_button("Download Mask", data=encoded.tobytes(), file_name="mask.png")

        # 2. contour visualization (your function must take defect_img + mask_np and return cv2 BGR image)
        annotated_bgr = draw_contours_and_labels(np.array(defect_img), mask_np)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.subheader("Contour Visualization with Labels")
        st.image(annotated_rgb, caption="Contour Visualization", use_column_width=True)
        # download annotated image
        success, encoded2 = cv2.imencode('.png', annotated_bgr)
        st.download_button("Download Annotated Image", data=encoded2.tobytes(), file_name="annotated.png")

        # 3. whole-image classification
        x = val_tfms(defect_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_idx = probs.argmax(1).item()
            pred_class = idx_to_class[pred_idx]
            confidence = probs[0,pred_idx].item()

        st.subheader("Predicted Defect Type for Entire Image")
        st.success(f"**{pred_class}** ({confidence*100:.1f}% confidence)")

st.info("Upload two images, then click 'Done' to run the pipeline (subtraction → contour → classification).")
