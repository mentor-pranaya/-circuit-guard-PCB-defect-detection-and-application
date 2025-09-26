import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
from efficientnet_pytorch import EfficientNet

# =========================
# ==== Device Setup =======
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# ==== Paths Setup ========
# =========================
MODEL_FILE = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\efficientnet_b4_best (2).pth"
SUBTRACTED_DIR = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\data\subtracted_images"
os.makedirs(SUBTRACTED_DIR, exist_ok=True)

# =========================
# ==== Load Checkpoint ====
# =========================
ckpt = torch.load(MODEL_FILE, map_location=device)
label_classes = ckpt.get("classes")

# =========================
# ==== Model Definition ===
# =========================
class PCBNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4")
        in_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)

model = PCBNet(num_classes=len(label_classes)).to(device)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)
model.eval()

# =========================
# ==== Image Transform ====
# =========================
img_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# ==== Streamlit UI =======
# =========================
st.title("⚡ PCB Defect Detection")
st.write("Upload a template and a test PCB image to identify defects.")

template_img_file = st.file_uploader("Template Image", type=["jpg", "png", "jpeg"])
test_img_file = st.file_uploader("Test Image", type=["jpg", "png", "jpeg"])

if template_img_file and test_img_file:
    # Open images
    template_img = Image.open(template_img_file).convert("RGB")
    test_img = Image.open(test_img_file).convert("RGB")

    st.image([template_img, test_img], caption=["Template", "Test"], width=300)

    # Convert to grayscale for subtraction
    template_gray = cv2.cvtColor(np.array(template_img), cv2.COLOR_RGB2GRAY)
    test_gray = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2GRAY)

    # Compute difference
    diff_img = cv2.absdiff(template_gray, test_gray)
    _, bin_img = cv2.threshold(diff_img, 50, 255, cv2.THRESH_BINARY)

    # Save and display
    out_path = os.path.join(SUBTRACTED_DIR, "subtracted.png")
    cv2.imwrite(out_path, bin_img)
    st.image(out_path, caption="Subtracted Image", width=400)

    # Prediction
    tensor_img = img_transform(test_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor_img)
        pred_class_idx = torch.argmax(output, 1).item()
        pred_class = label_classes[pred_class_idx]

    st.success(f"✅ Detected Defect: **{pred_class}**")
