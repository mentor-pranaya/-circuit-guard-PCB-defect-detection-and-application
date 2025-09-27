import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import os

# CONFIG - update paths
REFERENCE_DIR = r"C:/Users/devak/Downloads/PCB_Defect_Project/PCB USED"
MODEL_PATH = r"C:/Users/devak/Downloads/PCB_Defect_Project/efficientnet_b4.pth"
OUTPUT_DIR = r"C:/Users/devak/Downloads/PCB_Defect_Project/Annotated_Test_Images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model

@st.cache_resource
def load_model():
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = torch.nn.Linear(model._fc.in_features, len(CLASSES))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()


# Transform

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Utility functions

def make_mask(template_gray, test_gray):
    if template_gray.shape != test_gray.shape:
        template_gray = cv2.resize(template_gray, (test_gray.shape[1], test_gray.shape[0]))
    diff = cv2.absdiff(cv2.GaussianBlur(template_gray, (3,3), 0),
                       cv2.GaussianBlur(test_gray, (3,3), 0))
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def predict_roi(roi_bgr):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_pil = Image.fromarray(roi_rgb)
    tensor = transform(roi_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        pred = out.argmax(1).item()
    return CLASSES[pred]

def annotate_pcb(test_color, mask):
    annotated = test_color.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = test_color[y:y+h, x:x+w]
        pred = predict_roi(roi)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,0,255), 2)
        # Green text with black outline for readability
        cv2.putText(annotated, pred, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)  # outline
        cv2.putText(annotated, pred, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)  # main text
    return annotated

# Streamlit UI
st.title("PCB Defect Detection App")

# Load reference PCBs
ref_files = [f for f in os.listdir(REFERENCE_DIR) if f.lower().endswith((".jpg",".png",".jpeg"))]
if not ref_files:
    st.error("⚠️ No reference PCBs found in PCB USED folder!")
    st.stop()

selected_ref = st.selectbox("Select Reference PCB", ref_files)
reference_path = os.path.join(REFERENCE_DIR, selected_ref)
template_gray = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

uploaded = st.file_uploader("Upload a Test PCB Image", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    test_pil = Image.open(uploaded).convert("RGB")
    st.image(test_pil, caption="Uploaded Test PCB")

    if st.button("Run Detection"):
        st.info("🔍 Processing... Please wait")

        test_bgr = cv2.cvtColor(np.array(test_pil), cv2.COLOR_RGB2BGR)
        test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

        mask = make_mask(template_gray, test_gray)
        annotated = annotate_pcb(test_bgr, mask)

        save_path = os.path.join(OUTPUT_DIR, "annotated_result.jpg")
        cv2.imwrite(save_path, annotated)

        # Save path in session state
        st.session_state["annotated_path"] = save_path

# If annotated exists, always show + download
if "annotated_path" in st.session_state:
    annotated = cv2.imread(st.session_state["annotated_path"])
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Defects Annotated")
    with open(st.session_state["annotated_path"], "rb") as f:
        st.download_button("⬇️ Download Annotated Image", f, file_name="annotated_result.jpg")
