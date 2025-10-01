import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import os

# Paths
REFERENCE_DIR = r"C:/Users/devak/Downloads/PCB_Defect_Project/PCB USED"
MODEL_PATH = r"C:/Users/devak/Downloads/PCB_Defect_Project/efficientnet_b4.pth"
OUTPUT_DIR = r"C:/Users/devak/Downloads/PCB_Defect_Project/Annotated_Test_Images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
@st.cache_resource
def load_model():
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = torch.nn.Linear(model._fc.in_features, len(CLASSES))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Utility Functions
@st.cache_data
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
        pred = model(tensor).argmax(1).item()
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
        cv2.putText(annotated, pred, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
        cv2.putText(annotated, pred, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    return annotated

# Fast Similarity Check
@st.cache_data
def load_reference_histograms():
    """Precompute histograms of all golden PCBs"""
    ref_cache = {}
    ref_files = [f for f in os.listdir(REFERENCE_DIR) if f.lower().endswith((".jpg",".png",".jpeg"))]
    for f in ref_files:
        path = os.path.join(REFERENCE_DIR, f)
        ref_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ref_gray = cv2.resize(ref_gray, (256, 256))  # small size for fast computation
        hist = cv2.calcHist([ref_gray], [0], None, [256], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        ref_cache[f] = hist
    return ref_cache

@st.cache_data
def get_most_similar_reference(test_gray, ref_cache):
    test_gray_small = cv2.resize(test_gray, (256,256))
    test_hist = cv2.calcHist([test_gray_small], [0], None, [256], [0,256])
    test_hist = cv2.normalize(test_hist, test_hist).flatten()

    best_score, best_file = -1, None
    for f, hist in ref_cache.items():
        score = cv2.compareHist(hist, test_hist, cv2.HISTCMP_CORREL)
        if score > best_score:
            best_score = score
            best_file = f

    best_ref = cv2.imread(os.path.join(REFERENCE_DIR, best_file), cv2.IMREAD_GRAYSCALE)
    return best_ref, best_score, best_file

# Precompute golden PCB histograms once
ref_cache = load_reference_histograms()

# Streamlit UI
st.title("PCB Defect Detection")

uploaded = st.file_uploader("Upload a Test PCB Image", type=["jpg", "png", "jpeg"])

if uploaded:
    test_pil = Image.open(uploaded).convert("RGB")
    st.image(test_pil, caption="Uploaded Test PCB")

    if st.button("Run Detection"):
        st.info("🔍 Processing... Please wait")

        test_bgr = cv2.cvtColor(np.array(test_pil), cv2.COLOR_RGB2BGR)
        test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

        # Fast similarity check
        best_ref, similarity_score, best_file = get_most_similar_reference(test_gray, ref_cache)
        st.success(f"Most similar reference PCB: {best_file} (Score: {similarity_score:.4f})")

        # Mask and annotation
        mask = make_mask(best_ref, test_gray)
        annotated = annotate_pcb(test_bgr, mask)

        save_path = os.path.join(OUTPUT_DIR, "annotated_result.jpg")
        cv2.imwrite(save_path, annotated)
        st.session_state["annotated_path"] = save_path

# Display annotated image and download
if "annotated_path" in st.session_state:
    annotated = cv2.imread(st.session_state["annotated_path"])
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Defects Annotated")
    with open(st.session_state["annotated_path"], "rb") as f:
        st.download_button("⬇️ Download Annotated Image", f, file_name="annotated_result.jpg")
