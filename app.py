from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import os
import base64

# ---------- CONFIG ----------
REFERENCE_DIR = r"C:/Users/devak/Downloads/PCB_Defect_Project/PCB USED"
MODEL_PATH = r"C:/Users/devak/Downloads/PCB_Defect_Project/efficientnet_b4.pth"
OUTPUT_DIR = r"C:/Users/devak/Downloads/PCB_Defect_Project/Annotated_Test_Images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_COLORS = {
    'Missing_hole':    (0, 0, 255),
    'Mouse_bite':      (0, 255, 255),
    'Open_circuit':    (0, 255, 0),
    'Short':           (255, 0, 0),
    'Spur':            (255, 0, 255),
    'Spurious_copper': (203, 192, 255)
}

# ---------- LOAD MODEL ----------
model = EfficientNet.from_pretrained('efficientnet-b4')
model._fc = torch.nn.Linear(model._fc.in_features, len(CLASSES))
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------- UTILITY FUNCTIONS ----------
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
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    return CLASSES[pred_idx.item()], conf.item()

def annotate_pcb(test_color, mask, border_ignore=10):
    annotated = test_color.copy()
    alpha = 0.6
    defect_counts = {}

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if x <= border_ignore or y <= border_ignore or (x + w) >= (annotated.shape[1]-border_ignore) or (y + h) >= (annotated.shape[0]-border_ignore):
            continue

        roi = test_color[y:y+h, x:x+w]
        pred, conf = predict_roi(roi)

        # Count defects
        defect_counts[pred] = defect_counts.get(pred, 0) + 1

        label = f"{pred} {conf:.2f}"
        color = CLASS_COLORS.get(pred, (255, 255, 255))

        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,0,255), 4)
        rect_overlay = annotated.copy()
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_w, text_h = text_size
        top_y = max(y - text_h - 10, 0)
        cv2.rectangle(rect_overlay, (x, top_y), (x + text_w, y), color, -1)
        cv2.addWeighted(rect_overlay, alpha, annotated, 1 - alpha, 0, annotated)
        cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

    defects_list = [{"type": k, "count": v} for k, v in defect_counts.items()]
    return annotated, defects_list

ref_cache = {}
ref_files = [f for f in os.listdir(REFERENCE_DIR) if f.lower().endswith((".jpg",".png",".jpeg"))]
for f in ref_files:
    path = os.path.join(REFERENCE_DIR, f)
    ref_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ref_gray = cv2.resize(ref_gray, (256, 256))
    hist = cv2.calcHist([ref_gray], [0], None, [256], [0,256])
    hist = cv2.normalize(hist, hist).flatten()
    ref_cache[f] = hist

def get_most_similar_reference(test_gray):
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

def cv2_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

# ---------- FLASK APP ----------
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    file = request.files["file"]
    test_pil = Image.open(file).convert("RGB")
    test_bgr = cv2.cvtColor(np.array(test_pil), cv2.COLOR_RGB2BGR)
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    # Reference PCB
    best_ref, similarity_score, best_file = get_most_similar_reference(test_gray)

    # Mask + Annotate
    mask = make_mask(best_ref, test_gray)
    annotated, defects_list = annotate_pcb(test_bgr, mask, border_ignore=10)

    annotated_b64 = cv2_to_base64(annotated)
    reference_bgr = cv2.cvtColor(best_ref, cv2.COLOR_GRAY2BGR)
    reference_b64 = cv2_to_base64(reference_bgr)

    return jsonify({
        "annotated": annotated_b64,
        "reference": reference_b64,
        "similarity": similarity_score,
        "reference_name": best_file,
        "defects": defects_list
    })

if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True)
