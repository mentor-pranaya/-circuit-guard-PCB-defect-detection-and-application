from flask import Flask, render_template, request, send_file, url_for
import os
import cv2
import torch
import numpy as np
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from pathlib import Path
from skimage.metrics import structural_similarity as ssim  # for similarity

# ==== Paths ====
ROI_DIR = r"C:\vs code\PCB_circuit\PCB_DATASET\PCB_DATASET\roi_Output1"
GOLDEN_DIR = r"C:\vs code\PCB_circuit\PCB_DATASET\PCB_DATASET\PCB_USED"
MODEL_PATH = r"C:\vs code\PCB_circuit\PCB_circuit_output\pcb_model.pth"
OUTPUT_DIR = os.path.join("static", "annotated_images")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ==== Classes & Device ====
CLASSES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load Model ====
def load_model():
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = torch.nn.Linear(model._fc.in_features, len(CLASSES))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ==== Transform ====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ==== Utility Functions ====
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
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    return annotated

# ==== Golden PCB Selection ====
def get_most_similar_golden(test_gray):
    best_score = -1
    best_path = None
    for f in os.listdir(GOLDEN_DIR):
        if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        path = os.path.join(GOLDEN_DIR, f)
        golden = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if golden is None:
            continue
        golden_resized = cv2.resize(golden, (test_gray.shape[1], test_gray.shape[0]))
        score = ssim(test_gray, golden_resized)
        if score > best_score:
            best_score = score
            best_path = path
    if best_path is None:
        return None, None
    golden_img = cv2.imread(best_path, cv2.IMREAD_GRAYSCALE)
    return golden_img, os.path.basename(best_path)

# ==== Flask App ====
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_info = None
    annotated_url = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")
        try:
            img = Image.open(file).convert("RGB")
        except Exception:
            return render_template("index.html", error="Invalid image file")

        test_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

        # Select most similar golden PCB
        golden_gray, golden_file = get_most_similar_golden(test_gray)
        if golden_gray is None:
            return render_template("index.html", error="No valid golden PCB found.")

        # Mask + annotation
        mask = make_mask(golden_gray, test_gray)
        annotated = annotate_pcb(test_bgr, mask)

        annotated_path = os.path.join(OUTPUT_DIR, "annotated_result.jpg")
        cv2.imwrite(annotated_path, annotated)
        annotated_url = url_for('static', filename='annotated_images/annotated_result.jpg')

        prediction_info = f"Most similar golden PCB: {golden_file}"

    return render_template("index.html",
                           prediction=prediction_info,
                           annotated_url=annotated_url)

@app.route("/download")
def download():
    annotated_path = os.path.join(OUTPUT_DIR, "annotated_result.jpg")
    if not os.path.exists(annotated_path):
        return "No file to download."
    return send_file(annotated_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
