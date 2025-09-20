import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ==================================
# 1. SETUP & CONFIGURATION
# ==================================
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for easy frontend communication

# --- Paths ---
MODEL_PATH = os.path.join("outputs", "best_checkpoint.pth")
ROI_BASE_SAVE_DIR = "temp_rois"
os.makedirs(ROI_BASE_SAVE_DIR, exist_ok=True)

# ==================================
# 2. LOAD THE TRAINED MODEL
# ==================================
# This section is adapted from your test script to load the model on server startup.
# ==================================
# 2. LOAD THE TRAINED MODEL
# ==================================
import timm  # <-- Make sure to import timm at the top of your file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
classes = []

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    # CORRECTED: Use timm.create_model to match the training script architecture
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: Model checkpoint not found at '{MODEL_PATH}'. The prediction API will not work.")
except Exception as e:
    print(f"❌ ERROR: An error occurred while loading the model: {e}")

# Image transformation for the model
# Image transformation for the model
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # CORRECTED LINE: Matches the validation transform from your training script
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. FULL INFERENCE PIPELINE FUNCTIONS
# ==========================================
# These functions are adapted from your Streamlit app and combined here.

def subtract_images(defect_img, ref_img):
    # Convert PIL images to OpenCV format
    defect_color = np.array(defect_img.convert('RGB'))
    defect_gray = cv2.cvtColor(defect_color, cv2.COLOR_RGB2GRAY)
    ref_gray = np.array(ref_img.convert('L')) # Convert reference to grayscale

    # Align images by resizing
    if ref_gray.shape != defect_gray.shape:
        ref_gray = cv2.resize(ref_gray, (defect_gray.shape[1], defect_gray.shape[0]))

    # Image subtraction and mask generation
    defect_blur = cv2.GaussianBlur(defect_gray, (5, 5), 0)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    subtracted = cv2.absdiff(defect_blur, ref_blur)
    binary = cv2.adaptiveThreshold(subtracted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, -5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)
    return defect_color, cleaned

def extract_rois(orig_img, mask):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:  # Filter out small noise
            continue
        roi = orig_img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        rois.append({'image': roi, 'bbox': (x, y, x+w, y+h)})
    return rois

def predict_roi(roi_img, model, class_names, transform, device):
    # Convert OpenCV ROI back to PIL Image for transformation
    roi_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(roi_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    label = class_names[probs.argmax()]
    confidence = float(probs.max())
    return label, confidence

# ==================================
# 4. FLASK API ENDPOINTS
# ==================================

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    """The main API endpoint for the full detection pipeline."""
    if not model:
        return jsonify({"error": "Model is not loaded on the server."}), 500

    if 'ref_image' not in request.files or 'test_image' not in request.files:
        return jsonify({"error": "Missing reference or test image."}), 400

    ref_file = request.files['ref_image']
    test_file = request.files['test_image']
    
    try:
        ref_pil = Image.open(ref_file.stream).convert("RGB")
        test_pil = Image.open(test_file.stream).convert("RGB")

        # 1. Image Subtraction
        original_image, mask = subtract_images(test_pil, ref_pil)
        
        # 2. ROI Extraction
        rois = extract_rois(original_image, mask)
        
        annotated_image = original_image.copy()
        predictions = []

        # 3. Predict each ROI
        for roi_data in rois:
            label, conf = predict_roi(roi_data['image'], model, classes, transform, device)
            predictions.append({'label': label, 'confidence': f"{conf:.2%}"})
            
            # Draw bounding box and label on the image
            x1, y1, x2, y2 = roi_data['bbox']
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_image, f"{label} ({conf:.1%})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert annotated image to base64 to send in JSON
        _, buffer = cv2.imencode('.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'annotated_image': f'data:image/png;base64,{img_base64}',
            'predictions': predictions
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal error occurred during processing."}), 500

# ==================================
# 5. RUN THE APPLICATION
# ==================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

