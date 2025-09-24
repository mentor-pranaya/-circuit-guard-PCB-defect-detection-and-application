# pipeline.py

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image


# Mapping numeric labels to defect names
label_to_name = {
    0: "Missing Hole",
    1: "Mouse",
    2: "Open Circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious Copper"
}

# ----------------------------
# Load Trained Model
# ----------------------------
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    classes = checkpoint['classes']
    num_classes = len(classes)

    # Use torchvision EfficientNet (matches training)
    import torchvision.models as models

    model = models.efficientnet_b4(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, classes, device

# ----------------------------
# Model Prediction
# ----------------------------
def predict_roi(model, roi, device, classes):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),   # match training size
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(roi).unsqueeze(0).to(device)
        
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    # Convert numeric prediction to integer if necessary
    numeric_label = int(classes[pred.item()])

    # Map numeric label to defect name safely
    defect_name = label_to_name.get(numeric_label, "Unknown")
    return defect_name




# ----------------------------
# Annotate Image
# ----------------------------
def annotate_image(
    image, rois, predictions,
    bbox_color="#73170F",   
    label_color="#B82012",  
    bbox_thickness=5,
    label_font_scale=1.5,
    label_thickness=2
):
    # Convert hex color to BGR for OpenCV
    def hex2bgr(hex_color):
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))  # BGR order

    bbox_color_bgr = hex2bgr(bbox_color)
    label_color_bgr = hex2bgr(label_color)

    annotated = image.copy()
    for (roi, (x, y, w, h)), pred in zip(rois, predictions):
        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x+w, y+h), bbox_color_bgr, bbox_thickness)
        # Put defect name
        cv2.putText(annotated, str(pred), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    label_font_scale, label_color_bgr, label_thickness)
    return annotated


# ----------------------------
# Full Pipeline
# ----------------------------
def run_pipeline(ref_image, defect_image, model, classes, device,
                 bbox_color="#BD2315", label_color="#650C04"):
    # Convert PIL to OpenCV
    img_cv = np.array(defect_image)

    # Step 1: Image subtraction
    ref_cv = np.array(ref_image)
    diff = cv2.absdiff(ref_cv, img_cv)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 3: Extract ROIs
    rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h < 100:  # min area
            continue
        roi = img_cv[y-5:y+h+5, x-5:x+w+5]
        rois.append((roi, (x, y, w, h)))

    if len(rois) == 0:
        return defect_image, []

    # Step 4: Predict
    predictions = [predict_roi(model, roi, device, classes) for roi, _ in rois]

    # Step 5: Annotate
    annotated = annotate_image(img_cv, rois, predictions,
                               bbox_color=bbox_color, label_color=label_color)

    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    defects_info = [{"bbox": (x, y, x+w, y+h), "type": pred} for (roi, (x, y, w, h)), pred in zip(rois, predictions)]

    return annotated_pil, defects_info
