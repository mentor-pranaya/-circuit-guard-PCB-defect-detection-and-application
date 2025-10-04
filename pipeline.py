import os
import cv2
import torch
import timm
import torchvision.transforms as T
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ----------------------------
# Label mapping
# ----------------------------
label_to_name = {
    0: "Missing Hole",
    1: "Mouse Bite",
    2: "Open Circuit",
    3: "Short",
    4: "Spurious Copper",
    5: "Spur"
}

# ----------------------------
# Load trained model
# ----------------------------
def load_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint["num_classes"]
    label_map = checkpoint["label_map"]

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = [label_map[i] for i in range(num_classes)]
    return model, class_names, device

# ----------------------------
# Predict ROI
# ----------------------------
def predict_roi(model, roi, device):
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    tensor = transform(roi_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        pred_idx = out.argmax(1).item()
    return label_to_name[pred_idx]

# ----------------------------
# Annotate image
# ----------------------------
def annotate_image(image, rois, predictions, bbox_color="#73170F", label_color="#B82012"):
    def hex2bgr(hex_color):
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2],16) for i in (4,2,0))  # BGR
    annotated = image.copy()
    bbox_bgr = hex2bgr(bbox_color)
    label_bgr = hex2bgr(label_color)
    for idx, ((roi,(x,y,w,h)), pred) in enumerate(zip(rois, predictions), 1):
        cv2.rectangle(annotated, (x,y), (x+w,y+h), bbox_bgr, 5)
        cv2.putText(annotated, f"{pred}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_bgr, 5)
        print(f"[PRED] ROI {idx}: {pred}")
    return annotated

# ----------------------------
# Load reference PCBs
# ----------------------------
def load_reference_pcbs(reference_folder, size=(256,256)):
    ref_images, filenames = [], []
    for file in os.listdir(reference_folder):
        if file.lower().endswith((".png",".jpg",".jpeg")):
            path = os.path.join(reference_folder, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, size)
            ref_images.append(img)
            filenames.append(file)
    return ref_images, filenames

# ----------------------------
# Select best reference using SSIM
# ----------------------------
def select_best_reference(uploaded_img, ref_images, ref_filenames, size=(256,256)):
    uploaded_gray = cv2.cvtColor(np.array(uploaded_img), cv2.COLOR_RGB2GRAY)
    uploaded_gray = cv2.resize(uploaded_gray, size)
    best_score, best_idx = -1, 0
    for i, ref in enumerate(ref_images):
        score = ssim(uploaded_gray, ref)
        if score > best_score:
            best_score = score
            best_idx = i
    print(f"[INFO] Best reference PCB: {ref_filenames[best_idx]} (SSIM={best_score:.4f})")
    return best_idx, ref_filenames[best_idx]

# ----------------------------
# Full pipeline
# ----------------------------
def run_pipeline(uploaded_img, model, classes, device,
                 reference_folder=r"C:\Users\kavya\PCB_DEFECT_DETECTION\references",
                 bbox_color="#73170F", label_color="#B82012",
                 min_area=50, border_margin=10, padding=10):
    
    # Load references
    ref_images, ref_filenames = load_reference_pcbs(reference_folder)
    
    # Select best reference
    best_idx, best_file = select_best_reference(uploaded_img, ref_images, ref_filenames)
    ref_img = cv2.imread(os.path.join(reference_folder, best_file), cv2.IMREAD_GRAYSCALE)
    
    # Uploaded image grayscale
    uploaded_gray = np.array(uploaded_img.convert("L"))
    if uploaded_gray.shape != ref_img.shape:
        uploaded_gray = cv2.resize(uploaded_gray, (ref_img.shape[1], ref_img.shape[0]))
    
    # Subtraction + Gaussian + Otsu + morphology
    diff = cv2.absdiff(uploaded_gray, ref_img)
    diff_blur = cv2.GaussianBlur(diff, (5,5), 0)
    _, thresh = cv2.threshold(diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find all contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[INFO] Total contours found (before smart filtering): {len(contours)}")
    
    # Smart filtering (only remove tiny noise)
    filtered_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            filtered_contours.append(cnt)
    print(f"[INFO] Contours after smart filtering: {len(filtered_contours)}")
    
    # Extract ROIs
    rois = []
    h_img, w_img = uploaded_gray.shape[:2]
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x <= border_margin or y <= border_margin or (x+w) >= (w_img-border_margin) or (y+h) >= (h_img-border_margin):
            continue
        x1, y1 = max(0,x-padding), max(0,y-padding)
        x2, y2 = min(w_img, x+w+padding), min(h_img, y+h+padding)
        roi = cv2.cvtColor(np.array(uploaded_img), cv2.COLOR_RGB2BGR)[y1:y2, x1:x2]
        rois.append((roi,(x,y,w,h)))
    
    print(f"[INFO] ROIs extracted: {len(rois)}")
    
    # Predict defects
    predictions = [predict_roi(model, roi, device) for roi,_ in rois]
    
    # Annotate
    annotated = annotate_image(cv2.cvtColor(np.array(uploaded_img), cv2.COLOR_RGB2BGR),
                               rois, predictions, bbox_color, label_color)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    
    # Defect info
    defects_info = [{"bbox":(x,y,x+w,y+h), "type":pred} for (roi,(x,y,w,h)), pred in zip(rois,predictions)]
    
    # Save
    os.makedirs("outputs", exist_ok=True)
    save_path = os.path.join("outputs", "annotated_result.png")
    annotated_pil.save(save_path)
    
    return annotated_pil, defects_info, save_path
