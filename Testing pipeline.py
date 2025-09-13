# ===========================
# Full Automatic PCB Testing - Clean Final Version
# ===========================
!pip install torch torchvision matplotlib pillow opencv-python --quiet

import os, cv2, torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files

# ===========================
# FOLDER SETUP
# ===========================
BASE_FOLDER = "PCB_DATASET"
PCB_USED_PATH = os.path.join(BASE_FOLDER, "PCB_USED")
ROI_BASE = os.path.join(BASE_FOLDER, "Pipeline_ROIs")
SUBTRACTED_SAVE = os.path.join(BASE_FOLDER, "subtracted_images")
MODEL_PATH = os.path.join(BASE_FOLDER, "output/efficientnet_b4_best.pth")

os.makedirs(ROI_BASE, exist_ok=True)
os.makedirs(SUBTRACTED_SAVE, exist_ok=True)

# ===========================
# LOAD TRAINED MODEL
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint['classes']
num_classes = len(classes)

model = models.efficientnet_b4(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

idx_to_class = {v: k for k, v in checkpoint.get('class_to_idx', {}).items()}
print("✅ Model loaded with classes:", classes)

val_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# ===========================
# IMAGE SUBTRACTION
# ===========================
def subtract_images(defect_img_path, pcb_used_folder, save_mask_path=None):
    img_name = os.path.basename(defect_img_path)
    pcb_number = img_name.split("_")[0]  # e.g. "01"

    defect_gray = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)
    defect_color = cv2.imread(defect_img_path)
    if defect_gray is None:
        raise ValueError(f"❌ Could not load defect image: {defect_img_path}")

    # Match reference PCB (case-insensitive, ignore extension)
    ref_file = None
    for f in os.listdir(pcb_used_folder):
        if os.path.splitext(f)[0].lower() == pcb_number.lower():
            ref_file = os.path.join(pcb_used_folder, f)
            break
    if ref_file is None:
        raise ValueError(f"❌ No reference PCB found for {pcb_number}")

    ref_img = cv2.imread(ref_file, cv2.IMREAD_GRAYSCALE)
    if ref_img.shape != defect_gray.shape:
        ref_img = cv2.resize(ref_img, (defect_gray.shape[1], defect_gray.shape[0]))

    subtracted = cv2.absdiff(cv2.GaussianBlur(defect_gray, (5,5), 0),
                             cv2.GaussianBlur(ref_img, (5,5), 0))

    binary = cv2.adaptiveThreshold(subtracted, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   35, -5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)

    if save_mask_path:
        cv2.imwrite(save_mask_path, cleaned)

    return defect_color, cleaned

# ===========================
# ROI EXTRACTION
# ===========================
def extract_rois_and_save(orig_img, mask, save_base_folder, base_name, defect_type=None,
                          min_area=200, min_w=10, min_h=10):
    roi_sub_folder = os.path.join(save_base_folder, defect_type) if defect_type else save_base_folder
    os.makedirs(roi_sub_folder, exist_ok=True)

    contours_info = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info)==2 else contours_info[1]

    roi_paths = []
    roi_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < min_area or w < min_w or h < min_h:
            continue
        roi = orig_img[y:y+h, x:x+w]
        roi_filename = f"{os.path.splitext(base_name)[0]}_roi{roi_count}.png"
        roi_save_path = os.path.join(roi_sub_folder, roi_filename)
        cv2.imwrite(roi_save_path, roi)
        roi_paths.append(roi_save_path)
        roi_count += 1

    return roi_paths

# ===========================
# ROI PREDICTION
# ===========================
def predict_roi(roi_path, model, class_names, transform, device):
    img = Image.open(roi_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(class_names[i], float(probs[i])) for i in top3_idx]
    label, conf = top3[0]

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {label} ({conf:.2f})")
    plt.show()

    return label, conf, top3

# ===========================
# FULL PIPELINE
# ===========================
def process_single_image(img_path, save_mask_folder=SUBTRACTED_SAVE, roi_base=ROI_BASE):
    base_name = os.path.basename(img_path)
    defect_type = os.path.basename(os.path.dirname(img_path))  # folder name as defect type

    mask_save_path = os.path.join(save_mask_folder, f"{os.path.splitext(base_name)[0]}_mask.png")
    defect_color, mask = subtract_images(img_path, PCB_USED_PATH, save_mask_path=mask_save_path)

    roi_paths = extract_rois_and_save(defect_color, mask, roi_base, base_name, defect_type=defect_type)
    print(f"✅ Extracted {len(roi_paths)} ROIs for {base_name}")

    for roi_path in roi_paths:
        label, conf, top3 = predict_roi(roi_path, model, classes, val_transform, device)
        print("Final Prediction:", label, conf)
        print("Top-3 Predictions:", top3)
        print("----")

    return roi_paths

# ===========================
# UPLOAD + RUN
# ===========================
print("📂 Upload your test PCB image(s):")
uploaded_test = files.upload()

for fname in uploaded_test.keys():
    print("\n🔍 Processing:", fname)
    process_single_image(fname)
