import cv2
import os
import numpy as np

# Paths
BASE_DIR = r"C:\Users\devak\Downloads\PCB_DATASET"
REFERENCE_PCB_DIR = os.path.join(BASE_DIR, "PCB_USED")
DEFECT_IMAGES_DIR = os.path.join(BASE_DIR, "images")
MASKED_DIR = os.path.join(BASE_DIR, "Subtracted_Images", "Masked")

os.makedirs(MASKED_DIR, exist_ok=True)

# Function to preprocess and subtract with border mask

def preprocess_and_subtract(reference_path, defect_path, border_thickness=10):
    ref_img = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    defect_img = cv2.imread(defect_path, cv2.IMREAD_GRAYSCALE)

    if defect_img.shape != ref_img.shape:
        defect_img = cv2.resize(defect_img, (ref_img.shape[1], ref_img.shape[0]))

    # Optional Gaussian blur
    ref_blur = cv2.GaussianBlur(ref_img, (3, 3), 0)
    defect_blur = cv2.GaussianBlur(defect_img, (3, 3), 0)

    # Subtraction
    diff = cv2.absdiff(defect_blur, ref_blur)

    # Threshold to create mask
    _, defect_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply Border Mask (ignore edges of the PCB)
    h, w = defect_mask.shape
    border_mask = np.zeros_like(defect_mask, dtype=np.uint8)
    cv2.rectangle(
        border_mask,
        (border_thickness, border_thickness),
        (w - border_thickness, h - border_thickness),
        255,
        -1
    )

    defect_mask = cv2.bitwise_and(defect_mask, border_mask)

    return defect_mask

# Main loop

for defect_category in os.listdir(DEFECT_IMAGES_DIR):
    category_path = os.path.join(DEFECT_IMAGES_DIR, defect_category)
    if not os.path.isdir(category_path):
        continue

    masked_category = os.path.join(MASKED_DIR, defect_category)
    os.makedirs(masked_category, exist_ok=True)

    for img_name in os.listdir(category_path):
        defect_img_path = os.path.join(category_path, img_name)
        pcb_number = img_name.split("_")[0] + ".jpg"
        reference_img_path = os.path.join(REFERENCE_PCB_DIR, pcb_number)

        if not os.path.exists(reference_img_path):
            print(f"⚠️ Reference PCB not found for {img_name}")
            continue

        defect_mask = preprocess_and_subtract(reference_img_path, defect_img_path, border_thickness=10)
        cv2.imwrite(os.path.join(masked_category, img_name), defect_mask)

print("\n✅ Subtraction completed.")
