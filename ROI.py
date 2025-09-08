import cv2
import os
import numpy as np

# Base dataset folder
BASE_DIR = r"C:/Users/harsh/OneDrive/Documents/pythonvs/PCB_DATA"

# Golden reference (defect-free PCBs)
TEMPLATE_DIR = os.path.join(BASE_DIR, "PCB_USED")

# Defective PCBs
DEFECTS_DIR = r"C:/Users/harsh/OneDrive/Documents/pythonvs/DEFECTS/Spurious_copper" 

# Output folder for cropped ROIs
OUTPUT_DIR = os.path.join(BASE_DIR, "output/roi_crops")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(DEFECTS_DIR):
    if not fname.lower().endswith(".jpg"):
        continue

    template_id = fname.split("_")[0] + ".jpg"
    template_file = os.path.join(TEMPLATE_DIR, template_id)

    if not os.path.exists(template_file):
        print(f"[WARNING] Template {template_id} not found → Skipping {fname}")
        continue

    template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
    test_gray = cv2.imread(os.path.join(DEFECTS_DIR, fname), cv2.IMREAD_GRAYSCALE)
    test_color = cv2.imread(os.path.join(DEFECTS_DIR, fname))  # for cropping ROI

    if template is None or test_gray is None:
        print(f"[ERROR] Failed to read {fname}, skipping...")
        continue

    # Resize to match
    test_gray = cv2.resize(test_gray, (template.shape[1], template.shape[0]))
    test_color = cv2.resize(test_color, (template.shape[1], template.shape[0]))

    # Subtraction
    diff = cv2.absdiff(template, test_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Contours to crop ROI ---
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 50:
            roi = test_color[y:y+h, x:x+w]  # Crop region
            roi_path = os.path.join(OUTPUT_DIR, f"roi_{fname[:-4]}_{roi_count}.jpg")
            cv2.imwrite(roi_path, roi)
            roi_count += 1

    print(f"Saved {roi_count} ROIs for {fname}")

print("\nROI extraction completed! Cropped regions saved in:", OUTPUT_DIR)
