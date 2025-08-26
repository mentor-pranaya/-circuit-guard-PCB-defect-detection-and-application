
import cv2
import os
import numpy as np

BASE_DIR = r"C:/Users/harsh/OneDrive/Documents/pythonvs/PCB_DATA"

# just join with folder name, not full path again
TEMPLATE_DIR = os.path.join(BASE_DIR, "PCB_USED")      
DEFECTS_DIR  = os.path.join(BASE_DIR, "Missing_hole")  
OUTPUT_DIR   = os.path.join(BASE_DIR, "output/masks")  

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(DEFECTS_DIR):
    if not fname.lower().endswith(".jpg"):
        continue

    # Match test PCB with its template (e.g., "01_missing_hole_01.jpg" → "01.jpg")
    template_id = fname.split("_")[0] + ".jpg"
    template_file = os.path.join(TEMPLATE_DIR, template_id)

    if not os.path.exists(template_file):
        print(f"[WARNING] Template {template_id} not found → Skipping {fname}")
        continue

    # Load template and test PCB
    template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(os.path.join(DEFECTS_DIR, fname), cv2.IMREAD_GRAYSCALE)

    if template is None or test is None:
        print(f"[ERROR] Failed to read {fname}, skipping...")
        continue

    # Resize test to template size
    test = cv2.resize(test, (template.shape[1], template.shape[0]))

    # Subtraction
    diff = cv2.absdiff(template, test)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Save result
    out_path = os.path.join(OUTPUT_DIR, f"mask_{fname}")
    cv2.imwrite(out_path, mask_clean)
    print(f"[SAVED] {out_path}")

print("\n Subtraction completed! All masks saved in:", OUTPUT_DIR)
