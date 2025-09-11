import cv2, os
import numpy as np

# Paths
BASE_DIR = "PCB_DATASET"
TEMPLATE_DIR = os.path.join(BASE_DIR, "PCB_USED")
DEFECTS_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_MASKS = "output/masks"
os.makedirs(OUTPUT_MASKS, exist_ok=True)

for defect_type in os.listdir(DEFECTS_DIR):
    defect_path = os.path.join(DEFECTS_DIR, defect_type)
    if not os.path.isdir(defect_path):
        continue
    
    print(f"[Subtraction] Processing defect type: {defect_type}")
    defect_mask_dir = os.path.join(OUTPUT_MASKS, defect_type)
    os.makedirs(defect_mask_dir, exist_ok=True)

    for fname in os.listdir(defect_path):
        if not fname.endswith(".jpg"):
            continue

        # Find template image
        template_id = fname.split("_")[0] + ".jpg"
        template_file = os.path.join(TEMPLATE_DIR, template_id)

        if not os.path.exists(template_file):
            print(f"Template {template_file} not found, skipping {fname}")
            continue

        # Load images
        template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
        test = cv2.imread(os.path.join(defect_path, fname), cv2.IMREAD_GRAYSCALE)
        test = cv2.resize(test, (template.shape[1], template.shape[0]))

        # Subtract + Otsu threshold
        diff = cv2.absdiff(template, test)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Save mask
        out_path = os.path.join(defect_mask_dir, f"mask_{fname}")
        cv2.imwrite(out_path, mask_clean)


print("Subtraction completed. Masks saved in output/masks/")
