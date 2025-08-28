import cv2
import os
import numpy as np

# Base dataset folder
BASE_DIR = r"C:/Users/harsh/OneDrive/Documents/pythonvs/PCB_DATA"

# Golden reference (defect-free PCBs)
TEMPLATE_DIR = os.path.join(BASE_DIR, "PCB_USED")

# Defective PCBs
DEFECTS_DIR = r"C:/Users/harsh/OneDrive/Documents/pythonvs/DEFECTS/Spurious_copper"

# Output folder
OUTPUT_DIR = os.path.join(BASE_DIR, "output/masks1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through defective PCB images
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
    test_gray = cv2.imread(os.path.join(DEFECTS_DIR, fname), cv2.IMREAD_GRAYSCALE)
    test_color = cv2.imread(os.path.join(DEFECTS_DIR, fname))  # for drawing ROI

    if template is None or test_gray is None:
        print(f"[ERROR] Failed to read {fname}, skipping...")
        continue

    # Resize test to template size
    test_gray = cv2.resize(test_gray, (template.shape[1], template.shape[0]))
    test_color = cv2.resize(test_color, (template.shape[1], template.shape[0]))

    # Subtraction
    diff = cv2.absdiff(template, test_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Contour detection ---
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 50:  # ignore tiny noise
            cv2.rectangle(test_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(test_color, "Defect", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

    # Save mask and annotated image
    out_mask_path = os.path.join(OUTPUT_DIR, f"mask_{fname}")
    out_contour_path = os.path.join(OUTPUT_DIR, f"contours_{fname}")

    cv2.imwrite(out_mask_path, mask_clean)
    cv2.imwrite(out_contour_path, test_color)

    print(f"Saved: {out_mask_path}, {out_contour_path}")

print("\n Subtraction + Contour detection completed! Results saved in:", OUTPUT_DIR)
