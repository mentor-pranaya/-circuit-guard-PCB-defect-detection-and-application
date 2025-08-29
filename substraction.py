import cv2
import os
import numpy as np

# Define dataset paths

BASE_DIR = r"C:\Users\devak\Downloads\PCB_DATASET"
REFERENCE_PCB_DIR = os.path.join(BASE_DIR, "PCB_USED")
DEFECT_IMAGES_DIR = os.path.join(BASE_DIR, "images")
SUBTRACTED_DIR = os.path.join(BASE_DIR, "Subtracted_Images")
MASKED_DIR = os.path.join(SUBTRACTED_DIR, "Masked")
HIGHLIGHTED_DIR = os.path.join(SUBTRACTED_DIR, "Highlighted")

# Create folders if they don't exist
os.makedirs(MASKED_DIR, exist_ok=True)
os.makedirs(HIGHLIGHTED_DIR, exist_ok=True)

# Function to process PCB images

def process_defect_image(reference_path, defect_path, masked_dir, highlighted_dir, img_name):
    ref_img = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    defect_img = cv2.imread(defect_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None or defect_img is None:
        print(f" Skipping {img_name}")
        return

    if defect_img.shape != ref_img.shape:
        defect_img = cv2.resize(defect_img, (ref_img.shape[1], ref_img.shape[0]))

    # Absolute difference
    diff = cv2.absdiff(defect_img, ref_img)

    # Fixed threshold - detect real defects only
    _, defect_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Contours for bounding boxes
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pcb_with_boxes = cv2.cvtColor(defect_img, cv2.COLOR_GRAY2BGR)

    padding = 3
    min_area = 10
    max_area = pcb_with_boxes.shape[0] * pcb_with_boxes.shape[1] // 2  # ignore huge areas

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, pcb_with_boxes.shape[1] - 1)
            y2 = min(y + h + padding, pcb_with_boxes.shape[0] - 1)

            # Draw thick red rectangle
            cv2.rectangle(pcb_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save outputs
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(highlighted_dir, exist_ok=True)
    cv2.imwrite(os.path.join(masked_dir, img_name), defect_mask)
    cv2.imwrite(os.path.join(highlighted_dir, img_name), pcb_with_boxes)

# Main processing loop

for defect_category in os.listdir(DEFECT_IMAGES_DIR):
    category_path = os.path.join(DEFECT_IMAGES_DIR, defect_category)
    if not os.path.isdir(category_path):
        continue

    masked_category = os.path.join(MASKED_DIR, defect_category)
    highlighted_category = os.path.join(HIGHLIGHTED_DIR, defect_category)
    os.makedirs(masked_category, exist_ok=True)
    os.makedirs(highlighted_category, exist_ok=True)

    for img_name in os.listdir(category_path):
        defect_img_path = os.path.join(category_path, img_name)
        pcb_number = img_name.split("_")[0] + ".jpg"
        reference_img_path = os.path.join(REFERENCE_PCB_DIR, pcb_number)

        if not os.path.exists(reference_img_path):
            print(f" Reference PCB not found for {img_name}")
            continue

        process_defect_image(reference_img_path, defect_img_path,
                             masked_category, highlighted_category, img_name)

print(f"\n PCB defect detection completed. Masked and highlighted images saved under: {SUBTRACTED_DIR}")
