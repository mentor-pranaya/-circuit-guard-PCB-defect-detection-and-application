import cv2
import os
import numpy as np

# Paths
BASE_DIR = r"C:\Users\devak\Downloads\PCB_DATASET"
DEFECT_IMAGES_DIR = os.path.join(BASE_DIR, "images")
MASKED_DIR = os.path.join(BASE_DIR, "Subtracted_Images", "Masked")
HIGHLIGHTED_DIR = os.path.join(BASE_DIR, "Subtracted_Images", "Highlighted")

os.makedirs(HIGHLIGHTED_DIR, exist_ok=True)

# Function to extract ROI and highlight
def extract_contours_and_highlight(defect_img_color, defect_mask):
    pcb_with_boxes = defect_img_color.copy()  # keep original colors

    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    padding = 3
    min_area = 10
    max_area = pcb_with_boxes.shape[0] * pcb_with_boxes.shape[1] // 2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, pcb_with_boxes.shape[1] - 1)
            y2 = min(y + h + padding, pcb_with_boxes.shape[0] - 1)

            # Draw red rectangle on original color image
            cv2.rectangle(pcb_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return pcb_with_boxes

# Main loop
for defect_category in os.listdir(DEFECT_IMAGES_DIR):
    category_path = os.path.join(DEFECT_IMAGES_DIR, defect_category)
    if not os.path.isdir(category_path):
        continue

    highlighted_category = os.path.join(HIGHLIGHTED_DIR, defect_category)
    os.makedirs(highlighted_category, exist_ok=True)

    for img_name in os.listdir(category_path):
        defect_img_path = os.path.join(category_path, img_name)
        masked_img_path = os.path.join(MASKED_DIR, defect_category, img_name)

        if not os.path.exists(masked_img_path):
            print(f"⚠️ Masked image not found for {img_name}, skipping")
            continue

        # Load defect image in COLOR
        defect_img_color = cv2.imread(defect_img_path, cv2.IMREAD_COLOR)
        # Mask still in grayscale
        defect_mask = cv2.imread(masked_img_path, cv2.IMREAD_GRAYSCALE)

        highlighted_img = extract_contours_and_highlight(defect_img_color, defect_mask)
        cv2.imwrite(os.path.join(highlighted_category, img_name), highlighted_img)

print("\n✅ Highlighted images (ROI Part with Color) completed.")
