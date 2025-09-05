import cv2
import os
import numpy as np
import csv

# Paths
BASE_DIR = r"C:\Users\devak\Downloads\PCB_DATASET"
DEFECT_IMAGES_DIR = os.path.join(BASE_DIR, "images")
MASKED_DIR = os.path.join(BASE_DIR, "Subtracted_Images", "Masked")
HIGHLIGHTED_DIR = os.path.join(BASE_DIR, "Subtracted_Images", "Highlighted")
ROI_DIR = os.path.join(BASE_DIR, "Subtracted_Images", "ROIs")

os.makedirs(HIGHLIGHTED_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)

# CSV file for ROI labels
csv_path = os.path.join(ROI_DIR, "roi_labels.csv")
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["roi_filename", "defect_type"])

# Function to extract contours, highlight, and crop ROIs
def process_contours(defect_img, defect_mask, defect_category, img_name, roi_counter):
    pcb_with_boxes = defect_img.copy()  # keep colored PCB
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    padding = 3
    min_area = 20
    max_area = pcb_with_boxes.shape[0] * pcb_with_boxes.shape[1] // 2
    roi_saved = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)

            # Apply padding
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, pcb_with_boxes.shape[1] - 1)
            y2 = min(y + h + padding, pcb_with_boxes.shape[0] - 1)

            # Draw red rectangle on PCB
            cv2.rectangle(pcb_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Save cropped ROI
            roi_crop = defect_img[y1:y2, x1:x2]
            roi_folder = os.path.join(ROI_DIR, defect_category)
            os.makedirs(roi_folder, exist_ok=True)

            roi_filename = f"{defect_category}_{roi_counter}.jpg"
            roi_path = os.path.join(roi_folder, roi_filename)
            cv2.imwrite(roi_path, roi_crop)

            # Log in CSV
            csv_writer.writerow([roi_filename, defect_category])
            roi_counter += 1
            roi_saved += 1

    return pcb_with_boxes, roi_counter, roi_saved

# Main loop
roi_counter = 0
total_rois = 0

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

        defect_img = cv2.imread(defect_img_path, cv2.IMREAD_COLOR)   # Keep colored PCB
        defect_mask = cv2.imread(masked_img_path, cv2.IMREAD_GRAYSCALE)

        if defect_img is None or defect_mask is None:
            print(f"⚠️ Could not load {img_name}, skipping")
            continue

        highlighted_img, roi_counter, roi_saved = process_contours(
            defect_img, defect_mask, defect_category, img_name, roi_counter
        )

        total_rois += roi_saved

        # Save highlighted PCB
        cv2.imwrite(os.path.join(highlighted_category, img_name), highlighted_img)

csv_file.close()
print(f"\n✅ Highlighted images + ROI extraction completed.")
print(f"📊 Total ROIs saved: {total_rois}")
print(f"📝 ROI labels CSV saved at: {csv_path}")
