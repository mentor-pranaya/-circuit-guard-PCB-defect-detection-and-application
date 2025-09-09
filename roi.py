import cv2
import os
import numpy as np
import csv

# Paths
BASE_DIR = r"C:\Users\devak\Downloads\PCB_DATASET"
DEFECT_IMAGES_DIR = os.path.join(BASE_DIR, "images")
MASKED_DIR = os.path.join(BASE_DIR, "Subtracted_Images", "Masked")
ROI_DIR = os.path.join(BASE_DIR, "Subtracted_Images", "ROI_dataset")

os.makedirs(ROI_DIR, exist_ok=True)

# CSV file for ROI labels
csv_path = os.path.join(ROI_DIR, "roi_labels.csv")
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["roi_filename", "defect_type"])

# Function to extract and save cropped ROIs
def extract_and_save_rois(defect_img_color, defect_mask, defect_category, img_name, roi_counter):
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    padding = 3
    min_area = 10
    max_area = defect_img_color.shape[0] * defect_img_color.shape[1] // 2

    roi_saved = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, defect_img_color.shape[1] - 1)
            y2 = min(y + h + padding, defect_img_color.shape[0] - 1)

            roi_crop = defect_img_color[y1:y2, x1:x2]

            # Save ROI in defect type folder
            roi_category_dir = os.path.join(ROI_DIR, defect_category)
            os.makedirs(roi_category_dir, exist_ok=True)

            roi_filename = f"{os.path.splitext(img_name)[0]}_roi{roi_counter:04d}.jpg"
            roi_path = os.path.join(roi_category_dir, roi_filename)

            cv2.imwrite(roi_path, roi_crop)

            # Write to CSV: roi filename, defect type
            csv_writer.writerow([roi_filename, defect_category])

            roi_counter += 1
            roi_saved += 1

    return roi_counter, roi_saved

# Main loop
roi_counter = 0
total_rois = 0

for defect_category in os.listdir(DEFECT_IMAGES_DIR):
    category_path = os.path.join(DEFECT_IMAGES_DIR, defect_category)
    if not os.path.isdir(category_path):
        continue

    for img_name in os.listdir(category_path):
        defect_img_path = os.path.join(category_path, img_name)
        masked_img_path = os.path.join(MASKED_DIR, defect_category, img_name)

        if not os.path.exists(masked_img_path):
            print(f"⚠️ Masked image not found for {img_name}, skipping")
            continue

        defect_img_color = cv2.imread(defect_img_path, cv2.IMREAD_COLOR)
        defect_mask = cv2.imread(masked_img_path, cv2.IMREAD_GRAYSCALE)

        if defect_img_color is None or defect_mask is None:
            print(f"⚠️ Could not load {img_name}, skipping")
            continue

        roi_counter, roi_saved = extract_and_save_rois(
            defect_img_color, defect_mask, defect_category, img_name, roi_counter
        )

        total_rois += roi_saved

csv_file.close()

print(f"\n✅ ROI Extraction Completed.")
print(f"📊 Total ROIs saved: {total_rois}")
print(f"📝 ROI labels CSV saved at: {csv_path}")
