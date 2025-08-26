import cv2, os
import pandas as pd

MASKS_DIR = "output/masks"
DEFECTS_DIR = "PCB_DATASET/images"
OUTPUT_ROIS = "output/rois"
CSV_FILE = "output/roi_labels.csv"

os.makedirs(OUTPUT_ROIS, exist_ok=True)
labels = []

for defect_type in os.listdir(MASKS_DIR):
    mask_path = os.path.join(MASKS_DIR, defect_type)
    defect_path = os.path.join(DEFECTS_DIR, defect_type)
    if not os.path.isdir(mask_path):
        continue
    
    print(f"[ROI] Processing defect type: {defect_type}")
    defect_roi_dir = os.path.join(OUTPUT_ROIS, defect_type)
    os.makedirs(defect_roi_dir, exist_ok=True)

    for fname in os.listdir(mask_path):
        if not fname.endswith(".jpg"):
            continue

        mask = cv2.imread(os.path.join(mask_path, fname), cv2.IMREAD_GRAYSCALE)
        original_name = fname.replace("mask_", "")
        test_img = cv2.imread(os.path.join(defect_path, original_name))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h < 50:
                continue

            roi = test_img[y:y+h, x:x+w]
            roi_filename = f"{original_name}_roi{i}.jpg"
            cv2.imwrite(os.path.join(defect_roi_dir, roi_filename), roi)

            labels.append([roi_filename, defect_type])

df = pd.DataFrame(labels, columns=["filename", "label"])
df.to_csv(CSV_FILE, index=False)

print(" ROI extraction completed")
