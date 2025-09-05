import cv2
import os

# Base dataset folder
base_folder = r"C:\Users\Dell\Downloads\PCB_DATASET\PCB_DATASET"

# Input: subtracted images folder
subtracted_folder = os.path.join(base_folder, "subtracted_images")

# Output: ROI folder
roi_folder = os.path.join(base_folder, "roi_images")
os.makedirs(roi_folder, exist_ok=True)

# Loop through defect types inside subtracted images
for defect_type in os.listdir(subtracted_folder):
    defect_path = os.path.join(subtracted_folder, defect_type)
    if not os.path.isdir(defect_path):
        continue

    # Create corresponding ROI folder
    defect_roi_folder = os.path.join(roi_folder, defect_type)
    os.makedirs(defect_roi_folder, exist_ok=True)

    # Loop over subtracted images
    for img_name in os.listdir(defect_path):
        img_path = os.path.join(defect_path, img_name)

        # Load binary subtracted image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Find contours (defect regions)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Save each defect ROI separately
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)

            # Extract ROI
            roi = img[y:y+h, x:x+w]

            # Save with ROI index in filename
            roi_name = f"{os.path.splitext(img_name)[0]}_roi{i}.jpg"
            roi_path = os.path.join(defect_roi_folder, roi_name)
            cv2.imwrite(roi_path, roi)

print("✅ ROI images saved in:", roi_folder)
