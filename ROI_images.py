import cv2
import os

# Base dataset folder
base_folder = r"C:\Users\Dell\Downloads\PCB_DATASET\PCB_DATASET"

# Input folders
images_folder = os.path.join(base_folder, "images")              # defected images
subtracted_folder = os.path.join(base_folder, "subtracted_images") # binary masks

# Output folder (structured by defect type)
roi_folder = os.path.join(base_folder, "ROIs")
os.makedirs(roi_folder, exist_ok=True)

# Loop over defect type folders
for defect_type in os.listdir(subtracted_folder):
    defect_mask_path = os.path.join(subtracted_folder, defect_type)
    defect_img_path = os.path.join(images_folder, defect_type)

    if not os.path.isdir(defect_mask_path):
        continue

    # Create ROI folder for this defect type (class folder)
    roi_sub_folder = os.path.join(roi_folder, defect_type)
    os.makedirs(roi_sub_folder, exist_ok=True)

    # Loop over mask images
    for img_name in os.listdir(defect_mask_path):
        mask_path = os.path.join(defect_mask_path, img_name)
        orig_path = os.path.join(defect_img_path, img_name)

        # Load mask (binary) and original image (color)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        orig_img = cv2.imread(orig_path)

        if mask is None or orig_img is None:
            continue

        # Find contours (defect regions)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Filter very small boxes (noise)
            if w < 10 or h < 10:
                continue

            # Crop ROI from original image
            roi = orig_img[y:y+h, x:x+w]

            # Save ROI under correct defect class folder
            roi_filename = f"{os.path.splitext(img_name)[0]}_roi{roi_count}.png"
            roi_save_path = os.path.join(roi_sub_folder, roi_filename)
            cv2.imwrite(roi_save_path, roi)

            roi_count += 1

print("✅ ROI extraction completed", roi_folder)
