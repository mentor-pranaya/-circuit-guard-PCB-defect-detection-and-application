import cv2
import os

# Base dataset folder
base_folder = r"C:\Users\Dell\Downloads\PCB_DATASET\PCB_DATASET"

# Input: subtracted_images (binary defects)
subtracted_folder = os.path.join(base_folder, "subtracted_images")

# Output: ROI folder (in the main dataset)
roi_folder = os.path.join(base_folder, "ROI")
os.makedirs(roi_folder, exist_ok=True)

# Iterate over each defect type folder inside subtracted_images
for defect_type in os.listdir(subtracted_folder):
    defect_path = os.path.join(subtracted_folder, defect_type)
    if not os.path.isdir(defect_path):
        continue

    # Create subfolder for this defect type in ROI folder
    defect_roi_subfolder = os.path.join(roi_folder, defect_type)
    os.makedirs(defect_roi_subfolder, exist_ok=True)

    # Loop over subtracted images
    for img_name in os.listdir(defect_path):
        img_path = os.path.join(defect_path, img_name)

        # Load binary defect image
        defect_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if defect_img is None:
            continue

        # Convert to color so we can draw bounding boxes
        pcb_with_rois = cv2.cvtColor(defect_img, cv2.COLOR_GRAY2BGR)

        # Find contours (each defect = one contour)
        contours, _ = cv2.findContours(defect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes for every defect in this PCB
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(pcb_with_rois, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red box

        # Save single PCB image with all ROIs marked
        save_path = os.path.join(defect_roi_subfolder, f"ROI_{img_name}")
        cv2.imwrite(save_path, pcb_with_rois)

print("🎯 Single PCB images with all ROIs marked are saved in defect-wise folders inside:", roi_folder)
