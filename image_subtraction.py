import cv2
import os

# Base dataset folder
base_folder = r"E:\KARAN_PANCHAL\PCB_DATASET"

# Input folders
pcb_used_folder = os.path.join(base_folder, "PCB_USED")   # reference clean PCB images
images_folder = os.path.join(base_folder, "images")       # defected images

# Output folder
subtracted_folder = os.path.join(base_folder, "subtracted_images")
os.makedirs(subtracted_folder, exist_ok=True)

# Loop over defect type folders inside "images"
for defect_type in os.listdir(images_folder):
    defect_path = os.path.join(images_folder, defect_type)
    if not os.path.isdir(defect_path):
        continue

    # Create output folder for this defect type
    defect_sub_folder = os.path.join(subtracted_folder, defect_type)
    os.makedirs(defect_sub_folder, exist_ok=True)

    # Loop over defected images
    for img_name in os.listdir(defect_path):
        img_path = os.path.join(defect_path, img_name)

        # Load defected image (grayscale)
        defect_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if defect_img is None:
            continue

        # Extract PCB number from file name (e.g., "01_open_1.jpg" → "01.jpg")
        pcb_number = img_name.split("_")[0] + ".jpg"

        # Load corresponding reference PCB
        ref_path = os.path.join(pcb_used_folder, pcb_number)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            continue

        # Resize reference to match defect image if needed
        if ref_img.shape != defect_img.shape:
            ref_img = cv2.resize(ref_img, (defect_img.shape[1], defect_img.shape[0]))

        # Subtract images (absolute difference)
        subtracted = cv2.absdiff(defect_img, ref_img)

        # Threshold for binary defect highlighting
        _, binary = cv2.threshold(subtracted, 30, 255, cv2.THRESH_BINARY)

        # Save result using the SAME name as defected image (pcb + defect type + index)
        save_path = os.path.join(defect_sub_folder, img_name)
        cv2.imwrite(save_path, binary)

print("✅ Subtracted images saved in:", subtracted_folder)