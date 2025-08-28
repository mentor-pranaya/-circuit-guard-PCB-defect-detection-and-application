import cv2
import os

# Base dataset folder
base_folder = r"C:\Users\Dell\Downloads\PCB_DATASET\PCB_DATASET"

# Input folders
images_folder = os.path.join(base_folder, "images")       # Defected PCBs
subtracted_folder = os.path.join(base_folder, "subtracted_images")  # Binary defect masks

# Output folder for contour-marked PCBs
contour_folder = os.path.join(base_folder, "PCB_with_contours")
os.makedirs(contour_folder, exist_ok=True)

# Iterate over defect type folders
for defect_type in os.listdir(images_folder):
    defect_path = os.path.join(images_folder, defect_type)
    if not os.path.isdir(defect_path):
        continue

    # Create subfolder for this defect type
    defect_contour_subfolder = os.path.join(contour_folder, defect_type)
    os.makedirs(defect_contour_subfolder, exist_ok=True)

    # Loop over defect images
    for defect_img_name in os.listdir(defect_path):
        defect_img_path = os.path.join(defect_path, defect_img_name)

        # Load defected PCB (color)
        defected_pcb = cv2.imread(defect_img_path)

        # Load corresponding binary subtracted image
        subtracted_img_path = os.path.join(subtracted_folder, defect_type, defect_img_name)
        binary_img = cv2.imread(subtracted_img_path, cv2.IMREAD_GRAYSCALE)

        if defected_pcb is None or binary_img is None:
            print(f"⚠️ Skipping {defect_img_name}, missing PCB or subtracted image")
            continue

        # Find contours from binary image
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the defected PCB
        cv2.drawContours(defected_pcb, contours, -1, (0, 0, 255), 2)  # red contours

        # Save PCB with contours
        save_path = os.path.join(defect_contour_subfolder, f"Contour_{defect_img_name}")
        cv2.imwrite(save_path, defected_pcb)

print("🎯 Defect contours applied and saved on PCB images inside:", contour_folder)
