import cv2
import os

# ✅ Define the base dataset folder first
base_folder = r"C:\Users\devak\Downloads\PCB_DATASET"

# Folders
pcb_used_folder = os.path.join(base_folder, "PCB_USED")
images_folder = os.path.join(base_folder, "images")
subtracted_folder = os.path.join(base_folder, "subtracted_images")

# Ensure subtracted_images folder exists
os.makedirs(subtracted_folder, exist_ok=True)

# Iterate over defect type subfolders (like Missing_hole, Mouse_bite, etc.)
for defect_type in os.listdir(images_folder):
    defect_path = os.path.join(images_folder, defect_type)
    if not os.path.isdir(defect_path):
        continue

    # Create matching defect subfolder inside subtracted_images
    defect_subfolder = os.path.join(subtracted_folder, defect_type)
    os.makedirs(defect_subfolder, exist_ok=True)

    # Process all defect images
    for defect_img_name in os.listdir(defect_path):
        defect_img_path = os.path.join(defect_path, defect_img_name)

        # Extract PCB number from file name (e.g., "01_missing_hole_01.jpg" → "01.jpg")
        pcb_number = defect_img_name.split("_")[0] + ".jpg"
        pcb_img_path = os.path.join(pcb_used_folder, pcb_number)

        if not os.path.exists(pcb_img_path):
            print(f"⚠️ Skipping {defect_img_name}, PCB reference {pcb_number} not found.")
            continue

        # Load images
        defect_img = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)
        pcb_img = cv2.imread(pcb_img_path, cv2.IMREAD_GRAYSCALE)

        if defect_img is None or pcb_img is None:
            print(f"⚠️ Skipping {defect_img_name}, could not load images.")
            continue

        # Resize if dimensions don’t match
        if defect_img.shape != pcb_img.shape:
            defect_img = cv2.resize(defect_img, (pcb_img.shape[1], pcb_img.shape[0]))

        # Subtract images
        subtracted_img = cv2.absdiff(defect_img, pcb_img)

        # Save inside correct defect subfolder
        save_path = os.path.join(defect_subfolder, defect_img_name)
        cv2.imwrite(save_path, subtracted_img)

print(f"\n🎯 All subtracted images are organized into defect-type folders inside: {subtracted_folder}")

