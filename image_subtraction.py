import cv2
import os
# Base dataset folder
base_folder = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\data"
# Input folders
pcb_used_folder = os.path.join(base_folder, "pcb_used")      # reference clean PCB images
images_folder = os.path.join(base_folder, "images")          # defected images
# Output folder
subtracted_folder = os.path.join(base_folder, "subtracted_images")
os.makedirs(subtracted_folder, exist_ok=True)
print(f"Base folder: {base_folder}")
print(f"Reference PCB folder: {pcb_used_folder}")
print(f"Defected images folder: {images_folder}")
print("Starting subtraction process...\n")
# Loop over defect type folders inside "images"
for defect_type in os.listdir(images_folder):
    defect_path = os.path.join(images_folder, defect_type)
    if not os.path.isdir(defect_path):
        print(f"Skipping non-folder: {defect_path}")
        continue
    print(f"\nProcessing defect type: {defect_type}")
    # Create output folder for this defect type
    defect_sub_folder = os.path.join(subtracted_folder, defect_type)
    os.makedirs(defect_sub_folder, exist_ok=True)
    processed_count = 0
    skipped_count = 0
    # Loop over defected images
    for img_name in os.listdir(defect_path):
        img_path = os.path.join(defect_path, img_name)
        print(f"  Processing image: {img_name}")
        # Load defected image (grayscale)
        defect_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if defect_img is None:
            print(f"    [!] Failed to load defect image: {img_path}")
            skipped_count += 1
            continue
        # Extract PCB number from file name (e.g., "01_open_1.jpg" → "01.jpg")
        pcb_number = img_name.split("_")[0] + ".jpg"
        print(f"    Matched reference image: {pcb_number}")
        # Load corresponding reference PCB
        ref_path = os.path.join(pcb_used_folder, pcb_number)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            print(f"    [!] Failed to load reference PCB image: {ref_path}")
            skipped_count += 1
            continue

        # Resize reference to match defect image if needed
        if ref_img.shape != defect_img.shape:
            print(f"    Resizing reference ({ref_img.shape}) to defect image size ({defect_img.shape})")
            ref_img = cv2.resize(ref_img, (defect_img.shape[1], defect_img.shape[0]))

        # Subtract images (absolute difference)
        subtracted = cv2.absdiff(defect_img, ref_img)
        print(f"    Subtraction done.")

        # Threshold for binary defect highlighting
        _, binary = cv2.threshold(subtracted, 30, 255, cv2.THRESH_BINARY)
        print(f"    Thresholding applied.")

        # Save result using the SAME name as defected image (pcb + defect type + index)
        save_path = os.path.join(defect_sub_folder, img_name)
        cv2.imwrite(save_path, binary)
        print(f"    Saved subtracted image: {save_path}")

        processed_count += 1

    print(f"Summary for defect type '{defect_type}': {processed_count} processed, {skipped_count} skipped.")

print("\n✅ All subtracted images saved in:", subtracted_folder)
