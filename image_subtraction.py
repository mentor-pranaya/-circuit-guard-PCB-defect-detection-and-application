import cv2
import os

def perform_image_subtraction(base_folder):
    pcb_used_folder = os.path.join(base_folder, "PCB_USED")
    images_folder = os.path.join(base_folder, "images")
    subtracted_folder = os.path.join(base_folder, "subtracted_images")
    os.makedirs(subtracted_folder, exist_ok=True)

    # List of defect folders detected, debug print
    print(f"Defect folders found: {os.listdir(images_folder)}")

    for defect_type in os.listdir(images_folder):
        defect_path = os.path.join(images_folder, defect_type)
        if not os.path.isdir(defect_path):
            continue

        print(f"Processing defect folder: {defect_type}")

        # Create output folder matching exact defect folder name
        defect_subfolder = os.path.join(subtracted_folder, defect_type)
        os.makedirs(defect_subfolder, exist_ok=True)

        for defect_img_name in os.listdir(defect_path):
            defect_img_path = os.path.join(defect_path, defect_img_name)

            # Extract PCB number from filename prefix, e.g., "01_mouse_bite_01.jpg" → "01.jpg"
            pcb_number = defect_img_name.split("_")[0] + ".jpg"
            pcb_img_path = os.path.join(pcb_used_folder, pcb_number)

            if not os.path.exists(pcb_img_path):
                print(f"⚠️ Template PCB {pcb_number} not found for {defect_img_name}, skipping.")
                continue

            defect_img = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)
            pcb_img = cv2.imread(pcb_img_path, cv2.IMREAD_GRAYSCALE)

            if defect_img is None or pcb_img is None:
                print(f"⚠️ Failed to load images: {defect_img_name} or {pcb_number}, skipping.")
                continue

            if defect_img.shape != pcb_img.shape:
                defect_img = cv2.resize(defect_img, (pcb_img.shape[1], pcb_img.shape[0]))

            subtracted_img = cv2.absdiff(defect_img, pcb_img)
            save_path = os.path.join(defect_subfolder, defect_img_name)
            cv2.imwrite(save_path, subtracted_img)

            print(f"Saved subtracted image: {save_path}")
 
    print("\n🎯 Image subtraction completed successfully.")

if __name__ == "__main__":
    base_folder = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\data"
    perform_image_subtraction(base_folder)
