import cv2
import os

input_folder = "path_to_the_images_we_performed_image_subtraction"
output_folder = "path_to_store_the_output"         

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".png") or file_name.endswith(".jpg"):
        img_path = os.path.join(input_folder, file_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        defect_name = file_name.split(".")[0]   
        save_dir = os.path.join(output_folder, defect_name)
        os.makedirs(save_dir, exist_ok=True)

        roi_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w < 5 or h < 5:
                continue

            roi = img[y:y+h, x:x+w]

            roi_filename = f"{defect_name}_roi_{roi_count+1}.png"
            cv2.imwrite(os.path.join(save_dir, roi_filename), roi)

            roi_count += 1

        print(f"Extracted {roi_count} ROIs from {file_name}")
