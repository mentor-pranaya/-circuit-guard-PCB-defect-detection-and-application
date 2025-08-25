import cv2
import os
import numpy as np

correct_path = "path_to_the_used_images"    
defect_path =  "path_to_the_defected_images"      
output_path = "path_to_store_the_output"       

os.makedirs(output_path, exist_ok=True)

for correct_img_name in os.listdir(correct_path):
    correct_img_path = os.path.join(correct_path, correct_img_name)
    correct = cv2.imread(correct_img_path, cv2.IMREAD_GRAYSCALE)

    if correct is None:
        print(f"Skipping invalid image: {correct_img_path}")
        continue

    base_name = os.path.splitext(correct_img_name)[0]  

    for defect_category in os.listdir(defect_path):
        defect_category_path = os.path.join(defect_path, defect_category)

        for defect_img_name in os.listdir(defect_category_path):
            if defect_img_name.startswith(base_name):
                defect_img_path = os.path.join(defect_category_path, defect_img_name)
                defected = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)

                if defected is None:
                    print(f"Skipping invalid defect image: {defect_img_path}")
                    continue

                diff = cv2.absdiff(correct, defected)

                _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                kernel = np.ones((3,3), np.uint8)

                clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

                clean = cv2.medianBlur(clean, 3)

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
                min_area = 50  
                final = np.zeros_like(clean)
                for i in range(1, num_labels):  
                    if stats[i, cv2.CC_STAT_AREA] >= min_area:
                        final[labels == i] = 255

                save_name = f"{base_name}_{defect_category}_{defect_img_name}_diff.png"
                save_path = os.path.join(output_path, save_name)
                cv2.imwrite(save_path, final)

                print(f"Saved: {save_path}")
