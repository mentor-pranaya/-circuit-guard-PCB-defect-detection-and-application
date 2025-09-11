import cv2
import os
import csv

input_folder = r"C:\Users\kavya\PCB_Defect_Detection\Image subtraction"
output_folder = r"C:\Users\kavya\PCB_Defect_Detection\ROIs"
csv_file_path = r"C:\Users\kavya\PCB_Defect_Detection\roi_labels.csv"

os.makedirs(output_folder, exist_ok=True)
roi_counter = 0

# Mapping defect names to numeric labels
category_map = {
    "Missing_hole": 0,
    "Mouse_bite": 1,
    "Open_circuit": 2,
    "Spurious_copper": 4,  # put before "Spur" to avoid substring clash
    "Spur": 3,
    "Short": 5
}

# Filename variants for each defect type
filename_variants = {
    "Missing_hole": ["missing_hole", "missing-hole", "missing hole", "missinghole"],
    "Mouse_bite": ["mouse_bite", "mouse-bite", "mouse bite", "mousebite"],
    "Open_circuit": ["open_circuit", "open-circuit", "open circuit", "opencircuit"],
    "Spurious_copper": ["spurious_copper", "spurious-copper", "spurious copper", "spuriouscopper"],
    "Spur": ["spur"],
    "Short": ["short"]
}

# Save the mapping for reference
mapping_file = os.path.join(output_folder, "category_mapping.csv")
with open(mapping_file, mode='w', newline='') as map_file:
    writer = csv.writer(map_file)
    writer.writerow(["category_name", "label"])
    for name, label in category_map.items():
        writer.writerow([name, label])

with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["roi_filename", "category"])  # CSV header

    for file_name in os.listdir(input_folder):
        # Process all image files and "_diff" files
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")) or "_diff" in file_name.lower():
            img_path = os.path.join(input_folder, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read {file_name}")
                continue

            _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            category_num = -1
            file_name_lower = file_name.lower()
            for defect_name, variants in filename_variants.items():
                if any(variant in file_name_lower for variant in variants):
                    category_num = category_map[defect_name]
                    break

            if category_num == -1:
                print(f"Warning: Unknown defect category in {file_name}")
                continue

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 5 or h < 5:
                    continue

                roi = img[y:y+h, x:x+w]
                roi_filename = f"roi_{roi_counter+1}.png"
                cv2.imwrite(os.path.join(output_folder, roi_filename), roi)

                writer.writerow([roi_filename, category_num])
                roi_counter += 1

            print(f"Processed {file_name}, total ROIs so far: {roi_counter}")

print("ROI extraction and CSV generation complete!")
