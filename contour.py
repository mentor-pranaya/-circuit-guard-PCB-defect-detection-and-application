import cv2
import os
import csv

# Input: subtracted images
INPUT_DIR = "C:/Users/saida/Downloads/PCB_DATASET/subtracted"

# Output: marked images + ROI defects
OUTPUT_DIR = "C:/Users/saida/Downloads/PCB_DATASET/defects"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV file to store defect counts
CSV_PATH = os.path.join(OUTPUT_DIR, "defect_counts.csv")

# Allowed image formats
EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# Open CSV file for writing
with open(CSV_PATH, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Image_Name", "Defect_Count"])  # Header row

    count = 0
    for root, dirs, files in os.walk(INPUT_DIR):
        for file_name in files:
            if file_name.lower().endswith(EXTS):
                img_path = os.path.join(root, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"⚠ Could not read {img_path}")
                    continue

                # 1. Threshold to binary
                _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

                # 2. Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Prepare colored copy for marking
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                defect_num = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 50:  # Ignore tiny noise
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)

                    # Draw red rectangle
                    cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

                    # Extract ROI
                    roi = img[y:y+h, x:x+w]

                    # Save ROI
                    roi_name = f"{os.path.splitext(file_name)[0]}_defect{defect_num}.png"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, roi_name), roi)

                    defect_num += 1

                # Save marked image
                out_img_name = f"marked_{file_name}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, out_img_name), color_img)

                # Write image name + defect count into CSV
                writer.writerow([file_name, defect_num])

                count += 1
                print(f"✅ Processed {file_name}, found {defect_num} defects")

print(f"\n🎉 Done! Processed {count} images. Results saved in:")
print(f"- Marked images + ROIs → {OUTPUT_DIR}")
print(f"- CSV defect counts → {CSV_PATH}")