import cv2
import os


golden_path = r"C:\Users\phaneendra ybs\Downloads\PCB_DATASET\PCB_DATASET\PCB_USED\01.JPG" 
test_path = r"C:\Users\phaneendra ybs\Downloads\PCB_DATASET\PCB_DATASET\images" 
output_path = r"C:\Users\phaneendra ybs\Downloads\PCB_subtarction"

os.makedirs(output_path, exist_ok=True)

golden = cv2.imread(golden_path, cv2.IMREAD_GRAYSCALE)
if golden is None:
    raise FileNotFoundError("❌ Golden image not found! Check golden_path.")


valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


for defect_type in os.listdir(test_path):
    defect_folder = os.path.join(test_path, defect_type)
    if not os.path.isdir(defect_folder):
        continue

    defect_output = os.path.join(output_path, defect_type)
    os.makedirs(defect_output, exist_ok=True)

    for img_name in os.listdir(defect_folder):
        if not img_name.lower().endswith(valid_ext):
            continue

        test_img_path = os.path.join(defect_folder, img_name)
        test = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

        if test is None:
            print(f"Skipping {img_name}, cannot read file.")
            continue

        # Resize test image to match golden image (if needed)
        if test.shape != golden.shape:
            test = cv2.resize(test, (golden.shape[1], golden.shape[0]))

        # Subtract images (absolute difference)
        diff = cv2.absdiff(golden, test)

        # Threshold to highlight defects
        _, defect_mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        # Save output
        save_path = os.path.join(defect_output, f"sub_{img_name}")
        cv2.imwrite(save_path, defect_mask)

        print(f"[{defect_type}] Processed: {img_name}")

print("✅ Subtraction-based defect detection completed for whole dataset.")
