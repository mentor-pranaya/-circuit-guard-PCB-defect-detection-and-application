import cv2
import os

GOOD_DIR = "C:/Users/SIRIGIRI HARINI/Downloads/PCB_DATASET/PCB_DATASET/PCB_USED"
DEFECTIVE_DIR = "C:/Users/SIRIGIRI HARINI/Downloads/PCB_DATASET/PCB_DATASET/images"
OUTPUT_DIR = "C:/Users/SIRIGIRI HARINI/Downloads/PCB_DATASET/subtracted"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTS = (".png", ".jpg", ".jpeg", ".bmp")

good_images = [f for f in os.listdir(GOOD_DIR) if f.lower().endswith(EXTS)]
if not good_images:
    raise ValueError("⚠ No good PCB images found in pcb used/ folder!")

REFERENCE_IMG = os.path.join(GOOD_DIR, good_images[0])
print(f"✅ Using reference image: {REFERENCE_IMG}")

ref_img = cv2.imread(REFERENCE_IMG, cv2.IMREAD_GRAYSCALE)
if ref_img is None:
    raise ValueError("⚠ Could not load reference image!")

count = 0

for root, dirs, files in os.walk(DEFECTIVE_DIR):
    for file_name in files:
        if file_name.lower().endswith(EXTS):
            img_path = os.path.join(root, file_name)

            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠ Could not read {img_path}")
                continue

        
            if img.shape != ref_img.shape:
                img = cv2.resize(img, (ref_img.shape[1], ref_img.shape[0]))

           
            subtracted = cv2.subtract(ref_img, img)

            
            relative_path = os.path.relpath(img_path, DEFECTIVE_DIR)
            safe_name = relative_path.replace(os.sep, "_")
            out_path = os.path.join(OUTPUT_DIR, f"subtracted_{safe_name}")

            cv2.imwrite(out_path, subtracted)

            count += 1
            print(f"✅ Subtracted: {img_path}")

print(f"\n🎉 Done! Created {count} subtracted images in {OUTPUT_DIR}")