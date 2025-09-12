import cv2, os
import pandas as pd

# =========================
# Paths (use raw strings to avoid \r, \t issues)
# =========================
MASKS_DIR = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\data\subtracted_images"
DEFECTS_DIR = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\data\images"
OUTPUT_ROIS = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\data\roi_images"
CSV_FILE = r"D:\B-TECH\CERTIFICATES\Infosys Internship\CircuitGuard\data\roi_labels.csv"

# =========================
# Setup
# =========================
os.makedirs(OUTPUT_ROIS, exist_ok=True)
labels = []
roi_count = 0  # counter for total ROI images
per_defect_counts = {}  # dictionary to track per-defect ROI counts

print("[START] ROI extraction process initiated...\n")

# =========================
# Process each defect type
# =========================
for defect_type in os.listdir(MASKS_DIR):
    mask_path = os.path.join(MASKS_DIR, defect_type)
    defect_path = os.path.join(DEFECTS_DIR, defect_type)
    
    if not os.path.isdir(mask_path):
        print(f"[SKIP] {defect_type} is not a folder, skipping...")
        continue
    
    print(f"[ROI] Processing defect type: {defect_type}")
    defect_roi_dir = os.path.join(OUTPUT_ROIS, defect_type)
    os.makedirs(defect_roi_dir, exist_ok=True)

    defect_count = 0  # per-defect counter

    for fname in os.listdir(mask_path):
        if not fname.endswith(".jpg"):
            print(f"[SKIP] {fname} is not a .jpg file")
            continue

        print(f"  -> Processing mask file: {fname}")
        mask = cv2.imread(os.path.join(mask_path, fname), cv2.IMREAD_GRAYSCALE)
        original_name = fname.replace("mask_", "")
        test_img = cv2.imread(os.path.join(defect_path, original_name))

        if test_img is None:
            print(f"  [ERROR] Could not read original image: {original_name}")
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"     Found {len(contours)} contours in {fname}")

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 50:
                print(f"     [SKIP] Small ROI ignored (w*h={w*h})")
                continue

            roi = test_img[y:y+h, x:x+w]
            roi_filename = f"{original_name}_roi{i}.jpg"
            cv2.imwrite(os.path.join(defect_roi_dir, roi_filename), roi)

            roi_count += 1
            defect_count += 1
            labels.append([roi_filename, defect_type])
            print(f"     [SAVE] ROI saved: {roi_filename}")

    per_defect_counts[defect_type] = defect_count
    print(f"  ✅ Finished {defect_type} → {defect_count} ROIs\n")

# =========================
# Save labels to CSV
# =========================
df = pd.DataFrame(labels, columns=["filename", "label"])
df.to_csv(CSV_FILE, index=False)

# =========================
# Final Summary
# =========================
print("\n[END] ROI extraction completed")
print(f"📄 Labels saved to: {CSV_FILE}\n")

print("🔎 ROI Count Summary:")
for defect, count in per_defect_counts.items():
    print(f"   {defect}: {count} ROIs")

print(f"\n✅ Total ROI images generated: {roi_count}")
