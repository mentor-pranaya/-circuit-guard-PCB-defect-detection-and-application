import cv2
import os
import pathlib
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm   # progress bar

# Paths
DATA_ROOT = pathlib.Path(r"C:\Users\mdnou\OneDrive\Desktop\AI-CircuitGuard\PCB_DATASET\PCB_DATASET")
USED_DIR = DATA_ROOT / "PCB_USED"
DEFECT_IMAGES_DIR = DATA_ROOT / "images"

print("DATA_ROOT:", DATA_ROOT)
print("USED_DIR:", USED_DIR, "Exists?", USED_DIR.exists())
print("DEFECT_IMAGES_DIR:", DEFECT_IMAGES_DIR, "Exists?", DEFECT_IMAGES_DIR.exists())

OUTPUT_DIR = pathlib.Path("milestone1/outputs_pairs")
TEMPLATE_DIR = OUTPUT_DIR / "templates"
TEST_DIR = OUTPUT_DIR / "tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

def best_match(test_img, candidates):
    best_score = -1
    best_candidate = None
    for c_path in candidates:
        cand_img = cv2.imread(str(c_path), cv2.IMREAD_GRAYSCALE)
        cand_img = cv2.resize(cand_img, (test_img.shape[1], test_img.shape[0]))
        score = ssim(test_img, cand_img)
        if score > best_score:
            best_score = score
            best_candidate = c_path
    return best_candidate, best_score

records = []
candidate_templates = list(USED_DIR.glob("*.jpg"))

# Loop with progress bar
for class_dir in DEFECT_IMAGES_DIR.iterdir():
    if not class_dir.is_dir():
        continue
    print(f"Processing class: {class_dir.name}")
    for img_path in tqdm(class_dir.glob("*.jpg"), desc=f"{class_dir.name}"):
        test_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        match, score = best_match(test_img, candidate_templates)

        # Save test image
        out_test = TEST_DIR / img_path.name
        cv2.imwrite(str(out_test), cv2.imread(str(img_path)))

        # Save template image
        out_template = TEMPLATE_DIR / f"{img_path.stem}_template.jpg"
        cv2.imwrite(str(out_template), cv2.imread(str(match)))

        records.append({
            "test_image": str(out_test),
            "matched_template": str(out_template),
            "template_source_original": str(match),
            "ssim": score
        })

# Save mapping CSV
df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR / "pairs_mapping.csv", index=False)

print("✔ Template/Test pairs created at:", OUTPUT_DIR)
