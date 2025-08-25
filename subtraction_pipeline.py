import cv2, pathlib, os
import numpy as np

IN_DIR   = pathlib.Path("milestone1/outputs_aligned")
OUT_DIFF = pathlib.Path("milestone1/outputs_subtraction/diffs")
OUT_MASK = pathlib.Path("milestone1/outputs_subtraction/masks")

OUT_DIFF.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)

templates = sorted(IN_DIR.glob("*_template.png"))

print(f"Found {len(templates)} aligned pairs.")

for t_path in templates:
    test_path = pathlib.Path(str(t_path).replace("_template", "_test"))
    if not test_path.exists():
        print(f"❌ Missing test image for {t_path.name}")
        continue

    t_img = cv2.imread(str(t_path), cv2.IMREAD_GRAYSCALE)
    s_img = cv2.imread(str(test_path), cv2.IMREAD_GRAYSCALE)

    # subtraction
    diff = cv2.absdiff(t_img, s_img)

    # Otsu threshold
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean mask (remove noise + enlarge defects)
    kernel = np.ones((3,3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_DILATE, kernel)

    # save results
    cv2.imwrite(str(OUT_DIFF / t_path.name.replace("_template", "_diff")), diff)
    cv2.imwrite(str(OUT_MASK / t_path.name.replace("_template", "_mask")), mask_clean)

print("✔ Subtraction complete. Check outputs_subtraction/")
