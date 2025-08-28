import cv2, pathlib
import numpy as np

# Input: masks from subtraction step
IN_MASKS = pathlib.Path("milestone1/outputs_subtraction/masks")
IN_TESTS = pathlib.Path("milestone1/outputs_aligned")
OUT_ROI  = pathlib.Path("milestone1/outputs_subtraction/rois")

OUT_ROI.mkdir(parents=True, exist_ok=True)

masks = sorted(IN_MASKS.glob("*_mask.png"))
print(f"Found {len(masks)} masks for contour detection.")

for m_path in masks:
    test_path = pathlib.Path(str(m_path).replace("_mask", "_test").replace("masks", ""))
    if not test_path.exists():
        print(f"❌ Missing test image for {m_path.name}")
        continue

    # Load mask + test image
    mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(str(test_path), cv2.IMREAD_GRAYSCALE)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_idx = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # filter small noise
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Crop ROI from test image
        roi = test_img[y:y+h, x:x+w]

        roi_name = f"{test_path.stem}_roi{roi_idx}.png"
        cv2.imwrite(str(OUT_ROI / roi_name), roi)

        vis = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.imwrite(str(OUT_ROI / (test_path.stem + "_with_boxes.png")), vis)

        roi_idx += 1

print("✔ Contour detection + ROI extraction complete. Check outputs_subtraction/rois/")