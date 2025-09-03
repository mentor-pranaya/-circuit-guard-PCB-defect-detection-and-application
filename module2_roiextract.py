"""
Module 2: Contour Detection & ROI Extraction
--------------------------------------------
Inputs:
  - outputs_pairs/tests/   : test images
  - outputs_subtraction/masks/ : binary masks from Module 1

Outputs (created automatically):
  - outputs_module2/bbox/        : previews with bounding boxes
  - outputs_module2/rois/<class>/ : cropped ROI images (class auto from filename)
  - outputs_module2/roi_metadata.csv : metadata of all ROIs
"""

import os, sys, cv2
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
ROOT = "."   # run this script from milestone1/
TEST_DIR = os.path.join(ROOT, "outputs_pairs", "tests")
MASK_DIR = os.path.join(ROOT, "outputs_subtraction", "masks")
OUT_DIR  = os.path.join(ROOT, "outputs_module2")
BBOX_DIR = os.path.join(OUT_DIR, "bbox")
ROIS_DIR = os.path.join(OUT_DIR, "rois")
METADATA = os.path.join(OUT_DIR, "roi_metadata.csv")

# Filtering parameters
MIN_AREA = 100     # ignore contours smaller than this (pixels)
MIN_EXTENT = 0.0   # keep 0 if unsure; else use e.g. 0.1
PAD = 4            # padding around cropped ROI
MERGE_NEARBY = True
DILATE_KSZ = (3,3)
DILATE_ITERS = 1
# ----------------------------

# Ensure output dirs exist
os.makedirs(BBOX_DIR, exist_ok=True)
os.makedirs(ROIS_DIR, exist_ok=True)

def find_mask_for_base(base):
    """Find mask file for a test image base name."""
    candidate = f"{base}_mask.png"
    cand_path = os.path.join(MASK_DIR, candidate)
    if os.path.isfile(cand_path):
        return cand_path
    # fallback: substring match
    for m in os.listdir(MASK_DIR):
        if base.lower() in m.lower():
            return os.path.join(MASK_DIR, m)
    return None

def clean_mask(mask):
    """Apply binarization and morphology to mask."""
    if mask is None: return None
    if mask.ndim == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    if MERGE_NEARBY:
        kd = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KSZ)
        mask = cv2.dilate(mask, kd, iterations=DILATE_ITERS)
    return mask

def clamp(v, a, b): return max(a, min(b, v))

def get_class_from_filename(fname):
    """Extract defect class from filename (e.g. '01_missing_hole_01.jpg' -> 'missing_hole')."""
    name = os.path.splitext(fname)[0]
    parts = name.split("_")
    if len(parts) > 1:
        return parts[1]   # second part = class
    return "unlabeled"

# ------------------ MAIN -------------------
rows = []
gid = 0
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.bmp'))])
if not test_files:
    print("[ERROR] No test images found in", TEST_DIR)
    sys.exit(1)

for tname in test_files:
    base = os.path.splitext(tname)[0]
    tpath = os.path.join(TEST_DIR, tname)
    mask_path = find_mask_for_base(base)
    if mask_path is None:
        continue

    img = cv2.imread(tpath)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue

    H,W = img.shape[:2]
    pmask = clean_mask(mask)
    contours, _ = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = img.copy()

    defect_class = get_class_from_filename(tname)
    cls_dir = os.path.join(ROIS_DIR, defect_class)
    os.makedirs(cls_dir, exist_ok=True)

    for cid, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x,y,w,h = cv2.boundingRect(c)
        extent = area / float(w*h) if (w*h)>0 else 0
        if MIN_EXTENT > 0 and extent < MIN_EXTENT:
            continue

        x0 = clamp(x - PAD, 0, W-1)
        y0 = clamp(y - PAD, 0, H-1)
        x1 = clamp(x + w + PAD, 0, W-1)
        y1 = clamp(y + h + PAD, 0, H-1)
        w2 = x1 - x0
        h2 = y1 - y0

        roi_name = f"{base}_c{cid}.png"
        roi_path = os.path.join(cls_dir, roi_name)
        cv2.imwrite(roi_path, img[y0:y0+h2, x0:x0+w2])

        cv2.rectangle(annotated, (x0,y0), (x0+w2, y0+h2), (0,0,255), 2)

        rows.append({
            "row_id": gid,
            "image_name": tname,
            "mask_name": os.path.basename(mask_path),
            "contour_id": cid,
            "class": defect_class,
            "x": int(x0), "y": int(y0), "w": int(w2), "h": int(h2),
            "area": int(area),
            "roi_path": roi_path
        })
        gid += 1

    out_preview = os.path.join(BBOX_DIR, f"{base}_bbox.png")
    cv2.imwrite(out_preview, annotated)

# save metadata CSV
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(METADATA, index=False)

print("\n✅ Module 2 finished.")
print("ROIs saved in:", ROIS_DIR)
print("Previews saved in:", BBOX_DIR)
print("Metadata CSV:", METADATA)
