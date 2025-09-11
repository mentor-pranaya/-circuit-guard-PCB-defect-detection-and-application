import cv2, os

MASKS_DIR = "output/masks"          
OUTPUT_BBOX = "output/masksboxes"      
os.makedirs(OUTPUT_BBOX, exist_ok=True)

for defect_type in os.listdir(MASKS_DIR):
    mask_path = os.path.join(MASKS_DIR, defect_type)
    if not os.path.isdir(mask_path):
        continue

    print(f"[Contours] Processing defect type: {defect_type}")
    defect_bbox_dir = os.path.join(OUTPUT_BBOX, defect_type)
    os.makedirs(defect_bbox_dir, exist_ok=True)

    for fname in os.listdir(mask_path):
        if not fname.endswith(".jpg"):
            continue

        mask_file = os.path.join(mask_path, fname)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)


        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 50:
                continue

            
            cv2.rectangle(mask_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(mask_vis, defect_type, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

       
        out_path = os.path.join(defect_bbox_dir, f"bbox_{fname}")
        cv2.imwrite(out_path, mask_vis)

print("Contour detection completed (bounding boxes drawn on masks).")
