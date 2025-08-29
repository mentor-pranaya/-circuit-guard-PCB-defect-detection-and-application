import cv2
import os
import numpy as np

REF_DIR   = "dataset/reference_boards"    # defectfree boards
TEST_DIR  = "dataset/defected_boards"     # boards with defects
RESULTS   = "dataset/diff_results"        # where processed file will be stored

os.makedirs(RESULTS, exist_ok=True)

# Loop through reference (golden) boards
for ref_name in os.listdir(REF_DIR):
    ref_path = os.path.join(REF_DIR, ref_name)
    golden = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    if golden is None:
        print(f"[Skip] Could not read {ref_path}")
        continue

    # base file name without extention
    pcb_id = os.path.splitext(ref_name)[0]

    # Search inside every defect category
    for category in os.listdir(TEST_DIR):
        category_path = os.path.join(TEST_DIR, category)

        for test_name in os.listdir(category_path):
            if not test_name.startswith(pcb_id):
                continue

            test_path = os.path.join(category_path, test_name)
            faulty = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

            if faulty is None:
                print(f"[Skip] Could not read {test_path}")
                continue

            # Subtraction
            diff_img = cv2.absdiff(golden, faulty)

            # Thresholding
            _, mask = cv2.threshold(
                diff_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Morphological cleanup
            kernel = np.ones((3, 3), np.uint8)
            refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
            refined = cv2.GaussianBlur(refined, (3, 3), 0)

            # Connected components (filter out noise)
            num_labels, lbls, stats, _ = cv2.connectedComponentsWithStats(refined, 8)
            min_area = 40  # keep only significant blobs
            cleaned = np.zeros_like(refined)

            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[lbls == label] = 255

            # Save the result
            result_name = f"{pcb_id}__{category}__{test_name.replace('.png','')}_mask.png"
            result_path = os.path.join(RESULTS, result_name)
            cv2.imwrite(result_path, cleaned)

            print(f"[OK] Saved mask -> {result_path}")
