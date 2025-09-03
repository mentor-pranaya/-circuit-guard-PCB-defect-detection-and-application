import cv2
import pandas as pd
import pathlib

pairs_csv = pathlib.Path("milestone1/outputs_pairs/pairs_mapping.csv")
df = pd.read_csv(pairs_csv)

output_dir = pathlib.Path("milestone1/outputs_aligned")
output_dir.mkdir(parents=True, exist_ok=True)

for idx, row in df.iterrows():
    test_path = pathlib.Path(row["test_image"])
    template_path = pathlib.Path(row["matched_template"])

    test = cv2.imread(str(test_path), cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

    if test is None or template is None:
        continue

    # Resize test to match template size
    test_resized = cv2.resize(test, (template.shape[1], template.shape[0]))

    # Save aligned test and template
    cv2.imwrite(str(output_dir / f"{test_path.stem}_test.png"), test_resized)
    cv2.imwrite(str(output_dir / f"{test_path.stem}_template.png"), template)

print("✔ Alignment complete. Saved in:", output_dir)
