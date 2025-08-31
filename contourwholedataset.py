import cv2
import os

# === Paths ===
input_path = r"C:\Users\phaneendra ybs\Downloads\PCB_subtarction\Missing_hole"  
output_path = r"C:\Users\phaneendra ybs\Downloads\PCB_contour"

os.makedirs(output_path, exist_ok=True)

# Valid image extensions
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# === Loop over only images ===
for img_name in os.listdir(input_path):
    if not img_name.lower().endswith(valid_ext):  # skip non-image files
        print(f"Skipping non-image: {img_name}")
        continue

    img_path = os.path.join(input_path, img_name)

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping {img_name}, cannot read file.")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold (binary image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the image
    contour_img = image.copy()
    for c in contours:
        if cv2.contourArea(c) > 50:   # ignore tiny noise
            cv2.drawContours(contour_img, [c], -1, (0, 255, 0), 2)

    # Save output
    save_path = os.path.join(output_path, f"contour_{img_name}")
    cv2.imwrite(save_path, contour_img)

    print(f"Processed: {img_name}, contours found: {len(contours)}")

print("✅ Contour extraction completed for all valid images.")

