import cv2
import numpy as np

# Load images in grayscale
golden = cv2.imread(r"c:\Users\phaneendra ybs\OneDrive\Pictures\01.JPG", cv2.IMREAD_GRAYSCALE)
test   = cv2.imread(r"c:\Users\phaneendra ybs\OneDrive\Pictures\01_missing_hole_10.JPG", cv2.IMREAD_GRAYSCALE)

# Check if images loaded
if golden is None or test is None:
    print("Error: One or both images not found. Check file paths!")
    exit()

# Resize test image to match golden image size
test = cv2.resize(test, (golden.shape[1], golden.shape[0]))

# Subtract images
diff = cv2.absdiff(golden, test)

# Threshold to highlight defects
_, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

# Show results
cv2.imshow("Golden (Grayscale)", golden)
cv2.imshow("Test (Grayscale, Resized)", test)
cv2.imshow("Difference", diff)
cv2.imshow("Defects", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
