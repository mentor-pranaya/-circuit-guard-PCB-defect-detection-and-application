import cv2
import numpy as np

# ✅ Correct paths (only one PCB_DATASET folder)
img1 = cv2.imread(r"C:\infosys spring board intenship\PCB_DATASET\PCB_DATASET\PCB_USED\01.JPG")
img2 = cv2.imread(r"C:\infosys spring board intenship\PCB_DATASET\PCB_DATASET\images\Missing_hole\01_missing_hole_01.jpg")

# 🔍 Debug check for missing files
if img1 is None:
    print("Error: img1 not found")
    exit()
if img2 is None:
    print("Error: img2 not found")
    exit()

# Resize both images to same size (important for subtraction)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Compute absolute difference
diff = cv2.absdiff(gray1, gray2)

# Threshold the difference
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Dilate to make defects visible
kernel = np.ones((3,3), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=2)

# Find contours of defects
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes on differences
for cnt in contours:
    if cv2.contourArea(cnt) > 20:   # ignore tiny noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Show results
cv2.imshow("Original PCB", img1)
cv2.imshow("Defected PCB", img2)
cv2.imshow("Differences", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()