import cv2
import numpy as np

# Paths
good_pcb_path = r"Address of orginal pcb image "
defect_pcb_path = r"Adress of defected pcb image"
save_path = r"adress where the image is to saved"

# Load images
good_pcb = cv2.imread(good_pcb_path)
defective_pcb = cv2.imread(defect_pcb_path)

if good_pcb is None or defective_pcb is None:
    print("Error: Could not load one or both images.")
    exit()

# Convert to grayscale
good_pcb_gray = cv2.cvtColor(good_pcb, cv2.COLOR_BGR2GRAY)
defective_pcb_gray = cv2.cvtColor(defective_pcb, cv2.COLOR_BGR2GRAY)

# Resize defective image if shape mismatch
if good_pcb_gray.shape != defective_pcb_gray.shape:
    defective_pcb_gray = cv2.resize(defective_pcb_gray, (good_pcb_gray.shape[1], good_pcb_gray.shape[0]))

# Difference
diff = cv2.absdiff(good_pcb_gray, defective_pcb_gray)

# Threshold (highlight defects)
_, defect_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Create output image (white = defect, black = normal)
output = np.zeros_like(good_pcb_gray)
output[defect_mask > 0] = 255

# Save and show
cv2.imwrite(save_path, output)
cv2.imshow("Detected Defects", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
