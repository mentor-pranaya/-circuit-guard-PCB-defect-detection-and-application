import cv2
import os

def extract_rois(subtracted_folder, roi_folder, min_contour_area=50):
    os.makedirs(roi_folder, exist_ok=True)

    for defect_type in os.listdir(subtracted_folder):
        defect_path = os.path.join(subtracted_folder, defect_type)
        if not os.path.isdir(defect_path):
            continue
        
        roi_type_folder = os.path.join(roi_folder, defect_type)
        os.makedirs(roi_type_folder, exist_ok=True)

        for img_name in os.listdir(defect_path):
            img_path = os.path.join(defect_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for idx, contour in enumerate(contours):
                if cv2.contourArea(contour) > min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = img[y:y+h, x:x+w]
                    roi_path = os.path.join(roi_type_folder, f"{os.path.splitext(img_name)[0]}_roi_{idx}.png")
                    cv2.imwrite(roi_path, roi)
