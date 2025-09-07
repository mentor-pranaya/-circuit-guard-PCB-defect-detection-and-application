# module2_roiextract.py
import cv2, os, glob, pandas as pd, xml.etree.ElementTree as ET

anno_dir = "Annotations"                 # subfolders per class
test_dir = "outputs_pairs/tests"         # test images
output_bbox_dir = "outputs_module2/bbox"
output_rois_dir = "outputs_module2/rois"
os.makedirs(output_bbox_dir, exist_ok=True)
os.makedirs(output_rois_dir, exist_ok=True)

valid_classes = ["Missing_hole","Mouse_bite","Open_circuit","Short","Spur","Spurious_copper"]
metadata = []

print("🔎 Running ROI extraction from XML...")

for cls in valid_classes:
    xml_files = glob.glob(os.path.join(anno_dir, cls, "*.xml"))
    for xml_file in xml_files:
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        # find test image
        test_img_path = None
        for ext in (".jpg",".jpeg",".png"):
            candidate = os.path.join(test_dir, base_name + ext)
            if os.path.exists(candidate):
                test_img_path = candidate
                break
        if test_img_path is None:
            print(f"⚠ No test image found for {base_name}")
            continue

        img = cv2.imread(test_img_path)
        preview = img.copy()

        tree = ET.parse(xml_file)
        root = tree.getroot()
        roi_id = 0
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = int(float(bbox.findtext("xmin",0)))
            ymin = int(float(bbox.findtext("ymin",0)))
            xmax = int(float(bbox.findtext("xmax",0)))
            ymax = int(float(bbox.findtext("ymax",0)))
            # sanity
            if xmax<=xmin or ymax<=ymin: 
                continue
            # Draw box + label (label is folder name)
            cv2.rectangle(preview,(xmin,ymin),(xmax,ymax),(0,255,0),2)
            cv2.putText(preview, cls, (xmin, max(0,ymin-8)), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            # save ROI
            roi = img[ymin:ymax, xmin:xmax]
            roi_dir = os.path.join(output_rois_dir, cls); os.makedirs(roi_dir, exist_ok=True)
            roi_name = f"{base_name}_roi{roi_id}.png"
            cv2.imwrite(os.path.join(roi_dir, roi_name), roi)
            metadata.append([f"{cls}/{roi_name}", cls])
            roi_id += 1
        # save preview
        cv2.imwrite(os.path.join(output_bbox_dir, base_name + "_bbox.png"), preview)

# save CSV (filepath relative to rois/)
df = pd.DataFrame(metadata, columns=["filepath","label"])
df.to_csv("outputs_module2/roi_metadata.csv", index=False)
print("✅ ROI extraction complete. Total ROIs:", len(df))
