# inference_backend.py
import os
import shutil
import zipfile
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
import pandas as pd
import io

# Default classes used by your project (update if needed)
CLASSES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
IMG_SIZE = 128

# -------------------------
# Model utils
# -------------------------
def load_model(model_path: str, device: str = "cpu", model_name: str = "efficientnet_b4", num_classes: int = None):
    """
    Load timm model checkpoint, tolerant to common wrappers.
    Returns (model, torch.device)
    """
    device_obj = torch.device(device)
    if num_classes is None:
        num_classes = len(CLASSES)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(model_path, map_location=device_obj)

    # unwrap common wrappers
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            sd = ckpt["model_state"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    # normalize keys (remove module. if necessary)
    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            new_sd[nk] = v
    else:
        new_sd = sd

    try:
        model.load_state_dict(new_sd, strict=False)
    except Exception:
        # try fallback
        model.load_state_dict(sd if isinstance(sd, dict) else new_sd, strict=False)

    model.to(device_obj)
    model.eval()
    return model, device_obj

# -------------------------
# Image/Alignment utils
# -------------------------
def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def align_images_orb(template_bgr: np.ndarray, test_bgr: np.ndarray, max_features: int = 1500, downscale_max: int = 1600) -> Tuple[np.ndarray, bool]:
    """
    Align test -> template using ORB + homography.
    Downscale large images to speed up matching, then warp full size using computed homography.
    Returns (aligned_test_bgr, success_flag)
    """
    try:
        h_tpl, w_tpl = template_bgr.shape[:2]
        # optionally downscale for feature matching (keeps aspect)
        scale = 1.0
        max_dim = max(h_tpl, w_tpl)
        if max_dim > downscale_max:
            scale = downscale_max / float(max_dim)
            tpl_small = cv2.resize(template_bgr, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            test_small = cv2.resize(test_bgr, (tpl_small.shape[1], tpl_small.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            tpl_small = template_bgr.copy()
            test_small = cv2.resize(test_bgr, (tpl_small.shape[1], tpl_small.shape[0]), interpolation=cv2.INTER_AREA)

        gray1 = cv2.cvtColor(tpl_small, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(test_small, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=max_features)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return test_bgr, False

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 8:
            return test_bgr, False

        top_n = min(len(matches), 200)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:top_n]]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:top_n]]).reshape(-1,1,2)
        H_small, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H_small is None:
            return test_bgr, False

        # if we scaled images, convert H_small to original scale
        if scale != 1.0:
            # mapping from original test to original template: scale matrices
            S = np.array([[1/scale,0,0],[0,1/scale,0],[0,0,1]], dtype=float)
            H = S @ H_small @ np.array([[scale,0,0],[0,scale,0],[0,0,1]], dtype=float)
        else:
            H = H_small

        aligned = cv2.warpPerspective(test_bgr, H, (template_bgr.shape[1], template_bgr.shape[0]), flags=cv2.INTER_LINEAR)
        return aligned, True
    except Exception:
        return test_bgr, False

# -------------------------
# Mask & ROI detection
# -------------------------
def make_defect_mask(template_bgr: np.ndarray,
                     test_bgr: np.ndarray,
                     morph_iter: int = 1,
                     sensitivity: float = 1.0,
                     min_area: int = 30,
                     max_area: int = 10000,
                     max_rois: int = 10) -> Tuple[np.ndarray, List[tuple]]:
    """
    Create a clean binary defect mask (black background, white defects) and return
    a list of bounding boxes. Filters by area and keeps top-N largest blobs.
    """
    h, w = template_bgr.shape[:2]

    g1 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    g1_blur = cv2.GaussianBlur(g1, (7,7), 0)
    g2_blur = cv2.GaussianBlur(g2, (7,7), 0)

    diff = cv2.absdiff(g1_blur, g2_blur)

    # Otsu threshold to estimate a good threshold value
    retval, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_val = max(5, int(retval * float(sensitivity)))

    # apply threshold
    _, mask = cv2.threshold(diff, thr_val, 255, cv2.THRESH_BINARY)

    # morphological cleanup: small noise removed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

    # connected components to filter by area
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, num_labels):
        x, y, w_box, h_box, area = stats[i]
        if min_area <= area <= max_area:
            comps.append((i, x, y, w_box, h_box, area))

    # sort by area desc and keep top max_rois
    comps = sorted(comps, key=lambda c: c[5], reverse=True)[:max_rois]

    # rebuild filtered mask and boxes
    filtered_mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []
    for (lbl, x, y, w_box, h_box, area) in comps:
        filtered_mask[labels_im == lbl] = 255
        boxes.append((int(x), int(y), int(w_box), int(h_box)))

    return filtered_mask, boxes

# -------------------------
# Classification & annotation
# -------------------------
clf_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def classify_roi_pil(img_pil: Image.Image, model: torch.nn.Module, device: torch.device) -> Tuple[str, float, list]:
    """
    Classify a ROI (PIL) and return top label, confidence and top3 list.
    Uses mixed precision on CUDA to speed up if available.
    """
    x = clf_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        # use amp on CUDA
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = model(x)
        else:
            out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = int(probs.argmax())
    top_conf = float(probs[top_idx])
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(CLASSES[int(i)], float(probs[int(i)])) for i in top3_idx]
    return CLASSES[top_idx], top_conf, top3

def annotate_bboxes_on_image(image_bgr: np.ndarray, boxes: List[tuple], labels: List[str], scores: List[float]) -> np.ndarray:
    """
    Draw boxes and labels on a BGR image and return annotated BGR image.
    """
    out = image_bgr.copy()
    for (x,y,w,h), label, score in zip(boxes, labels, scores):
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        txt = f"{label} {score:.2f}"
        ((tw, th), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, max(0,y-22)), (x+tw+6, y), (0,0,0), -1)
        cv2.putText(out, txt, (x+2, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return out

# -------------------------
# Save outputs & logging
# -------------------------
def save_run_outputs(output_root: str,
                     run_id: str,
                     annotated_pil: Image.Image,
                     mask_np: np.ndarray,
                     boxes: List[tuple],
                     rois_pil: List[Image.Image],
                     predictions_df: pd.DataFrame) -> Dict[str,str]:
    """
    Save annotated image, mask, rois, predictions CSV and return saved paths.
    Output layout:
      output_root/run_{run_id}/annotated.png
                             /mask.png
                             /rois/*.png
                             /predictions.csv
    Also returns path to zipped folder when requested.
    """
    run_dir = os.path.join(output_root, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    # annotated
    ann_path = os.path.join(run_dir, "annotated.png")
    annotated_pil.save(ann_path)
    # mask (save as RGB so streamlit shows correctly)
    mask_rgb = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
    mask_pil = Image.fromarray(mask_rgb)
    mask_path = os.path.join(run_dir, "mask.png")
    mask_pil.save(mask_path)
    # rois
    rois_dir = os.path.join(run_dir, "rois")
    os.makedirs(rois_dir, exist_ok=True)
    for i, roi in enumerate(rois_pil):
        rpath = os.path.join(rois_dir, f"roi_{i}.png")
        roi.save(rpath)
    # predictions CSV
    csv_path = os.path.join(run_dir, "predictions.csv")
    predictions_df.to_csv(csv_path, index=False)

    # create a zip of run folder for easy download
    zip_path = os.path.join(output_root, f"run_{run_id}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(os.path.join(output_root, f"run_{run_id}"), 'zip', run_dir)

    return {
        "run_dir": run_dir,
        "annotated": ann_path,
        "mask": mask_path,
        "rois_dir": rois_dir,
        "predictions_csv": csv_path,
        "zip": zip_path
    }

def append_log(log_root: str, row: dict):
    os.makedirs(log_root, exist_ok=True)
    log_path = os.path.join(log_root, "inference_log.csv")
    df = pd.DataFrame([row])
    if not os.path.exists(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode="a", header=False, index=False)
    return log_path

# -------------------------
# High-level runner
# -------------------------
def run_single_pair(template: Image.Image,
                    test: Image.Image,
                    model: torch.nn.Module,
                    device: torch.device,
                    output_root: str = "outputs_module6",
                    run_id: Optional[str] = None,
                    align: bool = True,
                    morph_iter: int = 1,
                    sensitivity: float = 1.0,
                    min_area: int = 30,
                    max_area: int = 10000,
                    max_rois: int = 10) -> Dict[str, Any]:
    """
    Process a single template/test pair and return a dict with paths and dataframe.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    tpl_bgr = pil_to_bgr(template)
    test_bgr = pil_to_bgr(test)
    # align
    if align:
        aligned, ok = align_images_orb(tpl_bgr, test_bgr)
    else:
        aligned, ok = test_bgr, False

    # mask + boxes
    mask, boxes = make_defect_mask(tpl_bgr, aligned, morph_iter=morph_iter,
                                   sensitivity=sensitivity, min_area=min_area,
                                   max_area=max_area, max_rois=max_rois)

    rois_pil = []
    labels = []
    scores = []
    top3s = []
    rows = []

    for i, (x,y,w,h) in enumerate(boxes):
        roi_bgr = aligned[y:y+h, x:x+w]
        if roi_bgr.size == 0:
            continue
        roi_pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        lab, conf, top3 = classify_roi_pil(roi_pil, model, device)
        rois_pil.append(roi_pil)
        labels.append(lab); scores.append(conf); top3s.append(top3)
        rows.append({
            "roi_id": i, "x": int(x), "y": int(y),
            "w": int(w), "h": int(h),
            "pred_label": lab, "confidence": float(conf),
            "top3": ";".join([f"{n}:{p:.3f}" for n,p in top3])
        })

    df = pd.DataFrame(rows)
    annotated_bgr = annotate_bboxes_on_image(aligned, boxes, labels, scores)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))

    saved = save_run_outputs(output_root, run_id, annotated_pil, mask, boxes, rois_pil, df)

    # append to global log
    log_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "annotated_path": saved["annotated"],
        "mask_path": saved["mask"],
        "predictions_csv": saved["predictions_csv"],
        "rois_dir": saved["rois_dir"],
        "num_rois": len(df),
        "labels": ";".join(labels)
    }
    log_path = append_log(os.path.join(output_root, "log"), log_row)

    return {
        "run_id": run_id,
        "saved_paths": saved,
        "predictions_df": df,
        "log_path": log_path,
        "aligned_ok": ok
    }

# -------------------------
# Optional: batch runner that consumes a CSV with template,test columns
# -------------------------
def batch_run_from_pairs(pairs_csv: str, model, device, output_root: str = "outputs_module6",
                         align=True, morph_iter=1, sensitivity=1.0, min_area=30, max_area=10000, max_rois=10):
    pairs_df = pd.read_csv(pairs_csv)
    summary = []
    for idx, row in pairs_df.iterrows():
        tpl_path = str(row.get("template") or row.get("template_path") or row.get("template_file"))
        test_path = str(row.get("test") or row.get("test_path") or row.get("test_file"))
        if not os.path.exists(tpl_path) or not os.path.exists(test_path):
            continue
        tpl = Image.open(tpl_path).convert("RGB")
        tst = Image.open(test_path).convert("RGB")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{idx}"
        out = run_single_pair(tpl, tst, model, device, output_root=output_root,
                              run_id=run_id, align=align,
                              morph_iter=morph_iter, sensitivity=sensitivity,
                              min_area=min_area, max_area=max_area, max_rois=max_rois)
        summary.append(out)
    return summary
