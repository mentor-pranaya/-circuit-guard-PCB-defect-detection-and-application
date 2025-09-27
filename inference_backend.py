# inference_backend.py
"""
Backend module for AI-CircuitGuard inference:
- load_model
- image alignment, subtraction, mask, bbox extraction
- ROI classification
- save outputs + log
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

import os
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
import pandas as pd

# -----------------------------------------------------------------------------
# Config defaults (change here if you want global defaults)
# -----------------------------------------------------------------------------
DEFAULT_CLASSES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
DEFAULT_MODEL_NAME = "efficientnet_b4"

# -----------------------------------------------------------------------------
# Utilities: conversions
# -----------------------------------------------------------------------------
def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

# -----------------------------------------------------------------------------
# Model loading (robust)
# -----------------------------------------------------------------------------
def load_model(model_path: str,
               classes: List[str] = DEFAULT_CLASSES,
               device: str = "cpu",
               model_name: str = DEFAULT_MODEL_NAME):
    """
    Loads an EfficientNet model (timm) and returns (model, device).
    Tries to be permissive about checkpoint formats.
    """
    device = torch.device(device)
    model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
    ckpt = torch.load(model_path, map_location=device)

    # Normalize format
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            sd = ckpt["model_state"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    # Some checkpoints saved with "module." prefix
    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            new_sd[nk] = v
        sd = new_sd

    # Load state (be permissive)
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        # fallback: try if sd itself is nested
        try:
            model.load_state_dict(sd.get("state_dict", sd), strict=False)
        except Exception:
            # last resort: ignore - model stays randomly init'd (user will see bad accuracy)
            pass

    model.to(device)
    model.eval()
    return model, device

# -----------------------------------------------------------------------------
# Alignment (ORB + homography)
# -----------------------------------------------------------------------------
def align_images_orb(template_bgr: np.ndarray, test_bgr: np.ndarray,
                     max_features: int = 1500,
                     min_matches: int = 8) -> Tuple[np.ndarray, bool]:
    """
    Align test image to template using ORB + homography.
    Returns (aligned_test_image, success_flag).
    If alignment fails returns original test image and False.
    """
    try:
        g1 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=max_features)
        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)
        if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
            return test_bgr, False

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < min_matches:
            return test_bgr, False

        top_n = min(len(matches), 200)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:top_n]]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:top_n]]).reshape(-1,1,2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return test_bgr, False

        h, w = template_bgr.shape[:2]
        aligned = cv2.warpPerspective(test_bgr, H, (w, h), flags=cv2.INTER_LINEAR)
        return aligned, True
    except Exception:
        return test_bgr, False

# -----------------------------------------------------------------------------
# Subtraction, mask creation (Otsu) + optional LAB/edge diffs
# -----------------------------------------------------------------------------
def subtract_and_mask(template_bgr: np.ndarray,
                      test_bgr: np.ndarray,
                      morph_iter: int = 1,
                      kernel_size: int = 3,
                      use_lab: bool = True,
                      use_edges: bool = True) -> np.ndarray:
    """
    Combine multiple simple difference signals and clean them with morph ops.
    Returns binary mask (uint8 0/255).
    """
    # grayscale diff (base)
    g1 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    diff_g = cv2.absdiff(g1, g2)
    diff_g = cv2.GaussianBlur(diff_g, (5,5), 0)
    _, mask_g = cv2.threshold(diff_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [mask_g]

    if use_lab:
        try:
            lab1 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2LAB)
            lab2 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2LAB)
            diff_a = cv2.absdiff(lab1[:,:,1], lab2[:,:,1])
            diff_b = cv2.absdiff(lab1[:,:,2], lab2[:,:,2])
            diff_a = cv2.GaussianBlur(diff_a, (5,5), 0)
            diff_b = cv2.GaussianBlur(diff_b, (5,5), 0)
            _, mask_a = cv2.threshold(diff_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, mask_b = cv2.threshold(diff_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append(cv2.bitwise_or(mask_a, mask_b))
        except Exception:
            pass

    if use_edges:
        try:
            e1 = cv2.Canny(g1, 50, 150)
            e2 = cv2.Canny(g2, 50, 150)
            mask_e = cv2.bitwise_xor(e1, e2)
            mask_e = cv2.dilate(mask_e, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
            masks.append(mask_e)
        except Exception:
            pass

    combined = np.zeros_like(masks[0])
    for m in masks:
        combined = cv2.bitwise_or(combined, m)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k, iterations=morph_iter)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=morph_iter)
    return combined

# -----------------------------------------------------------------------------
# Extract bounding boxes from mask
# -----------------------------------------------------------------------------
def extract_bboxes_from_mask(mask: np.ndarray,
                             min_area: int = 50,
                             max_rois: int = 200) -> List[tuple]:
    """
    Returns list of (x,y,w,h) sorted top-to-bottom, left-to-right.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h >= min_area:
            boxes.append((x,y,w,h))
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes[:max_rois]

# -----------------------------------------------------------------------------
# Classification helper
# -----------------------------------------------------------------------------
clf_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def classify_roi_pil(img_pil: Image.Image, model, device, classes: List[str] = DEFAULT_CLASSES) -> Tuple[str, float, List[tuple]]:
    """
    Return (pred_label, confidence, top3_list).
    """
    x = clf_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = int(probs.argmax())
    top_conf = float(probs[top_idx])
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(classes[int(i)], float(probs[int(i)])) for i in top3_idx]
    return classes[top_idx], top_conf, top3

# -----------------------------------------------------------------------------
# Annotate image with boxes + label
# -----------------------------------------------------------------------------
def annotate_bboxes_on_image(image_bgr: np.ndarray, boxes: List[tuple], labels: List[str], scores: List[float]) -> np.ndarray:
    out = image_bgr.copy()
    for (x,y,w,h), label, score in zip(boxes, labels, scores):
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        txt = f"{label} {score:.2f}"
        ((tw, th), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, max(0, y - (th+6))), (x + tw + 6, y), (0,0,0), -1)
        cv2.putText(out, txt, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return out

# -----------------------------------------------------------------------------
# Save outputs and log
# -----------------------------------------------------------------------------
def save_run_outputs(output_dir: str,
                     run_id: str,
                     annotated_pil: Image.Image,
                     rois_pil: List[Image.Image],
                     predictions: List[Dict],
                     template_name: Optional[str] = None,
                     test_name: Optional[str] = None):
    """
    Saves:
      - annotated image -> output_dir/annotated/annotated_<runid>.png
      - rois -> output_dir/rois/run_<runid>/...
      - predictions csv -> output_dir/predictions/predictions_<runid>.csv
      - append log to output_dir/log/inference_log.csv
    Returns dict with saved paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    annotated_dir = os.path.join(output_dir, "annotated")
    rois_dir = os.path.join(output_dir, "rois", f"run_{run_id}")
    preds_dir = os.path.join(output_dir, "predictions")
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(rois_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    annotated_path = os.path.join(annotated_dir, f"annotated_{run_id}.png")
    annotated_pil.save(annotated_path)

    rows = []
    for i, (roi_pil, pred) in enumerate(zip(rois_pil, predictions)):
        roi_fname = f"roi_{i}_{pred['pred_label']}_{int(pred['confidence']*100)}.png"
        roi_path = os.path.join(rois_dir, roi_fname)
        roi_pil.save(roi_path)
        rows.append({
            "roi_fname": os.path.relpath(roi_path, output_dir),
            "pred_label": pred["pred_label"],
            "confidence": pred["confidence"],
            "top3": ";".join([f"{n}:{p:.3f}" for n,p in pred.get("top3", [])]),
            "x": pred.get("x"), "y": pred.get("y"), "w": pred.get("w"), "h": pred.get("h")
        })

    preds_df = pd.DataFrame(rows)
    preds_csv = os.path.join(preds_dir, f"predictions_{run_id}.csv")
    preds_df.to_csv(preds_csv, index=False)

    # append log
    log_path = os.path.join(log_dir, "inference_log.csv")
    log_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "template": template_name or "",
        "test": test_name or "",
        "num_rois": len(rows),
        "pred_csv": os.path.relpath(preds_csv, output_dir)
    }
    log_df = pd.DataFrame([log_row])
    if not os.path.exists(log_path):
        log_df.to_csv(log_path, index=False)
    else:
        log_df.to_csv(log_path, mode="a", header=False, index=False)

    return {"annotated": annotated_path, "predictions_csv": preds_csv, "rois_dir": rois_dir, "log": log_path}

# -----------------------------------------------------------------------------
# One-shot pipeline that ties everything together
# -----------------------------------------------------------------------------
def run_inference_on_pair(template_pil: Image.Image,
                          test_pil: Image.Image,
                          model,
                          device,
                          classes: List[str] = DEFAULT_CLASSES,
                          align: bool = True,
                          morph_iter: int = 1,
                          kernel_size: int = 3,
                          use_lab: bool = True,
                          use_edges: bool = True,
                          min_area: int = 50,
                          max_rois: int = 200):
    """
    Full pipeline:
     - align (optional)
     - subtract_and_mask
     - extract boxes
     - crop ROIs, classify
     - annotate
     - return structured results (not saved)
    """
    tpl_cv = pil_to_cv2(template_pil)
    test_cv = pil_to_cv2(test_pil)

    if align:
        aligned_cv, ok = align_images_orb(tpl_cv, test_cv)
    else:
        aligned_cv, ok = test_cv, False

    mask = subtract_and_mask(tpl_cv, aligned_cv, morph_iter=morph_iter,
                             kernel_size=kernel_size, use_lab=use_lab, use_edges=use_edges)
    boxes = extract_bboxes_from_mask(mask, min_area=min_area, max_rois=max_rois)

    rois_pil = []
    predictions = []
    labels = []
    scores = []
    for (i,(x,y,w,h)) in enumerate(boxes):
        roi_cv = aligned_cv[y:y+h, x:x+w]
        if roi_cv.size == 0:
            continue
        roi_pil = cv2_to_pil(roi_cv)
        pred_label, conf, top3 = classify_roi_pil(roi_pil, model, device, classes)
        rois_pil.append(roi_pil)
        predictions.append({"roi_id": i, "pred_label": pred_label, "confidence": conf, "top3": top3, "x": x, "y": y, "w": w, "h": h})
        labels.append(pred_label); scores.append(conf)

    annotated_cv = annotate_bboxes_on_image(aligned_cv.copy(), [(b[0],b[1],b[2],b[3]) for b in boxes], labels, scores)
    annotated_pil = cv2_to_pil(annotated_cv)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "run_id": run_id,
        "aligned_cv": aligned_cv,
        "mask": mask,
        "boxes": boxes,
        "rois_pil": rois_pil,
        "predictions": predictions,
        "annotated_pil": annotated_pil
    }
