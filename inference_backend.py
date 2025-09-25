# inference_backend.py
import os
import time
import io
import logging
from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms

# ---------------------------
# Logging & output dirs
# ---------------------------
OUT_DIR = "outputs_module6"
ANNOT_DIR = os.path.join(OUT_DIR, "annotated")
ROIS_DIR = os.path.join(OUT_DIR, "rois")
LOG_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(ANNOT_DIR, exist_ok=True)
os.makedirs(ROIS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ---------------------------
# Settings
# ---------------------------
DEFAULT_MODEL_NAME = "efficientnet_b4"  # change to b0 if CPU-only & slow
CLASS_NAMES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
IMG_SIZE = 128

# ---------------------------
# Model loader (singleton)
# ---------------------------
_model_cache = {}

def load_model(model_path: str, device: str = "cpu", model_name: str = DEFAULT_MODEL_NAME):
    key = (model_path, device, model_name)
    if key in _model_cache:
        return _model_cache[key]

    device_t = torch.device(device)
    model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(model_path, map_location=device_t)
    # handle different checkpoint shapes
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            sd = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        else:
            sd = checkpoint
    else:
        sd = checkpoint

    # normalize keys
    new_sd = {}
    if isinstance(sd, dict):
        for k, v in sd.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            new_sd[nk] = v
    else:
        new_sd = sd

    try:
        model.load_state_dict(new_sd)
    except Exception as e:
        # fallback - try direct if keys match
        model.load_state_dict(sd)

    model.to(device_t)
    model.eval()
    _model_cache[key] = (model, device_t)
    logging.info(f"Loaded model {model_name} from {model_path} on {device}")
    return model, device_t

# ---------------------------
# Image processing helpers
# ---------------------------
def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

def align_images_orb(template_bgr: np.ndarray, test_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Return aligned test and boolean success flag."""
    try:
        gray1 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
            return test_bgr, False
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 8:
            return test_bgr, False
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return test_bgr, False
        h, w = template_bgr.shape[:2]
        aligned = cv2.warpPerspective(test_bgr, H, (w, h), flags=cv2.INTER_LINEAR)
        return aligned, True
    except Exception as e:
        logging.warning(f"Alignment failed: {e}")
        return test_bgr, False

def subtract_and_mask(template_bgr: np.ndarray, test_aligned_bgr: np.ndarray, morph_iter: int = 1) -> np.ndarray:
    g1 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(test_aligned_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    diff = cv2.GaussianBlur(diff, (5,5), 0)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    if morph_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    return mask

def extract_bboxes_from_mask(mask: np.ndarray, min_area: int = 50) -> List[Tuple[int,int,int,int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h >= min_area:
            boxes.append((x,y,w,h))
    return sorted(boxes, key=lambda b:(b[1], b[0]))

# ---------------------------
# Classification helpers (batched)
# ---------------------------
clf_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def prepare_batch_from_pils(pil_list: List[Image.Image], device: torch.device) -> torch.Tensor:
    """Return tensor shape (N,C,H,W) on device."""
    if len(pil_list) == 0:
        return None
    tlist = [clf_transform(p).unsqueeze(0) for p in pil_list]  # list of [1,C,H,W]
    batch = torch.cat(tlist, dim=0).to(device)
    return batch

def classify_batch_rois(pil_list: List[Image.Image], model, device: torch.device, batch_size: int = 32):
    """Return list of (top_label, top_conf, top3list) for each roi (keeps order)."""
    results = []
    n = len(pil_list)
    if n == 0:
        return results
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_pils = pil_list[i:i+batch_size]
            batch = prepare_batch_from_pils(batch_pils, device)
            out = model(batch)
            probs = F.softmax(out, dim=1).cpu().numpy()
            for p in probs:
                top_idx = int(p.argmax())
                top_conf = float(p[top_idx])
                top3_idx = p.argsort()[-3:][::-1]
                top3 = [(CLASS_NAMES[int(ii)], float(p[int(ii)])) for ii in top3_idx]
                results.append((CLASS_NAMES[top_idx], top_conf, top3))
    return results

# ---------------------------
# Annotation & save helpers
# ---------------------------
def annotate_bboxes_on_image(image_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str], scores: List[float]) -> np.ndarray:
    out = image_bgr.copy()
    for (x,y,w,h), label, score in zip(boxes, labels, scores):
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        txt = f"{label} {score:.2f}"
        ((tw, th), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out, (x, max(0, y-18)), (x+tw+6, y), (0,0,0), -1)
        cv2.putText(out, txt, (x+2, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return out

def save_annotated_and_rois(base_name: str, annotated_pil: Image.Image, roi_pils: List[Image.Image], labels: List[str]):
    # Save annotated
    annot_path = os.path.join(ANNOT_DIR, f"{base_name}_annotated.png")
    annotated_pil.save(annot_path)
    # Save rois by label subfolder
    for i, (rp, lab) in enumerate(zip(roi_pils, labels)):
        lab_dir = os.path.join(ROIS_DIR, lab)
        os.makedirs(lab_dir, exist_ok=True)
        fname = f"{base_name}_roi{i}.png"
        rp.save(os.path.join(lab_dir, fname))
    return annot_path

# ---------------------------
# Top-level pipeline
# ---------------------------
def run_pipeline(template_pil: Image.Image,
                 test_pil: Image.Image,
                 model_path: str,
                 device: str = "cpu",
                 model_name: str = DEFAULT_MODEL_NAME,
                 min_area: int = 80,
                 morph_iter: int = 1,
                 batch_size: int = 32,
                 save_outputs: bool = True) -> Dict[str, Any]:
    """
    Run full pipeline and return dict:
    {
      "annotated_pil": PIL.Image,
      "mask_pil": PIL.Image,
      "roi_pils": [PIL.Image,...],
      "predictions": [{"roi_id":0,"x":..,"y":..,"w":..,"h":..,"label":..,"score":..,"top3":..}, ...],
      "annotated_path": path or None,
      "times": {...}
    }
    """
    t0 = time.time()
    tpl_cv = pil_to_cv2(template_pil)
    tst_cv = pil_to_cv2(test_pil)

    # 1) Align
    t1 = time.time()
    aligned_cv, ok_align = align_images_orb(tpl_cv, tst_cv)
    t_align = time.time() - t1
    logging.info(f"Alignment: ok={ok_align} time={t_align:.3f}s")

    # 2) Subtract + mask
    t2 = time.time()
    mask = subtract_and_mask(tpl_cv, aligned_cv, morph_iter=morph_iter)
    t_mask = time.time() - t2
    logging.info(f"Mask computed time={t_mask:.3f}s")

    # 3) Extract boxes
    boxes = extract_bboxes_from_mask(mask, min_area=min_area)
    logging.info(f"Found {len(boxes)} boxes (min_area={min_area})")

    # 4) Crop ROIs
    roi_pils = []
    coords = []
    for (x,y,w,h) in boxes:
        crop = aligned_cv[y:y+h, x:x+w].copy()
        roi_pils.append(cv2_to_pil(crop))
        coords.append((x,y,w,h))

    # 5) Load model
    t3 = time.time()
    model, device_t = load_model(model_path, device=device, model_name=model_name)
    t_model_load = time.time() - t3

    # 6) Classify ROIs (batch)
    t4 = time.time()
    preds = classify_batch_rois(roi_pils, model, device_t, batch_size=batch_size)
    t_classify = time.time() - t4
    logging.info(f"Classified {len(roi_pils)} ROIs time={t_classify:.3f}s")

    # 7) Build predictions list
    predictions = []
    labels = []
    scores = []
    for i, ((x,y,w,h), p) in enumerate(zip(coords, preds)):
        label, conf, top3 = p
        predictions.append({"roi_id": i, "x": x, "y": y, "w": w, "h": h, "label": label, "score": conf, "top3": top3})
        labels.append(label)
        scores.append(conf)

    # 8) Annotate
    annotated_cv = annotate_bboxes_on_image(aligned_cv, coords, labels, scores)
    annotated_pil = cv2_to_pil(annotated_cv)
    mask_pil = Image.fromarray(mask) if mask.ndim == 2 else Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

    # 9) Save outputs (optional)
    annot_path = None
    if save_outputs:
        base_name = f"run_{int(time.time())}"
        try:
            annot_path = save_annotated_and_rois(base_name, annotated_pil, roi_pils, labels)
        except Exception as e:
            logging.warning(f"Failed saving outputs: {e}")

    total_time = time.time() - t0
    logging.info(f"Total pipeline time={total_time:.3f}s (align {t_align:.3f}, mask {t_mask:.3f}, load {t_model_load:.3f}, classify {t_classify:.3f})")

    return {
        "annotated_pil": annotated_pil,
        "mask_pil": mask_pil,
        "roi_pils": roi_pils,
        "predictions": predictions,
        "annotated_path": annot_path,
        "ok_align": ok_align,
        "times": {
            "total": total_time,
            "align": t_align,
            "mask": t_mask,
            "model_load": t_model_load,
            "classify": t_classify
        }
    }
