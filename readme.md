AI-CircuitGuard – PCB Defect Detection

## Milestone 1: Preprocessing & Subtraction

In Milestone 1, the main goal is to prepare the dataset, separate template and test images, align them, apply image subtraction with Otsu’s thresholding, and finally visualize the detected defects.

The dataset contains four main folders:
1. Annotations/ – XML files describing the bounding boxes of defects for six classes (Missing_hole, Mouse_bite, Open_circuit, Short, Spur, Spurious_copper).
2. images/ – the defective PCB images divided per defect type.
3. PCB_USED/ – contains 10 defect-free template PCB images.
4. rotation/ – contains augmented rotated images and rotation angle text files.

---

### Step 1: Template and Test Separation
- Defect-free images in PCB_USED/ are used as templates, and defective images in images/ are used as test samples.
- Run the script:
  ```bash
  python milestone1/generate_template_test_pairs.py
  ```
- Outputs:
  - outputs_pairs/templates/ → template images
  - outputs_pairs/tests/ → test images
  - pairs_mapping.csv → mapping of templates to tests

---

### Step 2: Alignment
- Templates and test images are aligned to match pixel-to-pixel before subtraction.
- Even small shifts or rotations between template and test PCB images can produce false differences.
- Run the script:
  ```bash
  python milestone1/align_pairs.py
  ```
- Output: outputs_aligned/ containing aligned template–test pairs.

---

### Step 3: Subtraction with Otsu Thresholding
- Subtract aligned template and test images → obtain grayscale difference image.
- Apply Otsu’s thresholding → convert difference into a binary defect mask.
- Apply morphological filtering → remove noise and highlight only true defect regions.
- Run the script:
  ```bash
  python milestone1/subtraction_pipeline.py
  ```
- Outputs:
  - outputs_subtraction/diffs/ → difference images
  - outputs_subtraction/masks/ → binary masks

---

### Overall Workflow
Dataset → Separation → Alignment → Subtraction (Otsu)

---

### Deliverables
- Scripts:
  - generate_template_test_pairs.py
  - align_pairs.py
  - subtraction_pipeline.py
  - visualize_results.py
- Outputs:
  - outputs_pairs/
  - outputs_aligned/
  - outputs_subtraction/
  - outputs_visualizations/
- Report:
  - README.md

---

### Summary
Milestone 1 focuses entirely on preprocessing and defect isolation using classical image processing techniques (alignment, subtraction, thresholding) and does not yet involve training a deep learning model.

This ensures that:
- The dataset is clean,
- The defect regions are correctly isolated,
- The workflow for PCB defect detection is properly set up.

## Module 2: ROI Extraction

### Tasks
- Use OpenCV to detect contours of defects.
- Extract bounding boxes and crop individual defect regions (ROIs).
- Label defect ROIs for model training.

### Outputs
- Bounding box previews with defect labels → `outputs_module2/bbox/`
- Cropped ROI images → `outputs_module2/rois/`
- Metadata CSV with ROI filepaths and labels → `outputs_module2/roi_metadata.csv`

### Deliverables
- ROI extraction pipeline (module2_roiextract.py)
- Cropped and labeled defect samples
- Visualization of defect contours with labels

---

## Module 3: Model Training with EfficientNet-B4

### Tasks
- Implement EfficientNet-B4 using PyTorch.
- Preprocess and augment defect images (resize to 128×128).
- Train model using Adam optimizer and cross-entropy loss.

### Training Setup
- Data prepared from ROI crops → `outputs_module2/processed128/`
- Train/Val/Test splits created automatically.
- Training script:
  ```bash
  python train_efficientnet_b4.py --data_dir outputs_module2/processed128 --epochs 6 --batch_size 16 --lr 2e-4 --output_dir outputs_training
  ```

### Outputs
- Trained models → `efficientnet_b4_best.pth`, `efficientnet_b4_last.pth`
- Accuracy & loss plots → `training_acc.png`, `training_loss.png`
- Training history CSV → `training_history.csv`
- Confusion matrix → `confusion_matrix.png`
- Predictions on test set → `prediction_test.csv`
- Metrics summary → `metrics_summary.csv`

### Evaluation
- Achieved **~99% validation accuracy** and **~98–99% test accuracy**.
- Training is stable and repeatable.

---

## Module 4: Evaluation and Prediction Testing

### Tasks
- **Test model on unseen images**
  - Used `processed128/test/` images (held-out set).
  - Model predictions compared against true labels.

- **Run inference pipeline**
  - Implemented in `module4_inference.py`.
  - Outputs:
    - Annotated test images (class label + confidence written on image).
    - `predictions.csv` → file with predictions, ground truth, confidence.
    - `metrics_report.txt` → precision, recall, F1, and accuracy per class.
    - `confusion_matrix.png` → confusion matrix heatmap.

- **Compare with annotated ground truth**
  - Ground truth taken from `roi_metadata.csv`.
  - Code: `compare_with_gt.py`.
  - Matched predictions with ground-truth labels for accuracy.

---

### Deliverables
- ✅ Annotated output test images (`outputs_module4/annotated/`)
- ✅ Predictions CSV (`outputs_module4/predictions.csv`)
- ✅ Final evaluation report with metrics (`outputs_module4/metrics_report.txt`)
- ✅ Confusion matrix plot (`outputs_module4/confusion_matrix.png`)
- ✅ Comparison with ground truth (`outputs_module4/pred_vs_roi_metadata_merged.csv`)

---

### Evaluation Results
- **Accuracy on test set:** ~96.2%
- **Precision/Recall per class:** >0.92 for all classes
- **Low false positive/negative rate:** confirmed by high precision & recall
- **Prediction match rate with annotated truth:** ~96.4%

---

### Summary
Module 4 successfully validates the trained EfficientNet-B4 model on unseen data.
The pipeline produces:
- Annotated defect images with predictions,
- Quantitative evaluation metrics, and
- A confusion matrix showing classification performance.

---

## Module 5: Backend Integration & Inference Pipeline

### Objective
Build a modular backend pipeline for real-time PCB defect detection and classification using the trained EfficientNet-B4 model.

### Tasks
- Modularized image preprocessing, subtraction, and ROI extraction.
- Integrated model checkpoint for automated classification.
- Linked backend logic to frontend UI inputs using Streamlit.

### Implementation
- Scripts:
  - `inference_backend.py` – Handles model loading, masking, ROI detection, annotation, and saving outputs.
  - `batch_inference.py` – Automates inference on multiple image pairs using `pairs_mapping.csv`.
  - `app.py` – Streamlit-based web interface connecting frontend uploads to backend inference.

### Features
- Accepts both **template** and **test** images (via upload, folder, or path).
- Automatically generates **defect masks (black background + white blobs)**.
- Classifies ROIs and overlays defect labels + confidence scores.
- Logs all runs into `outputs_module6/log/inference_log.csv`.
- Saves outputs:
  - Annotated images → `outputs_module6/annotated/`
  - Cropped ROIs → `outputs_module6/rois/`
  - Predictions CSV → `outputs_module6/predictions/`
  - Logs → `outputs_module6/log/`

---

## Module 6: Web UI Integration

### Objective
Provide an easy-to-use **web interface** for users to upload PCB images and visualize results interactively.

### Implementation Details
**Frontend Tool:** Streamlit  
**File:** `app.py`

### Key Features
- Template selection from:
  - Templates folder
  - File upload
  - Custom path
- Test image upload
- Real-time defect mask visualization
- Annotated output image with predicted defect types
- Automatic logging and export

### Outputs
- **Annotated Image:** Highlighted ROIs with defect labels.
- **Defect Mask:** Binary black/white mask showing defect locations.
- **Predictions CSV:** Stores ROI coordinates, predicted labels, and confidence.
- **Inference Log:** Maintains timestamp, image names, and labels.

### Export Buttons in UI
✅ Download ROI predictions (CSV)  
✅ Download annotated image (PNG)

---

## Module 7: Testing, Evaluation & Export

### Objective
Validate app performance on multiple PCB pairs and ensure smooth export of results.

### Tasks
- Tested app using various defect types:
  - Missing_hole
  - Mouse_bite
  - Open_circuit
  - Short
  - Spur
  - Spurious_copper
- Reduced false positives through ROI filtering and adaptive thresholds.
- Optimized processing to minimize lag and memory use.

### Outputs
- Labeled images → Annotated test PCBs with bounding boxes and class names.
- Prediction logs → CSV + Log files saved under `outputs_module6/`.
- Export feature → Zipped folder (`run_YYYYMMDD_HHMMSS.zip`) containing all artifacts:
  - `annotated.png`
  - `mask.png`
  - `rois/` folder
  - `predictions.csv`

### Evaluation
✅ Fully functional, low-latency web app
✅ Accurate classification on all defect categories
✅ Successful export and auto-logging
✅ Consistent output structure

---

## Module 8: Documentation & Presentation

### Tasks
- **Prepare Technical Documentation & README** (this file).
- **Create User Guide:** Explains how to use backend and frontend.
- **Demo Video / Slides:** Walkthrough of the UI and backend architecture.
- **Organize Repository:** Folder structure for clean GitHub presentation.

### Deliverables
- `README.md` (this documentation)
- `AI-CircuitGuard-Final-Documentation.pdf`
- GitHub repository with:
  - `milestone1/` → classical processing
  - `outputs_module6/` → web app outputs
  - `outputs_training/` → model weights
  - `app.py`, `inference_backend.py`, `batch_inference.py`
- Recorded demo video or presentation slides

---

## Final Workflow Summary

Dataset → Preprocessing → ROI Extraction → Model Training  
↓  
Evaluation → Backend Integration → Streamlit Web UI  
↓  
Testing → Export → Documentation & Demo

**Result:**  
✅ End-to-end PCB defect detection system that:
- Detects and classifies defects accurately.
- Runs efficiently through a Streamlit-based interface.
- Produces clean visual outputs and logs.
- Is ready for presentation and deployment.

