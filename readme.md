AI-CircuitGuard – PCB Defect Detection

This project implements an end-to-end PCB defect detection pipeline in
three milestones:
1. Preprocessing & Subtraction
2. ROI Extraction & Dataset Preparation
3. EfficientNet Training & Evaluation

------------------------------------------------------------------------

📌 Milestone 1: Preprocessing & Subtraction

Goal

-   Prepare dataset
-   Separate template and test images
-   Align them
-   Apply image subtraction with Otsu’s thresholding
-   Visualize defects

Dataset Structure

    Annotations/          # XML defect labels (6 classes)
    images/               # Defective PCB images
    PCB_USED/             # Template defect-free images
    rotation/             # Augmented rotated images

Steps

1.  Template/Test Separation

    python milestone1/generate_template_test_pairs.py

Output → outputs_pairs/templates, outputs_pairs/tests, pairs_mapping.csv

2.  Alignment

    python milestone1/align_pairs.py

Output → outputs_aligned/

3.  Subtraction with Otsu Thresholding

    python milestone1/subtraction_pipeline.py

Output → outputs_subtraction/diffs, outputs_subtraction/masks

Deliverables

-   Subtracted defect masks
-   Contour visualizations

------------------------------------------------------------------------

📌 Milestone 2: ROI Extraction & Dataset Preparation

Goal

-   Detect contours of defects with OpenCV
-   Extract bounding boxes + crop defect ROIs
-   Assign labels from XML annotations
-   Build dataset for training

ROI Extraction

    python milestone1/module2_roiextract.py

Output →
- outputs_module2/bbox/ → defect previews with labels on bounding boxes
- outputs_module2/rois/ → cropped defect regions by class
- outputs_module2/roi_metadata.csv → ROI → label mapping

Dataset Preparation (128×128)

    python milestone1/prepare_dataset.py --input_dir outputs_module2/rois --output_dir outputs_module2/processed128 --img_size 128 --max_per_class 120

Output →
- processed128/train, processed128/val, processed128/test

Deliverables

-   Labeled ROI crops
-   Bounding-box previews with labels
-   Processed dataset (128×128 size)

------------------------------------------------------------------------

📌 Milestone 3: EfficientNet Training & Evaluation

Goal

-   Train an EfficientNet-B4 model with defect dataset
-   Use Adam optimizer + cross-entropy loss
-   Achieve ≥97% test accuracy

Training Command

    python milestone1/train_efficientnet_b4.py --data_dir outputs_module2/processed128 --epochs 6 --batch_size 16 --lr 2e-4 --output_dir outputs_training

Training Outputs

-   efficientnet_b4_best.pth → best model
-   efficientnet_b4_last.pth → last epoch model
-   training_loss.png → training loss curve
-   training_acc.png → training accuracy curve
-   confusion_matrix.png → evaluation confusion matrix
-   metrics_summary.csv → summary of loss/accuracy
-   prediction_test.csv → per-image predictions

Final Performance

✅ Stable & repeatable training
✅ Accuracy ≥97% on test set
✅ Clear evaluation plots

------------------------------------------------------------------------

📌 Repository Submission (GitHub)

Include:
- All scripts (.py)
- README.md (this file)
- requirements.txt
- Outputs (sample images + training plots + model weights)

Exclude:
- Full raw dataset (too large)
- All ROI crops (keep only a few samples per class)

------------------------------------------------------------------------

📊 Flowchart

    Dataset → Preprocessing → Subtraction → ROI Extraction → Processed Dataset → Training → Evaluation

------------------------------------------------------------------------

✅ Summary

-   Milestone 1 → Subtraction + defect isolation
-   Milestone 2 → ROI extraction + dataset preparation
-   Milestone 3 → Training EfficientNet-B4, accuracy & evaluation plots

This completes the pipeline for PCB defect detection.
