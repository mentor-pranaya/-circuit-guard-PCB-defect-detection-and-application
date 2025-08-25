AI-CircuitGuard – Milestone 1 (Preprocessing & Subtraction)

In Milestone 1, the main goal is to prepare the dataset, separate template and test images, align them, apply image subtraction with Otsu’s thresholding, and finally visualize the detected defects. 

The dataset contains four main folders:  
1. Annotations/ – XML files describing the bounding boxes of defects for six classes (Missing_hole, Mouse_bite, Open_circuit, Short, Spur, Spurious_copper).  
2. images/ – the defective PCB images divided per defect type.  
3. PCB_USED/ – contains 10 defect-free template PCB images.  
4. rotation/ – contains augmented rotated images and rotation angle text files.  

---

Step 1: Template and Test Separation  
- Defect-free images in PCB_USED/ are used as templates, and defective images in images/ are used as test samples.  
- Run the script:  
  python milestone1/generate_template_test_pairs.py
- Outputs:  
  - outputs_pairs/templates/ → template images  
  - outputs_pairs/tests/ → test images  
  - pairs_mapping.csv → mapping of templates to tests  

---

Step 2: Alignment  
- Templates and test images are aligned to match pixel-to-pixel before subtraction.  
- Even small shifts or rotations between template and test PCB images can produce false differences.  
- Run the script:  
  python milestone1/align_pairs.py
- Output: outputs_aligned/ containing aligned template–test pairs.  

---

Step 3: Subtraction with Otsu Thresholding  
- Subtract aligned template and test images → obtain grayscale difference image.  
- Apply Otsu’s thresholding → convert difference into a binary defect mask.  
- Apply morphological filtering → remove noise and highlight only true defect regions.  
- Run the script:  
  python milestone1/subtraction_pipeline.py
- Outputs:  
  - outputs_subtraction/diffs/ → difference images  
  - outputs_subtraction/masks/ → binary masks  

---



Overall Workflow  
Dataset → Separation → Alignment → Subtraction (Otsu) 

---

Deliverables  
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

Summary  
Milestone 1 focuses entirely on preprocessing and defect isolation using classical image processing techniques (alignment, subtraction, thresholding) and does not yet involve training a deep learning model.  

This ensures that:  
- The dataset is clean,  
- The defect regions are correctly isolated,  
- The workflow for PCB defect detection is properly set up.  
