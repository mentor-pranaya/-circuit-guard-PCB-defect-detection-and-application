# PCB Defect Detection System

## 📖 Introduction

This project is an AI-powered PCB (Printed Circuit Board) defect detection system.  
It automatically compares a golden (reference) PCB with a defected PCB, detects anomalies, extracts Regions of Interest (ROIs), classifies detected regions using a trained deep learning model, and produces:

- Annotated output images (with bounding boxes and defect labels)
- Prediction logs in CSV format

A user-friendly web interface built with Streamlit enables easy usage, visualization, and downloading of results.

---

## ⚙️ Tech Stack

- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Model:** EfficientNet-B4 (custom trained for PCB defects)  
- **Libraries:** OpenCV, Torchvision, Pandas, NumPy, Streamlit  
- **Frontend:** Streamlit (interactive UI)  
- **Backend:** Python functions for preprocessing, ROI extraction, model inference

---

## 🗂 Project Structure

```
PCB_DEFECT_DETECTION/
│── PCB_DATASET/              # Dataset folder
│── app.py                    # Main Streamlit application with full Pipeline(image_subtraction + roi_extraction + model prediction)                 
│── requirements.txt          # Required dependencies
│── README.md                 # Documentation       
│── model.py                  # Model Training Code                     
│── image_subtrcation.py      # Image Subtraction Code
│── roi_extraction.py         # Roi Extraction Code
│── testing_pipeline.py       # Testing model with Roi images
```

---

## 🔧 Setup Instructions

**Step 1: Clone Repository**
```bash
git clone https://github.com/your-username/pcb-defect-detection.git
cd pcb-defect-detection
```

**Step 2: Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Run Application**
```bash
streamlit run app.py
```

---

## 🚀 Usage Guide

### Frontend (Streamlit UI)

1. **Start the app:**  
   Access the Streamlit app in your browser (default: [http://localhost:8501](http://localhost:8501)).

2. **Upload Images:**  
   - Reference PCB (golden board)
   - Defected PCB

3. **Run Detection:**  
   Click **Run Detection**.

4. **Results:**  
   - The system subtracts the images, generates a defect mask, extracts ROIs, classifies ROIs using EfficientNet-B4, and annotates the PCB image.
   - Displays top-3 predictions per ROI.

5. **Download Results:**  
   - Annotated PCB Image (PNG)
   - Prediction Log (CSV)

### Backend (Core Functions & Pipeline)

- `subtract_images()`  
  Generates a defect mask by subtracting the reference and defected PCB images.

- `extract_rois_and_save()`  
  Extracts ROI patches from the defect mask, saves the images.

- `predict_roi()`  
  Classifies each ROI using EfficientNet-B4.

- `process_images()`  
  Orchestrates the entire workflow: image subtraction → ROI extraction → classification → annotation.

---





