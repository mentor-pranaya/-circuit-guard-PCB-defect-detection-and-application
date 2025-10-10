⚡ PCB Defect Detection System 📖 Overview

Printed Circuit Boards (PCBs) are critical components in modern electronics, and even minor defects can lead to system failures. This project presents an AI-powered defect detection system that automatically:

Compares a golden PCB (reference) with a defected PCB

Detects abnormal regions through image subtraction

Extracts Regions of Interest (ROIs) for detailed inspection

Classifies defects using a deep learning model (EfficientNet-B4)

Produces annotated images and prediction reports

A Streamlit web interface makes the system interactive, simple to use, and efficient for PCB quality inspection.

🛠 Features

📸 Automatic Defect Detection – Detects differences between reference and defected boards

🖼 ROI Extraction – Crops out defected regions for classification

🤖 AI Classification – Uses EfficientNet-B4 trained on PCB defect dataset

📊 Prediction Logs – Exports results as a CSV file

🌐 Streamlit UI – User-friendly frontend to upload, visualize, and download results

⚙️ Tech Stack

Language: Python

Deep Learning: PyTorch, Torchvision

Model: EfficientNet-B4 (fine-tuned)

Image Processing: OpenCV

Data Handling: Pandas, NumPy

Frontend: Streamlit

🚀 Getting Started

1️⃣ Clone Repository 

cd repository name

2️⃣ Create Virtual Environment python -m venv venv

Activate:

Windows:

venv\Scripts\activate

Linux/Mac:

source venv/bin/activate

3️⃣ Install Requirements pip install -r requirements.txt

4️⃣ Run the App streamlit run app.py

💻 Usage

Open the app at http://localhost:8501

Upload:

Reference PCB (Golden board)

Defected PCB

Click Run Detection

View results:

✅ Annotated PCB with bounding boxes + defect labels

✅ Downloadable output image

🔬 Core Workflow

Image Subtraction → Highlight differences between golden & defected PCBs

Defect Mask Generation → Binary mask of detected defects

ROI Extraction → Crop defected areas into patches

Classification → Run each ROI through EfficientNet-B4 model

Annotation & Reporting → Overlay bounding boxes & save logs

📊 Output Samples

✅ Annotated PCB Image

🌱 Future Enhancements

Real-time detection using camera feed

Support for multi-class PCB defects

Integration with manufacturing pipelines

Explainable AI for visualizing feature importance

✨ Conclusion

This project demonstrates the use of AI + Computer Vision for automated PCB defect detection. It reduces manual inspection time, increases accuracy, and provides a scalable solution for electronics manufacturing quality control

