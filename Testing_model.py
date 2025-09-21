# ========================================
# Testing / Inference Code
# ========================================

import torch
from torchvision import transforms,models
from PIL import Image
import os



# Label map (must match training!)
label_map = {
    0: "missinghole",
    1: "mousebites",
    2: "short",
    3: "spur",
    4: "spuriouscopper",
    5: "opencircuit"
}

# Transform for test images (same as validation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model again (make sure it matches training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b4(weights=None)  # weights=None since we load trained weights
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 6)
model.load_state_dict(torch.load("/content/drive/MyDrive/pcb_model.pth", map_location=device))
model = model.to(device)
model.eval()

print(" Model loaded for testing")

# ----------------------------------------
# 1. Function to test a single image
# ----------------------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = test_transform(image).unsqueeze(0).to(device)  # add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    label = label_map[predicted.item()]
    return label

# ----------------------------------------
# 2. Test on a single sample image
# ----------------------------------------
sample_img = "/content/drive/MyDrive/PCB_images/Missing_hole/roi_01_missing_hole_01_0.jpg"  # change this path
if os.path.exists(sample_img):
    prediction = predict_image(sample_img)
    print(f"Prediction for {sample_img}: {prediction}")
else:
    print(" Sample image not found. Please update path.")

# ----------------------------------------
# 3. Optional: Evaluate on whole test CSV
# ----------------------------------------
TEST_CSV = "/content/drive/MyDrive/test1.csv"  # <- if you have test CSV
if os.path.exists(TEST_CSV):
    import pandas as pd
    from sklearn.metrics import classification_report, accuracy_score

    test_data = pd.read_csv(TEST_CSV)
    y_true, y_pred = [], []

    for i in range(len(test_data)):
        img_path = test_data.iloc[i, 0]
        true_label = str(test_data.iloc[i, 1]).strip().lower().replace("_", "").replace(" ", "")
        if os.path.exists(img_path):
            pred_label = predict_image(img_path)
            y_true.append(true_label)
            y_pred.append(pred_label)

    print("\n Test Set Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred) * 100, "%")
    print(classification_report(y_true, y_pred))
else:
    print(" No test CSV found. Skipping dataset evaluation.")
