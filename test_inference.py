import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_name('efficientnet-b4')
model._fc = nn.Linear(model._fc.in_features, 6)  # 6 defect classes
model.load_state_dict(torch.load("model/efficientnet_b4.pth", map_location=device))
model.to(device)
model.eval()

# Classes (same order as training dataset)
classes = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(img_path):
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        return classes[pred.item()]

if __name__ == "__main__":
    test_img = "data/test.jpg"
    result = predict(test_img)
    print(f"Prediction: {result}")
