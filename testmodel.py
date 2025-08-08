import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from joblib import load
import matplotlib.pyplot as plt

# === Device Config ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Load your trained stacked model ===
stack_model = load('stacked_model.pkl')
print("Model loaded!")

# === Load and preprocess image ===
img_path = 'D:/image/test2.jpg'
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
image = cv2.resize(image, (224, 224))

# Convert to PIL and apply the same transforms used in training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# === Load MobileNetV2 (same as training) ===
model = models.mobilenet_v2(pretrained=True)
model.classifier = nn.Identity()  # Remove classification head
model.to(device)
model.eval()

# === Feature extraction ===
with torch.no_grad():
    feature = model(input_tensor).cpu().numpy()  # Shape: (1, 1280)

# === Make prediction ===
prediction = stack_model.predict(feature)[0]

# === Interpret result ===
if prediction == 0:
    result_text = "NotDrowsy"
elif prediction == 1:
    result_text = "Drowsy"
else:
    result_text = f"Class {prediction}"

print(f"Prediction: {result_text}")

# === Visualization ===
plt.imshow(image)
plt.title(f"Prediction: {result_text}")
plt.axis('off')
plt.show()
