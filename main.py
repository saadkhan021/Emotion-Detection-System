import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import platform

# === Path to your dataset ===
dataset_dir = r'D:\MLproject\train_data'  # <-- CHANGE THIS TO YOUR LOCAL FOLDER

# === Config ===
img_size = 224
batch_size = 64 # You can lower to 16 or 8 if running out of RAM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load dataset ===
dataset = datasets.ImageFolder(dataset_dir, transform=transform)
loader = DataLoader(dataset, 
                   batch_size=batch_size, 
                   shuffle=False,
                   num_workers=4,  # Use multiple workers for data loading
                   pin_memory=True)  # Enable pin memory for faster data transfer to GPU

# === Load MobileNetV2 ===
model = models.mobilenet_v2(pretrained=True)
model.classifier = nn.Identity()  # Remove classification head
model.to(device)
model.eval()

# === Extract features ===
features = []
labels = []

with torch.no_grad():
    for imgs, labs in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        feats = model(imgs).cpu().numpy()
        features.append(feats)
        labels.extend(labs.numpy())

features = np.vstack(features)
labels = np.array(labels)

# === Save results ===
np.save('features.npy', features)
np.save('labels.npy', labels)

print("Feature extraction completed.")
print("Feature shape:", features.shape)
print("Label shape:", labels.shape)
