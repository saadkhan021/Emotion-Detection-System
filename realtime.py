import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import mobilenet_v2
from joblib import load

# Load stacked model
stack_model = load('stacked_model.pkl')

# Load MobileNetV2 for feature extraction
mobilenet = mobilenet_v2(pretrained=True)
mobilenet.classifier = torch.nn.Identity()
mobilenet.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = mobilenet(input_tensor).numpy()

    prediction = stack_model.predict(features)[0]

    label = {0: "Drowsy", 1: "Not Drowsy"}.get(prediction, "Unknown")
    color = (0, 255, 0) if prediction == 1 else (0, 0, 255)

    # Show label on frame
    cv2.putText(frame, f"Emotion: {label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
