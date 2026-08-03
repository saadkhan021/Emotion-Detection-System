# Emotion Detection System

A computer vision system that classifies human emotional states from facial expressions.

## Overview
Applies classical ML/deep learning to a human-centered computer vision task: given a face image, predict the emotion being expressed (e.g. happy, sad, angry, neutral).

## What I Did
- Preprocessed facial image data: [grayscale conversion / resizing / normalization — confirm]
- Extracted features using [HOG / CNN — confirm approach]
- Trained a classifier ([SVM / CNN — confirm which]) across **[X]** emotion classes
- Achieved **[X%]** accuracy on the test set

## Key Findings
- [Which emotions were easiest/hardest to classify and why]
- [Any dataset imbalance issue]

## Tech Stack
Python · OpenCV · [Scikit-learn / TensorFlow / PyTorch — confirm] · [FER2013 or other dataset]

## How to Run
```bash
git clone https://github.com/saadkhan021/emotion-detection-system
cd emotion-detection-system
pip install -r requirements.txt
python predict.py --image path/to/face.jpg
```

## What I'd Improve
- Fine-tune a pretrained CNN instead of training from scratch for better accuracy
- Add real-time webcam inference


