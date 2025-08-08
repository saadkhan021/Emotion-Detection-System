import os
import cv2
import numpy as np

# Paths
train_path = r"C:\Users\saadk\Desktop\Ml project\train"
test_path = r"C:\Users\saadk\Desktop\Ml project\test"

classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
img_size = 48

class_map = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}

def load_data(folder_path):
    features = []
    labels = []

    for class_name in classes:
        class_folder = os.path.join(folder_path, class_name)
        label = class_map[class_name]

        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                features.append(img)
                labels.append(label)
            except:
                continue

    features = np.array(features).reshape(-1, img_size, img_size, 1) / 255.0
    labels = np.array(labels)
    return features, labels

print("🔄 Loading training data...")
X_train, y_train = load_data(train_path)

print("🔄 Loading test data...")
X_test, y_test = load_data(test_path)

np.save('features_train.npy', X_train)
np.save('labels_train.npy', y_train)
np.save('features_test.npy', X_test)
np.save('labels_test.npy', y_test)

print("✅ Done! Saved all .npy files.")

