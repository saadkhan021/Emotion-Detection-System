# Import Libraries

from joblib import dump
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load Features and Labels
features = np.load('features.npy')   # Extracted features
labels = np.load('labels.npy')       # Corresponding labels

print("Feature shape:", features.shape)
print("Label shape:", labels.shape)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Define Base Models
svm_model = SVC(probability=True, kernel='rbf', gamma='scale',class_weight='balanced')
dt_model = DecisionTreeClassifier(random_state=42,class_weight='balanced')
rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42,class_weight='balanced')
ann_model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42)

# Evaluate individual models
print("\n=== Individual Model Performance ===")
models = {
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'Neural Network': ann_model,
    'SVM': svm_model
}

# Use a smaller subset for quick evaluation
X_train_small = X_train[:50000]
y_train_small = y_train[:50000]

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_small, y_train_small)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Stacking Ensemble
stack_model = StackingClassifier(
    estimators=[
        ('svm', svm_model),
        ('dt', dt_model),
        ('rf', rf_model),
        ('ann', ann_model)
    ],
    final_estimator=LogisticRegression(),
    cv=3
)

# Train the Stacked Model
print("Training stacked model...")
X_train_small = X_train[:50000]
y_train_small = y_train[:50000]
stack_model.fit(X_train_small, y_train_small)

# Evaluate
y_pred = stack_model.predict(X_test)
print("\n=== Model Performance Metrics ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Create a summary of all metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

metrics_df = pd.DataFrame({
    'Class': ['Drowsy', 'Not Drowsy'],
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})

print("\nDetailed Metrics per Class:")
print(metrics_df.to_string(index=False))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
class_names = ['Drowsy', 'Not Drowsy']  # Update if you add more emotion classes

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Classification Report Plot
report = classification_report(y_test, y_pred, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
classes = [k for k in report.keys() if k.isdigit()]

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(classes))
width = 0.25

for i, metric in enumerate(metrics):
    values = [report[cls][metric] for cls in classes]
    ax.bar([pos + width * i for pos in x], values, width=width, label=metric)

ax.set_xticks([pos + width for pos in x])
ax.set_xticklabels([class_names[int(cls)] for cls in classes])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Classification Metrics per Class")
ax.legend()
plt.tight_layout()
plt.show()
dump(stack_model, 'stacked_model.pkl')
print(" Model saved successfully!")
proba = stack_model.predict_proba(features)[0]
print(f"Predicted: {labels}, Confidence: {proba}")
