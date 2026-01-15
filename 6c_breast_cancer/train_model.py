"""
Breast Cancer Prediction Model Training Script
Uses SYNTHETIC data - no network required
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import json

print("Generating synthetic Breast Cancer-like dataset...")
np.random.seed(42)
n_samples = 600

# Feature names matching real breast cancer dataset
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Generate benign samples (class 1)
n_benign = 350
benign = np.column_stack([
    np.random.normal(12, 2, n_benign), np.random.normal(18, 4, n_benign),
    np.random.normal(78, 12, n_benign), np.random.normal(460, 150, n_benign),
    np.random.normal(0.09, 0.01, n_benign), np.random.normal(0.08, 0.03, n_benign),
    np.random.normal(0.04, 0.02, n_benign), np.random.normal(0.02, 0.01, n_benign),
    np.random.normal(0.18, 0.02, n_benign), np.random.normal(0.06, 0.007, n_benign),
    np.random.normal(0.3, 0.1, n_benign), np.random.normal(1.2, 0.5, n_benign),
    np.random.normal(2.0, 1.0, n_benign), np.random.normal(25, 15, n_benign),
    np.random.normal(0.007, 0.002, n_benign), np.random.normal(0.02, 0.01, n_benign),
    np.random.normal(0.02, 0.01, n_benign), np.random.normal(0.01, 0.005, n_benign),
    np.random.normal(0.02, 0.007, n_benign), np.random.normal(0.003, 0.001, n_benign),
    np.random.normal(14, 2, n_benign), np.random.normal(24, 5, n_benign),
    np.random.normal(90, 15, n_benign), np.random.normal(600, 200, n_benign),
    np.random.normal(0.13, 0.02, n_benign), np.random.normal(0.2, 0.1, n_benign),
    np.random.normal(0.15, 0.1, n_benign), np.random.normal(0.07, 0.03, n_benign),
    np.random.normal(0.27, 0.04, n_benign), np.random.normal(0.08, 0.01, n_benign)
])

# Generate malignant samples (class 0)
n_malignant = 250
malignant = np.column_stack([
    np.random.normal(17, 3, n_malignant), np.random.normal(21, 4, n_malignant),
    np.random.normal(115, 20, n_malignant), np.random.normal(950, 300, n_malignant),
    np.random.normal(0.10, 0.02, n_malignant), np.random.normal(0.15, 0.06, n_malignant),
    np.random.normal(0.16, 0.08, n_malignant), np.random.normal(0.09, 0.04, n_malignant),
    np.random.normal(0.19, 0.03, n_malignant), np.random.normal(0.063, 0.008, n_malignant),
    np.random.normal(0.6, 0.3, n_malignant), np.random.normal(1.3, 0.6, n_malignant),
    np.random.normal(4.5, 2.5, n_malignant), np.random.normal(80, 50, n_malignant),
    np.random.normal(0.007, 0.003, n_malignant), np.random.normal(0.035, 0.02, n_malignant),
    np.random.normal(0.04, 0.025, n_malignant), np.random.normal(0.015, 0.008, n_malignant),
    np.random.normal(0.025, 0.01, n_malignant), np.random.normal(0.005, 0.002, n_malignant),
    np.random.normal(21, 4, n_malignant), np.random.normal(30, 6, n_malignant),
    np.random.normal(145, 30, n_malignant), np.random.normal(1400, 500, n_malignant),
    np.random.normal(0.16, 0.03, n_malignant), np.random.normal(0.45, 0.2, n_malignant),
    np.random.normal(0.45, 0.2, n_malignant), np.random.normal(0.18, 0.07, n_malignant),
    np.random.normal(0.35, 0.08, n_malignant), np.random.normal(0.1, 0.02, n_malignant)
])

X = np.vstack([benign, malignant])
y = np.array([1]*n_benign + [0]*n_malignant)

# Clip to positive values
X = np.clip(X, 0.001, None)

# Shuffle
idx = np.random.permutation(len(y))
X, y = X[idx], y[idx]

print(f"Dataset shape: {X.shape}")
print(f"Benign: {sum(y==1)}, Malignant: {sum(y==0)}")

with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train
print("\nTraining MLP Classifier...")
model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam',
                      max_iter=300, random_state=42, early_stopping=True, verbose=False)
model.fit(X_train_scaled, y_train)

print(f"Training Accuracy: {model.score(X_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test_scaled, y_test):.4f}")

with open('model.h5', 'wb') as f:
    pickle.dump(model, f)

# Save samples
with open('sample_data.json', 'w') as f:
    json.dump({'benign': benign[0].tolist(), 'malignant': malignant[0].tolist(), 'feature_names': feature_names}, f)

print("\n" + "="*50)
print("Training complete! Files: model.h5, scaler.pkl, feature_names.json, sample_data.json")
print("="*50)
