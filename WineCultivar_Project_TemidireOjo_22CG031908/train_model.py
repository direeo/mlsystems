"""
Wine Origin Prediction Model Training Script
Uses SYNTHETIC data - no network required
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import json

print("Generating synthetic Wine-like dataset...")
np.random.seed(42)

feature_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
class_names = ['class_0', 'class_1', 'class_2']

# Generate 3 wine classes with different characteristics
n_per_class = 60

class0 = np.column_stack([
    np.random.normal(14.2, 0.5, n_per_class), np.random.normal(1.7, 0.4, n_per_class),
    np.random.normal(2.4, 0.2, n_per_class), np.random.normal(15.5, 2, n_per_class),
    np.random.normal(127, 12, n_per_class), np.random.normal(2.8, 0.3, n_per_class),
    np.random.normal(3.0, 0.4, n_per_class), np.random.normal(0.28, 0.07, n_per_class),
    np.random.normal(2.3, 0.4, n_per_class), np.random.normal(5.6, 1.2, n_per_class),
    np.random.normal(1.04, 0.12, n_per_class), np.random.normal(3.9, 0.4, n_per_class),
    np.random.normal(1065, 200, n_per_class)
])

class1 = np.column_stack([
    np.random.normal(12.4, 0.5, n_per_class), np.random.normal(1.9, 0.6, n_per_class),
    np.random.normal(2.2, 0.25, n_per_class), np.random.normal(20, 3, n_per_class),
    np.random.normal(95, 15, n_per_class), np.random.normal(2.2, 0.4, n_per_class),
    np.random.normal(2.0, 0.5, n_per_class), np.random.normal(0.36, 0.1, n_per_class),
    np.random.normal(1.6, 0.4, n_per_class), np.random.normal(3.0, 1.0, n_per_class),
    np.random.normal(1.06, 0.1, n_per_class), np.random.normal(2.8, 0.4, n_per_class),
    np.random.normal(520, 150, n_per_class)
])

class2 = np.column_stack([
    np.random.normal(13.2, 0.5, n_per_class), np.random.normal(3.3, 1.0, n_per_class),
    np.random.normal(2.4, 0.2, n_per_class), np.random.normal(21, 2, n_per_class),
    np.random.normal(99, 12, n_per_class), np.random.normal(1.7, 0.4, n_per_class),
    np.random.normal(0.8, 0.3, n_per_class), np.random.normal(0.45, 0.1, n_per_class),
    np.random.normal(1.5, 0.4, n_per_class), np.random.normal(7.4, 2, n_per_class),
    np.random.normal(0.68, 0.12, n_per_class), np.random.normal(1.7, 0.4, n_per_class),
    np.random.normal(630, 150, n_per_class)
])

X = np.vstack([class0, class1, class2])
y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

# Shuffle
idx = np.random.permutation(len(y))
X, y = X[idx], y[idx]

print(f"Dataset shape: {X.shape}")
print(f"Classes: {class_names}")

with open('metadata.json', 'w') as f:
    json.dump({'feature_names': feature_names, 'class_names': class_names}, f)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train
print("\nTraining MLP Classifier...")
model = MLPClassifier(hidden_layer_sizes=(32, 16, 8), activation='relu', solver='adam',
                      max_iter=300, random_state=42, early_stopping=True, verbose=False)
model.fit(X_train_scaled, y_train)

print(f"Training Accuracy: {model.score(X_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test_scaled, y_test):.4f}")

with open('model.h5', 'wb') as f:
    pickle.dump(model, f)

# Save samples
with open('sample_data.json', 'w') as f:
    json.dump({'class_0': class0[0].tolist(), 'class_1': class1[0].tolist(), 'class_2': class2[0].tolist()}, f)

print("\n" + "="*50)
print("Training complete! Files: model.h5, scaler.pkl, metadata.json, sample_data.json")
print("="*50)
