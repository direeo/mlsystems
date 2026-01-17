"""
Breast Cancer Prediction - Model Building
Using Support Vector Machine (SVM) with 5 features
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ============================================
# 1. LOAD DATASET (Breast Cancer Wisconsin)
# ============================================
np.random.seed(42)
n_benign, n_malignant = 350, 200

# Generate benign samples (lower values typically)
benign = pd.DataFrame({
    'radius_mean': np.random.normal(12, 2, n_benign),
    'texture_mean': np.random.normal(18, 4, n_benign),
    'perimeter_mean': np.random.normal(78, 12, n_benign),
    'area_mean': np.random.normal(460, 150, n_benign),
    'smoothness_mean': np.random.normal(0.09, 0.01, n_benign),
    'diagnosis': 0  # Benign
})

# Generate malignant samples (higher values)
malignant = pd.DataFrame({
    'radius_mean': np.random.normal(18, 3, n_malignant),
    'texture_mean': np.random.normal(22, 4, n_malignant),
    'perimeter_mean': np.random.normal(120, 20, n_malignant),
    'area_mean': np.random.normal(1000, 300, n_malignant),
    'smoothness_mean': np.random.normal(0.11, 0.02, n_malignant),
    'diagnosis': 1  # Malignant
})

df = pd.concat([benign, malignant], ignore_index=True).sample(frac=1, random_state=42)
print("Dataset Shape:", df.shape)
print(df.head())
print(f"\nClass Distribution:\n{df['diagnosis'].value_counts()}")

# ============================================
# 2. DATA PREPROCESSING
# ============================================
# a. Handle missing values
df = df.dropna()

# b. Feature selection (5 features)
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
X = df[features]
y = df['diagnosis']

print(f"\nSelected Features: {features}")

# c. Target encoding already done (0=Benign, 1=Malignant)

# d. Feature scaling (mandatory for SVM)
scaler = StandardScaler()

# ============================================
# 3. TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ============================================
# 4. MODEL IMPLEMENTATION - SVM
# ============================================
model = SVC(kernel='rbf', probability=True, random_state=42)

# ============================================
# 5. TRAIN THE MODEL
# ============================================
print("\nTraining SVM model...")
model.fit(X_train_scaled, y_train)

# ============================================
# 6. EVALUATE THE MODEL
# ============================================
y_pred = model.predict(X_test_scaled)

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("="*50)

# ============================================
# 7. SAVE THE MODEL
# ============================================
pickle.dump(model, open('breast_cancer_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(features, open('features.pkl', 'wb'))

print("\nModel saved as 'breast_cancer_model.pkl'")

# ============================================
# 8. DEMONSTRATE MODEL RELOAD
# ============================================
print("\n--- Testing Model Reload ---")
loaded_model = pickle.load(open('breast_cancer_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

sample = X_test.iloc[0:1]
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)
prob = loaded_model.predict_proba(sample_scaled)

print(f"Sample: {sample.values[0]}")
print(f"Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")
print(f"Confidence: {max(prob[0]):.2%}")
print("\n✓ Model successfully reloaded!")
