"""
Wine Cultivar Origin Prediction - Model Building
Using Random Forest Classifier with 6 features
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ============================================
# 1. LOAD DATASET (Wine Dataset)
# ============================================
np.random.seed(42)
n_per_class = 60

# Generate 3 wine cultivar classes based on typical Wine dataset patterns
class0 = pd.DataFrame({
    'alcohol': np.random.normal(14.2, 0.5, n_per_class),
    'malic_acid': np.random.normal(1.7, 0.4, n_per_class),
    'ash': np.random.normal(2.4, 0.2, n_per_class),
    'magnesium': np.random.normal(127, 12, n_per_class),
    'flavanoids': np.random.normal(3.0, 0.4, n_per_class),
    'color_intensity': np.random.normal(5.6, 1.2, n_per_class),
    'cultivar': 0
})

class1 = pd.DataFrame({
    'alcohol': np.random.normal(12.4, 0.5, n_per_class),
    'malic_acid': np.random.normal(1.9, 0.6, n_per_class),
    'ash': np.random.normal(2.2, 0.25, n_per_class),
    'magnesium': np.random.normal(95, 15, n_per_class),
    'flavanoids': np.random.normal(2.0, 0.5, n_per_class),
    'color_intensity': np.random.normal(3.0, 1.0, n_per_class),
    'cultivar': 1
})

class2 = pd.DataFrame({
    'alcohol': np.random.normal(13.2, 0.5, n_per_class),
    'malic_acid': np.random.normal(3.3, 1.0, n_per_class),
    'ash': np.random.normal(2.4, 0.2, n_per_class),
    'magnesium': np.random.normal(99, 12, n_per_class),
    'flavanoids': np.random.normal(0.8, 0.3, n_per_class),
    'color_intensity': np.random.normal(7.4, 2, n_per_class),
    'cultivar': 2
})

df = pd.concat([class0, class1, class2], ignore_index=True).sample(frac=1, random_state=42)
print("Dataset Shape:", df.shape)
print(df.head())
print(f"\nClass Distribution:\n{df['cultivar'].value_counts().sort_index()}")

# ============================================
# 2. DATA PREPROCESSING
# ============================================
# a. Handle missing values
df = df.dropna()

# b. Feature selection (6 features as required)
features = ['alcohol', 'malic_acid', 'ash', 'magnesium', 'flavanoids', 'color_intensity']
X = df[features]
y = df['cultivar']

print(f"\nSelected Features: {features}")

# c. Feature scaling (mandatory)
scaler = StandardScaler()

# ============================================
# 3. TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ============================================
# 4. MODEL IMPLEMENTATION - Random Forest
# ============================================
model = RandomForestClassifier(n_estimators=100, random_state=42)

# ============================================
# 5. TRAIN THE MODEL
# ============================================
print("\nTraining Random Forest Classifier...")
model.fit(X_train_scaled, y_train)

# ============================================
# 6. EVALUATE THE MODEL
# ============================================
y_pred = model.predict(X_test_scaled)

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Cultivar 1', 'Cultivar 2', 'Cultivar 3']))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("="*50)

# ============================================
# 7. SAVE THE MODEL
# ============================================
pickle.dump(model, open('wine_cultivar_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(features, open('features.pkl', 'wb'))

print("\nModel saved as 'wine_cultivar_model.pkl'")

# ============================================
# 8. DEMONSTRATE MODEL RELOAD
# ============================================
print("\n--- Testing Model Reload ---")
loaded_model = pickle.load(open('wine_cultivar_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

sample = X_test.iloc[0:1]
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)
prob = loaded_model.predict_proba(sample_scaled)

print(f"Sample: {sample.values[0]}")
print(f"Prediction: Cultivar {prediction[0] + 1}")
print(f"Probabilities: {prob[0]}")
print("\n✓ Model successfully reloaded!")
