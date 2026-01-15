"""
House Price Prediction Model Training Script
Uses SYNTHETIC data - no network required
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pickle

print("Generating synthetic California Housing-like dataset...")
np.random.seed(42)
n_samples = 2000

# Generate synthetic housing data
X = np.column_stack([
    np.random.uniform(1, 15, n_samples),      # MedInc
    np.random.uniform(1, 52, n_samples),      # HouseAge
    np.random.uniform(2, 10, n_samples),      # AveRooms
    np.random.uniform(0.5, 3, n_samples),     # AveBedrms
    np.random.uniform(100, 5000, n_samples),  # Population
    np.random.uniform(1, 6, n_samples),       # AveOccup
    np.random.uniform(32, 42, n_samples),     # Latitude
    np.random.uniform(-124, -114, n_samples)  # Longitude
])

# Generate prices based on features (realistic formula)
y = (
    0.5 * X[:, 0] +                    # Income effect
    0.02 * X[:, 1] +                   # Age effect
    0.1 * X[:, 2] +                    # Rooms effect
    -0.05 * X[:, 3] +                  # Bedrooms effect
    0.0001 * X[:, 4] +                 # Population effect
    -0.1 * X[:, 5] +                   # Occupancy effect
    0.05 * (X[:, 6] - 35) +            # Latitude effect
    -0.02 * (X[:, 7] + 120) +          # Longitude effect
    np.random.normal(0, 0.5, n_samples) # Noise
)
y = np.clip(y, 0.5, 5.0)  # Clip to realistic range

feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(f"Features: {feature_names}")
print(f"Dataset shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved to scaler.pkl")

# Train model
print("\nTraining MLP Regressor model...")
model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    verbose=False
)

model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Training R² Score: {train_score:.4f}")
print(f"Test R² Score: {test_score:.4f}")

# Save model
with open('model.h5', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to model.h5")

# Test
sample = X_test_scaled[0:1]
prediction = model.predict(sample)
print(f"\nSample prediction: ${prediction[0] * 100000:.2f}")
print(f"Actual value: ${y_test[0] * 100000:.2f}")

print("\n" + "="*50)
print("Training complete! Files created: model.h5, scaler.pkl")
print("="*50)
