"""
Titanic Survival Prediction Model Training Script
Uses sklearn MLPClassifier
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import pickle

# Create simulated Titanic dataset
np.random.seed(42)
n_samples = 1000

data = {
    'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
    'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
    'Age': np.clip(np.random.normal(30, 14, n_samples), 0.5, 80),
    'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.68, 0.23, 0.05, 0.02, 0.01, 0.01]),
    'Parch': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.76, 0.13, 0.08, 0.02, 0.005, 0.005]),
    'Fare': np.clip(np.random.exponential(35, n_samples), 0, 512),
    'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
}

df = pd.DataFrame(data)

# Generate survival based on known patterns
survival_prob = np.zeros(n_samples)
survival_prob += (df['Sex'] == 'female') * 0.4
survival_prob += (df['Pclass'] == 1) * 0.2
survival_prob += (df['Pclass'] == 2) * 0.1
survival_prob += (df['Age'] < 16) * 0.15
survival_prob += (df['Fare'] > 50) * 0.1
survival_prob = np.clip(survival_prob + np.random.normal(0, 0.1, n_samples), 0, 1)
df['Survived'] = (np.random.random(n_samples) < survival_prob).astype(int)

print("Dataset created:")
print(df.head())
print(f"\nSurvival rate: {df['Survived'].mean():.2%}")

# Encode categorical variables
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])

# Prepare features
feature_cols = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
X = df[feature_cols].values
y = df['Survived'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and encoders
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump({'sex': le_sex, 'embarked': le_embarked}, f)
print("Scaler and encoders saved")

# Build and train MLP Classifier
print("\nTraining MLP Classifier...")
model = MLPClassifier(
    hidden_layer_sizes=(32, 16, 8),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    verbose=True
)

model.fit(X_train_scaled, y_train)

# Evaluate
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)
print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
with open('model.h5', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to model.h5")

# Test prediction
sample = X_test_scaled[0:1]
prediction = model.predict_proba(sample)
print(f"\nSample survival probability: {prediction[0][1]:.2%}")
print(f"Actual: {'Survived' if y_test[0] == 1 else 'Did not survive'}")

print("\n" + "="*50)
print("Training complete! Files created:")
print("  - model.h5, scaler.pkl, encoders.pkl")
print("="*50)
