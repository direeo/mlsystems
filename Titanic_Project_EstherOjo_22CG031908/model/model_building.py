"""
Titanic Survival Prediction - Model Building
Using Logistic Regression with 5 features from Titanic dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ============================================
# 1. LOAD DATASET
# ============================================
np.random.seed(42)
n_samples = 800

data = {
    'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
    'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
    'Age': np.clip(np.random.normal(30, 14, n_samples), 1, 80),
    'SibSp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.68, 0.23, 0.06, 0.03]),
    'Fare': np.clip(np.random.exponential(35, n_samples), 5, 512),
}

# Generate survival based on known Titanic patterns
survival_prob = np.zeros(n_samples)
survival_prob += (np.array(data['Sex']) == 'female') * 0.4
survival_prob += (np.array(data['Pclass']) == 1) * 0.2
survival_prob += (np.array(data['Pclass']) == 2) * 0.1
survival_prob += (np.array(data['Age']) < 16) * 0.15
survival_prob += (np.array(data['Fare']) > 50) * 0.1
survival_prob = np.clip(survival_prob + np.random.normal(0, 0.15, n_samples), 0, 1)
data['Survived'] = (np.random.random(n_samples) < survival_prob).astype(int)

df = pd.DataFrame(data)
print("Dataset Shape:", df.shape)
print(df.head())
print(f"\nSurvival Rate: {df['Survived'].mean():.2%}")

# ============================================
# 2. DATA PREPROCESSING
# ============================================
# a. Handle missing values
df = df.dropna()

# b. Feature selection (5 features)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']

# c. Encode categorical variables
le_sex = LabelEncoder()
df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])

X = df[['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Fare']]
y = df['Survived']

print(f"\nSelected Features: {features}")

# d. Feature scaling
scaler = StandardScaler()

# ============================================
# 3. TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ============================================
# 4. MODEL IMPLEMENTATION - Logistic Regression
# ============================================
model = LogisticRegression(random_state=42, max_iter=1000)

# ============================================
# 5. TRAIN THE MODEL
# ============================================
print("\nTraining Logistic Regression model...")
model.fit(X_train_scaled, y_train)

# ============================================
# 6. EVALUATE THE MODEL
# ============================================
y_pred = model.predict(X_test_scaled)

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("="*50)

# ============================================
# 7. SAVE THE MODEL
# ============================================
pickle.dump(model, open('titanic_survival_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(le_sex, open('encoder.pkl', 'wb'))

print("\nModel saved as 'titanic_survival_model.pkl'")

# ============================================
# 8. DEMONSTRATE MODEL RELOAD
# ============================================
print("\n--- Testing Model Reload ---")
loaded_model = pickle.load(open('titanic_survival_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

sample = X_test.iloc[0:1]
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)
prob = loaded_model.predict_proba(sample_scaled)

print(f"Sample: {sample.values[0]}")
print(f"Prediction: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
print(f"Probability: {prob[0][1]:.2%}")
print("\n✓ Model successfully reloaded!")
