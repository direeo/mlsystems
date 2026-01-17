"""
House Price Prediction - Model Building
Using Linear Regression with 6 features from House Prices dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ============================================
# 1. LOAD DATASET
# ============================================
# Using synthetic data based on House Prices features
# In production, load from: pd.read_csv('train.csv')
np.random.seed(42)
n_samples = 1000

data = {
    'OverallQual': np.random.randint(1, 10, n_samples),      # Overall quality 1-10
    'GrLivArea': np.random.randint(800, 3000, n_samples),    # Above ground living area
    'TotalBsmtSF': np.random.randint(0, 2000, n_samples),    # Total basement sq ft
    'GarageCars': np.random.randint(0, 4, n_samples),        # Garage car capacity
    'BedroomAbvGr': np.random.randint(1, 6, n_samples),      # Bedrooms above ground
    'YearBuilt': np.random.randint(1950, 2020, n_samples),   # Year built
}

# Generate SalePrice based on features
data['SalePrice'] = (
    data['OverallQual'] * 20000 +
    data['GrLivArea'] * 50 +
    data['TotalBsmtSF'] * 30 +
    data['GarageCars'] * 15000 +
    data['YearBuilt'] * 100 +
    np.random.normal(0, 20000, n_samples)
)

df = pd.DataFrame(data)
print("Dataset Shape:", df.shape)
print(df.head())

# ============================================
# 2. DATA PREPROCESSING
# ============================================
# a. Handle missing values
df = df.dropna()

# b. Feature selection (6 features as required)
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'BedroomAbvGr', 'YearBuilt']
X = df[features]
y = df['SalePrice']

print(f"\nSelected Features: {features}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# c. No categorical encoding needed (all numeric)

# d. Feature scaling
scaler = StandardScaler()

# ============================================
# 3. TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# ============================================
# 4. MODEL IMPLEMENTATION - Linear Regression
# ============================================
model = LinearRegression()

# ============================================
# 5. TRAIN THE MODEL
# ============================================
print("\nTraining Linear Regression model...")
model.fit(X_train_scaled, y_train)

# ============================================
# 6. EVALUATE THE MODEL
# ============================================
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
print("="*50)

# ============================================
# 7. SAVE THE MODEL
# ============================================
pickle.dump(model, open('house_price_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(features, open('features.pkl', 'wb'))

print("\nModel saved as 'house_price_model.pkl'")
print("Scaler saved as 'scaler.pkl'")

# ============================================
# 8. DEMONSTRATE MODEL RELOAD
# ============================================
print("\n--- Testing Model Reload ---")
loaded_model = pickle.load(open('house_price_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

sample = X_test.iloc[0:1]
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)

print(f"Sample input: {sample.values[0]}")
print(f"Predicted price: ${prediction[0]:,.2f}")
print(f"Actual price: ${y_test.iloc[0]:,.2f}")
print("\n✓ Model successfully reloaded and used for prediction!")
