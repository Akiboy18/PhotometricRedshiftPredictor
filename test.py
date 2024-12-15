import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('Model01.h5')

# Load new data
test_data = pd.read_csv('/content/SDSS_DR18.csv')

# Ensure necessary columns exist
required_columns = ['u', 'g', 'r', 'i', 'z', 'u-g', 'g-r', 'r-i', 'i-z', 'redshift']
if not all(col in test_data.columns for col in required_columns):
    raise ValueError(f"The dataset must contain these columns: {', '.join(required_columns)}")

# Remove rows with negative redshift values
test_data = test_data[test_data['redshift'] >= 0]

# Remove outliers using the Interquartile Range (IQR) method
Q1 = test_data.quantile(0.25)
Q3 = test_data.quantile(0.75)
IQR = Q3 - Q1
test_data = test_data[~((test_data < (Q1 - 25 * IQR)) | (test_data > (Q3 + 25 * IQR))).any(axis=1)]

# Separate features and target
X_test = test_data[['u', 'g', 'r', 'i', 'z', 'u-g', 'g-r', 'r-i', 'i-z']].apply(pd.to_numeric, errors='coerce')
y_true = test_data['redshift'].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
X_test = X_test.dropna()
y_true = y_true[X_test.index]

# Normalize features using the same scaling method as training
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Make predictions
y_pred = model.predict(X_test_scaled).flatten()

# Evaluate model performance
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Plot 1: True vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.5, label='Predictions')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('True Redshift')
plt.ylabel('Predicted Redshift')
plt.title('True vs Predicted Redshift')
plt.legend()
plt.show()

# Plot 2: Residuals Plot
residuals = y_true - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Redshift')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Redshift')
plt.show()

# Plot 3: Histogram of Residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.axvline(0, color='r', linestyle='--', lw=2)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

# Plot 4: Prediction Error Distribution
error = np.abs(y_true - y_pred)
plt.figure(figsize=(10, 6))
plt.hist(error, bins=30, edgecolor='k', alpha=0.7, label='Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.legend()
plt.show()
