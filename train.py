import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

# Load data (assuming your dataset is a pandas dataframe)
dataset = pd.read_csv('/content/drive/MyDrive/Train_Data.csv')

# Separate features and target
X = dataset[['u', 'g', 'r', 'i', 'z','u-g','g-r','r-i','i-z']].apply(pd.to_numeric, errors='coerce')
y = dataset[['redshift']].apply(pd.to_numeric, errors='coerce')

# Remove rows with negative redshift values
dataset = dataset[dataset['redshift'] >= 0]

# Remove outliers using the Interquartile Range (IQR) method
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1

# Define acceptable range for values (1.5 times the IQR above and below Q1 and Q3)
filtered_dataset = dataset[~((dataset < (Q1 - 25 * IQR)) | (dataset > (Q3 + 25 * IQR))).any(axis=1)]

# Separate features and target again after outlier removal
X = filtered_dataset[['u', 'g', 'r', 'i', 'z', 'u-g', 'g-r', 'r-i', 'i-z']].apply(pd.to_numeric, errors='coerce')
y = filtered_dataset[['redshift']].apply(pd.to_numeric, errors='coerce')


# Print the range of redshifts after outlier removal
redshift_min = filtered_dataset['redshift'].min()
redshift_max = filtered_dataset['redshift'].max()
print(f"Redshift range after outlier removal: {redshift_min:.4f} to {redshift_max:.4f}")

# Handle missing values by dropping rows with NaNs
X = X.dropna()
y = y.dropna()

# Make sure the dimensions of X and y are consistent
X, y = X.align(y, join='inner', axis=0)

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train, validation, test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Further split train data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define a custom callback to print MSE and R² after every epoch
class PrintMSEandR2Callback(Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs={}):
        # Predict on validation data
        y_val_pred = self.model.predict(self.X_val)

        # Calculate MSE and R²
        val_mse = mean_squared_error(self.y_val, y_val_pred)
        val_r2 = r2_score(self.y_val, y_val_pred)

        # Print metrics
        print(f'Epoch {epoch+1}: Validation MSE: {val_mse:.4f}, Validation R²: {val_r2:.4f}')

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  # Regression output

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with custom callback
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_val, y_val), verbose=1,
                    callbacks=[PrintMSEandR2Callback(X_val, y_val)])

# Evaluate the model
y_pred = model.predict(X_test)
y_test_clean, y_pred_clean = y_test.dropna(), y_pred[np.isfinite(y_pred)]

if len(y_test_clean) == len(y_pred_clean):
    mse = mean_squared_error(y_test_clean, y_pred_clean)
    r2 = r2_score(y_test_clean, y_pred_clean)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
else:
    print(f'Warning: Inconsistent lengths between predictions and actual values.')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Save the model
model.save('Model01.h5')

# Print final training and validation loss values
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
print(f'Final Training Loss: {final_train_loss}')
print(f'Final Validation Loss: {final_val_loss}')

# Check for overfitting or underfitting
if final_train_loss < final_val_loss * 0.8:
    print("The model is likely overfitting.")
elif final_train_loss > final_val_loss * 1.2:
    print("The model is likely underfitting.")
else:
    print("The model is likely correctly fitted.")
