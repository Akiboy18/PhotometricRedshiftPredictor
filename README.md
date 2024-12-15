# PhotometricRedshiftPredictor
An ML model designed to predict the value of redshift for a particular celestial object using its photometric filters u, g, r, i, and z
This model is a neural network-based regression model designed to predict the redshift of celestial objects using photometric features derived from the Sloan Digital Sky Survey (SDSS) dataset. 
It employs a deep learning approach to establish a relationship between the input features (magnitudes and colors) and the target variable (redshift) to provide accurate predictions.

# Key Features of the Model:

## Input Features:

The model takes 9 photometric features: magnitudes (u, g, r, i, z) and derived colors (u-g, g-r, r-i, i-z).
These features are selected for their astrophysical significance in estimating redshift.

## Data Preprocessing:

Outliers in the dataset are removed using an IQR-based filtering technique with an increased multiplier to retain a broader range of redshift values.
Missing values are handled by dropping rows with NaN values.
Features are standardized using StandardScaler to normalize the input data, ensuring consistent scaling for the neural network.

## Dataset Splitting:

The data is split into training (70%), validation (16%), and test (14%) sets to evaluate the model's performance.
Separate validation and test sets ensure robust evaluation during and after training.

## Model Architecture:

The model is implemented using Keras Sequential API.
It consists of:
An input layer with 128 neurons and ReLU activation.
A Dropout layer with a rate of 30% to prevent overfitting.
A hidden layer with 64 neurons and ReLU activation.
A single output neuron with a linear activation function, is suitable for regression.

## Loss Function and Optimization:

The model minimizes the Mean Squared Error (MSE), which is appropriate for regression tasks.
The optimizer used is Adam, known for its adaptive learning rate and efficiency in deep learning applications.

## Custom Callback:

A custom PrintMSEandR2Callback computes and prints the Mean Squared Error (MSE) and R² score on the validation set after each epoch, providing real-time feedback on the model's performance.

## Evaluation:

The test set predictions are evaluated using the MSE and R² score, providing quantitative measures of the model's predictive accuracy.
A plot of training and validation loss over epochs is generated to visualize the model's learning behavior.

## Outlier Removal and Redshift Range:

The redshift range after outlier removal is displayed, showing the data distribution retained for modeling.
Outlier removal is tuned to balance data quality with dataset size, using an increased IQR multiplier for flexibility.

## Model Performance Check:

The final training and validation losses are compared to detect overfitting or underfitting, ensuring the model generalizes well.

## Model Output:

The trained model is saved as a .h5 file for future use in predicting redshifts from new data.
This model is a step toward estimating photometric redshifts, which is crucial for large-scale astronomical surveys. It combines astrophysical insight with machine learning techniques to effectively handle noisy, high-dimensional data.
