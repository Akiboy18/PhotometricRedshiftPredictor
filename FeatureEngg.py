import pandas as pd

# Load the input CSV file
input_file = r"E:\Machine Learning and AI\Sloan Digital Sky Survey (14-18)\SDSS DR18\SDSS DR18.csv"  # Adjust as needed
output_file = r"E:\Machine Learning and AI\Sloan Digital Sky Survey (14-18)\SDSS DR18\SDSS_DR18.csv"

# Read the data into a DataFrame
data = pd.read_csv(input_file)  # Corrected to read CSV

# Ensure the required columns exist in the data
required_columns = ['u', 'g', 'r', 'i', 'z']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The input file must contain the columns: {', '.join(required_columns)}")

# Calculate the differences and add new columns to the DataFrame
data['u-g'] = data['u'] - data['g']
data['g-r'] = data['g'] - data['r']
data['r-i'] = data['r'] - data['i']
data['i-z'] = data['i'] - data['z']

# Save the updated DataFrame to a new CSV file
data.to_csv(output_file, index=False)

print(f"New CSV file with column differences saved as '{output_file}'")
