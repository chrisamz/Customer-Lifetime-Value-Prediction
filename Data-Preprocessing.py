# src/data_preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define file paths
raw_data_path = 'data/raw/transactions.csv'
processed_data_path = 'data/processed/processed_data.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

# Load raw data
print("Loading raw data...")
data = pd.read_csv(raw_data_path)

# Display the first few rows of the dataset
print("Raw Data:")
print(data.head())

# Data Cleaning
print("Cleaning data...")

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Convert date column to datetime
data['transaction_date'] = pd.to_datetime(data['transaction_date'], format='%Y-%m-%d')

# Feature Engineering
print("Performing feature engineering...")

# Extract year, month, day, and day of week from transaction_date
data['year'] = data['transaction_date'].dt.year
data['month'] = data['transaction_date'].dt.month
data['day'] = data['transaction_date'].dt.day
data['day_of_week'] = data['transaction_date'].dt.dayofweek

# Calculate recency, frequency, and monetary value for each customer
print("Calculating RFM values...")

# Define a reference date for recency calculation (e.g., today's date)
reference_date = data['transaction_date'].max() + pd.DateOffset(1)

# Aggregate data to calculate RFM metrics
rfm_data = data.groupby('customer_id').agg({
    'transaction_date': lambda x: (reference_date - x.max()).days,
    'transaction_id': 'count',
    'amount': 'sum'
}).reset_index()

# Rename columns for RFM metrics
rfm_data.rename(columns={
    'transaction_date': 'recency',
    'transaction_id': 'frequency',
    'amount': 'monetary_value'
}, inplace=True)

# Encode categorical features
print("Encoding categorical features...")
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'customer_id':
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Normalize numerical features
print("Normalizing numerical features...")
scaler = StandardScaler()
numerical_features = ['recency', 'frequency', 'monetary_value']
rfm_data[numerical_features] = scaler.fit_transform(rfm_data[numerical_features])

# Merge RFM data back with the original dataset
data = data.merge(rfm_data, on='customer_id', how='left')

# Save processed data
print("Saving processed data...")
data.to_csv(processed_data_path, index=False)

print("Data preprocessing completed!")
