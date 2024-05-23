# src/feature_engineering.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
feature_engineered_data_path = 'data/processed/feature_engineered_data.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(feature_engineered_data_path), exist_ok=True)

# Load processed data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# Display the first few rows of the dataset
print("Processed Data:")
print(data.head())

# Feature Engineering
print("Performing feature engineering...")

# Create features for customer tenure (days since first purchase)
data['first_purchase_date'] = data.groupby('customer_id')['transaction_date'].transform('min')
data['tenure_days'] = (pd.to_datetime(data['transaction_date']) - pd.to_datetime(data['first_purchase_date'])).dt.days

# Create features for purchase frequency
data['purchase_frequency'] = data.groupby('customer_id')['transaction_id'].transform('count')

# Create features for average order value
data['average_order_value'] = data.groupby('customer_id')['amount'].transform('mean')

# Create features for total spend
data['total_spend'] = data.groupby('customer_id')['amount'].transform('sum')

# Create features for recency (days since last purchase)
data['last_purchase_date'] = data.groupby('customer_id')['transaction_date'].transform('max')
reference_date = pd.to_datetime(data['transaction_date']).max() + pd.DateOffset(1)
data['recency_days'] = (reference_date - pd.to_datetime(data['last_purchase_date'])).dt.days

# Drop intermediate columns
data.drop(columns=['first_purchase_date', 'last_purchase_date'], inplace=True)

# Encode categorical features
print("Encoding categorical features...")
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column not in ['customer_id', 'transaction_id', 'transaction_date']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Normalize numerical features
print("Normalizing numerical features...")
scaler = StandardScaler()
numerical_features = ['recency_days', 'tenure_days', 'purchase_frequency', 'average_order_value', 'total_spend']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Save feature-engineered data
print("Saving feature-engineered data...")
data.to_csv(feature_engineered_data_path, index=False)

print("Feature engineering completed!")
