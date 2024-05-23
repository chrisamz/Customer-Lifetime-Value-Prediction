# src/exploratory_data_analysis.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
processed_data_path = 'data/processed/processed_data.csv'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(figures_path, exist_ok=True)

# Load processed data
print("Loading processed data...")
data = pd.read_csv(processed_data_path)

# Display the first few rows of the dataset
print("Processed Data:")
print(data.head())

# Summary statistics
print("Summary Statistics:")
print(data.describe())

# Distribution of numerical features
numerical_features = ['recency', 'frequency', 'monetary_value', 'amount']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'{feature.capitalize()} Distribution')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(figures_path, f'{feature}_distribution.png'))
    plt.show()

# Correlation matrix
print("Correlation Matrix:")
correlation_matrix = data.corr()
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(figures_path, 'correlation_matrix.png'))
plt.show()

# Box plots for RFM features
rfm_features = ['recency', 'frequency', 'monetary_value']
for feature in rfm_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(y=feature, data=data)
    plt.title(f'{feature.capitalize()} Box Plot')
    plt.ylabel(feature.capitalize())
    plt.savefig(os.path.join(figures_path, f'{feature}_boxplot.png'))
    plt.show()

# Time series plot of sales
plt.figure(figsize=(15, 6))
data.set_index('transaction_date', inplace=True)
data['amount'].resample('M').sum().plot()
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.savefig(os.path.join(figures_path, 'monthly_sales_over_time.png'))
plt.show()

# Customer segmentation based on RFM
data['RFM_Score'] = data['recency'] + data['frequency'] + data['monetary_value']
rfm_segments = pd.qcut(data['RFM_Score'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
data['RFM_Segment'] = rfm_segments

plt.figure(figsize=(10, 6))
sns.countplot(x='RFM_Segment', data=data, order=['Low', 'Medium', 'High', 'Very High'])
plt.title('Customer Segments Based on RFM Score')
plt.xlabel('RFM Segment')
plt.ylabel('Count')
plt.savefig(os.path.join(figures_path, 'rfm_segments.png'))
plt.show()

print("Exploratory Data Analysis completed!")
