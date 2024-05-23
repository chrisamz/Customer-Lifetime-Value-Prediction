# src/cohort_analysis.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
feature_engineered_data_path = 'data/processed/feature_engineered_data.csv'
results_path = 'results/cohort_analysis_results.txt'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(results_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Load feature-engineered data
print("Loading feature-engineered data...")
data = pd.read_csv(feature_engineered_data_path)

# Display the first few rows of the dataset
print("Processed Data:")
print(data.head())

# Extract year and month from transaction_date for cohort analysis
data['transaction_date'] = pd.to_datetime(data['transaction_date'])
data['year_month'] = data['transaction_date'].dt.to_period('M')

# Create cohort: the month of a customer's first purchase
data['cohort_month'] = data.groupby('customer_id')['transaction_date'].transform('min').dt.to_period('M')

# Calculate the difference in months between the transaction and the cohort
data['cohort_index'] = (data['year_month'] - data['cohort_month']).apply(attrgetter('n'))

# Create a cohort table for number of customers
cohort_data = data.groupby(['cohort_month', 'cohort_index']).agg({
    'customer_id': pd.Series.nunique
}).reset_index()

cohort_pivot = cohort_data.pivot_table(index='cohort_month', columns='cohort_index', values='customer_id')

# Create retention matrix
cohort_size = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_size, axis=0)

# Plot the retention matrix
plt.figure(figsize=(12, 8))
sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='Blues')
plt.title('Cohort Analysis - Customer Retention')
plt.xlabel('Cohort Index')
plt.ylabel('Cohort Month')
plt.savefig(os.path.join(figures_path, 'cohort_retention_matrix.png'))
plt.show()

# Cohort analysis for average purchase value
cohort_value = data.groupby(['cohort_month', 'cohort_index']).agg({
    'amount': np.mean
}).reset_index()

cohort_value_pivot = cohort_value.pivot_table(index='cohort_month', columns='cohort_index', values='amount')

# Plot the cohort average purchase value
plt.figure(figsize=(12, 8))
sns.heatmap(cohort_value_pivot, annot=True, fmt='.2f', cmap='Blues')
plt.title('Cohort Analysis - Average Purchase Value')
plt.xlabel('Cohort Index')
plt.ylabel('Cohort Month')
plt.savefig(os.path.join(figures_path, 'cohort_average_purchase_value.png'))
plt.show()

# Save cohort analysis results
with open(results_path, 'w') as f:
    f.write("Cohort Analysis Results:\n")
    f.write("Customer Retention Matrix:\n")
    f.write(retention_matrix.to_string())
    f.write("\n\nAverage Purchase Value Matrix:\n")
    f.write(cohort_value_pivot.to_string())

print("Cohort analysis completed!")
