# src/survival_analysis.py

import os
import pandas as pd
import numpy as np
from lifelines import Kaplan-MeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import joblib

# Define file paths
feature_engineered_data_path = 'data/processed/feature_engineered_data.csv'
kaplan_meier_model_path = 'models/kaplan_meier_model.pkl'
cox_model_path = 'models/cox_model.pkl'
results_path = 'results/survival_model_results.txt'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(os.path.dirname(kaplan_meier_model_path), exist_ok=True)
os.makedirs(os.path.dirname(cox_model_path), exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Load feature-engineered data
print("Loading feature-engineered data...")
data = pd.read_csv(feature_engineered_data_path)

# Display the first few rows of the dataset
print("Processed Data:")
print(data.head())

# Data Preprocessing for Survival Analysis
print("Preprocessing data for survival analysis...")

# Assuming 'recency_days' indicates the time since the last purchase and 'is_churned' indicates whether the customer has churned
# Here, we define churn as no purchase in the last 365 days
data['is_churned'] = (data['recency_days'] > 365).astype(int)
data['duration'] = data['recency_days']
data['event_observed'] = data['is_churned']

# Columns needed for survival analysis
survival_data = data[['duration', 'event_observed', 'recency_days', 'frequency', 'monetary_value', 'tenure_days']]

# Kaplan-Meier Estimator
print("Fitting Kaplan-Meier model...")
kmf = Kaplan-MeierFitter()
kmf.fit(durations=survival_data['duration'], event_observed=survival_data['event_observed'])

# Plot the survival function
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Function')
plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.savefig(os.path.join(figures_path, 'kaplan_meier_survival_function.png'))
plt.show()

# Save Kaplan-Meier model
print("Saving Kaplan-Meier model...")
joblib.dump(kmf, kaplan_meier_model_path)

# Cox Proportional Hazards Model
print("Fitting Cox Proportional Hazards model...")
cph = CoxPHFitter()
cph.fit(survival_data, duration_col='duration', event_col='event_observed')

# Print the summary of the Cox model
print("Cox Proportional Hazards Model Summary:")
print(cph.summary)

# Plot the survival functions for the first few individuals
plt.figure(figsize=(10, 6))
cph.plot_partial_effects_on_outcome(covariates='monetary_value', values=[survival_data['monetary_value'].quantile(0.25), survival_data['monetary_value'].quantile(0.5), survival_data['monetary_value'].quantile(0.75)], cmap='coolwarm')
plt.title('Survival Function by Monetary Value')
plt.savefig(os.path.join(figures_path, 'cox_survival_function.png'))
plt.show()

# Save Cox model
print("Saving Cox Proportional Hazards model...")
joblib.dump(cph, cox_model_path)

# Model Evaluation
print("Evaluating Cox model...")
c_index = concordance_index(survival_data['duration'], -cph.predict_partial_hazard(survival_data), survival_data['event_observed'])
print(f"Cox Model Concordance Index: {c_index:.4f}")

with open(results_path, 'w') as f:
    f.write("Cox Proportional Hazards Model Summary:\n")
    f.write(str(cph.summary))
    f.write(f"\nCox Model Concordance Index: {c_index:.4f}\n")

print("Survival analysis model training and evaluation completed!")
