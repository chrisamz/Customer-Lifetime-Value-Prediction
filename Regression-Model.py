# src/regression_models.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Define file paths
feature_engineered_data_path = 'data/processed/feature_engineered_data.csv'
linear_model_path = 'models/linear_regression_model.pkl'
ridge_model_path = 'models/ridge_regression_model.pkl'
lasso_model_path = 'models/lasso_regression_model.pkl'
results_path = 'results/regression_model_results.txt'

# Create directories if they don't exist
os.makedirs(os.path.dirname(linear_model_path), exist_ok=True)
os.makedirs(os.path.dirname(ridge_model_path), exist_ok=True)
os.makedirs(os.path.dirname(lasso_model_path), exist_ok=True)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

# Load feature-engineered data
print("Loading feature-engineered data...")
data = pd.read_csv(feature_engineered_data_path)

# Define features and target variable
X = data.drop(columns=['customer_id', 'transaction_id', 'transaction_date', 'amount'])
y = data['amount']

# Split data into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate and save model
def evaluate_and_save_model(model, model_name, model_path):
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    with open(results_path, 'a') as f:
        f.write(f"{model_name} Performance:\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write("\n")
    
    print(f"Saving {model_name}...")
    joblib.dump(model, model_path)

# Linear Regression
linear_model = LinearRegression()
evaluate_and_save_model(linear_model, 'Linear Regression', linear_model_path)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
evaluate_and_save_model(ridge_model, 'Ridge Regression', ridge_model_path)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
evaluate_and_save_model(lasso_model, 'Lasso Regression', lasso_model_path)

print("Regression model training and evaluation completed!")
