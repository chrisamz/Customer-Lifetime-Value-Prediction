# src/model_deployment.py

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Define file paths
feature_engineered_data_path = 'data/processed/feature_engineered_data.csv'
linear_model_path = 'models/linear_regression_model.pkl'
ridge_model_path = 'models/ridge_regression_model.pkl'
lasso_model_path = 'models/lasso_regression_model.pkl'

# Load models
print("Loading models...")
linear_model = joblib.load(linear_model_path)
ridge_model = joblib.load(ridge_model_path)
lasso_model = joblib.load(lasso_model_path)

# Load feature-engineered data
print("Loading feature-engineered data...")
data = pd.read_csv(feature_engineered_data_path)

# Define the feature set
features = data.drop(columns=['customer_id', 'transaction_id', 'transaction_date', 'amount'])

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Customer Lifetime Value Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        json_data = request.get_json()
        
        # Convert JSON data to DataFrame
        input_data = pd.DataFrame(json_data)
        
        # Ensure input data has the same features as the training data
        input_data = input_data.reindex(columns=features.columns, fill_value=0)
        
        # Make predictions using the loaded models
        linear_prediction = linear_model.predict(input_data)
        ridge_prediction = ridge_model.predict(input_data)
        lasso_prediction = lasso_model.predict(input_data)
        
        # Prepare the response
        response = {
            'linear_regression_prediction': linear_prediction.tolist(),
            'ridge_regression_prediction': ridge_prediction.tolist(),
            'lasso_regression_prediction': lasso_prediction.tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
