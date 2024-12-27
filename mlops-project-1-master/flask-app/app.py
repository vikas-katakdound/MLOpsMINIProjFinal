from flask import Flask, request, jsonify, render_template, url_for
import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib
import os

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static')

# Setup MLflow
mlflow.set_tracking_uri("https://dagshub.com/AdityaThakare72/mlops-project-1.mlflow")

# Get the absolute path to the models directory within flask-app
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_model_and_scaler():
    """Load the production model and scaler"""
    try:
        # Load model from MLflow
        model = mlflow.pyfunc.load_model(f"models:/loan_approval_model/Production")
        
        # Load scaler from local file
        scaler_path = os.path.join(MODELS_DIR, 'minmax_scaler.pkl')
        print(f"Looking for scaler at: {scaler_path}")  # Debug print
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            
        scaler = joblib.load(scaler_path)
        
        logging.info("Model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model and scaler: {e}")
        raise e

def prepare_features(form_data, scaler):
    """Prepare features for prediction using the saved scaler"""
    try:
        # Create feature dictionary with exact column names (notice the spaces after underscores)
        features = {
            'no_of_dependents': int(form_data['no_of_dependents']),
            'income_annum': float(form_data['income_annum']),
            'loan_amount': float(form_data['loan_amount']),
            'loan_term': int(form_data['loan_term']),
            'cibil_score': int(form_data['cibil_score']),
            'residential_assets_value': float(form_data['residential_assets_value']),
            'commercial_assets_value': float(form_data['commercial_assets_value']),
            'luxury_assets_value': float(form_data['luxury_assets_value']),
            'bank_asset_value': float(form_data['bank_asset_value']),
            'education_ Graduate': 1 if form_data['education'] == 'Graduate' else 0,
            'education_ Not Graduate': 1 if form_data['education'] == 'Not Graduate' else 0,
            'self_employed_ No': 1 if form_data['self_employed'] == 'No' else 0,
            'self_employed_ Yes': 1 if form_data['self_employed'] == 'Yes' else 0
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Use the loaded scaler
        scaled_features = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_features, columns=df.columns)
        
        logging.info("Features prepared successfully")
        return scaled_df
    except Exception as e:
        logging.error(f"Error preparing features: {e}")
        raise e

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        form_data = {
            "no_of_dependents": request.form.get("no_of_dependents"),
            "education": request.form.get("education"),
            "self_employed": request.form.get("self_employed"),
            "income_annum": request.form.get("income_annum"),
            "loan_amount": request.form.get("loan_amount"),
            "loan_term": request.form.get("loan_term"),
            "cibil_score": request.form.get("cibil_score"),
            "residential_assets_value": request.form.get("residential_assets_value"),
            "commercial_assets_value": request.form.get("commercial_assets_value"),
            "luxury_assets_value": request.form.get("luxury_assets_value"),
            "bank_asset_value": request.form.get("bank_asset_value"),
        }
        
        # Load model and scaler
        model, scaler = load_model_and_scaler()
        
        # Prepare features using the loaded scaler
        features = prepare_features(form_data, scaler)
        
        # Make prediction
        prediction = model.predict(features)
        result = "Approved" if prediction[0] == " Approved" else "Rejected"
        
        # Render result page
        return render_template(
            'result.html',
            prediction=result,
            form_data=form_data
        )
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)



