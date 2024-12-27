import pandas as pd
import numpy as np
import pickle
import json
import logging
import mlflow
import dagshub
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_mlflow():
    """Initialize MLflow tracking"""
    try:
        mlflow.set_tracking_uri("https://dagshub.com/AdityaThakare72/mlops-project-1.mlflow")
        dagshub.init(repo_owner='AdityaThakare72', repo_name='mlops-project-1', mlflow=True)
        mlflow.set_experiment('model_evaluation')
        logging.info("MLflow tracking setup completed")
    except Exception as e:
        logging.error(f"Error setting up MLflow: {e}")
        raise e

def load_data(file_path):
    """Load the processed test data"""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise e

def load_model(model_path):
    """Load the trained model"""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e

def evaluate_model(model, X_test, y_test):
    """Evaluate model and calculate metrics"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, pos_label=' Approved'),
            'precision': precision_score(y_test, y_pred, pos_label=' Approved'),
            'recall': recall_score(y_test, y_pred, pos_label=' Approved')
        }
        
        logging.info("Model evaluation completed successfully")
        return metrics
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise e

def save_metrics(metrics, output_path):
    """Save metrics to JSON file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise e

def save_run_info(run_id, model_uri, output_path='reports/run_info.json'):
    """Save MLflow run information"""
    try:
        run_info = {
            'run_id': run_id,
            'model_uri': model_uri
        }
        
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(run_info, f, indent=4)
        logging.info(f"Run info saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving run info: {e}")
        raise e

def main():
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Load test data
        test_data = load_data('./data/processed/test_processed.csv')
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        
        # Load model
        model = load_model('models/model.pkl')
        
        # Start MLflow run
        with mlflow.start_run(run_name="model_evaluation") as run:
            # Log model parameters
            model_params = model.get_params()
            mlflow.log_params(model_params)
            
            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # Log the model
            model_path = "random_forest_model"
            mlflow.sklearn.log_model(model, model_path)
            
            # Save metrics locally
            save_metrics(metrics, 'reports/metrics.json')
            
            # Save run information
            model_uri = f"runs:/{run.info.run_id}/{model_path}"
            save_run_info(run.info.run_id, model_uri)
            
            logging.info("Model evaluation and logging completed successfully")
            
    except Exception as e:
        logging.error(f"Model evaluation process failed: {e}")
        raise e

if __name__ == "__main__":
    main()
