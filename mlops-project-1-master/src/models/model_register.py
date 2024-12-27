 
import json
import logging
import dagshub
import mlflow
from mlflow.tracking import MlflowClient

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_mlflow():
    """Initialize MLflow tracking"""
    try:
        mlflow.set_tracking_uri("https://dagshub.com/AdityaThakare72/mlops-project-1.mlflow")
        dagshub.init(repo_owner='AdityaThakare72', repo_name='mlops-project-1', mlflow=True)
        logging.info("MLflow tracking setup completed")
        return MlflowClient()
    except Exception as e:
        logging.error(f"Error setting up MLflow: {e}")
        raise e

def load_run_info(run_info_path='reports/run_info.json'):
    """Load run information from JSON file"""
    try:
        with open(run_info_path, 'r') as f:
            run_info = json.load(f)
        logging.info("Run info loaded successfully")
        return run_info
    except Exception as e:
        logging.error(f"Error loading run info: {e}")
        raise e

def register_model(client, model_uri, model_name="loan_approval_model"):
    """Register the model in MLflow Model Registry"""
    try:
        # Register the model
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        logging.info(f"Model registered successfully with name: {result.name}, version: {result.version}")
        
        # Transition the model to 'Production' stage
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )
        
        logging.info(f"Model {result.name} version {result.version} transitioned to Production stage")
        return result
    except Exception as e:
        logging.error(f"Error registering model: {e}")
        raise e

def main():
    try:
        # Setup MLflow
        client = setup_mlflow()
        
        # Load run information
        run_info = load_run_info()
        
        # Register model
        register_model(client, run_info['model_uri'])
        
        logging.info("Model registration completed successfully")
        
    except Exception as e:
        logging.error(f"Model registration process failed: {e}")
        raise e

if __name__ == "__main__":
    main()