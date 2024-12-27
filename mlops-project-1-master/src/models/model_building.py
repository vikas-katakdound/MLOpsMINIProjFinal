import pandas as pd
import numpy as np
import yaml
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise e
    except Exception as e:
        logging.error(f"Error occurred while loading data: {e}")
        raise e

# Function to prepare data
def prepare_data(data):
    try:
        x = data.iloc[:, 0:-1].values
        y = data.iloc[:, -1].values
        logging.info("Data prepared successfully.")
        return x, y
    except Exception as e:
        logging.error(f"Error occurred while preparing data: {e}")
        raise e

# Function to load model parameters
def load_model_params(file_path):
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)['model_building']
        logging.info("Model parameters loaded successfully.")
        return params
    except FileNotFoundError as e:
        logging.error(f"Parameter file not found: {file_path}")
        raise e
    except Exception as e:
        logging.error(f"Error occurred while loading parameters: {e}")
        raise e

# Function to train the model
def train_model(x_train, y_train, params):
    try:
        rf = RandomForestClassifier(random_state=params['random_state'],
                                    n_estimators=params['n_estimators'],
                                    max_depth=10,
                                    )
        rf.fit(x_train, y_train)
        logging.info("Model trained successfully.")
        return rf
    except Exception as e:
        logging.error(f"Error occurred while training the model: {e}")
        raise e

# Function to save the model
def save_model(model, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving the model: {e}")
        raise e

def main():
    try:
        # Load data
        train_data = load_data('./data/processed/train_processed.csv')

        # Prepare data
        x_train, y_train = prepare_data(train_data)

        # Load model parameters
        params = load_model_params('params.yaml')

        # Train model
        rf_model = train_model(x_train, y_train, params)

        # Save model
        save_model(rf_model, 'models/model.pkl')

    except Exception as e:
        logging.error(f"Process failed: {e}")

if __name__ == "__main__":
    main()
