import pandas as pd
import numpy as np
import os
import yaml
import logging
from sklearn.model_selection import train_test_split

def setup_logging(log_file='data_ingestion.log'):
    """Set up logging configuration"""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

def load_params(param_file='params.yaml'):
    """Load parameters from a YAML file."""
    try:
        with open(param_file, 'r') as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully.")
        return params
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.replace(' ', '')
        logging.info("Data loaded successfully from %s", file_path)
        return df
    except FileNotFoundError:
        logging.error(f"Data file {file_path} not found.")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error reading the CSV file: {e}")
        raise

def split_data(df, test_size):
    """Split the data into train and test sets."""
    try:
        x = df.drop(['loan_id', 'loan_status'], axis=1)
        y = df['loan_status']
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=0, stratify=y, test_size=test_size
        )
        logging.info("Data split into training and testing sets successfully.")
        return x_train, x_test, y_train, y_test
    except KeyError as e:
        logging.error(f"Missing required columns in the dataframe: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during data splitting: {e}")
        raise

def save_data(train_data, test_data, data_dir='data/raw'):
    """Save train and test data to CSV files."""
    try:
        os.makedirs(data_dir, exist_ok=True)
        train_data.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
        logging.info("Train and test data saved successfully to %s", data_dir)
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    """Main function to execute the data ingestion pipeline."""
    setup_logging()
    logging.info("Data ingestion pipeline started.")
    try:
        # Load parameters
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        
        # Load data
        data_file_path = r"data\external\loan_approval_dataset.csv"
        df = load_data(data_file_path)
        
        # Split data
        x_train, x_test, y_train, y_test = split_data(df, test_size)
        
        # Concatenate and save
        train_data = pd.concat([x_train, y_train], axis=1)
        test_data = pd.concat([x_test, y_test], axis=1)
        save_data(train_data, test_data, data_dir='data/raw')
        
        logging.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Data ingestion pipeline failed: {e}")

if __name__ == "__main__":
    main()
