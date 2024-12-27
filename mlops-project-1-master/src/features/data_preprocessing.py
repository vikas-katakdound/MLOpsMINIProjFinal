import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import logging

def setup_logging(log_file='data_processing.log'):
    """Set up logging configuration"""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

def load_data(train_path, test_path):
    """Load train and test data from CSV files."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Train and test data loaded successfully.")
        return train_data, test_data
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(train_data, test_data):
    """Preprocess the data: one-hot encoding, alignment, and scaling."""
    try:
        # Separate features and target
        X_train = train_data.drop(columns=['loan_status'])
        y_train = train_data['loan_status']
        X_test = test_data.drop(columns=['loan_status'])
        y_test = test_data['loan_status']
        logging.info("Separated features and target from train and test data.")

        # One-hot encoding
        X_train_encoded = pd.get_dummies(X_train, drop_first=False)
        X_test_encoded = pd.get_dummies(X_test, drop_first=False)
        logging.info("Applied one-hot encoding on categorical features.")

        # Align test data with train data columns
        X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
        X_train_encoded.fillna(0, inplace=True)
        X_test_encoded.fillna(0, inplace=True)
        logging.info("Aligned test data columns with train data.")

        # Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)
        logging.info("Applied MinMax scaling to numerical features.")

        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_encoded.columns)
        logging.info("Converted scaled arrays back to DataFrame.")

        # Recombine features with target
        train_preprocessed = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
        test_preprocessed = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
        logging.info("Recombined preprocessed features with target variable.")

        return train_preprocessed, test_preprocessed
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def save_data(train_preprocessed, test_preprocessed, output_dir='data/processed'):
    """Save preprocessed train and test data to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_preprocessed.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
        test_preprocessed.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
        logging.info("Preprocessed data saved successfully to %s", output_dir)
    except Exception as e:
        logging.error(f"Error saving preprocessed data: {e}")
        raise

def main():
    """Main function to execute data preprocessing pipeline."""
    setup_logging()
    logging.info("Data preprocessing pipeline started.")
    try:
        # File paths
        train_path = './data/raw/train.csv'
        test_path = './data/raw/test.csv'
        output_dir = 'data/processed'

        # Load data
        train_data, test_data = load_data(train_path, test_path)

        # Preprocess data
        train_preprocessed, test_preprocessed = preprocess_data(train_data, test_data)

        # Save preprocessed data
        save_data(train_preprocessed, test_preprocessed, output_dir)
        logging.info("Data preprocessing pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Data preprocessing pipeline failed: {e}")

if __name__ == "__main__":
    main()
