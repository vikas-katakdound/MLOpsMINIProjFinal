import mlflow
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
import dagshub
import joblib
import os
import pickle

# Load and preprocess data
df = pd.read_csv("data/external/loan_approval_dataset.csv")
df.columns = df.columns.str.replace(' ', '')

x = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']
x = pd.get_dummies(x)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Fit and save the scaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Define the relative path
relative_path = os.path.join("flask-app", "models", "minmax_scaler.pkl")

# Ensure the directory exists
os.makedirs(os.path.dirname(relative_path), exist_ok=True)

# Save the scaler
with open(relative_path, "wb") as file:
    pickle.dump(scaler, file)


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Define parameter grid
params_grid = {
    "n_estimators": [15, 20, 30],
    "max_depth": [10, 20],
}

# Initialize MLflow tracking
mlflow.set_tracking_uri("https://dagshub.com/AdityaThakare72/mlops-project-1.mlflow")
dagshub.init(repo_owner='AdityaThakare72', repo_name='mlops-project-1', mlflow=True)
mlflow.set_experiment("exp3_minmax_rf_hpt1")

# Start the parent run for hyperparameter tuning
with mlflow.start_run(run_name="exp3_minmax_rf") as parent_run:
    grid_search = GridSearchCV(RandomForestClassifier(), params_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    
    # Log child runs for each parameter combination
    for params, mean_f1 in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        with mlflow.start_run(run_name=f"exp3_minmax_rf_{params}", nested=True) as child_run:
            mlflow.log_params(params)
            mlflow.log_metric("mean_f1_score", mean_f1)
    
    # Log the best model and metrics
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label=' Approved')  # Change pos_label if necessary
    precision = precision_score(y_test, y_pred, pos_label=' Approved')
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("best_f1_score", grid_search.best_score_)
    mlflow.sklearn.log_model(best_model, "best_model")

    # Save the script file
    mlflow.log_artifact(__file__)
