import pandas as pd
import numpy as np
import mlflow
import dagshub

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv(r'D:\MIT\dec 26\mlops-project\data\external\loan_approval_dataset.csv')

df.columns = df.columns.str.strip()

x = df.drop(['loan_id', 'loan_status'], axis=1)

y = df['loan_status']

x = pd.get_dummies(x)

scalers = {'standard': StandardScaler(), 'minmax': MinMaxScaler()}

models = {'logistic_regression': LogisticRegression(), 'naive_bayes': GaussianNB(),
          'random_forest': RandomForestClassifier()}

# set tracking uri
mlflow.set_tracking_uri("https://dagshub.com/AdityaThakare72/mlops-project-1.mlflow")

dagshub.init(repo_owner='AdityaThakare72', repo_name='mlops-project-1', mlflow=True)
mlflow.set_experiment('exp2_strategy')

with mlflow.start_run() as parent_run:
    for algo, model in models.items():
        for scaler_name, scaler in scalers.items():
            # start the child run
            with mlflow.start_run(run_name=f'{algo}_{scaler_name}', nested=True) as child_run:

                x_scaled = scaler.fit_transform(x)

                x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

                model.fit(x_train, y_train)

                y_pred = model.predict(x_test)

                # log scaler and algo used
                mlflow.log_param('scaler', scaler_name)
                mlflow.log_param('algo', algo)  

                mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
                mlflow.log_metric('f1_score', f1_score(y_test, y_pred, pos_label= ' Approved'))
                mlflow.log_metric('precision', precision_score(y_test, y_pred, pos_label= ' Approved'))
                mlflow.log_metric('recall', recall_score(y_test, y_pred, pos_label= ' Approved'))

                # log the model
                mlflow.sklearn.log_model(model, 'model')

                # log the python file
                mlflow.log_artifact(__file__)



