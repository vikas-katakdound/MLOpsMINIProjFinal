import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/AdityaThakare72/mlops-project-1.mlflow")

dagshub.init(repo_owner='AdityaThakare72', repo_name='mlops-project-1', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
