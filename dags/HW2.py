from airflow.models import DAG, Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago, timedelta
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import json
import time
import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
import os

DEFAULT_ARGS = {
    "owner": "Lapshin Nikita",
    "retries": 3,
    "retry_delay": timedelta(minutes=1)
}

BUCKET = 'mlops-2'
model_names = ["random_forest", "linear_regression", "decision_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ])
)

def init():
    timestamp = time.time()
    print(f"Init DAG for all models")
    return {"timestamp": timestamp}

def get_data(**kwargs):
    ti = kwargs['ti']
    
    start_time = time.time()
    
    data = fetch_california_housing(as_frame=True)
    
    end_time = time.time()
    
    dataset_info = {
        "start_time": start_time,
        "end_time": end_time,
        "size": len(data.data),
        "features": list(data.data.columns)
    }

    s3_hook = S3Hook("s3_connection")
    path = f"datasets/california_housing.csv"
    s3_hook.load_string(data.data.to_csv(), path, bucket_name=BUCKET, replace=True)
    
    return dataset_info

def prepare_data(**kwargs):
    s3_hook = S3Hook("s3_connection")
    path = f"datasets/california_housing.csv"
    data = pd.read_csv(s3_hook.download_file(path, bucket_name=BUCKET))
    
    start_time = time.time()
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:, :-1])  
    y = data.iloc[:, -1].values  # Ensure y is a numpy array
    
    end_time = time.time()

    prepared_data_info = {
        "start_time": start_time,
        "end_time": end_time,
        "features": list(data.columns[:-1])
    }
    
    # Combine scaled features and target into a single DataFrame
    prepared_data = pd.DataFrame(X, columns=data.columns[:-1])
    prepared_data['target'] = y  # Add the target column
    
    prepared_path = f"datasets/prepared_data.csv"
    # Save without the index to keep the CSV clean
    s3_hook.load_string(prepared_data.to_csv(index=False), prepared_path, bucket_name=BUCKET, replace=True)
    
    return prepared_data_info


def train_and_log_model(**kwargs):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
    os.environ["AWS_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
    os.environ["AWS_ACCESS_KEY_ID"] = "YCAJEL1fYA0Ik72-Ypm-sJ7dm"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "YCPAbWM-NBrKe5gPX7xRZLKPuNOlyK_VL91TLd4G"
    os.environ["AWS_DEFAULT_REGION"] = "ru-central1"

    ti = kwargs['ti']
    s3_hook = S3Hook("s3_connection")
    prepared_path = f"datasets/prepared_data.csv"
    data = pd.read_csv(s3_hook.download_file(prepared_path, bucket_name=BUCKET))
    
    X = data.drop(columns=['target'])
    y = data['target']
    
    start_time = time.time()
    
    mlflow.set_tracking_uri("http://mlflow-service:5000")
    
    experiment_name = "Lapshin Nikita"
    mlflow.set_experiment(experiment_name)
    print('result1')
    
    with mlflow.start_run(run_name="@GodSiemens", nested=False) as parent_run:
        model_metrics = {}
        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"{model_name}_run", nested=True):
                model.fit(X, y)
                print('result2', model_name)

                # Provide an input example
                input_example = X.head(1)

                mlflow.sklearn.log_model(model, model_name, input_example=input_example)

                
                # Get the model URI
                model_uri = mlflow.get_artifact_uri(model_name)
                
                # Evaluate the model
                eval_results = mlflow.evaluate(
                    model=model_uri,
                    data=data,  # Pass the DataFrame including features and target
                    model_type="regressor",
                    targets="target",  # Specify the target column name
                    evaluators="default",
                    evaluator_config={"log_artifacts": True}
                )

                # Log evaluation metrics
                mlflow.log_metrics(eval_results.metrics)
            
                end_time = time.time()

                model_metrics[model_name] = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "model_score": model.score(X, y)
                }

                # Save the model to S3
                model_path = f"results/{model_name}_model.pkl"
                s3_hook.load_bytes(pickle.dumps(model), key=model_path, bucket_name=BUCKET, replace=True)
                print('result3', model_name)

        return model_metrics


def save_results(**kwargs):
    ti = kwargs['ti']
    model_metrics = ti.xcom_pull(task_ids='train_and_log_model')
    
    s3_hook = S3Hook("s3_connection")
    results_path = f"results/metrics.json"
    s3_hook.load_string(json.dumps(model_metrics), results_path, bucket_name=BUCKET, replace=True)
    
    print("Results saved successfully!")

# Создаем один DAG
with DAG(
    dag_id="Nikita_Lapshin_experiment",
    schedule_interval="0 1 * * *",
    start_date=days_ago(1),
    default_args=DEFAULT_ARGS,
    tags=["mlops"]
) as dag:

    task_init = PythonOperator(
        task_id="init",
        python_callable=init
    )

    task_get_data = PythonOperator(
        task_id="get_data",
        python_callable=get_data,
    )

    task_prepare_data = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )

    # Тренировка и логирование всех моделей
    task_train_models = PythonOperator(
        task_id="train_and_log_model",
        python_callable=train_and_log_model,
    )

    task_save_results = PythonOperator(
        task_id="save_results",
        python_callable=save_results,
    )

    # Устанавливаем зависимости
    task_init >> task_get_data >> task_prepare_data >> task_train_models >> task_save_results
