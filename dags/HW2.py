import os
import json
import time
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn

from airflow.models import DAG, Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago, timedelta
from airflow.operators.python_operator import PythonOperator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing

DEFAULT_ARGS = {
    "owner": "Nikita Lapshin",
    "retries": 3,
    "retry_delay": timedelta(minutes=1)
}

BUCKET = Variable.get("S3_BUCKET")
model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ])
)

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)

def get_data(**kwargs):
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
    y = data.iloc[:, -1].values  
    
    end_time = time.time()

    prepared_data_info = {
        "start_time": start_time,
        "end_time": end_time,
        "features": list(data.columns[:-1])
    }
    
    prepared_data = pd.DataFrame(X, columns=data.columns[:-1])
    prepared_data['target'] = y
    prepared_path = f"datasets/prepared_data.csv"
    s3_hook.load_string(prepared_data.to_csv(index=False), prepared_path, bucket_name=BUCKET, replace=True)
    
    return prepared_data_info

def train_model(m_name: str, **kwargs):
    configure_mlflow()
    
    s3_hook = S3Hook("s3_connection")
    prepared_path = f"datasets/prepared_data.csv"
    data = pd.read_csv(s3_hook.download_file(prepared_path, bucket_name=BUCKET))
    
    X = data.drop(columns=['target'])
    y = data['target']
    
    experiment_name = f"Nikita_Lapshin"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{m_name}_run") as run:
        model = models[m_name]
        model.fit(X, y)

        mlflow.sklearn.log_model(model, m_name, input_example=X.head(1))

        model_uri = mlflow.get_artifact_uri(m_name)
        
        eval_results = mlflow.evaluate(
            model=model_uri,
            data=data,
            model_type="regressor",
            targets="target",
            evaluators="default",
            evaluator_config={"log_artifacts": True}
        )

        mlflow.log_metrics(eval_results.metrics)
        
        model_path = f"results/{m_name}_model.pkl"
        s3_hook.load_bytes(pickle.dumps(model), key=model_path, bucket_name=BUCKET, replace=True)
        
    return {"model_score": model.score(X, y)}


def save_results(**kwargs):
    ti = kwargs['ti']
    model_metrics = ti.xcom_pull(task_ids='train_model')
    
    s3_hook = S3Hook("s3_connection")
    results_path = f"results/metrics.json"
    s3_hook.load_string(json.dumps(model_metrics), results_path, bucket_name=BUCKET, replace=True)
    
    print("Results saved successfully!")

def create_dag(dag_id: str, m_name: str):
    with DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(1),
        default_args=DEFAULT_ARGS,
        tags=["mlops"]
    ) as dag:

        task_get_data = PythonOperator(
            task_id="get_data",
            python_callable=get_data,
        )

        task_prepare_data = PythonOperator(
            task_id="prepare_data",
            python_callable=prepare_data,
        )

        task_train_model = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
            op_kwargs={"m_name": m_name},  
        )

        task_save_results = PythonOperator(
            task_id="save_results",
            python_callable=save_results,
        )

        task_get_data >> task_prepare_data >> task_train_model >> task_save_results

        return dag

for model_name in models.keys():
    dag_id = f"lapshin_nikita_{model_name}"
    globals()[dag_id] = create_dag(dag_id, model_name)
