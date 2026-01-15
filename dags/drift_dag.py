from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import numpy as np
from sklearn.datasets import load_wine
import pandas as pd

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# проверка дрифта по среднему значению фичи 'alcohol'
def check_data_drift(**kwargs):
    # Загружаем данные
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    reference_mean = df['alcohol'].mean()
    
    # Добавляем шум, чтобы спровоцировать дрифт
    current_data = df['alcohol'] + np.random.normal(0.5, 0.1, size=len(df))
    #current_data = df['alcohol']
    current_mean = current_data.mean()

    # Считаем изменение в процентах
    drift_score = abs(current_mean - reference_mean) / reference_mean
    print(f"Drift Score (Mean Shift): {drift_score}")
    
    # Порог дрифта 2%
    THRESHOLD = 0.02 
    
    if drift_score > THRESHOLD:
        print("DRIFT DETECTED! Initiating retraining.")
        return 'trigger_retraining'
    else:
        print("No drift detected.")
        return 'no_drift'

with DAG(
    'drift_monitoring_automl',
    default_args=default_args,
    description='Monitor data drift and retrain model',
    schedule_interval='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    # Задача 1: Проверка дрифта
    check_drift_task = BranchPythonOperator(
        task_id='check_data_drift',
        python_callable=check_data_drift,
    )

    # Задача 2: Переобучение
    retrain_task = BashOperator(
        task_id='trigger_retraining',
        bash_command='python /opt/airflow/scripts/train_model.py '
    )

    # Задача 3: дрифта нет
    no_drift_task = BashOperator(
        task_id='no_drift',
        bash_command='echo "Data is stable"'
    )

    check_drift_task >> [retrain_task, no_drift_task]