import mlflow
import mlflow.sklearn
import pandas as pd
from pycaret.classification import *
from sklearn.datasets import load_wine
from mlflow.tracking import MlflowClient
import time

# Настройка MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
EXPERIMENT_NAME = "wine_quality_automl"
mlflow.set_experiment(EXPERIMENT_NAME)

def train_and_register():
    print("Загрузка данных...")
    wine = load_wine()
    data = pd.DataFrame(wine.data, columns=wine.feature_names)
    data['target'] = wine.target
    # data['target'] = data['target'].sample(frac=1).values
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Инициализация PyCaret
    s = setup(data, target='target', session_id=123, 
              verbose=False, html=False, log_experiment=False,
              n_jobs=1)

    print("Обучение...")
    best_model = compare_models(include=['rf', 'dt', 'lr'], fold=3, n_select=1, sort='Accuracy')
    final_model = finalize_model(best_model)
    
    results_df = pull()
    accuracy = results_df.iloc[0]['Accuracy']
    f1 = results_df.iloc[0]['F1']

    print("Логирование и регистрация в MLflow...")
    with mlflow.start_run(run_name="Retrain_Standard") as run:
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("f1", float(f1))
        
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name="WineQualityModel"
        )
        print("Модель отправлена на сервер.")

    # Перевод в Staging
    print("Обновление стадии модели...")
    client = MlflowClient()
    model_name = "WineQualityModel"
    
    latest_version_info = client.get_latest_versions(model_name, stages=["None"])
    if latest_version_info:
        latest_version = latest_version_info[0].version
        print(f"Перевод версии {latest_version} в Staging...")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging"
        )
        print("Успешно завершено.")
    else:
        print("Версия модели не найдена.")

if __name__ == "__main__":
    try:
        train_and_register()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit(1)