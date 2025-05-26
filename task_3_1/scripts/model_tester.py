import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import json
import os

def test_model():
    # Чтение тестовых данных
    test_data = pd.read_csv('/opt/airflow/datasets/processed/iris_test.csv')
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Загрузка модели
    model = joblib.load('/opt/airflow/models/iris/model.pkl')
    predictions = model.predict(X_test)
    
    # Расчет метрик
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    # Сохранение метрик
    os.makedirs('/opt/airflow/datasets/reports', exist_ok=True)
    with open('/opt/airflow/datasets/reports/model_metrics.json', 'w') as f:
        json.dump({'accuracy': accuracy, 'report': report}, f)
    print("Метрики сохранены в /opt/airflow/datasets/reports/model_metrics.json")