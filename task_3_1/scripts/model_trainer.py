import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model():
    # Чтение данных с абсолютным путем
    train_data = pd.read_csv('/opt/airflow/datasets/processed/iris_train.csv')
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    
    # Обучение модели
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Сохранение модели с абсолютным путем
    os.makedirs('/opt/airflow/models/iris', exist_ok=True)
    joblib.dump(model, '/opt/airflow/models/iris/model.pkl')
    print("Модель успешно сохранена в /opt/airflow/models/iris/model.pkl")