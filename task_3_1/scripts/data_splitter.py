import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    # Чтение с абсолютным путем
    df = pd.read_csv('/opt/airflow/datasets/raw/iris.csv')
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Создаем папку и сохраняем с абсолютными путями
    os.makedirs('/opt/airflow/datasets/processed', exist_ok=True)
    train.to_csv('/opt/airflow/datasets/processed/iris_train.csv', index=False)
    test.to_csv('/opt/airflow/datasets/processed/iris_test.csv', index=False)
    print("Данные успешно разделены и сохранены в /opt/airflow/datasets/processed/")