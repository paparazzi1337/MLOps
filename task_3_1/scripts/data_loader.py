from sklearn.datasets import load_iris
import pandas as pd
import os

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Создаем папку и сохраняем с абсолютным путем
    os.makedirs('/opt/airflow/datasets/raw', exist_ok=True)
    df.to_csv('/opt/airflow/datasets/raw/iris.csv', index=False)
    print("Данные успешно сохранены в /opt/airflow/datasets/raw/iris.csv")