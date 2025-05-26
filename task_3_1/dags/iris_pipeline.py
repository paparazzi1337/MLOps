from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator  # Измененный импорт
import sys
import os

# Добавляем путь к скриптам в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG(
    'iris_classification',
    default_args=default_args,
    description='Ирисы Фишера: загрузка, обработка, обучение и тестирование',
    schedule_interval=None,
    catchup=False,
    tags=['mlops'],
) as dag:

    load_data = PythonOperator(
        task_id='load_data',
        python_callable=lambda: __import__('data_loader').load_data()
    )

    prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=lambda: __import__('data_splitter').prepare_data()
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=lambda: __import__('model_trainer').train_model()
    )

    test_model = PythonOperator(
        task_id='test_model',
        python_callable=lambda: __import__('model_tester').test_model()
    )

    load_data >> prepare_data >> train_model >> test_model