from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()  # Загружаем переменные окружения из .env

default_args = {
    'owner': 'airflow',
    'retries': 3,
    'retry_delay': timedelta(minutes=1)
}

with DAG(
    'weather_data_pipeline',
    default_args=default_args,
    description='DAG for collecting weather data',
    schedule_interval='*/1 * * * *',  # Каждую минуту
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    
    def collect_weather_data():
        from scripts.weather import save_weather_to_csv
        api_key = os.getenv("OPENWEATHER_API_KEY")  # Получаем ключ из переменных окружения
        save_weather_to_csv(api_key)  # Явно передаем ключ
    
    fetch_task = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=collect_weather_data
    )