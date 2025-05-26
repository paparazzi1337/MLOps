# Используем официальный образ Airflow с Python 3.8
FROM apache/airflow:2.7.3

USER root

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Создание рабочих директорий
RUN mkdir -p /opt/airflow/dags && \
    mkdir -p /opt/airflow/scripts && \
    mkdir -p /opt/airflow/data && \
    chown -R airflow:root /opt/airflow

USER airflow

# Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/airflow