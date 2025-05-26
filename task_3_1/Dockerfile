# Используем официальный образ Airflow с Python 3.8
FROM apache/airflow:2.7.3

USER root

# Создаем необходимые директории
RUN mkdir -p /opt/airflow/datasets && \
    mkdir -p /opt/airflow/models && \
    chown -R airflow:root /opt/airflow

# Устанавливаем зависимости
USER airflow
WORKDIR /opt/airflow
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt