import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, 
                            confusion_matrix, roc_auc_score,
                            ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import numpy as np
from config import config
from data import get_data

def plot_and_save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """Создает и сохраняет матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()
    return filename

def main():
    # Настройка MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Logistic Regression")
    
    # Создаем и настраиваем модель
    model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
        penalty=config["logistic_regression"].get("penalty", "l2"),
        C=config["logistic_regression"].get("C", 1.0),
        random_state=config["random_state"]
    )

    # Загружаем данные
    data = get_data()
    
    # Основной запуск эксперимента
    with mlflow.start_run() as run:
        print(f"Started run with ID: {run.info.run_id}")
        
        # 1. Обучение модели и логирование параметров
        model.fit(data["x_train"], data["y_train"])
        mlflow.log_params(model.get_params())
        mlflow.log_metric("intercept", model.intercept_[0])
        
        # Логируем коэффициенты
        for i, coef in enumerate(model.coef_[0]):
            mlflow.log_metric(f"coef_{i}", coef)

        # 2. Тестирование модели и логирование метрик
        y_pred = model.predict(data["x_test"])
        y_proba = model.predict_proba(data["x_test"])[:, 1]
        
        # Вычисляем метрики
        metrics = {
            "accuracy": accuracy_score(data["y_test"], y_pred),
            "f1": f1_score(data["y_test"], y_pred, average='weighted')
        }
        
        # ROC-AUC только для бинарной классификации
        if len(np.unique(data["y_test"])) == 2:
            metrics["roc_auc"] = roc_auc_score(data["y_test"], y_proba)
        
        # Логируем метрики в основной run
        mlflow.log_metrics(metrics)
        print(f"Logged metrics: {metrics}")

        # 3. Логируем матрицу ошибок
        cm_path = plot_and_save_confusion_matrix(data["y_test"], y_pred)
        mlflow.log_artifact(cm_path)
        
        # 4. Логируем модель
        mlflow.sklearn.log_model(model, "model")
        
        print("Experiment completed successfully!")

if __name__ == "__main__":
    main()