from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np

from config import config
from data import get_data
from clearml import Task

# Инициализация задачи ClearML
task = Task.init(project_name='Model Tracking', task_name='Logistic Regression')

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)
    # Логируем параметры модели
    task.connect(model.get_params(), name='Logistic Regression Parameters')
    # Логируем коэффициенты регрессии
    if hasattr(model, 'coef_'):
        for i, coef in enumerate(model.coef_[0]):
            task.get_logger().report_scalar(title='Coefficients', series=f'Feature_{i}', value=coef, iteration=0)

def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    
    # Вычисляем метрики
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')  # Исправлено
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")  # Добавлен вывод accuracy в консоль
    
    # Логируем метрики
    task.get_logger().report_scalar(title='Metrics', series='Accuracy', value=accuracy, iteration=1)
    task.get_logger().report_scalar(title='Metrics', series='F1 Score', value=f1, iteration=1)
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(x_test)
        try:
            if y_proba.shape[1] == 2:  # Бинарная классификация
                roc_auc = roc_auc_score(y_true=y_test, y_score=y_proba[:, 1])
            else:  # Мультиклассовая
                roc_auc = roc_auc_score(y_true=y_test, y_score=y_proba, multi_class='ovr')
            task.get_logger().report_scalar(title='Metrics', series='AUC-ROC', value=roc_auc, iteration=1)
        except Exception as e:
            print(f"Could not calculate ROC-AUC: {str(e)}")
    
    # Логируем матрицу ошибок
    task.get_logger().report_confusion_matrix(
        title='Confusion Matrix',
        series='Actual vs Predicted',
        matrix=cm,
        xaxis='Predicted',  # Исправлено
        yaxis='Actual'
    )

if __name__ == "__main__":
    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
        penalty=config["logistic_regression"].get("penalty", "l2"),
        C=config["logistic_regression"].get("C", 1.0),
        random_state=config["random_state"]
    )

    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"])