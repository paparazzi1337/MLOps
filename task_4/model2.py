from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from clearml import Task
import numpy as np

from config import config
from data import get_data

# Инициализация задачи ClearML
task = Task.init(project_name='Model Tracking', task_name='Decision Tree')

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)
    # Логируем параметры модели
    params = model.get_params()
    task.connect(params, name='Decision Tree Parameters')
    
    # Логируем специфичные параметры дерева
    task.get_logger().report_scalar(title='Tree Info', series='Depth', value=model.get_depth(), iteration=0)
    task.get_logger().report_scalar(title='Tree Info', series='Leaves', value=model.get_n_leaves(), iteration=0)
    
    # Логируем важность признаков
    if hasattr(model, 'feature_importances_'):
        for i, importance in enumerate(model.feature_importances_):
            task.get_logger().report_scalar(title='Feature Importance', series=f'Feature_{i}', value=importance, iteration=0)

def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    
    # Вычисляем метрики с учетом мультиклассовости
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')  # Для мультиклассовой классификации
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    # Логируем метрики
    task.get_logger().report_scalar(title='Metrics', series='Accuracy', value=accuracy, iteration=1)
    task.get_logger().report_scalar(title='Metrics', series='F1 Score', value=f1, iteration=1)
    
    # Обработка ROC-AUC для мультиклассового случая
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(x_test)
        try:
            # Для бинарной классификации
            if y_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_true=y_test, y_score=y_proba[:, 1])
            # Для мультиклассовой
            else:
                roc_auc = roc_auc_score(y_true=y_test, y_score=y_proba, multi_class='ovr')
            task.get_logger().report_scalar(title='Metrics', series='AUC-ROC', value=roc_auc, iteration=1)
        except Exception as e:
            print(f"Could not calculate ROC-AUC: {str(e)}")
    
    # Логируем матрицу ошибок
    task.get_logger().report_confusion_matrix(
        title='Confusion Matrix',
        series='Actual vs Predicted',
        matrix=cm,
        xaxis='Predicted',
        yaxis='Actual'
    )

if __name__ == "__main__":
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"],
        criterion=config["decision_tree"].get("criterion", "gini"),
        min_samples_split=config["decision_tree"].get("min_samples_split", 2)
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    test(decision_tree_model, data["x_test"], data["y_test"])