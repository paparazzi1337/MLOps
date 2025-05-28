import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
from config import config
from data import get_data
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    return "confusion_matrix.png"

def train(model, x_train, y_train):
    with mlflow.start_run():
        model.fit(x_train, y_train)
        
        # Log parameters
        mlflow.log_params(model.get_params())
        mlflow.log_metric("max_depth", model.get_depth())
        mlflow.log_metric("n_leaves", model.get_n_leaves())
        
        # Log feature importance
        for i, importance in enumerate(model.feature_importances_):
            mlflow.log_metric(f"feature_importance_{i}", importance)

def test(model, x_test, y_test):
    with mlflow.start_run():
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) == 2 else 0
        })
        
        # Log confusion matrix
        mlflow.log_artifact(plot_confusion_matrix(cm, np.unique(y_test)), "confusion_matrix.png")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    mlflow.set_experiment("Decision Tree")
    
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"],
        criterion=config["decision_tree"].get("criterion", "gini"),
        min_samples_split=config["decision_tree"].get("min_samples_split", 2)
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    test(decision_tree_model, data["x_test"], data["y_test"])