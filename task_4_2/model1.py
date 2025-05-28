from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import wandb
from config import config
from data import get_data

def train(model, x_train, y_train):
    model.fit(x_train, y_train)
    # Log parameters to W&B
    wandb.log({
        "coef": model.coef_[0],
        "intercept": model.intercept_[0],
        "params": model.get_params()
    })

def test(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else None
    
    # Log metrics to W&B
    wandb.log({
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test, preds=y_pred,
            class_names=[str(i) for i in np.unique(y_test)]
        )
    })

if __name__ == "__main__":
    # Initialize W&B
    wandb.init(project="ml-project", name="logistic-regression")
    
    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
        penalty=config["logistic_regression"].get("penalty", "l2"),
        C=config["logistic_regression"].get("C", 1.0),
        random_state=config["random_state"]
    )

    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"])
    
    wandb.finish()