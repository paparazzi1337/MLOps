from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import wandb
from config import config
from data import get_data

def train(model, x_train, y_train):
    model.fit(x_train, y_train)
    # Log tree-specific parameters
    wandb.log({
        "max_depth": model.get_depth(),
        "n_leaves": model.get_n_leaves(),
        "feature_importances": model.feature_importances_,
        "params": model.get_params()
    })

def test(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) == 2 else None
    
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
    wandb.init(project="ml-project", name="decision-tree")
    
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"],
        criterion=config["decision_tree"].get("criterion", "gini"),
        min_samples_split=config["decision_tree"].get("min_samples_split", 2)
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    test(decision_tree_model, data["x_test"], data["y_test"])
    
    wandb.finish()