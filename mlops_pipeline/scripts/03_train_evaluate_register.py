
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

def load_splits(data_dir: str, target_col: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    X_train, y_train = train.drop(columns=[target_col]), train[target_col]
    X_test,  y_test  = test.drop(columns=[target_col]),  test[target_col]
    return X_train, X_test, y_train, y_test

def build_pipeline(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Default model: RandomForest baseline
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced_subsample"
    )

    clf = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    return clf

def evaluate_binary(y_true, y_prob, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7)),
    }
    return metrics

def evaluate_multiclass(y_true, y_prob, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7)),
    }
    return metrics

def main(data_dir: str, target_col: str, model_name: str = "fashion-classifier"):
    mlflow.set_experiment("Fashion - Train/Evaluate/Register")
    with mlflow.start_run():
        mlflow.set_tag("ml.step", "train_evaluate_register")
        mlflow.log_params({"data_dir": data_dir, "target_col": target_col, "model_name": model_name})

        X_train, X_test, y_train, y_test = load_splits(data_dir, target_col)
        num_classes = int(pd.Series(y_train).nunique())
        is_binary = num_classes == 2

        clf = build_pipeline(X_train)
        clf.fit(X_train, y_train)

        if is_binary:
            prob = clf.predict_proba(X_test)[:,1]
        else:
            prob = clf.predict_proba(X_test)

        pred = clf.predict(X_test)

        if is_binary:
            metrics = evaluate_binary(y_test, prob, pred)
        else:
            # Convert to 2D probs for log_loss if needed
            if prob.ndim == 1: # just in case
                prob = np.vstack([1-prob, prob]).T
            metrics = evaluate_multiclass(y_test, prob, pred)

        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        mlflow.sklearn.log_model(clf, artifact_path="model", registered_model_name=model_name)
        print(json.dumps({"model_name": model_name, "metrics": metrics, "classes": num_classes}, indent=2))

if __name__ == "__main__":
    # Usage: python scripts/03_train_evaluate_register.py <data_dir> <target_col> [model_name]
    if len(sys.argv) < 3:
        print("Usage: python scripts/03_train_evaluate_register.py <data_dir> <target_col> [model_name]")
        sys.exit(1)
    data_dir = sys.argv[1]
    target_col = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else "fashion-classifier"
    main(data_dir, target_col, model_name)
