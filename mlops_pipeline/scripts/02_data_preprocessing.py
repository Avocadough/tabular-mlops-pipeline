
import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import mlflow

def preprocess(csv_path: str, target_col: str, out_dir: str = "processed_data", test_size: float = 0.2, random_state: int = 42):
    mlflow.set_experiment("Fashion - Preprocessing")
    with mlflow.start_run():
        mlflow.set_tag("ml.step", "preprocessing")
        mlflow.log_params({"csv_path": csv_path, "target_col": target_col, "test_size": test_size, "random_state": random_state})

        df = pd.read_csv(csv_path)
        assert target_col in df.columns, f"target_col '{target_col}' not in CSV"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

        num_pipe = Pipeline(steps=[("scaler", StandardScaler())])
        cat_pipe = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ]
        )

        # Fit on full data (or only train after split); we will fit on train to avoid leakage.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None)

        # Persist splits
        out = os.path.join(os.path.dirname(os.path.dirname(csv_path)), out_dir) if not os.path.isabs(out_dir) else out_dir
        os.makedirs(out, exist_ok=True)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df  = pd.concat([X_test, y_test], axis=1)
        train_path = os.path.join(out, "train.csv")
        test_path = os.path.join(out, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        mlflow.log_param("num_cols", ",".join(num_cols))
        mlflow.log_param("cat_cols", ",".join(cat_cols))
        mlflow.log_artifact(train_path, artifact_path="processed_data")
        mlflow.log_artifact(test_path, artifact_path="processed_data")
        print(json.dumps({"train_path": train_path, "test_path": test_path, "num_cols": num_cols, "cat_cols": cat_cols}, indent=2))

if __name__ == "__main__":
    # Usage: python scripts/02_data_preprocessing.py <csv_path> <target_col> [out_dir]
    if len(sys.argv) < 3:
        print("Usage: python scripts/02_data_preprocessing.py <csv_path> <target_col> [out_dir]")
        sys.exit(1)
    csv_path = sys.argv[1]
    target_col = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "processed_data"
    preprocess(csv_path, target_col, out_dir)
