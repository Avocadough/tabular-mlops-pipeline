
import os
import sys
import json
import pandas as pd
import mlflow


def validate_data(csv_path: str, target_col: str):
    mlflow.set_experiment("Fashion - Data Validation")
    with mlflow.start_run():
        mlflow.set_tag("ml.step", "data_validation")
        mlflow.log_param("csv_path", csv_path)
        mlflow.log_param("target_col", target_col)

        df = pd.read_csv(csv_path)
        n_rows, n_cols = df.shape

        summary = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": df.columns.tolist(),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing_total": int(df.isnull().sum().sum()),
            "missing_by_col": {c: int(v) for c, v in df.isnull().sum().items()},
        }

        if target_col in df.columns:
            y = df[target_col]
            summary["target_unique"] = int(y.nunique())
            summary["target_counts"] = {str(k): int(v) for k, v in y.value_counts(dropna=False).to_dict().items()}

        mlflow.log_text(json.dumps(summary, indent=2), "validation_summary.json")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    # Usage: python scripts/01_data_validation.py <csv_path> <target_col>
    if len(sys.argv) != 3:
        print("Usage: python scripts/01_data_validation.py <csv_path> <target_col>")
        sys.exit(1)
    validate_data(sys.argv[1], sys.argv[2])
