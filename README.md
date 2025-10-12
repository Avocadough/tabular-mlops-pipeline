
# Fashion MLflow MLOps (Tabular) — with FastAPI Inference

This repo is a **dataset‑agnostic** MLOps template built around **MLflow** for tabular data (e.g., a fashion dataset in CSV).
It contains a 3‑step training pipeline (validation → preprocessing → train/eval/register) and a **FastAPI** app for serving predictions from the **MLflow Model Registry**.

> If you want to serve **images** (e.g., Fashion‑MNIST), you'll need extra dependencies and a different training script. This template targets **tabular** data by default.

---

## Project Structure

```
mlflow_github/
├─ data/                      # put your raw CSV here (not versioned in this template)
├─ processed_data/            # train/test splits written here by step 02
├─ mlops_pipeline/
│  ├─ scripts/
│  │  ├─ 01_data_validation.py
│  │  ├─ 02_data_preprocessing.py
│  │  ├─ 03_train_evaluate_register.py
│  │  └─ 04_transition_model.py
│  └─ requirements.txt
├─ mlruns/                    # local MLflow runs & registry (if using local backend)
├─ app.py                     # FastAPI inference app (serves from MLflow registry)
└─ README.md
```

---

## Environment

Python 3.10+ recommended.

Install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r mlops_pipeline/requirements.txt
```

If you plan to run the FastAPI server:
```bash
pip install fastapi uvicorn
# or: pip install -r mlops_pipeline/requirements.txt  # already includes fastapi & uvicorn if you kept the updated file
```

---

## 1) Data Validation

**Input:** CSV file + target column name

```bash
python mlops_pipeline/scripts/01_data_validation.py data/fashion.csv target
```

**What it does**
- Reads the CSV
- Reports columns, dtypes, missing values, class distribution (if target is present)
- Logs a `validation_summary.json` to MLflow

Set your tracking location if needed:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000   # example using remote MLflow server
```

---

## 2) Preprocessing & Split

```bash
python mlops_pipeline/scripts/02_data_preprocessing.py data/fashion.csv target
```

**What it does**
- Infers numeric vs categorical columns
- Applies `StandardScaler` (numeric) and `OneHotEncoder` (categorical)
- Generates `processed_data/train.csv` and `processed_data/test.csv`
- Logs artifacts to MLflow

---

## 3) Train, Evaluate, Register

```bash
python mlops_pipeline/scripts/03_train_evaluate_register.py processed_data target fashion-classifier
```

**What it does**
- Builds a `ColumnTransformer` + `RandomForestClassifier` pipeline
- Evaluates (binary or multiclass)
- Logs metrics
- **Registers** the model to MLflow Model Registry as `fashion-classifier`

> If using a remote tracking server + registry, ensure envs:
> - `MLFLOW_TRACKING_URI`
> - `MLFLOW_S3_ENDPOINT_URL` (if using S3/minio for artifact store)
> - Credentials for artifact store (e.g., AWS/MINIO keys)

---

## 4) Transition Model Alias (optional)

Mark the latest production model with an alias, e.g. `prod`:

```bash
python mlops_pipeline/scripts/04_transition_model.py fashion-classifier prod
```

Now your server can load `models:/fashion-classifier/prod`.

---

## FastAPI Inference Server

### Run
```bash
# Make sure you've set the MLflow URI if using a registry (optional if local default):
# export MLFLOW_TRACKING_URI=http://localhost:5000

# Optional: set which model to load
# export MODEL_NAME=fashion-classifier
# export MODEL_STAGE=prod          # or "staging" or a numeric VERSION
# Optional: override with a direct model URI:
# export MODEL_URI=models:/fashion-classifier/prod

uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoints
- `GET /health` → health check
- `GET /model` → which model URI is currently loaded
- `POST /predict`
  ```json
  {
    "instances": [
      {"feature_a": 1.0, "feature_b": "x", "feature_c": 0.5},
      {"feature_a": 2.0, "feature_b": "y", "feature_c": 1.5}
    ]
  }
  ```
  **Response**
  ```json
  {"predictions": [0, 2]}
  ```
- `POST /predict_proba` (if model exposes `predict_proba`)
  ```json
  {"probabilities": [[0.1, 0.9], [0.7, 0.3]]}
  ```
- `POST /reload` → Reload model (optionally with new `model_name`, `model_stage` or `model_uri` in body)

### Notes
- The server expects **tabular** inputs (list of JSON objects). It converts them to a Pandas `DataFrame` before calling the model.
- Use `MODEL_URI` env var to hard‑point to a specific run’s artifact if you don’t have a registry.
- If your registry is local (inside this repo’s `mlruns/`), the default MLflow client should find it without extra envs.

---

## Troubleshooting

- **Model not found**: Ensure the registry has a registered model/version or set `MODEL_URI` to a direct artifact path.
- **predict_proba missing**: Some models don’t implement it. Use `/predict` or pick a classifier that supports probabilities.
- **Schema mismatch**: Your request JSON keys must align with training features (after preprocessing).

---

## License

MIT (adjust as needed).
