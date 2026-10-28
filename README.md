<div align="center">
  <h1>Tabular Classification with MLflow</h1>
  <p>An MLOps prototype that validates a labelled CSV, trains a scikit-learn classifier and serves a registered model through FastAPI.</p>
  <p>
    <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square" alt="scikit-learn">
    <img src="https://img.shields.io/badge/MLflow-Registry-0194E2?style=flat-square" alt="MLflow registry">
    <img src="https://img.shields.io/badge/FastAPI-Inference-009688?style=flat-square" alt="FastAPI inference">
  </p>
  <p><a href="#workflow">Workflow</a> · <a href="#run-locally">Run locally</a> · <a href="#limits">Limits</a></p>
</div>

## Workflow

The pipeline logs a data summary, makes a stratified train/test split, then fits numeric scaling and categorical encoding inside a Random Forest pipeline. It records classification metrics and registers the model with MLflow. A separate script assigns an alias to the latest model version.

- [`mlops_pipeline/scripts/`](mlops_pipeline/scripts/) — validation, splitting, training/evaluation/registration and alias assignment.
- [`mlops_pipeline/requirements.txt`](mlops_pipeline/requirements.txt) — pipeline dependencies.
- [`app.py`](app.py) — FastAPI endpoints for model information, predictions and reload.
- [`.github/workflows/main.yml`](.github/workflows/main.yml) — configured GitHub Actions workflow.
- `mlruns/` and `processed_data/` — historical tracked artifacts; they do not establish performance on a new dataset.

## Run locally

The original example CSV is not included. Supply your own labelled CSV, including a `target` column; use suitable class counts for a stratified split. The binary metric code assumes labels compatible with scikit-learn's default positive class. Commands below run from the repository root. Use a disposable copy: scripts write or replace split files and MLflow state. This workflow has not been rerun for this README.

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r mlops_pipeline/requirements.txt
# FastAPI and Uvicorn are not included in this revision's requirements file:
python -m pip install fastapi uvicorn

python mlops_pipeline/scripts/01_data_validation.py data/fashion.csv target
python mlops_pipeline/scripts/02_data_preprocessing.py data/fashion.csv target "$(pwd)/processed_data"
python mlops_pipeline/scripts/03_train_evaluate_register.py processed_data target fashion-classifier
python mlops_pipeline/scripts/04_transition_model.py fashion-classifier prod
```

The absolute output path in the preprocessing command avoids the script's path-relative output resolution. In PowerShell, use `"$((Get-Location).Path)/processed_data"` instead of `"$(pwd)/processed_data"`.

The app's default URI uses `/prod` as a stage-style path; use the alias URI explicitly after registration:

```bash
# bash / zsh; in PowerShell use $env:MODEL_URI = 'models:/fashion-classifier@prod'
export MODEL_URI='models:/fashion-classifier@prod'
uvicorn app:app --host 127.0.0.1 --port 8000
```

Visit `/docs` for request schemas. `GET /health` reports whether the model loaded; `POST /predict` expects an `instances` list of objects whose keys match the training CSV features. `/predict_proba` requires the loaded model to expose probabilities; `/reload` attempts to load a new model URI or configuration.

## Limits

The project illustrates a pipeline, not a verified production service. Dataset quality, suitable splits, external data rights, model quality and serving compatibility depend on the supplied data and environment.

The checked-in Actions workflow does not match these scripts: validation and preprocessing are invoked without their required CSV path and target arguments. Training references a preprocessing run ID instead of its required data directory and target, but this preprocessing script does not publish a run ID as a workflow output. The alias step also names `wine-classifier-prod`, not the default model in this project. The workflow was not run in this review.

The previous README stated `MIT (adjust as needed)`; no separate LICENSE file is present. Confirm the intended license terms before reuse.
