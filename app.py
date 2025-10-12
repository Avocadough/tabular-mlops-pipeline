
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import mlflow
import mlflow.pyfunc

app = FastAPI(title="Fashion Tabular Inference API", version="1.0.0")

# -----------------------------
# Config / Environment
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "fashion-classifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "prod")  # staging, prod, or numeric version
MODEL_URI = os.getenv("MODEL_URI")  # if set, overrides registry lookup

_current_model_uri: Optional[str] = None
_model: Optional[mlflow.pyfunc.PyFuncModel] = None

def _resolve_model_uri() -> str:
    if MODEL_URI:
        return MODEL_URI
    # If user supplied a numeric version, use version; otherwise use stage alias
    if str(MODEL_STAGE).isdigit():
        return f"models:/{MODEL_NAME}/{int(MODEL_STAGE)}"
    return f"models:/{MODEL_NAME}/{MODEL_STAGE}"

def _load_model():
    global _model, _current_model_uri
    uri = _resolve_model_uri()
    try:
        _model = mlflow.pyfunc.load_model(uri)
        _current_model_uri = uri
        return uri
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{uri}': {e}")

# Load on startup
@app.on_event("startup")
def startup_event():
    try:
        uri = _load_model()
        print(f"[startup] Loaded model from {uri}")
    except Exception as e:
        # Don't crash; allow /health to report failure state
        print(f"[startup] WARN: {e}")

# -----------------------------
# Schemas
# -----------------------------
class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")

class ReloadRequest(BaseModel):
    model_name: Optional[str] = None
    model_stage: Optional[str] = None  # alias or version
    model_uri: Optional[str] = None

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    ok = _model is not None
    return {"status": "ok" if ok else "model_not_loaded", "model_uri": _current_model_uri}

@app.get("/model")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_uri": _current_model_uri or _resolve_model_uri(),
    }

@app.post("/reload")
def reload_model(req: ReloadRequest):
    global MODEL_NAME, MODEL_STAGE, MODEL_URI
    if req.model_name is not None:
        MODEL_NAME = req.model_name
    if req.model_stage is not None:
        MODEL_STAGE = req.model_stage
    if req.model_uri is not None:
        MODEL_URI = req.model_uri

    try:
        uri = _load_model()
        return {"status": "reloaded", "model_uri": uri}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(body: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        df = pd.DataFrame(body.instances)
        preds = _model.predict(df)
        # Ensure JSON-serializable
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.post("/predict_proba")
def predict_proba(body: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        # Try to call predict_proba on the underlying sklearn pipeline if available
        # mlflow.pyfunc returns a wrapper; attempt to access _model._model_impl.python_model.model
        py_model = None
        try:
            py_model = _model._model_impl.python_model  # type: ignore[attr-defined]
        except Exception:
            pass
        clf = None
        if py_model is not None:
            # For sklearn flavor
            inner = getattr(py_model, "model", None)
            if inner is not None and hasattr(inner, "predict_proba"):
                clf = inner

        df = pd.DataFrame(body.instances)
        if clf is None:
            # Some pyfunc models also expose predict_proba directly
            if hasattr(_model, "predict_proba"):
                prob = _model.predict_proba(df)  # type: ignore
            else:
                raise RuntimeError("Model does not expose predict_proba")
        else:
            prob = clf.predict_proba(df)

        if hasattr(prob, "tolist"):
            prob = prob.tolist()
        return {"probabilities": prob}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Probability prediction failed: {e}")
