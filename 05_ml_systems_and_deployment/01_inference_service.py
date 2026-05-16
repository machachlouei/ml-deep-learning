"""
FastAPI inference service for the fraud model.

Loads the serialized pipeline from `artifacts/fraud_pipeline.joblib` (built by
01_classical_ml/train_pipeline.py) and serves a single `/score` endpoint with
Pydantic input validation.

Run:
    uvicorn 05_ml_systems_and_deployment.01_inference_service:app --reload
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "fraud_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


class TransactionInput(BaseModel):
    """One transaction's features. Mirrors the training schema exactly."""

    account_age_days: float = Field(..., ge=0)
    txn_amount: float = Field(..., ge=0)
    velocity_1h: int = Field(..., ge=0)
    device_entropy: float = Field(..., ge=0, le=1)
    email_risk: float = Field(..., ge=0, le=1)
    ip_country_mismatch: Literal[0, 1]
    device_type: Literal["ios", "android", "web_chrome", "web_safari", "web_other"]
    country: str
    noise_1: float = 0.0
    noise_2: float = 0.0
    noise_3: float = 0.0


class ScoreResponse(BaseModel):
    fraud_probability: float
    decision: Literal["allow", "review", "block"]
    model_version: str
    latency_ms: float


app = FastAPI(title="Fraud Scoring Service", version="0.1.0")

# Globals populated at startup
_MODEL = None
_TRAINING_METRICS: dict = {}
_MODEL_VERSION = "unknown"

# Decision thresholds (would normally come from a config service or feature flags).
REVIEW_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.80


@app.on_event("startup")
def _load_model() -> None:
    global _MODEL, _TRAINING_METRICS, _MODEL_VERSION
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing model artifact at {MODEL_PATH}. "
            "Run `python 01_classical_ml/train_pipeline.py` first."
        )
    _MODEL = joblib.load(MODEL_PATH)
    if METRICS_PATH.exists():
        _TRAINING_METRICS = json.loads(METRICS_PATH.read_text())
    # In a real system this would be the artifact's content hash or registry ID.
    _MODEL_VERSION = f"sha-{abs(hash(MODEL_PATH.stat().st_mtime)) % 10**8:08d}"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _MODEL is not None else "loading",
        "model_version": _MODEL_VERSION,
        "training_metrics": _TRAINING_METRICS,
    }


def _decision(p: float) -> str:
    if p >= BLOCK_THRESHOLD:
        return "block"
    if p >= REVIEW_THRESHOLD:
        return "review"
    return "allow"


@app.post("/score", response_model=ScoreResponse)
def score(txn: TransactionInput) -> ScoreResponse:
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    # Pipeline expects a DataFrame with the same column names/dtypes as training.
    row = pd.DataFrame([txn.model_dump()])
    proba = float(_MODEL.predict_proba(row)[:, 1][0])
    latency_ms = (time.perf_counter() - t0) * 1000

    return ScoreResponse(
        fraud_probability=round(proba, 6),
        decision=_decision(proba),
        model_version=_MODEL_VERSION,
        latency_ms=round(latency_ms, 2),
    )
