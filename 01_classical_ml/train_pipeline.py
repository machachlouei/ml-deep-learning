"""
Production-shaped training pipeline.

Trains an XGBoost fraud classifier wrapped in a scikit-learn Pipeline so the
fitted preprocessor and model are serialized as a single artifact. Module 05's
inference service loads exactly this artifact.

Run:
    python 01_classical_ml/train_pipeline.py

Writes:
    artifacts/fraud_pipeline.joblib
    artifacts/metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "transactions.parquet"
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

NUMERIC_COLS = [
    "account_age_days", "txn_amount", "velocity_1h",
    "device_entropy", "email_risk", "ip_country_mismatch",
    "noise_1", "noise_2", "noise_3",
]
CATEGORICAL_COLS = ["device_type", "country"]


def build_pipeline() -> Pipeline:
    """Construct the preprocessing + model pipeline."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_COLS),
        ("cat", categorical, CATEGORICAL_COLS),
    ])
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=SEED,
    )
    return Pipeline([("prep", preprocessor), ("clf", model)])


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Missing dataset at {DATA_PATH}. "
            "Run `python data/synthetic_fraud.py` first."
        )

    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    pipeline = build_pipeline()
    # scale_pos_weight on the inner classifier
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    pipeline.set_params(clf__scale_pos_weight=spw)

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean()),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / "fraud_pipeline.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"
    joblib.dump(pipeline, artifact_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved pipeline -> {artifact_path}")
    print(f"Saved metrics  -> {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
