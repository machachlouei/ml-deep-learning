"""
Model-agnostic permutation importance.

Why this matters for interviews:
- XGBoost's built-in `gain` importance is **biased** toward high-cardinality
  features and depends on training-time tree structure.
- Permutation importance asks the right question: "how much does the model's
  test-set performance drop if I scramble this column?"
- It's model-agnostic — works for trees, MLPs, linear models, anything with a
  scoring function.

Run:
    python 04_model_selection_and_representation/feature_importance.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "transactions.parquet"


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}. Run `python data/synthetic_fraud.py` first.")

    df = pd.read_parquet(DATA_PATH)
    for col in ["device_type", "country"]:
        df[col] = df[col].astype("category")

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        random_state=SEED,
    )
    model.fit(X_tr, y_tr)
    baseline = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"Baseline test PR-AUC: {baseline:.4f}\n")

    # 1) Built-in gain
    gain = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)

    # 2) Permutation importance — uses test data and the scoring function
    perm = permutation_importance(
        model, X_te, y_te,
        scoring="average_precision",
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
    )
    perm_series = pd.Series(perm.importances_mean, index=X_te.columns).sort_values(ascending=False)
    perm_std = pd.Series(perm.importances_std, index=X_te.columns)

    side_by_side = pd.DataFrame({
        "gain": gain.round(4),
        "perm_mean_pr_drop": perm_series.round(4),
        "perm_std": perm_std.round(4),
    }).sort_values("perm_mean_pr_drop", ascending=False)

    print("Side-by-side: built-in gain vs permutation importance (PR-AUC drop)")
    print("--------------------------------------------------------------")
    print(side_by_side.to_string())
    print()
    print("Interpretation:")
    print("  - High `perm_mean_pr_drop` = scrambling this column hurts the model = real signal.")
    print("  - The noise_* columns should sit near zero, with std overlapping zero.")
    print("  - When `gain` and permutation importance disagree, trust permutation.")


if __name__ == "__main__":
    main()
