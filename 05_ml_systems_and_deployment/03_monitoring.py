"""
Feature drift monitoring with Population Stability Index (PSI).

PSI quantifies how much a feature's distribution has shifted between a
reference window (e.g., training data) and a current window (production
traffic). It's the workhorse drift metric in finance and fraud:

    PSI = sum_i (p_curr_i - p_ref_i) * log(p_curr_i / p_ref_i)

Rules of thumb:
    < 0.10  : no significant change
    0.10-0.25 : moderate shift -- investigate
    > 0.25  : significant shift -- likely retrain

Run:
    python 05_ml_systems_and_deployment/03_monitoring.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "transactions.parquet"


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Compute PSI between reference and current distributions of a numeric feature.

    Uses fixed quantile-based bin edges from `reference` so the comparison is
    apples-to-apples. Adds a tiny epsilon to avoid log(0).
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    # Quantile-based edges from reference, with -inf and +inf so current values
    # outside the reference range still fall into a bin.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    p_ref = (ref_counts / ref_counts.sum()) + eps
    p_cur = (cur_counts / cur_counts.sum()) + eps

    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI for categorical features (no binning)."""
    all_cats = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    eps = 1e-6
    p_ref = reference.value_counts(normalize=True).reindex(all_cats, fill_value=0).values + eps
    p_cur = current.value_counts(normalize=True).reindex(all_cats, fill_value=0).values + eps
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def severity(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if value < 0.10:
        return "ok"
    if value < 0.25:
        return "watch"
    return "alert"


def simulate_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Build a synthetic 'current window' where some features have drifted.
    In production this would just be the latest day's traffic.
    """
    current = df.sample(frac=0.5, random_state=42).copy()

    # Inject realistic drifts:
    #   - Average txn amount drifts +20% (e.g., merchant mix change)
    current["txn_amount"] = current["txn_amount"] * rng.normal(1.20, 0.05, size=len(current))
    #   - Country distribution shifts toward higher-risk regions
    shift_mask = rng.random(len(current)) < 0.15
    current.loc[shift_mask, "country"] = rng.choice(["NG", "RU"], size=shift_mask.sum())
    #   - email_risk distribution unchanged (control)
    return current


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}. Run `python data/synthetic_fraud.py` first.")

    reference = pd.read_parquet(DATA_PATH)
    rng = np.random.default_rng(7)
    current = simulate_drift(reference, rng)

    numeric_cols = ["account_age_days", "txn_amount", "velocity_1h",
                    "device_entropy", "email_risk", "ip_country_mismatch",
                    "noise_1", "noise_2", "noise_3"]
    categorical_cols = ["device_type", "country"]

    rows = []
    for col in numeric_cols:
        value = psi(reference[col].values, current[col].values)
        rows.append({"feature": col, "type": "numeric", "psi": round(value, 4), "severity": severity(value)})
    for col in categorical_cols:
        value = psi_categorical(reference[col], current[col])
        rows.append({"feature": col, "type": "categorical", "psi": round(value, 4), "severity": severity(value)})

    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    print("Population Stability Index — reference vs simulated-current window")
    print("=" * 64)
    print(report.to_string(index=False))
    print()
    alerts = report[report["severity"] == "alert"]
    if not alerts.empty:
        print(f"⚠️  {len(alerts)} feature(s) above the alert threshold (0.25). Investigate.")
    else:
        print("✅ No alert-level drift detected.")


if __name__ == "__main__":
    main()
