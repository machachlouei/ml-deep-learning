"""
Synthetic fraud / identity-risk dataset.

Generates a tabular dataset that resembles a transaction-level fraud signal
problem on an identity-trust platform. Highly imbalanced (~2% positives),
mixes numeric and categorical features, with realistic noise and a small set
of "signal" features the models should learn to rely on.

Run once:
    python data/synthetic_fraud.py

Writes:
    data/transactions.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_ROWS = 50_000
FRAUD_RATE = 0.02

DEVICE_TYPES = ["ios", "android", "web_chrome", "web_safari", "web_other"]
COUNTRIES = ["US", "CA", "GB", "MX", "BR", "IN", "NG", "RU", "OTHER"]


def generate(n_rows: int = N_ROWS, fraud_rate: float = FRAUD_RATE, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Latent fraud label first; features will be conditioned on it.
    is_fraud = rng.binomial(1, fraud_rate, size=n_rows).astype(np.int8)

    # --- Signal features (correlated with fraud) ---
    # Account age in days: fraudsters tend to use newer accounts.
    account_age_days = np.where(
        is_fraud == 1,
        rng.exponential(scale=30, size=n_rows),
        rng.exponential(scale=400, size=n_rows),
    ).clip(0, 3650)

    # Transaction amount: fraud skews higher with a fat tail.
    txn_amount = np.where(
        is_fraud == 1,
        rng.lognormal(mean=5.5, sigma=1.2, size=n_rows),
        rng.lognormal(mean=3.5, sigma=1.0, size=n_rows),
    ).round(2)

    # Velocity: txns in the past hour. Fraud rings burst.
    velocity_1h = np.where(
        is_fraud == 1,
        rng.poisson(lam=6, size=n_rows),
        rng.poisson(lam=1, size=n_rows),
    )

    # Device entropy: low = device fingerprint looks suspicious (emulator-like).
    device_entropy = np.where(
        is_fraud == 1,
        rng.beta(2, 5, size=n_rows),
        rng.beta(5, 2, size=n_rows),
    )

    # Email risk score from upstream signal (0-1).
    email_risk = np.where(
        is_fraud == 1,
        rng.beta(4, 2, size=n_rows),
        rng.beta(2, 6, size=n_rows),
    )

    # IP / country mismatch flag.
    ip_country_mismatch = rng.binomial(
        1, np.where(is_fraud == 1, 0.45, 0.05), size=n_rows
    ).astype(np.int8)

    # --- Categorical features ---
    device_type = rng.choice(DEVICE_TYPES, size=n_rows, p=[0.25, 0.30, 0.20, 0.15, 0.10])
    # Fraud is more likely from higher-risk geographies.
    fraud_country_probs = np.array([0.10, 0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.15, 0.10])
    legit_country_probs = np.array([0.55, 0.10, 0.10, 0.05, 0.05, 0.05, 0.02, 0.03, 0.05])
    country = np.array([
        rng.choice(COUNTRIES, p=fraud_country_probs if f else legit_country_probs)
        for f in is_fraud
    ])

    # --- Pure noise features (the model must learn to ignore) ---
    noise_1 = rng.normal(0, 1, size=n_rows)
    noise_2 = rng.normal(0, 1, size=n_rows)
    noise_3 = rng.uniform(0, 100, size=n_rows)

    df = pd.DataFrame({
        "account_age_days": account_age_days.round(1),
        "txn_amount": txn_amount,
        "velocity_1h": velocity_1h,
        "device_entropy": device_entropy.round(4),
        "email_risk": email_risk.round(4),
        "ip_country_mismatch": ip_country_mismatch,
        "device_type": device_type,
        "country": country,
        "noise_1": noise_1.round(4),
        "noise_2": noise_2.round(4),
        "noise_3": noise_3.round(2),
        "is_fraud": is_fraud,
    })

    # Inject ~3% missing values into a couple of numeric columns (realistic).
    for col in ["email_risk", "device_entropy"]:
        mask = rng.random(n_rows) < 0.03
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    out_dir = Path(__file__).parent
    out_path = out_dir / "transactions.parquet"
    df = generate()
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}  ({len(df):,} rows, fraud rate = {df.is_fraud.mean():.2%})")


if __name__ == "__main__":
    main()
