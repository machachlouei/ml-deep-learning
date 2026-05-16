"""
Batch scorer.

Streams a large Parquet file through the trained pipeline in chunks. Useful
patterns demonstrated:
  - Lazy iteration so memory stays flat regardless of file size.
  - Same model artifact as the online service (no training-serving skew).
  - Joblib parallelism across chunks if you have many cores.

Run:
    python 05_ml_systems_and_deployment/02_batch_scoring.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "transactions.parquet"
DEFAULT_OUTPUT = ROOT / "artifacts" / "scored.parquet"
MODEL_PATH = ROOT / "artifacts" / "fraud_pipeline.joblib"

FEATURE_COLS = [
    "account_age_days", "txn_amount", "velocity_1h",
    "device_entropy", "email_risk", "ip_country_mismatch",
    "device_type", "country",
    "noise_1", "noise_2", "noise_3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--batch-size", type=int, default=10_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Missing model at {MODEL_PATH}. "
            "Run `python 01_classical_ml/train_pipeline.py` first."
        )
    if not args.input.exists():
        raise SystemExit(f"Missing input file: {args.input}")

    pipeline = joblib.load(MODEL_PATH)
    pf = pq.ParquetFile(args.input)
    print(f"Scoring {args.input} ({pf.metadata.num_rows:,} rows) in batches of {args.batch_size:,}")

    scored_chunks: list[pd.DataFrame] = []
    total = 0
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=FEATURE_COLS):
        df = batch.to_pandas()
        probs = pipeline.predict_proba(df)[:, 1]
        df_out = df.copy()
        df_out["fraud_probability"] = probs
        scored_chunks.append(df_out)
        total += len(df_out)
        print(f"  scored {total:,} rows")

    out = pd.concat(scored_chunks, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"\nDone. Wrote {len(out):,} rows -> {args.output}")
    print(out["fraud_probability"].describe().round(4))


if __name__ == "__main__":
    main()
