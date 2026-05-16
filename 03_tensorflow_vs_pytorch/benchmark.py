"""
Quick training-time benchmark: PyTorch vs TensorFlow on identical architecture.

Not meant to declare a winner — runtime depends on hardware, batch size, and
graph compilation. Useful as a 'hello world' for spinning up both frameworks
on the same workload.

Run:
    python 03_tensorflow_vs_pytorch/benchmark.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "transactions.parquet"

EPOCHS = 5
BATCH_SIZE = 256
HIDDEN = (64, 32)


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(DATA_PATH)
    df = pd.get_dummies(df, columns=["device_type", "country"], drop_first=True)
    for c in ["email_risk", "device_entropy"]:
        df[c] = df[c].fillna(df[c].median())
    y = df["is_fraud"].values.astype(np.float32)
    X = df.drop(columns=["is_fraud"]).values.astype(np.float32)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    return (scaler.transform(X_tr).astype(np.float32),
            scaler.transform(X_va).astype(np.float32), y_tr, y_va)


def bench_pytorch(X_tr, y_tr) -> float:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(SEED)

    in_dim = X_tr.shape[1]
    layers, prev = [], in_dim
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    model = nn.Sequential(*layers)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return time.perf_counter() - t0


def bench_tensorflow(X_tr, y_tr) -> float:
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(SEED)

    inputs = keras.Input(shape=(X_tr.shape[1],))
    x = inputs
    for h in HIDDEN:
        x = keras.layers.Dense(h)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    return time.perf_counter() - t0


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}. Run `python data/synthetic_fraud.py` first.")

    X_tr, _X_va, y_tr, _y_va = load()
    print(f"Train shape: {X_tr.shape}  | epochs={EPOCHS}  | batch={BATCH_SIZE}\n")

    pt_time = bench_pytorch(X_tr, y_tr)
    print(f"PyTorch     : {pt_time:.2f}s")

    tf_time = bench_tensorflow(X_tr, y_tr)
    print(f"TensorFlow  : {tf_time:.2f}s")

    print("\nReminder: this is one CPU run on a toy net. Real benchmarks need")
    print("warmups, multiple runs, and matched data pipelines. Use as a sanity")
    print("check that both frameworks are correctly installed.")


if __name__ == "__main__":
    main()
