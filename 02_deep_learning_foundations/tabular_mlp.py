"""
PyTorch MLP for tabular fraud detection.

Demonstrates the building blocks you'd actually ship: batch norm, dropout,
weight decay, BCEWithLogitsLoss for stability, early stopping on val PR-AUC.

Run:
    python 02_deep_learning_foundations/tabular_mlp.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "transactions.parquet"

torch.manual_seed(SEED)
np.random.seed(SEED)


class FraudMLP(nn.Module):
    """A small, well-regularized MLP for tabular fraud signals."""

    def __init__(self, in_dim: int, hidden=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(DATA_PATH)
    df = pd.get_dummies(df, columns=["device_type", "country"], drop_first=True)
    for c in ["email_risk", "device_entropy"]:
        df[c] = df[c].fillna(df[c].median())
    y = df["is_fraud"].values.astype(np.float32)
    X = df.drop(columns=["is_fraud"]).values.astype(np.float32)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr).astype(np.float32), scaler.transform(X_va).astype(np.float32), y_tr, y_va


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        probs = torch.sigmoid(logits).numpy().ravel()
    return (
        roc_auc_score(y, probs),
        average_precision_score(y, probs),
        nn.BCEWithLogitsLoss()(logits, torch.from_numpy(y).unsqueeze(1)).item(),
    )


def train() -> None:
    X_tr, X_va, y_tr, y_va = load_data()
    print(f"train: {X_tr.shape}  val: {X_va.shape}  fraud rate: {y_tr.mean():.4f}")

    # Address imbalance via pos_weight in the loss (cleaner than resampling).
    pos_weight = torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum(), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = FraudMLP(in_dim=X_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    tr_loader = make_loader(X_tr, y_tr)

    best_pr = -1.0
    best_epoch = 0
    patience = 5
    epochs = 30

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item() * xb.size(0)
        sched.step()
        train_loss = running / len(tr_loader.dataset)

        roc, pr, val_loss = evaluate(model, X_va, y_va)
        print(f"ep {ep:02d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"ROC {roc:.4f}  PR {pr:.4f}")

        if pr > best_pr:
            best_pr, best_epoch = pr, ep
        elif ep - best_epoch >= patience:
            print(f"Early stopping: no PR-AUC improvement for {patience} epochs.")
            break

    print(f"\nBest val PR-AUC: {best_pr:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Missing dataset at {DATA_PATH}. Run `python data/synthetic_fraud.py` first."
        )
    train()
