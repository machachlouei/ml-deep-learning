# ml-deep-learning

An educational repo covering the machine learning and deep learning foundations most relevant to **fraud, risk, and identity-trust platforms** — framed for senior IC / staff-level data science interviews.

Each module mixes:
- **Concept READMEs** — overview, pros/cons, when-to-use, common pitfalls.
- **Concept notebooks (`.ipynb`)** — small, runnable demos on a synthetic fraud dataset.
- **System `.py` files** — production-shaped code (training pipelines, inference services, monitoring).

The domain backdrop throughout is **digital identity & transaction fraud detection** — noisy tabular data, severe class imbalance, latency-sensitive inference, model + concept drift, and human-in-the-loop case review.

## Modules

| # | Module | Focus |
|---|--------|-------|
| 01 | [Classical ML with scikit-learn & XGBoost](./01_classical_ml/) | Logistic regression, tree ensembles, calibration, class imbalance |
| 02 | [Deep Learning Foundations](./02_deep_learning_foundations/) | Forward/backprop, optimizers, regularization, when DL beats trees |
| 03 | [TensorFlow & PyTorch](./03_tensorflow_vs_pytorch/) | Same tabular model in both; ecosystem trade-offs |
| 04 | [Model Selection & Representation Learning](./04_model_selection_and_representation/) | CV strategies, leakage, embeddings, contrastive learning |
| 05 | [Practical ML Systems & Deployment](./05_ml_systems_and_deployment/) | Serving APIs, batch scoring, drift monitoring, MLOps |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Generate the shared synthetic dataset once:

```bash
python data/synthetic_fraud.py
```

That writes `data/transactions.parquet` (~50k rows) — all notebooks and scripts use it.

## How to use this repo

- **Skimming for an interview?** Read the module READMEs end-to-end (~30 min).
- **Brushing up hands-on?** Run the notebooks in order — each is self-contained.
- **Studying production patterns?** Focus on module 05 plus `train_pipeline.py` in module 01.

## Conventions

- Random seeds fixed at `42` everywhere for reproducibility.
- All examples use the same synthetic dataset so you can compare apples to apples across modules.
- Code is written for clarity over cleverness — interview-explainable, not Kaggle-optimized.
