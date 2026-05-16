# 04 — Model Selection & Representation Learning

> *Why this module matters in interviews:* Staff-level interviews probe **how you validate**, not just what model you pick. Model selection done wrong is the #1 way good models become bad products. Representation learning is the modern lever for unlocking gains beyond what hand-crafted features give you.

## Concepts

### Model Selection

#### Cross-validation strategies — pick the right one
| Strategy | When to use | Risk if misused |
|---|---|---|
| **K-Fold** | IID data, no time or grouping | Optimistic on temporal data |
| **Stratified K-Fold** | Classification with imbalance | Doesn't fix temporal leakage |
| **Group K-Fold** | Repeated entities (same user/device across rows) | Without it, you leak identity across folds |
| **Time Series Split** | Anything with a temporal label | Don't use random splits; you'll publish a non-shippable model |
| **Purged + embargoed CV** | Financial / temporal with target overlap | Overlapping label windows otherwise leak |

**Heuristic for fraud problems**: time-based split first; group split if you have repeated entities; stratify only as a tiebreaker.

#### Hyperparameter search
- **Grid search**: exhaustive, slow, ignores interactions.
- **Random search**: typically beats grid for the same compute. Default for ≤30 trials.
- **Bayesian optimization** (Optuna, scikit-optimize): smarter for expensive evals; use when each trial costs >5 minutes.
- **Hyperband / ASHA**: when you can afford many cheap trials and need early stopping; great for neural nets.

#### Choosing the operating point
Models output scores; **decisions need thresholds.** Three common framings:

1. **Fixed-FPR**: pick the threshold that hits a target false-positive rate. Aligns with capacity constraints (review queue size).
2. **Precision-at-K**: top-K most suspicious cases per day. Aligns with analyst throughput.
3. **Expected-loss minimization**: needs calibrated probabilities × cost matrix. Aligns with finance.

### Leakage — the senior interview probe
You will be asked to *find* leakage in a scenario, not just define it. Patterns to recognize:

- **Future information in features**: time-windowed aggregates that include the current row.
- **Label-derived features**: post-decision flags, override notes, downstream outcomes.
- **Target encoding without out-of-fold averaging**: classic mean-encoded category leakage.
- **Same-entity rows across train/test**: handles, devices, sessions.
- **Preprocessing fit on the full dataset**: scaling/imputation using test statistics.

### Representation Learning

Why we care: hand-crafted features have a ceiling. Learned representations transfer across tasks and capture interactions you wouldn't think to engineer.

#### Categorical embeddings
Replace one-hot encoding with a learned dense vector per category. Works the same as word embeddings: a `nn.Embedding(num_categories, dim)` layer trained jointly with the downstream task. Wins big for **high-cardinality** categoricals (zip codes, device IDs, merchant IDs).

#### Self-supervised pre-training on tabular data
Methods like TabNet, SAINT, VIME pre-train on unlabeled rows (masked feature reconstruction, contrastive views) then fine-tune on labels. Useful when labels are scarce — common in fraud, where confirmed-fraud labels lag by weeks.

#### Contrastive learning
Pull similar entities together in embedding space, push dissimilar ones apart. In fraud: same user across sessions → close; different users → far. The resulting user/device embeddings are reusable across downstream models (login risk, payment risk, account-takeover detection).

#### Transfer learning
For text / image content (KYC documents, selfies, support tickets), starting from pretrained encoders (BERT, ViT, CLIP) and fine-tuning a small head is almost always better than training from scratch.

## What's in this module

| File | What you'll see |
|------|-----------------|
| [`01_cross_validation_and_metrics.ipynb`](./01_cross_validation_and_metrics.ipynb) | Stratified K-Fold vs Group K-Fold vs Time Series Split on the same dataset. See how each shifts the score. |
| [`02_embeddings_for_categorical.ipynb`](./02_embeddings_for_categorical.ipynb) | Learned embeddings for `country` and `device_type` in a small PyTorch model, contrasted with one-hot encoding. |
| [`feature_importance.py`](./feature_importance.py) | Permutation importance — model-agnostic, more reliable than gain-based importance. |

## Likely interview questions

1. *Walk me through how you'd validate a fraud model in production.*
2. *I have a feature that's the user's average txn amount over the last 30 days. What's a subtle leakage risk?*
3. *When would you use learned embeddings vs target encoding vs one-hot for a high-cardinality categorical?*
4. *We have 100M unlabeled events and 50k labeled fraud cases. How would you exploit that?*
5. *Why is permutation importance more trustworthy than XGBoost's `gain`?*
