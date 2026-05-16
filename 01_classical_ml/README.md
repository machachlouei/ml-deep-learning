# 01 — Classical ML with scikit-learn & XGBoost

> *Why this module matters in interviews:* For tabular fraud / risk data, gradient-boosted trees still beat deep learning on most problems. Senior interviews expect fluency in **logistic regression as a baseline**, **GBMs as the workhorse**, and the surrounding plumbing (calibration, imbalance, leakage).

## Concepts

### Logistic Regression
A linear model on the log-odds of the positive class. Trained by maximizing log-likelihood (equivalently, minimizing cross-entropy).

**Pros**
- Cheap to train and serve, fully interpretable (coefficients = log-odds contributions).
- Probabilities are reasonably well-calibrated out of the box.
- Strong baseline; if a complex model can't beat regularized LR, that's a signal to fix data, not architecture.

**Cons**
- Cannot capture interactions or non-linearities without manual feature engineering.
- Sensitive to feature scale and outliers.
- Multicollinearity destabilizes coefficients (use L2 / elastic net).

**When to reach for it**
- Need explainability for regulators or model risk reviewers.
- Small data or very high-dimensional sparse data (e.g., text bag-of-words).
- As the **always-include baseline** in any modeling exercise.

### Gradient-Boosted Trees (XGBoost, LightGBM, CatBoost)
Sequential ensembles of shallow trees, each fitting the gradient of the loss w.r.t. the previous ensemble's prediction. Today's default for tabular data.

**Pros**
- Handles mixed types, missing values, and non-linearities natively.
- State-of-the-art on tabular benchmarks; few-shot tuning gets you 90% of the way.
- Built-in feature importance and partial dependence are interview-friendly explanations.

**Cons**
- Probabilities are **not well-calibrated** — always check & calibrate before using as risk scores.
- Many hyperparameters; easy to overfit small data.
- Slower inference than a linear model — matters at high QPS.

**When to reach for it**
- Tabular data with mixed numeric/categorical features.
- Class imbalance (combine with `scale_pos_weight` or downsampling).
- Any time you don't have a strong reason to use deep learning.

### Class imbalance — the fraud-specific concern
Fraud is typically 0.5–3% of traffic. Plain accuracy is useless. What works:

1. **Pick the right metric.** PR-AUC, recall at fixed FPR, or precision at top-K (operationally most useful for case review queues).
2. **Resample carefully.** Random undersampling is the simplest fix; SMOTE is usually a trap (it interpolates in feature space and creates leaky-looking synthetic positives). Prefer class weighting first.
3. **Calibrate after.** If you resample, your model's predicted probabilities will be biased — fix with Platt scaling or isotonic regression on a held-out set.

### Data leakage — the silent killer
The #1 reason a "great" fraud model dies in prod. Watch for:
- **Temporal leakage**: training on features computed *after* the label time. Always split by time.
- **Group leakage**: same user/device in train and test. Use grouped splits.
- **Target leakage**: a feature that's only populated post-decision (e.g., a chargeback flag).

## What's in this module

| File | What you'll see |
|------|-----------------|
| [`01_logistic_regression_fraud.ipynb`](./01_logistic_regression_fraud.ipynb) | LR baseline end-to-end: preprocessing, class weighting, PR-AUC, coefficient interpretation. |
| [`02_xgboost_fraud_detection.ipynb`](./02_xgboost_fraud_detection.ipynb) | XGBoost on the same data; tuning, early stopping, calibration, threshold selection at fixed FPR. |
| [`train_pipeline.py`](./train_pipeline.py) | Production-shaped sklearn `Pipeline` with persisted preprocessor + model, ready for module 05's serving code. |

## Likely interview questions

1. *Why might XGBoost beat a deep net on tabular fraud data?*
2. *What's wrong with using accuracy or ROC-AUC on a 1% fraud problem?*
3. *Walk me through how you'd calibrate a model whose probabilities are used to set review queue thresholds.*
4. *I have a feature `account_age_days_since_last_chargeback`. What's wrong with that feature?*
5. *When would you choose logistic regression over a GBM in production?*
