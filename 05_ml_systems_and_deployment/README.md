# 05 — Practical ML Systems & Deployment

> *Why this module matters in interviews:* Staff-level expectation is that you can ship and operate models, not just train them. Expect questions on serving latency, drift detection, A/B testing, and the boring-but-critical glue: feature stores, monitoring, rollback. The role description for staff roles will say "production-grade pipelines" — that's this module.

## Concepts

### The serving spectrum
| Pattern | Latency | When |
|---|---|---|
| **Batch scoring** | minutes-hours | Daily risk scores, model retraining, offline analytics |
| **Near-real-time (NRT)** | seconds | Periodic risk refresh, async fraud queues |
| **Real-time (RT) sync** | <100ms | Login risk, checkout decisioning |
| **Streaming** | <1s | Continuous transaction monitoring |

For fraud decisioning at the point of action, you almost always need **synchronous RT** with strict p99 latency budgets.

### Feature stores — why they exist
The training-serving skew problem: your training set computes features from historical batches; your prod system computes the same features from a streaming source. Slightly different logic = silent model degradation.

A feature store solves this by:
1. Defining each feature **once**, in a single transformation.
2. Materializing it to both an **offline** store (Parquet/BigQuery for training) and an **online** store (Redis/DynamoDB for serving).
3. Providing point-in-time correct historical reads ("what was this user's velocity at 3:14 PM last Tuesday?") — prevents temporal leakage.

Open source: Feast, Tecton (commercial), Hopsworks.

### Monitoring — what to track
| Layer | Examples |
|---|---|
| **Operational** | p50/p99 latency, error rate, request volume, model artifact version |
| **Data drift** | Feature distribution shifts (PSI, KS test, JS divergence) |
| **Prediction drift** | Distribution of model outputs over time |
| **Performance** | PR-AUC, precision@K, recall@K — *requires labels, often delayed* |
| **Business** | Approval rate, chargeback rate, analyst override rate |

The two layers most teams underbuild: **prediction drift** (catchable in real time) and **delayed performance** (most teams treat it as offline-only — wire it back to alerts).

### Drift detection — quick reference
- **PSI** (Population Stability Index): simple, interpretable. >0.1 watch, >0.25 alert.
- **KS test**: for continuous features; sensitive to large samples.
- **JS divergence**: symmetric, bounded; nice for dashboards.

### Deployment patterns
| Pattern | When |
|---|---|
| **Shadow mode** | New model runs in parallel without affecting decisions; compare to incumbent |
| **Canary** | Route X% traffic to new model; watch ops + business metrics |
| **A/B test** | Random split, measure causal lift; needs sufficient power |
| **Blue/green** | Two full environments; flip after validation |

For high-risk models (fraud, lending), **always start with shadow mode**. The cost of a bad rollout is high.

### MLOps essentials
- **Reproducible training**: pinned deps, fixed seeds, versioned data, versioned code.
- **Model registry**: every artifact has a unique ID, lineage, metrics, owner.
- **CI/CD for models**: tests on data contracts, training reproducibility, evaluation gates.
- **Rollback plan**: previous model is one config flip away.

## What's in this module

| File | What you'll see |
|------|-----------------|
| [`01_inference_service.py`](./01_inference_service.py) | FastAPI service that loads the artifact from module 01 and serves a `/score` endpoint with input validation. |
| [`02_batch_scoring.py`](./02_batch_scoring.py) | Batch scorer over a Parquet file with parallel chunking. |
| [`03_monitoring.py`](./03_monitoring.py) | PSI-based drift detector you can run on incoming features vs a training reference. |
| [`Dockerfile`](./Dockerfile) | Minimal Docker image for the inference service. |

## Run the inference service

```bash
# 1. Generate data and train the model (one-time)
python data/synthetic_fraud.py
python 01_classical_ml/train_pipeline.py

# 2. Start the API
uvicorn 05_ml_systems_and_deployment.01_inference_service:app --reload

# 3. Try it
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"account_age_days": 5, "txn_amount": 750.0, "velocity_1h": 8, \
       "device_entropy": 0.12, "email_risk": 0.85, "ip_country_mismatch": 1, \
       "device_type": "android", "country": "NG", \
       "noise_1": 0.1, "noise_2": -0.3, "noise_3": 50.0}'
```

## Likely interview questions

1. *Walk me through what happens between a `POST /score` request and the JSON response.*
2. *How would you detect that your fraud model is degrading **before** chargeback labels come in?*
3. *Your p99 latency just spiked from 30ms to 250ms. Where do you look?*
4. *You retrained the model. How do you roll it out safely?*
5. *Explain training-serving skew with a concrete example from this codebase.*
