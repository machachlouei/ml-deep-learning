# 04 — Model Selection & Representation Learning

> *Why this module matters in interviews:* Staff-level interviews probe **how you validate**, not just what model you pick. Model selection done wrong is the #1 way good models become bad products. Representation learning is the modern lever for unlocking gains beyond what hand-crafted features give you.

## Concepts

### Model Selection

#### Cross-validation strategies — pick the right one
| Strategy | When to use | Risk if misused | 2026 status in fraud |
|---|---|---|---|
| **K-Fold** | IID data, no time or grouping | Optimistic on temporal data | Mostly dead; sanity-check baseline only |
| **Stratified K-Fold** | Classification with imbalance | Doesn't fix temporal leakage | Useful only as a tiebreaker, never the final eval |
| **Group K-Fold** | Repeated entities (same user/device across rows) | Without it, you leak identity across folds | More critical than ever once embeddings get reused downstream |
| **Time Series Split** | Anything with a temporal label | Don't use random splits; you'll publish a non-shippable model | The default; 3–5 forward folds with 1-week–1-month eval windows |
| **Purged + embargoed CV** | Financial / temporal with target overlap | Overlapping label windows otherwise leak | Mainstream now: chargeback labels land 30–90 days late, so purging pending-label rows is mandatory |

**Heuristic for fraud problems**: time-based split first; group split if you have repeated entities; stratify only as a tiebreaker. In 2026 the standard is **group-then-time** (group by user_id, then time-split within), wired into a feature store with point-in-time joins (Feast, Tecton) so the leakage isn't reintroduced at serving time.

#### Hyperparameter search
- **Grid search**: exhaustive, slow, ignores interactions. Reserved for 2–3 critical knobs.
- **Random search**: typically beats grid for the same compute. Default for ≤30 trials on XGBoost/LightGBM/LR.
- **Bayesian optimization** (Optuna): production default when each trial costs >5 minutes. TPE sampler is the de facto choice; integrates with MLflow.
- **Hyperband / ASHA**: when you can afford many cheap trials and need early stopping; great for neural nets. Standard via Ray Tune or Optuna's Hyperband pruner.
- **Population-Based Training (PBT)**: when GPU headroom is plentiful and you want learning-rate schedules co-evolved with weights.
- **AutoML platforms** (SageMaker Autopilot, Vertex AI AutoML, H2O Driverless AI): used as a baseline in mid-tier teams, then hand-tuned to beat it.

The bottleneck in fraud HPO is usually **validation latency**, not training — re-scoring 50M historical transactions per trial dominates. Cache scored validation slices and only re-evaluate the threshold during the search.

#### Choosing the operating point
Models output scores; **decisions need thresholds.** Three common framings:

1. **Fixed-FPR**: pick the threshold that hits a target false-positive rate. Aligns with capacity constraints (review queue size). Still the default at most banks ("give me 5,000 alerts/day").
2. **Precision-at-K**: top-K most suspicious cases per day. Aligns with analyst throughput. Common for batch fraud queues and merchant-side platforms.
3. **Expected-loss minimization**: needs calibrated probabilities × cost matrix. The most defensible framing, but only works after you **calibrate** — isotonic regression or beta calibration is now the standard post-hoc step in production pipelines.

**New in 2026:** multi-objective threshold selection. Teams now optimize jointly over (cost, customer friction, fairness gap across demographic groups) because regulators ask. NSGA-II-style Pareto-front exploration is the becoming-standard tooling.

### Leakage — the senior interview probe
You will be asked to *find* leakage in a scenario, not just define it. Patterns to recognize:

- **Future information in features**: time-windowed aggregates that include the current row.
- **Label-derived features**: post-decision flags, override notes, downstream outcomes.
- **Target encoding without out-of-fold averaging**: classic mean-encoded category leakage.
- **Same-entity rows across train/test**: handles, devices, sessions.
- **Preprocessing fit on the full dataset**: scaling/imputation using test statistics.

**New 2026 patterns to watch for:**
- **Embedding leakage**: if you pretrain user/device embeddings on data overlapping the test window, your evaluation is compromised even when labels look clean.
- **Feature-store leakage**: feature stores make point-in-time joins easier *and* create novel leak shapes (a feature that depends on a row inserted milliseconds after the event of interest). Time-travel / replay audit tooling is now mandatory.
- **LLM-generated feature leakage**: when an LLM labels or featurizes training data, its pretraining may have seen the test cases — common with chargeback narratives that get scraped. See [`papers/2506.02703`](./papers/2506.02703_data_leakage_critique.pdf) for the field-wide critique.

### Representation Learning

Why we care: hand-crafted features have a ceiling. Learned representations transfer across tasks and capture interactions you wouldn't think to engineer.

#### Categorical embeddings
Replace one-hot encoding with a learned dense vector per category. Works the same as word embeddings: a `nn.Embedding(num_categories, dim)` layer trained jointly with the downstream task. Wins big for **high-cardinality** categoricals (zip codes, device IDs, merchant IDs, bank BINs, IP-AS numbers).

**2026 usage patterns** (in increasing order of sophistication):
1. **Trained jointly with the task** — the simple `nn.Embedding` setup. Still works.
2. **Pretrained then frozen** — train embeddings on unlabeled user/device sequences (skip-gram or contrastive), then plug into many downstream models. Reusable across fraud, churn, recommendation.
3. **Pretrained then fine-tuned** — middle ground; most common in production. Modern deep-tabular models (FT-Transformer, TabPFN, SAINT) default to this.

#### Self-supervised pre-training on tabular data
Methods like TabNet, SAINT, VIME pre-train on unlabeled rows (masked feature reconstruction, contrastive views) then fine-tune on labels. Useful when labels are scarce — common in fraud, where confirmed-fraud labels lag by weeks.

**The field moved on in 2024–2026.** What's relevant today:
- **TabPFN** — in-context learning for tabular data; runs inference without per-task training. Surprisingly competitive on small labeled sets (<10k rows) — useful for new merchant onboarding or synthetic-ID detection.
- **CARTE, XTab** — cross-table pretraining; one model that generalizes across schemas.
- **CT-BERT-style masked column reconstruction** — MLM, but for tables.

The classic SSL win in fraud is unchanged: *millions of unlabeled events + thousands of confirmed-fraud labels.* Pretrain on the full stream (masked feature reconstruction or next-event prediction), fine-tune the labeled head. Typical lift over a from-scratch XGBoost is +2–5 PRAUC on hard fraud subtypes.

#### Contrastive learning
Pull similar entities together in embedding space, push dissimilar ones apart. In fraud: same user across sessions → close; different users → far. The resulting user/device embeddings are reusable across downstream models (login risk, payment risk, account-takeover detection).

**Probably the most under-used technique in production fraud teams.** Concrete 2026 use cases:
- **Synthetic identity detection** — embed (name, address, email, phone) tuples; near-duplicate clusters in embedding space are usually one fraudster running many applications.
- **Account takeover** — pre-attack and post-attack session embeddings drift apart even when no individual feature is anomalous.
- **Merchant scam detection** — embed merchant description + product images via SigLIP; cluster against known scam templates.
- **Behavioral sequence embeddings** — SimCLR / BYOL / DINO-style training on session traces; identifies "this device behaves like this other device" without labels.

#### Transfer learning
For text / image content (KYC documents, selfies, support tickets), starting from pretrained encoders (BERT, ViT, CLIP) and fine-tuning a small head is almost always better than training from scratch.

**The specific encoders refreshed in 2024–2026:**

| Modality | 2020-era go-to | 2026 go-to | Fraud use case |
|---|---|---|---|
| Text (general) | BERT-base | **ModernBERT-base** (8192 context, faster) | Support tickets, chargeback narratives |
| Text (embeddings) | Sentence-BERT | **E5 / BGE / GTE** | Similarity search, retrieval for analyst review |
| Text (multilingual) | mBERT | **XLM-R** or **Cohere Embed v3 multilingual** | Cross-region KYC, multilingual scams |
| Images (general) | ResNet-50 / ViT-B | **DINOv2** | Selfie liveness, document layout |
| Vision-language | CLIP | **SigLIP-2** | Zero-shot doc-type classification, merchant scam detection |
| Domain-specific | n/a | **FinBERT, LegalBERT, BioBERT** | Start here when the domain matches |

**Parameter-efficient fine-tuning (PEFT)** is the single biggest practical change since this module was first written. LoRA / QLoRA / adapters let you fine-tune a 1B-parameter encoder on the compute budget of a small CNN. Workflow: frozen encoder + linear head (1 hour) → LoRA fine-tune (half a day) → full fine-tune only if you have >100k labels and LoRA falls short.

For the empirical evidence behind this section — what works, what doesn't, where the leakage hides — see the annotated bibliography in [`papers/`](./papers/).

### Embedding stores — the operational lever

Representation learning isn't useful in production until the embeddings can be served. The under-emphasized bottleneck in 2026 fraud-ML programs isn't model quality — it's whether you can:

1. **Compute** embeddings in real time for new entities (fast tokenizer + cached encoder).
2. **Look them up** by ID with millisecond latency (vector DB, Redis, or in-memory store).
3. **Refresh** them on a schedule without breaking downstream models (versioning + shadow scoring).

Teams that get the *representation* right but skip the *store* end up with great embeddings that never reach a decision. Worth treating as a first-class engineering concern alongside the model itself.

## What's in this module

| File | What you'll see |
|------|-----------------|
| [`01_cross_validation_and_metrics.ipynb`](./01_cross_validation_and_metrics.ipynb) | Stratified K-Fold vs Group K-Fold vs Time Series Split on the same dataset. See how each shifts the score. |
| [`02_embeddings_for_categorical.ipynb`](./02_embeddings_for_categorical.ipynb) | Learned embeddings for `country` and `device_type` in a small PyTorch model, contrasted with one-hot encoding. |
| [`feature_importance.py`](./feature_importance.py) | Permutation importance — model-agnostic, more reliable than gain-based importance. |
| [`papers/`](./papers/) | Annotated bibliography of 9 arXiv papers on Transformer / LLM use in fraud detection — surveys, methodological critique, and empirical models, with usage / strengths / weaknesses per paper. |

## Likely interview questions

1. *Walk me through how you'd validate a fraud model in production.*
2. *I have a feature that's the user's average txn amount over the last 30 days. What's a subtle leakage risk?*
3. *When would you use learned embeddings vs target encoding vs one-hot for a high-cardinality categorical?*
4. *We have 100M unlabeled events and 50k labeled fraud cases. How would you exploit that?*
5. *Why is permutation importance more trustworthy than XGBoost's `gain`?*
