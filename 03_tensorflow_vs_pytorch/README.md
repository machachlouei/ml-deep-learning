# 03 — TensorFlow & PyTorch

> *Why this module matters in interviews:* You'll be asked which you'd use and why. The honest answer in 2026 is **"PyTorch by default, TensorFlow if I inherit it or need TF Serving / TFX,"** but you should be able to defend that and translate between the two.

## Concepts

### The frameworks at a glance

|  | **PyTorch** | **TensorFlow / Keras** |
|---|---|---|
| **Execution** | Eager by default; `torch.compile` for graphs | Eager by default; `tf.function` for graphs |
| **Authoring** | Pythonic, define-by-run | Keras high-level API; functional/sequential/subclassed |
| **Ecosystem** | Research-dominant; HuggingFace, Lightning, vLLM | Production-mature; TFX, TF Serving, TFLite, TF.js |
| **Distributed** | DDP, FSDP, DeepSpeed | `tf.distribute.Strategy` |
| **Mobile / edge** | ExecuTorch, ONNX → CoreML | TFLite (most mature for mobile) |
| **Industry trend** | Dominant for new LLM / GenAI work | Strong in established prod stacks, GCP, mobile |

### When to pick which

**PyTorch when:**
- New project, modern stack (transformers, diffusion, RLHF).
- Hiring from ML/research talent pools.
- You want the most active ecosystem for GenAI tooling.

**TensorFlow / Keras when:**
- Existing TF pipeline you're not replacing.
- Need TFLite for mobile / on-device inference.
- Heavy GCP + Vertex AI integration.
- Strict TFX-style production pipelines.

### Things that look different but aren't
- **Autograd**: same idea, different syntax. PyTorch: `.backward()`. TF: `tf.GradientTape`.
- **Layers**: `nn.Linear` ↔ `keras.layers.Dense`. PyTorch is `(in, out)`; Keras infers `in`.
- **Training loop**: PyTorch makes you write it (good for control); Keras gives you `model.fit()` (good for speed).
- **Data**: `torch.utils.data.DataLoader` ↔ `tf.data.Dataset`.

### The interoperability tools
- **ONNX** — common intermediate format. PyTorch → ONNX → TF, or PyTorch → ONNX → ONNX Runtime / CoreML / TensorRT.
- **HuggingFace Transformers** — same model checkpoints, dual-implemented in PT and TF (PT increasingly the only one).

## What's in this module

| File | What you'll see |
|------|-----------------|
| [`01_pytorch_fraud_mlp.ipynb`](./01_pytorch_fraud_mlp.ipynb) | Same model as the systems file in module 02, in notebook form. |
| [`02_tensorflow_fraud_mlp.ipynb`](./02_tensorflow_fraud_mlp.ipynb) | The exact same architecture, in Keras. Side-by-side comparison. |
| [`benchmark.py`](./benchmark.py) | Toy training-time benchmark between PT and TF on identical architecture. |

## Likely interview questions

1. *Which framework would you pick for a new fraud-decisioning project and why?*
2. *Explain `tf.function` and `torch.compile`. When do they help, when do they hurt?*
3. *You have a PyTorch model and need to ship it to iOS. Walk me through your options.*
4. *Compare `DataLoader` with `num_workers > 0` to `tf.data.Dataset` with `prefetch`. Where do they diverge?*
5. *Why is `BCEWithLogitsLoss` (or `from_logits=True` in Keras) preferable to applying sigmoid then BCE separately?*
