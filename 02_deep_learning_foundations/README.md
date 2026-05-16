# 02 — Deep Learning Foundations

> *Why this module matters in interviews:* Even on tabular fraud problems where you'll deploy XGBoost, you'll be asked to **explain neural networks fluently** — forward pass, backprop, why batch norm helps, when DL actually beats GBMs. This module is a brisk refresher with two demos.

## Concepts

### The neuron and the forward pass
A single layer computes `y = activation(W·x + b)`. Stack layers, and you have an MLP. Universal approximation says a sufficiently wide one-hidden-layer net can approximate any continuous function — but depth gives you compositional features that wide-shallow nets need exponentially more parameters to match.

### Backpropagation
Just the chain rule on the computation graph. The gradient of the loss w.r.t. any parameter is computed by:
1. **Forward pass:** evaluate the network, caching intermediate activations.
2. **Backward pass:** propagate ∂L/∂output backwards through each op using cached values.

The right mental model: backprop is *not* a learning algorithm. It's an efficient way to compute gradients. **Gradient descent is the learning algorithm.**

### Activations
| Function | When to use | Gotcha |
|---|---|---|
| **ReLU** | Default for hidden layers | "Dying ReLU" — neurons stuck at 0. Use LeakyReLU / GELU if it bites. |
| **GELU** | Modern default, esp. transformers | Slightly more compute than ReLU. |
| **Sigmoid** | Binary output head | Saturating gradients — never use in hidden layers. |
| **Softmax** | Multi-class output head | Numerically combine with cross-entropy (`logsoftmax`) for stability. |

### Optimizers
| Optimizer | Notes |
|---|---|
| **SGD + momentum** | Simple, well-understood, often the best generalizer with proper schedule. |
| **Adam / AdamW** | Adaptive per-parameter learning rates. AdamW decouples weight decay from the gradient — preferred for transformers. |
| **RMSProp** | Largely superseded by Adam. |

Rule of thumb: **AdamW with cosine LR schedule and warmup** is a safe default for modern deep nets. SGD+momentum with careful tuning can outperform on vision.

### Regularization
- **Weight decay (L2)** — universally useful, ~1e-4 to 1e-2.
- **Dropout** — randomly zero activations during training. Tabular: 0.1–0.3; vision: 0.0–0.1; transformers: 0.0–0.1.
- **Batch norm / Layer norm** — normalize activations. BN reduces internal covariate shift and acts as a regularizer; LN is the standard for transformers (works for any batch size).
- **Early stopping** — stop when val loss stops improving. Free and very effective.
- **Data augmentation** — domain-specific (rotations for vision, dropout for tabular features, etc.).

### Why GBMs usually beat DL on tabular data
1. Tabular features are already pre-engineered; DL's main strength is *learning* features, which is wasted here.
2. Trees handle mixed types and missingness natively; nets require careful preprocessing.
3. Sample efficiency: GBMs need less data per dollar of accuracy.
4. Interpretability and ops simplicity.

**When DL wins on tabular:**
- Very large datasets (10M+ rows) with rich interactions.
- Multi-task / multi-modal setups (fraud signal + text claim + image).
- You need learned representations (embeddings) that downstream systems can reuse — e.g., user embeddings for cold-start cases.

## What's in this module

| File | What you'll see |
|------|-----------------|
| [`01_neural_net_from_scratch.ipynb`](./01_neural_net_from_scratch.ipynb) | NumPy-only 2-layer MLP with hand-derived backprop. Sanity check against PyTorch autograd. |
| [`02_backprop_and_optimizers.ipynb`](./02_backprop_and_optimizers.ipynb) | Compare SGD, SGD+momentum, Adam, AdamW on a toy fraud subset. Visualize loss curves. |
| [`tabular_mlp.py`](./tabular_mlp.py) | A clean PyTorch MLP for the fraud dataset with batch norm, dropout, weight decay, early stopping. |

## Likely interview questions

1. *Derive the gradient of binary cross-entropy with a sigmoid output. Why is it numerically nicer than computing sigmoid then BCE separately?*
2. *What's the difference between Adam and AdamW, and why does it matter?*
3. *Why does batch norm work? What changes when you switch to layer norm?*
4. *On a 100K-row tabular fraud problem, would you reach for a neural net or XGBoost? Defend.*
5. *I see your loss is decreasing but accuracy is flat. What's likely happening?*
