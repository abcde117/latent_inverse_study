
---

# Neural Latent Mapping Experiment

**Bidirectional Latent Modeling with Encoder / Decoder / Inverse Encoder**

This project explores a bidirectional latent framework designed to reinterpret the latent representation *z* as a **structural variable** connecting input *x* and output *y*.
Instead of relying solely on standard end-to-end training (x → z → y), the model introduces an inverse path and studies how training order affects representation quality and model behavior.

---

## Overview

Traditional neural networks treat latent space as an opaque intermediate.
This project instead aims to:

* Align latent representations from both directions (**f(x) ≈ g(y)**)
* Use a shared Decoder to unify forward and inverse mappings
* Interpret x → y as a **measure transformation** mediated by latent space
* Improve interpretability by decoupling x-side and y-side processing
* Compare different training flows that combine Encoder, Decoder, and Inverse Encoder

This framework resembles an autoencoder structure but focuses on **x–y structural understanding**, not self-reconstruction.

---

## Project Components

* **exp.ipynb** – Demonstrates the full experimental flow, including both paths, training order comparisons, and evaluation.
* **modular_nn_experiment1/** – Contains modular Python implementations (MiniViT, MiniUNet, MLP, PCA+MLP), experiment notebooks, and recorded results.
* **idea_explanation.pdf** – Detailed theoretical and architectural explanation.
* **idea_summary.pdf** – Concise summary of concepts and findings (basis for this README).

---

## Method

### Bidirectional Structure

* **Forward:** x → Encoder → z → Decoder → y
* **Inverse:** y → Inverse Encoder → z → Decoder → y

Both flows share the same Decoder, encouraging structural consistency in the latent space.

### Two Training Flows

**Path 1**

1. Train Encoder + Inverse Encoder
2. Train Decoder
3. Fine-tune all modules

**Path 2**

1. Train Inverse Encoder + Decoder
2. Freeze Inverse Encoder
3. Train Encoder
4. Fine-tune all modules

**Path 2 performs best overall** and frequently exceeds end-to-end training.

---

## Key Results

### Main Findings

* Joint training of all three modules is unstable and leads to poor loss reduction.
* Training order significantly impacts model behavior.
* Path 2 achieves **equal or better** performance compared to end-to-end.
* An overly strong early Decoder makes Encoder optimization difficult, but final loss may still improve.
* ViT and MLP often show **no statistically significant difference** from end-to-end (t-test & KS test).
* Decoder design contributes more to performance than minimizing latent alignment losses.

---

## Models

The project evaluates multiple architectures:

| Model        | Characteristics                          |
| ------------ | ---------------------------------------- |
| **MiniViT**  | Global attention, strong structural bias |
| **MiniUNet** | Local convolutional structure            |
| **MLP**      | Dense compression                        |
| **PCA+MLP**  | Explicit low-dimensional basis + MLP     |

Implementation uses:

* **jaxtyping** (type safety)
* **einops** (tensor manipulation)
* Custom **Dense** layers for consistent initialization
* Fourier time embeddings + label embeddings for conditioning

---

## Experimental Highlights

| Model    | End-to-End Loss | Path 2 Loss | End-to-End Cos | Path 2 Cos |
| -------- | --------------- | ----------- | -------------- | ---------- |
| MiniViT  | 1136.6          | **1128.6**  | 0.640          | **0.643**  |
| MiniUNet | 1177.4          | **1134.8**  | 0.623          | **0.640**  |
| MLP      | 1172.8          | **1161.6**  | 0.625          | **0.629**  |
| PCA+MLP  | 1250.3          | 1256.2      | 0.594          | 0.590      |

**Path 2 consistently offers improved or comparable representation quality.**

---

## Summary

This project provides a modular framework for analyzing latent representations through bidirectional mapping.
Key contributions:

* A unified latent space jointly reachable from both x and y
* Structural interpretation of latent variables
* Demonstration that **non-end-to-end flows can outperform end-to-end**
* Modular, interpretable architectures suitable for analysis and future extensions

The results suggest that latent-focused, flow-dependent training offers a promising alternative for understanding and improving neural representation learning.

---

