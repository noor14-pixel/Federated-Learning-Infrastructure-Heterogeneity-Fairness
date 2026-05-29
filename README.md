# 🌐 Federated Learning: Infrastructure, Heterogeneity & Fairness

> A from-scratch implementation of federated learning — no FL library used. Covers FedAvg, FedProx, client drift analysis, real-world distracted driver detection, and a custom fairness-aware aggregation algorithm.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20%7C%20Colab-20BEFF?style=flat-square)

---

## 📌 Overview

This project builds a complete federated learning system from the ground up, then stress-tests it under realistic data heterogeneity. Four parts progress from theory to real-world deployment:

| Part | Focus |
|------|-------|
| **A** | FL infrastructure: data partitioning, server–client loop, convergence verification |
| **B** | FedProx, client drift anatomy, training diagnostics (sharpness, gradient divergence) |
| **C** | Real-world FL on the State Farm Distracted Driver Detection (SFD3) dataset |
| **D** | Custom fairness-aware aggregation to improve worst-client accuracy |

---

## Part A — FL Infrastructure from Scratch

### A1 · Data Partitioning

Three heterogeneity regimes implemented on CIFAR-10 (10% fraction, 10 clients):

| Regime | Description | Size range |
|--------|-------------|------------|
| **IID** | Uniform shuffle and equal split | 500 each |
| **Dirichlet non-IID** (α = 0.5) | Class allocations drawn from Dirichlet; realistic label skew | 272 – 661 |
| **Quantity skew** | Sizes from LogNormal(0, 1); extreme imbalance, balanced within-client classes | 61 – 1746 |

### A2 · Server–Client Communication

Full FedAvg built from scratch:

- `FederatedClient.local_train(E, lr, mu)` — E epochs of SGD with optional FedProx proximal term `(μ/2)‖w − w_global‖²`
- `FederatedServer.broadcast()` — deep-copies global weights to selected clients
- `FederatedServer.aggregate()` — weighted FedAvg over `(state_dict, n_samples)` pairs
- Per-round logging: weight norms, gradient norms, upload payload in bytes

### A3 · Gradient-Averaging Equivalence Verification

Numerically verified on a 1-D quadratic system (K=5 clients, `F_k(w) = (aₖ/2)(w − wₖ*)²`):

> **With E=1, FedAvg is bit-exact with centralised gradient descent**
> `max |w_FedAvg − w_cent| < 1e-10` across all 50 rounds

Divergence grows monotonically with E — at E=10 the drift plateaus near 1e-1 and never recovers to the true global minimiser, demonstrating the client drift problem empirically.

### A4 · Instrumentation Dashboard

30 rounds of FedAvg on the Dirichlet partition (E=5, lr=0.01):

```
Panel 1 · Global test accuracy vs round
Panel 2 · Per-client weight norm (mean ± 1 std)
Panel 3 · Per-client gradient norm (mean ± 1 std) — peaks early, decays as model converges
Panel 4 · Cumulative upload payload (annotated total MB)
```

---

## Part B — FedProx, Drift & Diagnostics

### B1 · FedProx μ Sweep

Sweep over **μ ∈ {0, 0.01, 0.1, 1, 10}** — Dirichlet partition, E=10, T=30 rounds.

**1-D bias analysis:**

| Limit | Behaviour | Bias |
|-------|-----------|------|
| μ → 0 | Drifts toward weighted avg of local minimisers | 0.024 |
| μ → ∞ | Frozen at initialisation — learns nothing | 1.67 |

Neither extreme is desirable. The proximal term trades off bias against variance reduction.

### B2 · Client Drift Anatomy

Per-client drift `‖wₖ(t) − w_global(t)‖` over 20 rounds, FedAvg vs FedProx (μ=0.1):

- FedProx reduces mean drift: **2.45 → 1.93**
- Cosine similarity heatmaps (clients × rounds) reveal outlier clients updating in opposing directions — most pronounced under FedAvg
- Gradient divergence `∑ₖ pₖ ‖∇Fₖ(w) − ∇F(w)‖²` tracked each round; FedProx slightly higher (22.0 vs 19.95) as the proximal term reshapes local loss landscapes

### B3 · Training Diagnostics

Recorded every 5 rounds for FedAvg and FedProx:

| Metric | What it measures |
|--------|-----------------|
| **Gradient divergence** | Heterogeneity-induced disagreement between client and global gradients |
| **Per-client loss variance** | How unevenly the global model fits different clients |
| **Sharpness** | Dominant Hessian eigenvalue via power iteration — peaks ~round 10, then decays |

> FedProx is slightly flatter on average (mean eigenvalue **157 vs 182**), suggesting modest convergence to wider minima.

---

## Part C — Real-World FL: Distracted Driver Detection

**Dataset:** State Farm SFD3 — 10 distraction classes across 24 drivers, partitioned by driver identity (20 train clients, 4 test drivers).

### C1 · Centralised Baseline

- ResNet-18 (ImageNet pretrained) fine-tuned on pooled training data
- **Centralised ceiling: 84.8% test accuracy**
- Driver-level partitioning is a realistic heterogeneity model: each driver is recorded performing a subset of behaviors at varying frequencies

### C2 · Federated Fine-Tuning

10 communication rounds, E=5 local epochs, ResNet-18 backbone:

| Metric | FedAvg | FedProx (μ=0.1) |
|--------|--------|-----------------|
| Global test accuracy | 82.8% | — |
| **Worst-driver val accuracy** | 37.1% | **67.1%** |
| Δ vs centralised ceiling | −2.0pp | — |

FedProx nearly doubles worst-driver accuracy (D7: 37% → 67%), confirming proximal regularisation protects minority clients from being overwhelmed by dominant data holders.

### C3 · GradCAM Analysis

Applied to centralised model, FedAvg global model, best client (D0), and worst client (D7) across 3 classes:

- 🟢 **Centralised ≈ FedAvg global** — near-identical spatial attention (steering wheel for c0, right hand for c1, face area for c6)
- 🟢 **Best driver D0** — tight, discriminative heatmaps; closely mirrors centralised model
- 🔴 **Worst driver D7** — diffuse attention spreading to seat and background; evidence of sparse or class-imbalanced local data

---

## Part D — Fairness-Aware Aggregation

### Motivation

Standard FedAvg minimises the weighted-average loss `F(w) = ∑ₖ pₖ Fₖ(w)`, naturally favouring clients with more data (`pₖ = nₖ/n`). This systematically underserves clients with small or unusual distributions.

### Algorithm: `FairServer`

Exponentiated gradient update on the aggregation weight simplex with uniform mixing:

```
q_k^(t+1)  ∝  q_k^(t) · exp(η_q · F_k(w^(t)))
p_k  =  (1 − α) · q_k  +  α/K
```

High-loss clients receive more weight next round. Uniform mixing (α=0.1) prevents collapse to a single dominant client. Hyperparameters: **η_q = 2.0, α = 0.1**.

### Results (T=30, Dirichlet CIFAR-10, E=5)

| Metric | FedAvg | FedProx | **FairAgg** |
|--------|--------|---------|-------------|
| Mean client accuracy | 0.89 | — | **0.89** |
| Worst-client accuracy | 0.76 | — | **0.85 (+9pp)** |
| Client acc variance | 0.0042 | — | **0.0005 (↓88%)** |

**Verification — all three conditions pass:**
- ✅ Worst-client accuracy improves by ≥ 2pp over FedAvg
- ✅ Per-client accuracy variance strictly lower than FedAvg
- ✅ Mean accuracy within 5pp of FedAvg

### The Privacy–Fairness–Accuracy Trilemma

Improving worst-client accuracy requires evaluating per-client losses each round — which leaks information about local distributions. A client with consistently high loss implicitly reveals it belongs to a minority or unusual group. Differential privacy mitigations (gradient clipping, noise injection) would weaken exactly the signal FairAgg depends on.

> In a real deployment, **average accuracy should be sacrificed first**. The core motivation for federated learning is often to serve users who cannot share data precisely because they are vulnerable or marginalised. Worst-client accuracy and variance reduction should be primary objectives; privacy a non-negotiable constraint.

---

## 🛠️ Setup

```bash
pip install torch torchvision matplotlib seaborn scikit-learn pandas pillow
```

- **Parts A–B:** CIFAR-10 (auto-downloaded via `torchvision`)
- **Part C:** [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) dataset (Kaggle)
- **Hardware:** GPU strongly recommended — tested on Kaggle T4

---

## 📁 Repository Structure

```
├── pa6_2.ipynb              # Full notebook: Parts A–D
│
├── a3_divergence.png        # FedAvg vs centralised GD convergence plot
├── a4_dashboard.png         # 4-panel instrumentation dashboard
├── b1_mu_sweep.png          # FedProx μ sweep (3-panel)
├── b2_drift.png             # Client drift + cosine similarity heatmaps
├── b3_dashboard.png         # Training diagnostics dashboard
├── c2_sfd3_results.png      # SFD3 FL results (accuracy + per-driver bar)
├── c3_gradcam.png           # GradCAM grid (4 models × 3 classes)
└── d2_comparison.png        # FairAgg vs FedAvg vs FedProx (3-panel)
```

---

## 📚 References

- McMahan et al. (2017) — [*Communication-Efficient Learning of Deep Networks from Decentralized Data*](https://arxiv.org/abs/1602.05629) (FedAvg)
- Li et al. (2020) — [*Federated Optimization in Heterogeneous Networks*](https://arxiv.org/abs/1812.06127) (FedProx)
- Selvaraju et al. (2017) — [*Grad-CAM: Visual Explanations from Deep Networks*](https://arxiv.org/abs/1610.02391)
- Cho & Hariharan (2019) — [*On the Efficacy of Knowledge Distillation*](https://arxiv.org/abs/1910.01348)
