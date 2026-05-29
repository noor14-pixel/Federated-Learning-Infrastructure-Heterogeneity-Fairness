# Federated-Learning-Infrastructure-Heterogeneity-Fairness
A from-scratch implementation of federated learning covering FedAvg, FedProx, client drift analysis, real-world deployment on a distracted driver detection dataset, and a custom fairness-aware aggregation algorithm.

Overview
This project builds a complete federated learning system without any FL library, then stress-tests it under realistic data heterogeneity conditions. Four parts progress from theory to real-world application:
PartFocusAFL infrastructure from scratch: data partitioning, server–client loop, convergence verificationBFedProx, client drift anatomy, training diagnostics (sharpness, gradient divergence)CReal-world FL on State Farm Distracted Driver Detection (SFD3) with GradCAM analysisDCustom fairness-aware aggregation to improve worst-client accuracy

Part A — FL Infrastructure from Scratch
A1: Data Partitioning (DatasetPartitioner)
Implements three heterogeneity regimes on CIFAR-10 (10% fraction, 10 clients):

IID — uniform shuffle and split; all clients get equal samples across all classes
Dirichlet non-IID (α = 0.5) — class allocations drawn from Dirichlet distribution; produces realistic label skew across clients
Quantity skew — client sizes drawn from LogNormal(0, 1); extreme size imbalance (min: 61, max: 1746 samples) with balanced within-client class distribution

A2: Server–Client Communication (FederatedServer, FederatedClient)
Full FedAvg implementation with:

FederatedClient.local_train(E, lr, mu) — E epochs of SGD with optional FedProx proximal term (μ/2)‖w − w_global‖²
FederatedServer.broadcast() — deep-copies global model to selected clients
FederatedServer.aggregate() — weighted FedAvg over (state_dict, n_samples) pairs
Per-round logging of weight norms, gradient norms, and upload payload in bytes

A3: Gradient-Averaging Equivalence Verification
Proves numerically on a 1-D quadratic system (K=5 clients, F_k(w) = (aₖ/2)(w − wₖ*)²) that:

With E=1, FedAvg is bit-exact with centralised gradient descent (max |w_FedAvg − w_cent| < 1e-10)

Divergence grows monotonically with E — at E=10 the drift plateaus around 1e-1 and never recovers to the true global minimiser.
A4: Instrumentation Dashboard
30 rounds of FedAvg on Dirichlet partition (E=5, lr=0.01). 4-panel dashboard:

Global test accuracy vs round
Mean per-client weight norm ± std
Mean per-client gradient norm ± std (peaks early then decays as model converges)
Cumulative upload payload (annotated total MB)


Part B — FedProx, Drift, and Diagnostics
B1: FedProx μ Sweep
Sweep over μ ∈ {0, 0.01, 0.1, 1, 10} on Dirichlet partition, E=10, T=30 rounds.
1-D bias analysis:

μ → 0: aggregate drifts toward weighted average of local minimisers (small but nonzero bias = 0.024)
μ → ∞: model frozen at initialisation (bias = 1.67, learns nothing)
Neither extreme is desirable — the proximal term trades bias for variance reduction

B2: Client Drift Anatomy
Per-client drift ‖wₖ(t) − w_global(t)‖ over 20 rounds for FedAvg vs FedProx (μ=0.1):

FedProx reduces mean drift from 2.45 to 1.93
Cosine similarity heatmaps (clients × rounds) reveal clients with outlier label distributions update in opposing directions — most visible under FedAvg
Gradient divergence ∑ₖ pₖ ‖∇Fₖ(w) − ∇F(w)‖² measured each round; FedProx slightly higher (22.0 vs 19.95) because the proximal term changes local loss landscapes

B3: Training Diagnostics Dashboard
Diagnostic metrics recorded every 5 rounds for both algorithms:

Gradient divergence — measures heterogeneity-induced disagreement between client and global gradients
Per-client loss variance — tracks how unevenly the global model fits different clients
Sharpness — dominant Hessian eigenvalue estimated via power iteration; peaks ~round 10 then decays for both algorithms; FedProx slightly flatter on average (157 vs 182 mean eigenvalue)


Part C — Real-World FL: Distracted Driver Detection (SFD3)
Dataset: State Farm Distracted Driver Detection — 10 distraction classes (safe driving, texting, phone use, radio, drinking, etc.), partitioned by driver identity (20 train drivers, each = one FL client).
C1: Centralised Baseline

ResNet-18 (ImageNet pretrained) fine-tuned on pooled data
Centralised ceiling: 84.8% accuracy
Driver-level partitioning is a realistic FL heterogeneity model: in practice each driver's data stays on their device; label skew is mild in the standard split (all 10 classes per driver) but becomes significant under label-sparse partitioning

C2: Federated Fine-Tuning (FedAvg vs FedProx)
10 communication rounds, E=5 local epochs, ResNet-18 pretrained backbone.
MetricFedAvgFedProxGlobal test accuracy82.8%—Worst-driver val accuracy37.1%67.1%Δ vs centralised ceiling−2.0pp—
FedProx dramatically improves the worst-performing driver (D7: 37% → 67%), confirming proximal regularisation protects minority clients from being drowned out by dominant data holders.
C3: GradCAM Analysis
GradCAM applied to centralised model, FedAvg global model, best client (D0), and worst client (D7) across 3 classes.

Centralised and FedAvg global models attend to near-identical spatial regions (steering wheel for c0, right hand for c1, face/hand area for c6)
Best client D0 heatmaps are tight and discriminative — closely mirrors the centralised model
Worst client D7 shows diffuse, non-discriminative attention spreading to background and seat regions — evidence of insufficient or imbalanced training data


Part D — Fairness-Aware Aggregation (FairServer)
Motivation
Standard FedAvg minimises the weighted average loss, naturally favouring clients with more data. This systematically underserves clients with small or unusual distributions.
Algorithm
Exponentiated gradient update on the aggregation weight simplex, mixed with uniform distribution:
q_k^(t+1) ∝ q_k^(t) · exp(η_q · F_k(w^(t)))
p_k = (1 − α) · q_k + α/K
High-loss clients receive more weight next round; uniform mixing (α=0.1) prevents any single client from dominating. Hyperparameters: η_q = 2.0, α = 0.1.
Results (T=30, Dirichlet CIFAR-10, E=5)
MetricFedAvgFedProxFairAggGlobal test accuracy——comparableMean client accuracy0.89—0.89Worst-client accuracy0.76—0.85 (+9pp)Client acc variance0.0042—0.0005 (↓88%)
FairAgg passes all three verification conditions:

Worst-client accuracy improves by ≥ 2pp over FedAvg ✓
Per-client accuracy variance strictly lower than FedAvg ✓
Mean accuracy within 5pp of FedAvg ✓

Reflection: The Privacy–Fairness–Accuracy Trilemma
Improving worst-client accuracy requires evaluating per-client losses each round, which leaks information about local data distributions — a client with consistently high loss implicitly reveals it belongs to a minority or unusual group. Differential privacy mitigations (gradient clipping, noise injection) would weaken the loss signal FairAgg depends on. In a real deployment, average accuracy should be sacrificed first: the core motivation for federated learning is often to serve users who cannot share data precisely because they are vulnerable or marginalised. Worst-client accuracy and variance reduction should be primary objectives, with privacy treated as a non-negotiable constraint.

Setup
bashpip install torch torchvision matplotlib seaborn scikit-learn pandas pillow

Parts A–B use CIFAR-10 (auto-downloaded via torchvision)
Part C requires the State Farm Distracted Driver Detection dataset (Kaggle)
GPU strongly recommended; tested on Kaggle T4


Repository Structure
├── pa6_2.ipynb            # Full notebook: Parts A–D
├── a3_divergence.png      # FedAvg vs centralised GD convergence
├── a4_dashboard.png       # Instrumentation dashboard
├── b1_mu_sweep.png        # FedProx μ sweep
├── b2_drift.png           # Client drift heatmaps
├── b3_dashboard.png       # Training diagnostics
├── c2_sfd3_results.png    # SFD3 FL results
├── c3_gradcam.png         # GradCAM grid
└── d2_comparison.png      # FairAgg vs baselines

References

McMahan et al. (2017) — Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)
Li et al. (2020) — Federated Optimization in Heterogeneous Networks (FedProx)
Cho & Hariharan (2019) — On the Efficacy of Knowledge Distillation
Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks
