# Deep Operator Network (DeepONet) for the 1D Poisson Equation

A physics-informed Deep Operator Network trained to learn the solution operator of the 1D Poisson equation with Dirichlet boundary conditions:

$$-u''(x) = f(x), \quad x \in [0, 1], \quad u(0) = u(1) = 0$$

The network learns to map a **forcing function** $f$ → **solution** $u$ without any labelled solution data, supervised purely by the PDE residual and boundary conditions.

---

## Overview

| Component | Description |
|---|---|
| **Framework** | PyTorch (CUDA) |
| **Architecture** | DeepONet (Branch + Trunk MLP) |
| **PDE** | 1D Poisson: $-u'' = f$ |
| **Training** | Physics-informed (no ground-truth $u$ labels) |
| **Validation** | Finite difference method (FDM) reference solutions |

---

## Notebook Structure

### 1. Imports
Standard imports: `torch`, `numpy`, `matplotlib`, and PyTorch's `nn`/`optim` modules.

---

### 2. Function Space Definition — Random Forcing Functions
Defines the class of input functions $f$ used during training.

- `sample_polynomial_coeffs(batch_size, degree, scale)` — samples random polynomial coefficients from a Gaussian distribution to generate diverse forcing functions.
- `eval_polynomial(coeffs, x)` — evaluates the batched polynomial $f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3$ at query points $x$.

---

### 3. Model Architecture — DeepONet

#### `MLP`
A generic multi-layer perceptron with configurable hidden layers and activation (default: Tanh).

#### `DeepONet`
Implements the standard DeepONet architecture:
- **Branch network** — encodes the input function $f$ sampled at $m = 10$ fixed sensor points into a $p$-dimensional embedding.
- **Trunk network** — encodes the query coordinate $x \in [0,1]$ into the same $p$-dimensional space.
- **Output** — the dot product of branch and trunk embeddings, multiplied by $x(1-x)$ to **hard-enforce** the Dirichlet boundary conditions $u(0) = u(1) = 0$.

---

### 4. Automatic Differentiation — Second Derivative
`second_derivative(u, x)` computes $u''(x)$ using two successive calls to `torch.autograd.grad` with `create_graph=True`, enabling the PDE residual to flow gradients back through the network during training.

---

### 5. Hyperparameters & Setup

| Parameter | Value |
|---|---|
| Sensor points $m$ | 10 |
| Latent dimension $p$ | 32 |
| Collocation points per batch | 100 |
| Batch size | 64 |
| PDE loss weight $w_\text{pde}$ | 1.0 |
| BC loss weight $w_\text{bc}$ | 10.0 |
| Optimizer | Adam, lr = 1e-3 |

Sensor locations and collocation points are fixed uniform grids on $[0, 1]$.

---

### 6. Training Loop — `train_step()`
Each step:
1. Samples a batch of random polynomial forcing functions $f$.
2. Evaluates $f$ at sensor locations to form the branch input.
3. Runs a forward pass to obtain $u(x)$.
4. Computes the **PDE residual loss**: $\mathcal{L}_\text{pde} = \| {-u'' - f} \|^2$
5. Computes the **boundary condition loss**: $\mathcal{L}_\text{bc} = u(0)^2 + u(1)^2$ (auxiliary check; BCs are also hard-enforced by the architecture).
6. Minimises $\mathcal{L} = w_\text{pde} \cdot \mathcal{L}_\text{pde} + w_\text{bc} \cdot \mathcal{L}_\text{bc}$.

Training runs for **10,000 steps** with a loss printout every 500 steps.

---

### 7. Qualitative Visualisation
After training, plots 3 random test cases side-by-side:
- **Left panel** — the forcing function $f(x)$ with sensor observations marked.
- **Right panel** — the predicted solution $u(x)$ from DeepONet.

---

### 8. Finite Difference Reference Solver
`solve_poisson_fd(x, f)` assembles and solves the standard second-order finite difference system for $-u'' = f$ with Dirichlet BCs, producing a ground-truth reference solution for validation.

---

### 9. Quantitative Evaluation
For each of 3 held-out test functions, reports:
- **Residual $L^2$** — how well the network satisfies the PDE.
- **Residual max** — worst-case pointwise PDE violation.
- **FDM relative $L^2$ error** — distance between DeepONet prediction and the FDM reference solution.

---

### 10. Comparison Plots
Overlays the DeepONet prediction against the FDM reference for each test case, with the relative $L^2$ error and residual $L^2$ annotated in the title.

---

## Results Interpretation
Low residual $L^2$ confirms the network satisfies the PDE. Low FDM relative $L^2$ error confirms the solution matches the classical numerical solver — achieved with **zero labelled training data**.
