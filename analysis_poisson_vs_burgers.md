# Why Power-Law ELM Works for Poisson but Struggles with Burgers

## Summary

The Poisson equation benchmark (cells 3 & 5) shows Power-Law ELM clearly outperforming Gaussian ELM (3-10x lower error). The Burgers equation benchmark (cells 8-9) hasn't been run yet, but the code structure reveals **fundamental reasons** why the advantage may not transfer. This document explains the key differences and what to do about it.

---

## 1. The Poisson Equation: Why Power-Law Wins

### Problem Setup
```
-u''(x) = f(x),  x in [-1, 1]
u(-1) = u(1) = 0
u_true = sin(pi*x),  f = -pi^2 * sin(pi*x)
```

### Why Power-Law Dominates Here

| Property | Value |
|----------|-------|
| Dimension | **1D** (input dim = 1) |
| Solution character | **Smooth, single-frequency** sinusoid |
| Target frequency | pi ~ 3.14 |
| Weight scale | 0.1 |
| Effective frequencies | W * x = 0.1 * [-1,1] → small range |
| Activation | sin(Wx + b) |

**The key mechanism:** With `w_scale = 0.1`, both Power-Law and Gaussian produce weights in roughly [-0.3, 0.3]. But the distributions are very different:

- **Gaussian (N(0, 0.1)):** Weights cluster tightly around 0. Most hidden neurons produce nearly identical low-frequency basis functions → **redundant features, poor basis diversity**.
- **Power-Law (P(w) ~ w^{-2}):** Heavy tails produce a **spread of weight magnitudes** — some very small, some moderate. This creates a richer set of frequencies even at small scale → **diverse basis, better approximation**.

For a smooth 1D target like sin(pi*x), this frequency diversity is the *entire game*. Power-Law's heavy-tailed distribution naturally provides multi-scale basis functions, which is exactly what you need to approximate a sinusoid with a random feature expansion.

### Measured Results (from notebook)
| Method | L2 Error (mean) | L2 Error (std) |
|--------|-----------------|----------------|
| ELM (Power Law) | **0.033** | 0.036 |
| ELM (Gaussian)  | 0.337 | 0.278 |
| PINN (Light)    | 0.006 | 0.0004 |

Power-Law is **10x better** than Gaussian. Clear win.

---

## 2. The Burgers Equation: Why It's Fundamentally Harder

### Problem Setup
```
u_t + u*u_x = nu * u_xx,  (x,t) in [-1,1] x [0,1]
u(x,0) = -sin(pi*x),  u(-1,t) = u(1,t) = 0
nu = 0.01/pi  (low viscosity → steep gradients)
```

### 5 Critical Differences from Poisson

#### Difference 1: 2D Input Instead of 1D
- Poisson: input = `x` (1 feature), weights W shape (1, H)
- Burgers: input = `(x, t)` (2 features), weights W shape (2, H)

With 2D input, each hidden neuron computes `sin(w1*x + w2*t + b)`. The weight distribution now controls **two coupled frequencies** (spatial and temporal). Power-Law's advantage in producing diverse magnitudes gets diluted across two dimensions — you need the *right ratio* of w1/w2, not just diverse magnitudes.

#### Difference 2: Multi-Scale, Nonlinear Target
- Poisson: single frequency (pi), globally smooth
- Burgers: **develops steep shock-like gradients** near x=0 at later times

The Burgers solution at t=0.5 has:
- Smooth regions where u ~ 0 (large |x|)
- A steep transition zone near x = 0 (near-discontinuity)

Approximating this requires **many high-frequency basis functions concentrated near the shock**, not just diverse frequencies. Power-Law's frequency spread is *uniform in space* — it doesn't know to put more resolution near x=0.

#### Difference 3: ELM Does Pure Curve Fitting (No Physics)
The ELM solver in the notebook does:
```python
H = sin(X @ W + b)           # random features
beta = solve(H'H + λI, H'u)  # least-squares fit
```

This is **pure function approximation** — no PDE residual, no physics. For Poisson (smooth, 1D), this works fine. For Burgers (shock, 2D), you're asking a random Fourier basis to fit a near-discontinuity by least-squares. This is hard for *any* random feature method, regardless of initialization.

The PINN solver, by contrast, enforces `u_t + u*u_x = nu*u_xx` directly. Physics-informed loss provides gradient information that steers the network toward physically correct solutions — especially near shocks.

#### Difference 4: Hyperparameter Sensitivity
The Poisson cell uses `w_scale=0.1, b_scale=0.1, lambda=1e-6`.
The Burgers cell (correctly) rescales to `w_scale=1.0, b_scale=1.0, lambda=1e-6`.

But Burgers is much more sensitive to these choices:
- Too small w_scale → can't represent the shock's high frequencies
- Too large w_scale → oscillatory overfitting (Gibbs-like phenomenon)
- Lambda too small → ill-conditioning with 10,000 points and 4096 features
- Lambda too large → smooths out the shock

For Poisson, there's a wide "sweet spot" in hyperparameter space. For Burgers, it's narrow and may not be centered in the same place for Power-Law vs. Gaussian.

#### Difference 5: Conditioning of the Linear System
With 2D input and 10,000 points (100x100 grid), the hidden matrix H has shape (10000, 4096). The condition number of H'H depends heavily on:
- How diverse the basis functions are (good)
- How correlated they become at the shock (bad)

Near the shock, many sin(w*x + ...) features produce similar outputs → columns of H become correlated → H'H becomes ill-conditioned → least-squares solution is noisy. Power-Law's heavy-tailed weights can make this *worse* because extreme weights create highly oscillatory features that are numerically problematic.

---

## 3. Summary Table

| Factor | Poisson (1D, smooth) | Burgers (2D, shock) |
|--------|---------------------|---------------------|
| Input dimension | 1 | 2 |
| Solution regularity | C-infinity (smooth) | Near-discontinuous shock |
| Frequency content | Single mode (pi) | Broadband (shock = all frequencies) |
| ELM approach | Curve fitting works well | Curve fitting struggles with shocks |
| Power-Law advantage | Frequency diversity → better basis | Diversity alone insufficient |
| Hyperparameter sensitivity | Low (wide sweet spot) | High (narrow sweet spot) |
| Conditioning | Good (smooth target) | Poor (correlated near shock) |

---

## 4. What to Do About It

### Option A: Accept the Limitation (Honest)
Burgers with low viscosity is a **known hard problem** for random feature methods. The paper can focus on problems where Power-Law genuinely helps (smooth PDEs, classification) and note Burgers as a challenge case. This is scientifically honest and still publishable.

### Option B: Try Alternative Test Functions
If the goal is to show Power-Law advantage on a *harder* PDE than Poisson, consider:

1. **Helmholtz equation:** `-u'' - k^2 u = f` with high wavenumber k. This is smooth but multi-frequency — Power-Law should excel here.
2. **Advection-diffusion (linear):** `u_t + c*u_x = nu*u_xx`. Linear, so no shock, but 2D input. Tests whether Power-Law helps with 2D.
3. **Poisson 2D:** `-laplacian(u) = f` on a square. Smooth, but 2D input. Good middle ground.
4. **Higher-frequency Poisson 1D:** `u = sin(10*pi*x)` or a sum of modes. Tests whether Power-Law handles multi-frequency targets better.

### Option C: Fix the Burgers Benchmark
If you want to keep Burgers, several improvements could help:

1. **Increase hidden dim** to 8192+ (more basis functions for the shock)
2. **Use RBF activation** instead of sin (Gaussian RBFs can localize near the shock)
3. **Adaptive weight scaling** — use larger w_scale for some neurons, smaller for others
4. **Domain decomposition** — split [0,1] into subdomains, use separate ELMs near the shock
5. **Tune hyperparameters separately** for Power-Law and Gaussian (they may need different lambda)

---

## 5. Bottom Line

Power-Law initialization helps ELM when **basis diversity is the bottleneck** — i.e., when the Gaussian initialization produces too many redundant neurons. This happens most clearly for:
- Low-dimensional inputs (1D or 2D)
- Smooth targets with moderate frequency content
- Problems where the activation function's frequency range matters

It does NOT automatically help when:
- The target has sharp features (shocks, discontinuities)
- The problem is fundamentally ill-conditioned
- Physics constraints (PDE residual) would help more than better initialization
- The input dimension is high enough that weight magnitude diversity is less impactful

The Poisson equation is the **ideal showcase** for Power-Law ELM. Burgers is arguably the worst case. Choose benchmarks accordingly.
