# Paper Weakness Audit — EM-ELM (Updated 2026-03-31)

Based on full read of `PINN_ELM_final/main.tex`.

---

## Critical (would cause rejection)

### 1. Weight scale `* 0.1` invalidates the spectral bias claim
**Location:** Methods §2.1 (Line 77–79)

The paper claims Power-Law creates a "multiscale basis," but the code multiplies all weights by `0.1`. With `sin(0.1 * w * x)`, the max effective frequency is ~0.3 — both Power-Law AND Gaussian are stuck at low frequencies. The "80% improvement" over Gaussian is real but comes from distribution **shape** at the same tiny scale, not from reaching higher frequencies.

The Helmholtz benchmark (k=10, freq=31.4) only works because the notebook there implicitly uses a different effective scale. A reviewer who reads the code will catch this contradiction.

**Fix:** Either (a) acknowledge that `*0.1` limits both methods and the improvement is from tail shape, or (b) remove the fixed scaling and tune `w_scale` per problem (as done in the fixed notebook).

---

### 2. Burgers table has "---" placeholders
**Location:** Table 2 (Lines 200–204)

Cannot submit with missing data. Needs actual numbers from the rerun.

**Fix:** Run Burgers cell 9 in the fixed notebook and fill in the table.

---

### 3. PINN comparison is unfair
**Location:** Methods §2.2 (Line 89)

The main Poisson benchmark uses a PINN with **400 iterations, 128 neurons, 1 layer** — this is a deliberately weak baseline. The paper then claims a 416× speedup over a **4-layer deep PINN with 2000 iterations**, which is still undertrained for nonlinear PDEs. Literature standard (Raissi et al. 2019) uses 10,000+ Adam steps + L-BFGS.

A reviewer will say: "you're comparing against an undertrained PINN."

**Fix:** Increase PINN iterations to at least 5,000 Adam + L-BFGS, or explicitly acknowledge the light PINN baseline and justify why (e.g., "we compare against a single-layer PINN with comparable parameter count to the ELM's output layer").

---

### 4. ~~ELM does supervised regression, PINN does physics~~ → CONVERT TO PIELM
**Location:** Methods §2.2 (Lines 86–88), all PDE benchmark cells

**The Problem:**
In the current PDE benchmarks, the ELM minimizes `||H*β − u_true||²` — it fits the **known analytical solution** directly. The PINN minimizes the **PDE residual** without access to the solution. This is not a fair comparison:

| | Current ELM (supervised) | PINN |
|---|---|---|
| **Needs the answer?** | Yes — needs `u_true` at grid points | No — only knows the PDE |
| **What it minimizes** | `||prediction − u_true||²` | `||PDE residual||²` |

**Note:** This does NOT apply to the MNIST/FashionMNIST classification benchmark, where both ELM and backprop receive the same `(image, label)` training pairs — a fair comparison.

**The Fix — Implement PIELM (Physics-Informed ELM):**

Convert all PDE benchmarks from supervised ELM to PIELM ([Dwivedi & Srinivasan 2020](https://arxiv.org/abs/1907.03507)). PIELM applies the PDE operator to the random basis analytically, then solves for β using physics — no `u_true` needed during training.

#### How PIELM works

For `φ_j(x) = sin(w_j·x + b_j)`, the derivatives are analytic:
```
φ_j'(x)  =  w_j · cos(w_j·x + b_j)
φ_j''(x) = -w_j² · sin(w_j·x + b_j)
```

**Poisson: `-u''(x) = f(x)`**
```python
H     = sin(X @ W + b)                    # shape (N, h)
H_xx  = -(W**2) * sin(X @ W + b)          # analytic second derivative
# PDE: -u'' = f  →  -H_xx @ β = f  →  (W² ⊙ H) @ β = f
H_pde = (W**2) * H                        # = -H_xx

# Boundary conditions: u(-1) = 0, u(1) = 0
H_bc = sin(X_bc @ W + b)                  # shape (2, h)
u_bc = [0, 0]

# Stack into one linear system:
A = vstack([H_pde, λ_bc * H_bc])          # shape (N+2, h)
b = vstack([f_source, u_bc])              # shape (N+2, 1)

# Solve: β = argmin ||Aβ - b||²  (one-shot, same as before)
beta = solve(A^T A + λI, A^T @ b)
```

**Helmholtz: `-u''(x) - k²u(x) = f(x)`**
```python
H_pde = (W**2 - k**2) * H                 # combines -u'' and -k²u
# Same BC stacking, same one-shot solve
```

**Multi-Frequency Poisson:** Same as Poisson (different f).

**Burgers (nonlinear): `u_t + u·u_x = ν·u_xx`**
The `u·u_x` term is **nonlinear in β** → PIELM cannot be applied directly.
Options:
- (a) Keep supervised regression for Burgers only (acknowledge in paper)
- (b) Use iterative Newton-like PIELM (Dong & Li 2021 approach)
- (c) Drop Burgers from paper (simplest)

#### Key advantage of PIELM approach
- ELM and PINN now solve the **same problem** (PDE from physics only)
- Speed comparison becomes fair: both discover the solution, ELM just does it in one shot
- `u_true` is only used for **evaluation** (computing L2 error), never for training
- This is exactly what Dwivedi (2020) and Dong & Li (2021) do — standard in the field

#### Implementation checklist for `ELM_Energy_Minimization_fixed.ipynb`

- [ ] Add `solve_pielm_poisson(x, f_source, h_dim, ...)` — uses H_pde + BCs
- [ ] Add `solve_pielm_helmholtz(x, f_source, k, h_dim, ...)` — uses (W²-k²)⊙H + BCs
- [ ] Add `solve_pielm_multifreq(x, f_source, h_dim, ...)` — same as Poisson
- [ ] Update Cell 3 (main Poisson benchmark) — replace `solve_elm` with `solve_pielm`
- [ ] Update Cell 11 (Multi-Freq Poisson) — replace `solve_elm_1d` with `solve_pielm`
- [ ] Update Cell 13 (Helmholtz) — replace `solve_elm_1d` with `solve_pielm`
- [ ] Cell 9 (Burgers) — keep supervised OR implement nonlinear PIELM
- [ ] Keep `u_true` only for error evaluation, never in training
- [ ] Update paper text to describe PIELM formulation instead of supervised regression
- [ ] Add Dwivedi (2020) citation for PIELM methodology

---

## Major (would require revision)

### 5. "Critical state" α=2.0 claim is unsupported
**Location:** Methods §2.1 (Line 79)

> "This specific value is significant because it represents a 'critical state' in statistical mechanics"

No citation, no proof. α=2.0 produces a Cauchy-like distribution, not a phase transition. No ablation over α values is presented.

**Fix:** Either cite a specific result connecting α=2.0 to criticality (e.g., Lévy flights, sandpile models), or soften to "we empirically select α=2.0 as it provides a good balance between frequency coverage and numerical conditioning."

---

### 6. Cholesky complexity claim is misleading
**Location:** Methods §2.2 (Lines 88–89)

> "reduces computational complexity from O(n³) to O(⅓n³)"

These are the **same complexity class** O(n³). The ⅓ is a constant factor, not a complexity reduction. A reviewer will flag this as overstating the contribution.

**Fix:** Change to: "Cholesky decomposition reduces the constant factor from ~n³ (full LU) to ~⅓n³, providing a practical speedup without changing the asymptotic complexity."

---

### 7. No α ablation study

Paper claims α=2.0 is the right choice but never tests α=1.5, 2.5, 3.0. Without this, a reviewer cannot evaluate whether the choice is principled or lucky.

**Fix:** Add a small table or figure showing L2 error vs α ∈ {1.5, 2.0, 2.5, 3.0} on the Poisson benchmark. Even 5 seeds each would suffice.

---

### 8. Extended benchmark numbers are stale/inconsistent

- Table says Multi-Freq PL/G ratio = **1.5×** but text (line 220) says **"1.2×"**
- Table says Helmholtz ratio = **3.7×** but text (line 237) says **"3.5×"**
- Burgers text (line 254) quotes old numbers (0.207, 0.269) that don't match current "---"

**Fix:** After rerunning all benchmarks, do a single pass to update all numbers consistently in both the table and the prose.

---

## Minor (but worth fixing)

### 9. No Limitations section

Paper doesn't discuss when Power-Law fails (localized features, 2D problems, Burgers shock). Honest limitations strengthen a paper. Every top venue expects this.

**Fix:** Add a short "Limitations" subsection in Discussion acknowledging:
- ELM requires reference data (unlike PINN)
- Global sinusoidal basis struggles with localized features (boundary layers, shocks)
- Weight scale must be tuned per problem (no universal default)

---

### 10. MNIST/FashionMNIST in abstract but missing from body

Abstract mentions "classification experiment on MNIST and FashionMNIST confirms the generality" but the Results section only discusses PDE benchmarks. The classification results are in the notebook but never appear in the paper.

**Fix:** Either add a brief subsection with the MNIST/FashionMNIST table, or remove the claim from the abstract.

---

### 11. "Real-Time Digital Twin" overclaim
**Location:** Discussion §5.2 (Lines 272–273)

Solving a 1D Poisson equation doesn't demonstrate digital twin capability. This is marketing language a reviewer will push back on.

**Fix:** Soften to: "The sub-second latency of the EM-ELM suggests potential for real-time applications such as surrogate modeling in digital twin pipelines, though validation on production-scale systems remains future work."

---

### 12. Text/table number mismatches

| Location | Table says | Text says |
|----------|-----------|-----------|
| Multi-Freq Poisson ratio | 1.5× | 1.2× |
| Helmholtz ratio | 3.7× | 3.5× |
| Helmholtz PL error | 0.183 | 0.159 |
| Helmholtz G error | 0.668 | 0.557 |

**Fix:** Single pass to sync all numbers after final rerun.

---

## Priority Order

| Priority | Issue | Effort |
|----------|-------|--------|
| 1 | Fill Burgers table (blocker) | Rerun notebook |
| 2 | Fix weight scale story (Critical #1) | Text edit |
| 3 | Add ELM vs PINN fairness disclaimer (Critical #4) | 1 paragraph |
| 4 | Sync all numbers (Major #8) | One pass after rerun |
| 5 | Soften α=2.0 and Cholesky claims (Major #5, #6) | Text edits |
| 6 | Add Limitations section (Minor #9) | 1 paragraph |

---

## Previously Fixed Issues (from earlier audit)

These were already resolved in the current version:
- ✅ arXiv:2105.14354 phantom citation → fixed to 2012.02895
- ✅ Missing Rahaman et al. (2019) spectral bias reference → added
- ✅ Missing Dwivedi & Srinivasan (2020) PIELM reference → added
- ✅ Missing Wang et al. (2022) NTK spectral bias reference → added
- ✅ FNO title "arbitrary" → "Parametric" → fixed
- ✅ Draft artifacts (Change #1, #2, NEED TO CHECK) → removed from final version
- ✅ Contradictory speedup numbers → unified to 7.3×/416×
