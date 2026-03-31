# Literature Review: ELM for PDEs — What the Field Actually Does

## 1. Dwivedi & Srinivasan (2020) — PIELM
**Paper:** "Physics Informed Extreme Learning Machine", Neurocomputing Vol. 391
**arXiv:** https://arxiv.org/abs/1907.03507

- **Activation:** tanh (exclusively)
- **Weight init:** "Fixed randomly" — paper is vague on distribution/range (reproducibility weakness)
- **BC enforcement:** Augmented linear system with penalty weight lambda (soft constraint)
- **Solver:** Moore-Penrose pseudoinverse: `c = pinv(H) * K`
- **Regularization:** None explicit
- **PDEs:** Linear and quasi-linear only (advection, diffusion/Poisson, advection-diffusion 1D/2D)
- **No nonlinear PDEs**

Later work (Dwivedi 2025, curriculum learning) switches to Gaussian RBF activation.

---

## 2. Dong & Li (2021) — locELM
**Paper:** "Local Extreme Learning Machines", CMAME Vol. 387
**arXiv:** https://arxiv.org/abs/2012.02895

- **Activation:** tanh
- **Weight init:** Uniform on [-Rm, Rm], where Rm is user-provided hyperparameter
- **Rm is the single most important tuning parameter**
- **Nonlinear PDEs:** Newton-LLSQ (linearize per iteration, solve linear least-squares each step)
- **Domain decomposition:** Partition domain into sub-domains, each with local ELM

---

## 3. Dong (2022) — On Computing the Hyperparameter Rm
**Paper:** "On Computing the Hyperparameter of Extreme Learning Machines", J. Comput. Phys.
**arXiv:** https://arxiv.org/abs/2110.14121
**ScienceDirect:** https://www.sciencedirect.com/science/article/pii/S0021999122003527

### This is the key paper on weight scale for ELM-PDEs.

- **Activation:** Gaussian RBF `sigma(x) = exp(-x^2)`
- **Weight init:** Uniform on [-Rm, Rm]
- **Core finding:** ELM performance is **highly sensitive to Rm**. Best accuracy comes from a moderate range. Very large or very small Rm → poor accuracy.
- **Method:** Uses differential evolution to find optimal Rm: `Rm_0 = argmin_Rm ||r(Rm)||`
- **Two configs:** Single-Rm-ELM (one Rm for all) vs Multi-Rm-ELM (different Rm per layer)
- **Critical insight: scale matters more than distribution shape.** Dong keeps uniform distribution fixed and only varies Rm — implicitly showing that distribution shape is secondary.

---

## 4. Huang, Zhu & Siew (2006) — Original ELM
**Paper:** "Extreme learning machine: Theory and applications", Neurocomputing Vol. 70

- **Theoretical result:** Weights can come from **any continuous probability distribution**
- **Practical convention:** Uniform on [-1, 1] for weights, [0, 1] for biases (de facto standard)
- **Activation:** Sigmoid and other infinitely differentiable functions
- **Universal approximation:** Holds for random weights provided sufficient hidden neurons
- **Weight scale:** Paper claims "minimal impact" — but this is overly optimistic for PDE applications (contradicted by Dong 2022)

---

## 5. LSE-ELM (2025) — BC Enforcement Comparison
**arXiv:** https://arxiv.org/abs/2503.19185

Clearest comparison of BC strategies:

| Method | BC Enforcement | Weakness |
|--------|---------------|----------|
| PIELM (Dwivedi 2020) | Soft: penalty lambda | lambda tuning, instability |
| XTFC (Leake & Mortari) | Hard: modified basis | Limited to simple geometries |
| LSE-ELM (this paper) | Hard: SVD nullspace projection | Complex implementation |

- **Weight init:** Uniform on [-M, M] with M = 3
- **Activation:** tanh

---

## 6. Activation Functions Used in PIELM/ELM for PDEs

| Paper | Activation |
|-------|-----------|
| Dwivedi & Srinivasan 2020 | tanh |
| Dong & Li 2021 (locELM) | tanh |
| Dong 2022 (Rm paper) | Gaussian RBF |
| LSE-ELM 2025 | tanh |
| Dwivedi 2025 (curriculum) | Gaussian RBF |
| Sitzmann 2020 (SIREN, not ELM) | sin |

**No PIELM paper uses sin(wx+b) as activation.** Sin activation comes from SIREN (Sitzmann et al. 2020), which is deep learning, not ELM.

---

## 7. Weight Initialization Summary

- **Standard:** Uniform on [-Rm, Rm] or [-1, 1]
- **No paper uses Gaussian initialization** as default for ELM
- **No paper uses Power-Law initialization** — novel to our work
- **Dong (2022) shows Rm (scale) is the critical hyperparameter**, not distribution shape
- **Distribution shape is secondary to scale**

---

## 8. "ELM for PDEs" = PIELM (Terminology Clarification)

The literature uses different names for the same thing:

| Author | What they call it | What it actually does |
|---|---|---|
| Huang (2006) | ELM | Supervised regression (needs u_true) |
| Dwivedi (2020) | PIELM | ELM + PDE operator (no u_true) |
| Dong & Li (2021) | locELM | ELM + PDE operator + domain decomposition |
| Dong (2022) | "ELM for computational PDEs" | ELM + PDE operator (no u_true) |
| LSE-ELM (2025) | LSE-ELM | ELM + PDE operator + hard BC constraints |

**They are all doing the same core thing:**

Replace `H @ beta = u_true` with `L[H] @ beta = f_source`, where `L` is the PDE differential operator applied analytically to the basis functions.

- Standard ELM (Huang 2006) is **supervised** — always needs target values (u_true). It's a general ML method with no concept of PDEs, physics, or differential operators. It just fits data.
- Every paper that uses "ELM" to solve PDEs actually uses the physics-informed variant — applying the PDE operator to the basis, stacking BCs, solving without u_true.
- **There is no paper that solves PDEs with standard supervised ELM** — that would be pointless since you'd already need the analytical solution.
- The only differences between papers are: activation function (tanh vs Gaussian RBF vs others), BC enforcement (soft penalty vs hard constraint), domain decomposition (yes/no), and how they tune the weight scale Rm.

**The real workflow (all papers):**
```
Known:    PDE operator L, source term f, boundary conditions g
Unknown:  solution u(x)

Training (no u_true):
  [L[H_interior]]           [f_source]
  [              ] @ beta  = [        ]
  [H_boundary    ]           [g_bc    ]

  beta = pinv(A) @ rhs    (one-shot solve)

Evaluation only (u_true used here):
  u_pred = H @ beta
  error = ||u_pred - u_true|| / ||u_true||
```

---

## 9. Implications for Our Paper

| Aspect | Standard PIELM | Our current notebook |
|---|---|---|
| Activation | **tanh** | sin |
| Weight init | **Uniform [-Rm, Rm]** | Power-Law * 0.1 |
| Key param | **Rm tuned per problem** | Fixed 0.1 |
| BC enforcement | Augmented system + penalty | Fits u_true directly (supervised) |
| Solver | Pseudoinverse (pinv) | Cholesky / linalg.solve |
| Regularization | None (pinv implicit) | Ridge (lambda) |

### Key issues:
1. `*0.1` scale (Rm=0.1) is far too small — Dong (2022) would classify this as expected-poor
2. Sin activation is non-standard — need to cite SIREN and justify departure from tanh
3. Power-Law vs Gaussian comparison may be measuring noise if Dong (2022) is right that only scale matters
4. Supervised ELM (fitting u_true) is not PIELM — unfair comparison with PINN
5. If we use Power-Law, must show benefit is from heavy tail, not just effective scale
