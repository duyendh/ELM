# Paper Audit: "Accelerating Physics-Informed Learning via Power-Law Extreme Learning Machines"

---

## 🔴 CRITICAL ISSUES

### 1. Reference [7] — arXiv:2105.14354 Does NOT Exist

The paper cites:
> *S. Dong, Z. Li, "Extreme learning machine-based predictors for accelerating physics-informed neural networks," arXiv:2105.14354 (2021).*

**This arXiv ID does not resolve to any paper.** The actual Dong & Li (2021) paper on ELM for PDEs is:

- **Correct paper:** S. Dong, Z. Li, *"Local extreme learning machines and domain decomposition for solving linear and nonlinear partial differential equations,"* *Computer Methods in Applied Mechanics and Engineering*, 387, 114129, 2021. arXiv: **2012.02895**

The title in [7] is also wrong. This is either a phantom citation or a fabricated arXiv ID. **Must be corrected before any submission.**

---

### 2. Contradictory Speedup Numbers in the Abstract

The abstract contains two unreconciled speedup claims left side-by-side:

- **Version 1 (original):** "7.3× over single-layer PINN and 416× over four-layer deep PINN"
- **Version 2 (Change #2):** "150× speedup on benchmark PDEs"

A reviewer seeing both in the final abstract would immediately flag this as a serious inconsistency. **One version must be chosen and the other deleted entirely.**

---

### 3. Internally Acknowledged Measurement Inconsistency

The paper openly states:
> *"The original submitted figure reported a 7.3× speedup. When re-running the notebook, we obtain 10.7×"*

This means the reported figure comes from an **unreproducible run**, and the authors consciously retained an older number "to stay consistent with the published figure." For venues like NeurIPS or ICML this is a reproducibility red flag. Either re-run the experiment and update all numbers consistently, or explicitly characterize run-to-run variance with error bars in the results section.

---

## 🟠 SIGNIFICANT CONCERNS

### 4. Reference [2] — Misuse of Basri et al. (2020) for Spectral Bias

The paper uses Basri et al. as its **sole** primary citation for "spectral bias." However, the Basri et al. (2020) paper specifically addresses *"Frequency Bias in Neural Networks for Input of Non-Uniform Density"* — a specialized extension of spectral bias theory for non-uniform data distributions, **not** the foundational spectral bias paper.

The **canonical primary citation** that reviewers will expect is:

> Rahaman, N. et al. *"On the Spectral Bias of Neural Networks,"* ICML 2019. arXiv: 1806.08734.

Using only Basri et al. for a general spectral bias claim will appear as either a literature gap or an error to expert reviewers. **Add Rahaman et al. (2019) as the primary reference.**

---

### 5. "Critical State" at α = 2.0 — Unsubstantiated Claim

The paper states:
> *"α = 2.0 represents a 'critical state' in statistical mechanics, creating a multiscale basis."*

This invokes a serious and specific concept from physics (criticality, phase transitions) but **no citation or derivation is provided.** A reviewer will immediately challenge this. Either:

- Cite a specific paper connecting α = 2 to a critical state in the relevant context (e.g., Barabási–Albert networks, heavy-tailed random matrix theory), **or**
- Soften the language to something like: *"a value associated with heavy-tailed, multiscale behavior"*

---

### 6. Cholesky Complexity Claim is Misleading

The paper states:
> *"Cholesky Decomposition reduces computational complexity from O(n³) to O(⅓n³)"*

This is **technically incorrect as stated.** Cholesky has cost ~⅓n³ FLOPS vs. LU at ~⅔n³ FLOPS — a ~2× constant-factor improvement. Both remain **O(n³)**. The paper implies this unlocks a new complexity class, which it does not.

**Rewrite as:** *"Cholesky decomposition halves the constant factor compared to general LU decomposition, reducing the solve cost from ⅔n³ to ⅓n³ FLOPS."*

---

### 7. The 416× Speedup Is Not Actually Derived from the Formula

The theoretical speedup formula gives:

$$S \approx O(I \cdot k \cdot L)$$

But the paper never specifies what values of *I*, *k*, and *L* yield 416×. Without plugging in the actual experimental parameters (e.g., I = 400 iterations, k = derivative order multiplier, L = 4 layers), the theoretical section does not "prove" the observed speedup as claimed. It only shows the speedup scales with these factors.

**Either plug in the actual parameter values to verify the formula predicts ~416×, or qualify the claim as "consistent with" rather than "derived from" the analysis.**

---

## 🟡 MINOR / EDITORIAL CONCERNS

### 8. Draft Artifacts Still Visible in Manuscript

The following internal notes must be removed entirely before any submission — they make it obvious the document is an unfinished draft:

| Artifact | Location |
|---|---|
| `NEED TO CHECK` | Appears twice (Sections 2.1, 2.2) |
| `Change #1`, `Change #2` | Abstract / Keywords section |
| `DATA NOTE` | Page 1 |
| `Review Note #1`, `Review Note #2` | Sections 2.1, 2.2 |
| `R1` flags | Multiple locations |
| Duplicate abstract text | Page 1 |

---

### 9. Missing Key Related Work

For NeurIPS/ICML, reviewers will expect citation of the following highly relevant prior work that is currently absent:

| Missing Citation | Why It Matters |
|---|---|
| Dwivedi & Srinivasan (2020), *PIELM*, arXiv:1907.03507 | Most directly comparable prior method (ELM for PDEs); not citing it is a major gap |
| Wang, Yu & Perdikaris (2022), *"When and Why PINNs Fail to Train"*, J. Comput. Phys. | Direct NTK-based proof of spectral bias in PINNs; central to your motivation |
| Rahaman et al. (2019), *"On the Spectral Bias of Neural Networks"*, ICML | Foundational spectral bias reference (see Issue #4) |

---

### 10. Reference [4] — FNO Title Incorrect

The FNO citation reads:
> *"Fourier neural operator for **arbitrary** partial differential equations"*

The correct title is:
> *"Fourier Neural Operator for **Parametric** Partial Differential Equations"*

---

## Summary Priority Table

| # | Issue | Severity | Required Action |
|---|---|---|---|
| 1 | arXiv:2105.14354 doesn't exist | 🔴 Critical | Replace with correct citation (arXiv:2012.02895) |
| 2 | Contradictory speedup numbers in abstract | 🔴 Critical | Choose one version, delete the other |
| 3 | Acknowledged irreproducible figure | 🔴 Critical | Re-run or formally characterize variance |
| 4 | Wrong primary citation for spectral bias | 🟠 Significant | Add Rahaman et al. (2019) |
| 5 | "Critical state" at α=2.0 is unsupported | 🟠 Significant | Cite supporting theory or soften claim |
| 6 | Cholesky complexity claim is misleading | 🟠 Significant | Reframe as constant-factor improvement |
| 7 | 416× not actually derived from formula | 🟠 Significant | Plug in actual values or soften claim |
| 8 | Draft artifacts still in manuscript | 🟡 Minor | Delete all `NEED TO CHECK`, `R1`, `Change #` notes |
| 9 | Missing PIELM and key related work | 🟡 Minor | Add Dwivedi & Srinivasan (2020), Wang et al. (2022) |
| 10 | FNO title slightly wrong | 🟡 Minor | Fix "arbitrary" → "parametric" |

---

## Corrected Reference List

| Ref | Original | Corrected |
|---|---|---|
| [2] | Basri et al. (2020) only | Add: Rahaman et al., "On the Spectral Bias of Neural Networks," ICML 2019, arXiv:1806.08734 |
| [4] | "...for arbitrary PDEs" | "Fourier Neural Operator for Parametric Partial Differential Equations" |
| [7] | arXiv:2105.14354 (does not exist) | S. Dong, Z. Li, CMAME 387, 114129, 2021. arXiv:2012.02895 |
| [new] | — | V. Dwivedi, B. Srinivasan, "Physics Informed Extreme Learning Machine (PIELM)," arXiv:1907.03507, 2020 |
| [new] | — | S. Wang, X. Yu, P. Perdikaris, "When and Why PINNs Fail to Train," J. Comput. Phys. 449, 2022 |
