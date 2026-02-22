# PINN_ELM Paper — Project Instructions

## Overview

This project contains the LaTeX manuscript for the paper **"Accelerating Physics-Informed Learning via Power-Law Extreme Learning Machines"**, along with a supporting Jupyter notebook for the ELM energy minimization experiments.

---

## Project Structure

```
ELM/
├── .venv/                          # Python virtual environment (local)
├── .vscode/
│   ├── settings.json               # VS Code Python interpreter config
│   └── tasks.json                  # LaTeX build tasks
├── PINN_ELM/
│   ├── main.tex                    # ⭐ Main manuscript (single-file, elsarticle class)
│   ├── references.bib              # ⭐ BibTeX references
│   └── figure2_boxplot.png         # Figure used in the paper
├── ELM_Energy_Minimization.ipynb   # Jupyter notebook for experiments
└── INSTRUCTIONS.md                 # ← You are here
```

---

## 1. Environment Setup

### Python Virtual Environment

A local `.venv` is already created in the project root. To activate:

```bash
source .venv/bin/activate
```

Installed packages: `jupyter`, `numpy`, `matplotlib`, `scipy`, `torch`, `scikit-learn`, `seaborn`, `pandas`.

### LaTeX (System)

LaTeX is available system-wide via MacTeX:
- `pdflatex` → `/Library/TeX/texbin/pdflatex`
- `bibtex` → `/Library/TeX/texbin/bibtex`

No additional LaTeX installation needed.

---

## 2. Building the PDF

### Option A: VS Code Task (Recommended)

Press `Cmd+Shift+B` → select **"Build LaTeX PDF"**.

This runs the full build cycle:
```
pdflatex main.tex → bibtex main → pdflatex main.tex → pdflatex main.tex
```

For quick iterations (no bibliography changes), use **"Quick LaTeX (no bib)"**.

### Option B: Terminal

```bash
cd PINN_ELM
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Output: `PINN_ELM/main.pdf`

---

## 3. Editing the Manuscript

### ⚠️ IMPORTANT: Follow the Current Structure

The manuscript is a **single-file LaTeX document** (`PINN_ELM/main.tex`) using the `elsarticle` document class (Elsevier journal format). **Keep it this way for simplicity.**

#### Current Document Structure in `main.tex`:

| Section | Description |
|---------|-------------|
| **Frontmatter** | Title, author (Quang Nguyen), abstract, keywords |
| **§1 Introduction** | PINNs limitations, EM-ELM proposal, key contributions |
| **§2 (Related Work)** | PINNs, ELMs, Neural Operators — embedded in intro |
| **§3 Methods** | Power-Law initialization, energy functional, spectral bias resolution |
| **§4 (Theoretical Analysis)** | Complexity derivation, speedup factor proof |
| **Table 1** | PINN vs. ELM vs. EM-ELM paradigm comparison |
| **§5 Results** | Statistical performance, field reconstruction, box-whisker plot |
| **Figure 1** | Box-and-whisker plot (`figure2_boxplot.png`) |
| **§6 Discussion** | Power-Law advantage, real-time deployment, energy minimization |
| **§7 Conclusions** | Currently placeholder ("x") — **needs writing** |
| **Bibliography** | Uses `elsarticle-num` style, `references.bib` |

#### Rules When Editing:

1. **Do NOT split into multiple `.tex` files** — keep everything in `main.tex`.
2. **Do NOT change the document class** — it must remain `elsarticle` (Elsevier format).
3. **Use `\cite{}` for citations** — all references are in `references.bib`.
4. **Run the full build** (with bibtex) after adding/changing any `\cite{}` commands.
5. **Figures go in `PINN_ELM/`** alongside `main.tex`.

---

## 4. References (`references.bib`)

### Current References:

| Key | Paper |
|-----|-------|
| `raissi2019physics` | Raissi et al. — PINNs (Journal of Computational Physics, 2019) |
| `huang2006extreme` | Huang et al. — ELM Theory (Neurocomputing, 2006) |
| `jacot2018neural` | Jacot et al. — Neural Tangent Kernel (NeurIPS, 2018) |
| `dong2021extreme` | Dong & Li — ELM for PINNs (arXiv, 2021) |
| `li2020fourier` | Li et al. — Fourier Neural Operator (arXiv, 2020) |
| `cuomo2022scientific` | Cuomo et al. — SciML Survey (J. Scientific Computing, 2022) |
| `tikhonov1977solutions` | Tikhonov & Arsenin — Ill-Posed Problems (1977) |
| `basri2020frequency` | Basri et al. — Frequency Bias (ICML, 2020) |

#### Rules When Adding References:

1. **Add new entries to `references.bib`** — standard BibTeX format.
2. **Use descriptive cite keys** like `authorYEARkeyword` (e.g., `smith2023deeplearning`).
3. **Avoid duplicate entries** — the file currently has some duplicates (`raissi2019physics`, `huang2006extreme`, `jacot2018neural` appear twice). Clean these up if editing.
4. **The bibliography uses `\nocite{*}`** — all entries in `.bib` are printed regardless of in-text citations.

---

## 5. Known Issues to Fix

1. **`\Section` typo** (line ~99 area): `\Section{Theoretical Complexity Analysis...}` should be `\section{...}` (lowercase `s`).
2. **Conclusions section is empty** — currently just "x".
3. **Duplicate bib entries** — `raissi2019physics`, `huang2006extreme`, `jacot2018neural` are defined twice in `references.bib`.
4. **Long unbroken paragraphs** — Some sections in Methods/Introduction are wall-of-text without proper LaTeX paragraph breaks.
5. **Missing `\cite{}` commands** — The text references Raissi et al., Huang et al. etc. by name but many lack actual `\cite{}` calls.

---

## 6. Running the Jupyter Notebook

```bash
source .venv/bin/activate
jupyter notebook ELM_Energy_Minimization.ipynb
```

Or open it directly in VS Code (the `.venv` kernel will be auto-detected).

---

## Quick Reference

| Task | Command |
|------|---------|
| Activate venv | `source .venv/bin/activate` |
| Build PDF (full) | `Cmd+Shift+B` → "Build LaTeX PDF" |
| Build PDF (quick) | `Cmd+Shift+B` → "Quick LaTeX (no bib)" |
| Open notebook | Open `ELM_Energy_Minimization.ipynb` in VS Code |
| Add a figure | Place `.png/.pdf` in `PINN_ELM/`, use `\includegraphics` |
| Add a reference | Add to `references.bib`, use `\cite{key}` in `main.tex` |
