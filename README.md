# ELM — Power-Law Extreme Learning Machines for Physics-Informed Learning

Paper: *Accelerating Physics-Informed Learning via Power-Law Extreme Learning Machines*

---

## Prerequisites

- **macOS** (tested on Apple Silicon)
- **Python 3.9+**
- **TeX Live** (BasicTeX or full MacTeX)
- **VS Code**

---

## Setup Instructions

### 1. Install TeX Live (LaTeX)

If you don't have LaTeX installed:

```bash
# Option A: BasicTeX (lightweight, ~300MB)
brew install --cask basictex

# Option B: Full MacTeX (~5GB)
brew install --cask mactex
```

After installing, add TeX to your PATH and install required packages:

```bash
eval "$(/usr/libexec/path_helper)"
sudo tlmgr update --self
sudo tlmgr install courier
```

Verify:

```bash
pdflatex --version
bibtex --version
```

### 2. Clone & Set Up Python Environment

```bash
git clone <repo-url> ELM
cd ELM
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyter numpy matplotlib scipy torch scikit-learn seaborn
```

### 3. Install VS Code Extensions

Open the project in VS Code, then install:

- **LaTeX Workshop** (`james-yu.latex-workshop`) — LaTeX editing, live PDF preview, build-on-save

```bash
code --install-extension james-yu.latex-workshop
```

The project already includes `.vscode/settings.json` with LaTeX Workshop configured (build recipes, PDF viewer, word wrap, etc.).

### 4. Build the PDF

**Option A — VS Code (recommended):**

1. Open `PINN_ELM/main.tex`
2. Press `Cmd+S` to save → auto-builds the PDF
3. Press `Cmd+Option+V` to open the PDF preview side-by-side
4. Use the **TEX sidebar** (Σ icon) to navigate sections

**Option B — Terminal:**

```bash
cd PINN_ELM
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Output: `PINN_ELM/main.pdf`

### 5. Run the Jupyter Notebook

```bash
source .venv/bin/activate
jupyter notebook ELM_Energy_Minimization.ipynb
```

Or open `ELM_Energy_Minimization.ipynb` directly in VS Code.

---

## Project Structure

```
ELM/
├── .venv/                          # Python virtual environment
├── .vscode/                        # VS Code + LaTeX Workshop config
├── PINN_ELM/                       # ⛔ ORIGINAL — do not edit
│   ├── main.tex                    # Original submitted manuscript
│   └── references.bib
├── PINN_ELM_review/                # 🔴 REVIEW — working draft with review notes
│   ├── main.tex                    # Manuscript + red "NEED TO CHECK" boxes
│   └── references.bib
├── PINN_ELM_final/                 # ✅ FINAL — clean version after corrections
│   ├── main.tex                    # Post-review corrected manuscript
│   └── references.bib
├── ELM_Energy_Minimization.ipynb   # Experiment notebook
├── INSTRUCTIONS.md                 # Detailed editing guide
├── REVIEW_CHECKLIST.md             # Review checklist & notes
└── README.md                       # ← You are here
```

> **Workflow:** Edit `PINN_ELM_review/` to address review notes → once resolved, produce the clean version in `PINN_ELM_final/`. Never modify `PINN_ELM/` (the original submission).

---

## Editing Guide

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for detailed rules on editing the manuscript, adding references, and maintaining the document structure.

**Key rules:**
- Keep everything in the single `main.tex` file (elsarticle class)
- Use `references.bib` for all citations
- Run full build (`pdflatex → bibtex → pdflatex × 2`) after changing citations

---

## VS Code Shortcuts (LaTeX Workshop)

| Action | Shortcut |
|--------|----------|
| Build PDF | `Cmd+S` (auto) or `Cmd+Option+B` |
| Open PDF preview | `Cmd+Option+V` |
| Jump to PDF from source | `Cmd+Option+J` |
| Section navigator | TEX sidebar (Σ icon) |
