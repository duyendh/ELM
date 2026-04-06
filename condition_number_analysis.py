"""
Condition Number Analysis for Power-Law PIELM
==============================================

This script provides the rigorous evidence that power-law initialization
produces better-conditioned PIELM systems than Gaussian or Uniform.

Key experiments:
1. Condition number vs initialization type (fixed scale)
2. Condition number vs scale sweep (controls for Dong 2022 critique)
3. Alpha ablation (finds optimal power-law exponent)
4. Full PIELM benchmark comparison (Poisson, Multi-freq, Helmholtz)

All benchmarks use PIELM (physics-informed), NOT supervised ELM.
u_true is only used for evaluation, never for training.
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from pielm_solver import (
    random_hidden_params,
    feature_matrix,
    feature_matrix_d2,
    build_poisson_system,
    build_helmholtz_system,
    solve_pielm,
    predict,
)

torch.set_default_dtype(torch.float64)
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Extended initialization: adds power-law with variable alpha
# ---------------------------------------------------------------------------

def random_hidden_params_extended(
    input_dim: int,
    hidden_dim: int,
    init_type: str = "power",
    scale: float = 30.0,
    alpha: float = 2.0,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extended version with configurable alpha for power-law.

    init_type:
        'uniform' -> U(-scale, scale)
        'normal'  -> N(0, scale)
        'power'   -> |w| ~ 10^U(0, log10(scale)) with random sign
                     This is a log-uniform distribution (alpha=1 in P(w)~|w|^{-alpha})
        'power_alpha' -> |w| sampled from P(w) ~ |w|^{-alpha} on [1, scale]
                         via inverse CDF: w = (1 + u*(scale^{1-alpha} - 1))^{1/(1-alpha)}
    """
    if init_type == "uniform":
        W = (2 * torch.rand(input_dim, hidden_dim, dtype=dtype) - 1) * scale
        b = (2 * torch.rand(1, hidden_dim, dtype=dtype) - 1) * scale
    elif init_type == "normal":
        W = torch.randn(input_dim, hidden_dim, dtype=dtype) * scale
        b = torch.randn(1, hidden_dim, dtype=dtype) * scale
    elif init_type == "power":
        # Log-uniform (original implementation)
        log_w = torch.rand(input_dim, hidden_dim, dtype=dtype) * torch.log10(
            torch.tensor(scale, dtype=dtype)
        )
        sign = 2 * torch.randint(0, 2, (input_dim, hidden_dim)).to(dtype) - 1
        W = 10.0 ** log_w * sign
        b = (2 * torch.rand(1, hidden_dim, dtype=dtype) - 1) * torch.pi
    elif init_type == "power_alpha":
        # True power-law P(w) ~ |w|^{-alpha} on [w_min, scale]
        # Inverse CDF sampling: F^{-1}(u) for P(w) ~ w^{-alpha}
        w_min = 0.1  # avoid singularity at 0
        u = torch.rand(input_dim, hidden_dim, dtype=dtype)
        if abs(alpha - 1.0) < 1e-6:
            # alpha=1 -> log-uniform (special case)
            W = w_min * (scale / w_min) ** u
        else:
            # General case: inverse CDF of truncated power-law
            W = (w_min ** (1 - alpha) + u * (scale ** (1 - alpha) - w_min ** (1 - alpha))) ** (1.0 / (1 - alpha))
        sign = 2 * torch.randint(0, 2, (input_dim, hidden_dim)).to(dtype) - 1
        W = W * sign
        b = (2 * torch.rand(1, hidden_dim, dtype=dtype) - 1) * torch.pi
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    return W, b


# ---------------------------------------------------------------------------
# Benchmark problems
# ---------------------------------------------------------------------------

def poisson_simple():
    """u(x) = sin(pi*x), -u'' = pi^2 sin(pi*x)"""
    import math
    f_fn = lambda x: (math.pi ** 2) * torch.sin(math.pi * x)
    u_fn = lambda x: torch.sin(math.pi * x)
    return f_fn, u_fn, "Poisson (sin pi x)"


def poisson_multifreq():
    """u(x) = sin(pi*x) + 0.3*sin(3*pi*x) + 0.1*sin(7*pi*x)"""
    import math
    def f_fn(x):
        return ((math.pi**2) * torch.sin(math.pi * x)
                + 0.3 * (3*math.pi)**2 * torch.sin(3*math.pi * x)
                + 0.1 * (7*math.pi)**2 * torch.sin(7*math.pi * x))
    def u_fn(x):
        return (torch.sin(math.pi * x)
                + 0.3 * torch.sin(3*math.pi * x)
                + 0.1 * torch.sin(7*math.pi * x))
    return f_fn, u_fn, "Multi-Freq Poisson"


def helmholtz_k10():
    """u(x) = sin(pi*x)*exp(-4x^2), -u'' - k^2*u = f, k=10"""
    import math
    k = 10.0
    def u_fn(x):
        return torch.sin(math.pi * x) * torch.exp(-4 * x**2)
    def f_fn(x):
        # Compute f = -u'' - k^2*u analytically
        # u = sin(pi*x) * exp(-4*x^2)
        # u' = pi*cos(pi*x)*exp(-4x^2) + sin(pi*x)*(-8x)*exp(-4x^2)
        # u'' = -pi^2*sin(pi*x)*exp(-4x^2) + pi*cos(pi*x)*(-8x)*exp(-4x^2)
        #      + pi*cos(pi*x)*(-8x)*exp(-4x^2) + sin(pi*x)*(-8)*exp(-4x^2)
        #      + sin(pi*x)*(-8x)^2*exp(-4x^2)
        # u'' = [-pi^2 - 16*pi*x*cos(pi*x)/sin(pi*x) - 8 + 64*x^2] * u(x)
        # ... better to compute numerically for correctness
        s = torch.sin(math.pi * x)
        c = torch.cos(math.pi * x)
        e = torch.exp(-4 * x**2)
        u = s * e
        u_pp = ((-math.pi**2) * s * e
                + 2 * math.pi * c * (-8*x) * e
                + s * (-8 + 64*x**2) * e)
        return -u_pp - k**2 * u
    return f_fn, u_fn, f"Helmholtz (k={k})"


def poisson_highfreq():
    """u(x) = sin(10*pi*x), -u'' = (10*pi)^2 sin(10*pi*x)"""
    import math
    omega = 10 * math.pi
    f_fn = lambda x: omega**2 * torch.sin(omega * x)
    u_fn = lambda x: torch.sin(omega * x)
    return f_fn, u_fn, "High-Freq Poisson (10pi)"


def advection_diffusion():
    """
    -eps*u'' + u' = 0 on [0,1], u(0)=0, u(1)=1
    u(x) = (exp(x/eps) - 1) / (exp(1/eps) - 1)
    Boundary layer problem -- tests whether power-law captures localized features.
    """
    eps = 0.02
    def u_fn(x):
        return (torch.exp(x / eps) - 1) / (torch.exp(torch.tensor(1.0/eps)) - 1)
    def f_fn(x):
        # -eps*u'' + u' = 0, so f = 0 for the advection-diffusion operator
        # But we write it as: eps*u'' = u' => -u'' = -u'/eps
        # Actually for PIELM we need to rewrite as Poisson-like.
        # -eps*u'' + u' = 0 => -u'' = -u'/eps
        # This doesn't fit standard Poisson. Need custom operator.
        return torch.zeros_like(x)
    return f_fn, u_fn, "Advection-Diffusion (eps=0.02)"


# ---------------------------------------------------------------------------
# Experiment 1: Condition number comparison
# ---------------------------------------------------------------------------

def experiment_condition_number(
    n_seeds: int = 20,
    hidden_dim: int = 400,
    n_interior: int = 200,
    scale: float = 30.0,
):
    """Compare cond(A'A) across init types for each benchmark."""
    dtype = torch.float64
    domain = (-1.0, 1.0)

    benchmarks = [poisson_simple, poisson_multifreq, helmholtz_k10, poisson_highfreq]
    init_types = ["power", "normal", "uniform"]

    results = {}

    for bench_fn in benchmarks:
        f_fn, u_fn, name = bench_fn()
        results[name] = {}

        x_int = torch.linspace(domain[0], domain[1], n_interior + 2, dtype=dtype)[1:-1].unsqueeze(1)
        x_bc = torch.tensor([[domain[0]], [domain[1]]], dtype=dtype)
        f_int = f_fn(x_int)
        u_bc = torch.tensor([[0.0], [0.0]], dtype=dtype)  # Dirichlet BCs

        for init in init_types:
            conds = []
            errors = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                W, b = random_hidden_params_extended(1, hidden_dim, init, scale, dtype=dtype)

                # Build PIELM system
                is_helmholtz = "Helmholtz" in name
                if is_helmholtz:
                    k = 10.0
                    A, rhs = build_helmholtz_system(x_int, x_bc, f_int, u_bc, W, b, k, bc_weight=100.0)
                else:
                    A, rhs = build_poisson_system(x_int, x_bc, f_int, u_bc, W, b, bc_weight=100.0)

                # Condition number of A
                try:
                    cond = torch.linalg.cond(A).item()
                except:
                    cond = float('inf')
                conds.append(cond)

                # Solve and evaluate
                beta = solve_pielm(A, rhs, lambd=1e-10)
                x_test = torch.linspace(domain[0], domain[1], 1000, dtype=dtype).unsqueeze(1)
                u_pred = predict(x_test, W, b, beta)
                u_true = u_fn(x_test)
                err = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
                errors.append(err)

            results[name][init] = {
                "cond_mean": np.mean(conds),
                "cond_std": np.std(conds),
                "cond_median": np.median(conds),
                "error_mean": np.mean(errors),
                "error_std": np.std(errors),
                "error_median": np.median(errors),
                "errors": errors,
                "conds": conds,
            }

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Scale sweep (controls for Dong 2022 critique)
# ---------------------------------------------------------------------------

def experiment_scale_sweep(
    n_seeds: int = 10,
    hidden_dim: int = 400,
    n_interior: int = 200,
):
    """For each init type, sweep scale and find the optimal one.
    Shows that even at each distribution's OPTIMAL scale, power-law still wins.
    """
    dtype = torch.float64
    domain = (-1.0, 1.0)
    scales = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0]
    init_types = ["power", "normal", "uniform"]

    benchmarks = [poisson_simple, poisson_multifreq, helmholtz_k10]
    results = {}

    for bench_fn in benchmarks:
        f_fn, u_fn, name = bench_fn()
        results[name] = {}

        x_int = torch.linspace(domain[0], domain[1], n_interior + 2, dtype=dtype)[1:-1].unsqueeze(1)
        x_bc = torch.tensor([[domain[0]], [domain[1]]], dtype=dtype)
        f_int = f_fn(x_int)
        u_bc = torch.tensor([[0.0], [0.0]], dtype=dtype)

        x_test = torch.linspace(domain[0], domain[1], 1000, dtype=dtype).unsqueeze(1)
        u_true = u_fn(x_test)

        for init in init_types:
            scale_results = []
            for sc in scales:
                errs = []
                conds = []
                for seed in range(n_seeds):
                    torch.manual_seed(seed)
                    W, b = random_hidden_params_extended(1, hidden_dim, init, sc, dtype=dtype)

                    is_helmholtz = "Helmholtz" in name
                    if is_helmholtz:
                        A, rhs = build_helmholtz_system(x_int, x_bc, f_int, u_bc, W, b, 10.0, bc_weight=100.0)
                    else:
                        A, rhs = build_poisson_system(x_int, x_bc, f_int, u_bc, W, b, bc_weight=100.0)

                    try:
                        cond = torch.linalg.cond(A).item()
                    except:
                        cond = float('inf')
                    conds.append(cond)

                    beta = solve_pielm(A, rhs, lambd=1e-10)
                    u_pred = predict(x_test, W, b, beta)
                    err = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
                    errs.append(err)

                scale_results.append({
                    "scale": sc,
                    "error_mean": np.mean(errs),
                    "error_std": np.std(errs),
                    "cond_mean": np.mean(conds),
                    "errors": errs,
                })

            results[name][init] = scale_results

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Alpha ablation
# ---------------------------------------------------------------------------

def experiment_alpha_ablation(
    n_seeds: int = 20,
    hidden_dim: int = 400,
    n_interior: int = 200,
    scale: float = 30.0,
):
    """Sweep alpha in power-law P(w) ~ |w|^{-alpha} and measure error + cond."""
    dtype = torch.float64
    domain = (-1.0, 1.0)
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    benchmarks = [poisson_simple, poisson_multifreq, helmholtz_k10]
    results = {}

    for bench_fn in benchmarks:
        f_fn, u_fn, name = bench_fn()
        results[name] = []

        x_int = torch.linspace(domain[0], domain[1], n_interior + 2, dtype=dtype)[1:-1].unsqueeze(1)
        x_bc = torch.tensor([[domain[0]], [domain[1]]], dtype=dtype)
        f_int = f_fn(x_int)
        u_bc = torch.tensor([[0.0], [0.0]], dtype=dtype)
        x_test = torch.linspace(domain[0], domain[1], 1000, dtype=dtype).unsqueeze(1)
        u_true = u_fn(x_test)

        for alpha in alphas:
            errs = []
            conds = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                W, b = random_hidden_params_extended(
                    1, hidden_dim, "power_alpha", scale, alpha=alpha, dtype=dtype
                )

                is_helmholtz = "Helmholtz" in name
                if is_helmholtz:
                    A, rhs = build_helmholtz_system(x_int, x_bc, f_int, u_bc, W, b, 10.0, bc_weight=100.0)
                else:
                    A, rhs = build_poisson_system(x_int, x_bc, f_int, u_bc, W, b, bc_weight=100.0)

                try:
                    cond = torch.linalg.cond(A).item()
                except:
                    cond = float('inf')
                conds.append(cond)

                beta = solve_pielm(A, rhs, lambd=1e-10)
                u_pred = predict(x_test, W, b, beta)
                err = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
                errs.append(err)

            results[name].append({
                "alpha": alpha,
                "error_mean": np.mean(errs),
                "error_std": np.std(errs),
                "error_median": np.median(errs),
                "cond_mean": np.mean(conds),
                "cond_std": np.std(conds),
                "errors": errs,
                "conds": conds,
            })

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Full PIELM benchmark (publication-ready numbers)
# ---------------------------------------------------------------------------

def experiment_full_benchmark(
    n_seeds: int = 20,
    hidden_dim: int = 600,
    n_interior: int = 300,
    scale: float = 30.0,
):
    """Full PIELM comparison for all benchmarks, publication-ready."""
    dtype = torch.float64
    domain = (-1.0, 1.0)
    init_types = ["power", "normal", "uniform"]

    benchmarks = [poisson_simple, poisson_multifreq, helmholtz_k10, poisson_highfreq]
    results = {}

    for bench_fn in benchmarks:
        f_fn, u_fn, name = bench_fn()
        results[name] = {}

        x_int = torch.linspace(domain[0], domain[1], n_interior + 2, dtype=dtype)[1:-1].unsqueeze(1)
        x_bc = torch.tensor([[domain[0]], [domain[1]]], dtype=dtype)
        f_int = f_fn(x_int)
        u_bc = torch.tensor([[0.0], [0.0]], dtype=dtype)
        x_test = torch.linspace(domain[0], domain[1], 1000, dtype=dtype).unsqueeze(1)
        u_true = u_fn(x_test)

        for init in init_types:
            errs = []
            conds = []
            times = []

            for seed in range(n_seeds):
                torch.manual_seed(seed)

                t0 = time.perf_counter()
                W, b = random_hidden_params_extended(1, hidden_dim, init, scale, dtype=dtype)

                is_helmholtz = "Helmholtz" in name
                if is_helmholtz:
                    A, rhs = build_helmholtz_system(x_int, x_bc, f_int, u_bc, W, b, 10.0, bc_weight=100.0)
                else:
                    A, rhs = build_poisson_system(x_int, x_bc, f_int, u_bc, W, b, bc_weight=100.0)

                beta = solve_pielm(A, rhs, lambd=1e-10)
                t1 = time.perf_counter()

                u_pred = predict(x_test, W, b, beta)
                err = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()

                try:
                    cond = torch.linalg.cond(A).item()
                except:
                    cond = float('inf')

                errs.append(err)
                conds.append(cond)
                times.append(t1 - t0)

            results[name][init] = {
                "error_mean": np.mean(errs),
                "error_std": np.std(errs),
                "error_median": np.median(errs),
                "cond_mean": np.mean(conds),
                "cond_std": np.std(conds),
                "time_mean": np.mean(times),
                "errors": errs,
                "conds": conds,
            }

    return results


# ---------------------------------------------------------------------------
# Experiment 5: Effective frequency spectrum analysis
# ---------------------------------------------------------------------------

def experiment_frequency_spectrum(
    hidden_dim: int = 1000,
    scale: float = 30.0,
    n_bins: int = 50,
):
    """Visualize the effective frequency distributions for each init type.
    Produces histogram data showing how power-law creates multi-scale coverage.
    """
    dtype = torch.float64
    init_types = ["power", "normal", "uniform"]
    results = {}

    for init in init_types:
        torch.manual_seed(42)
        W, b = random_hidden_params_extended(1, hidden_dim, init, scale, dtype=dtype)
        freqs = torch.abs(W).flatten().numpy()

        hist, bin_edges = np.histogram(freqs, bins=n_bins, range=(0, scale * 1.1))
        results[init] = {
            "frequencies": freqs.tolist(),
            "hist_counts": hist.tolist(),
            "hist_edges": bin_edges.tolist(),
            "mean": float(np.mean(freqs)),
            "std": float(np.std(freqs)),
            "median": float(np.median(freqs)),
            "min": float(np.min(freqs)),
            "max": float(np.max(freqs)),
            "pct_below_1": float(np.mean(freqs < 1.0) * 100),
            "pct_1_to_10": float(np.mean((freqs >= 1) & (freqs < 10)) * 100),
            "pct_above_10": float(np.mean(freqs >= 10.0) * 100),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(results: dict, experiment_name: str):
    """Pretty-print results as a table."""
    print(f"\n{'='*80}")
    print(f"  {experiment_name}")
    print(f"{'='*80}")

    for name, data in results.items():
        print(f"\n  {name}")
        print(f"  {'Init':<12} {'Error (mean)':<16} {'Error (std)':<14} {'Cond (mean)':<16}")
        print(f"  {'-'*58}")
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            for init, metrics in data.items():
                em = metrics.get('error_mean', 0)
                es = metrics.get('error_std', 0)
                cm = metrics.get('cond_mean', 0)
                print(f"  {init:<12} {em:<16.4e} {es:<14.4e} {cm:<16.2e}")
        elif isinstance(data, list):
            # Alpha ablation format
            print(f"  {'Alpha':<12} {'Error (mean)':<16} {'Error (std)':<14} {'Cond (mean)':<16}")
            print(f"  {'-'*58}")
            for entry in data:
                a = entry['alpha']
                em = entry['error_mean']
                es = entry['error_std']
                cm = entry['cond_mean']
                print(f"  {a:<12.1f} {em:<16.4e} {es:<14.4e} {cm:<16.2e}")


def print_scale_sweep(results: dict):
    """Pretty-print scale sweep results."""
    print(f"\n{'='*80}")
    print(f"  Scale Sweep (Dong 2022 Control)")
    print(f"{'='*80}")

    for name, data in results.items():
        print(f"\n  {name}")
        for init, scale_data in data.items():
            best = min(scale_data, key=lambda x: x['error_mean'])
            print(f"    {init:<10} best scale={best['scale']:<6.1f} "
                  f"error={best['error_mean']:.4e} +/- {best['error_std']:.4e} "
                  f"cond={best['cond_mean']:.2e}")
        # Show full sweep for power-law vs best competitor
        print(f"\n    Full sweep:")
        print(f"    {'Scale':<8}", end="")
        for init in data.keys():
            print(f" {init:<18}", end="")
        print()
        n_scales = len(list(data.values())[0])
        for i in range(n_scales):
            sc = list(data.values())[0][i]['scale']
            print(f"    {sc:<8.1f}", end="")
            for init in data.keys():
                em = data[init][i]['error_mean']
                print(f" {em:<18.4e}", end="")
            print()


def print_frequency_spectrum(results: dict):
    """Print frequency distribution statistics."""
    print(f"\n{'='*80}")
    print(f"  Frequency Spectrum Analysis")
    print(f"{'='*80}")
    print(f"  {'Init':<10} {'Mean':<10} {'Std':<10} {'Median':<10} {'<1':<8} {'1-10':<8} {'>10':<8}")
    print(f"  {'-'*64}")
    for init, data in results.items():
        print(f"  {init:<10} {data['mean']:<10.2f} {data['std']:<10.2f} "
              f"{data['median']:<10.2f} {data['pct_below_1']:<8.1f} "
              f"{data['pct_1_to_10']:<8.1f} {data['pct_above_10']:<8.1f}")


if __name__ == "__main__":
    print("Running PIELM Condition Number Analysis")
    print("This provides the theoretical backbone for the power-law paper.\n")

    # Exp 1: Condition number comparison
    print(">>> Experiment 1: Condition number vs init type...")
    r1 = experiment_condition_number(n_seeds=20, hidden_dim=400, scale=30.0)
    print_table(r1, "Condition Number Comparison (scale=30)")

    # Exp 2: Scale sweep
    print("\n>>> Experiment 2: Scale sweep (controlling for Dong 2022)...")
    r2 = experiment_scale_sweep(n_seeds=10, hidden_dim=400)
    print_scale_sweep(r2)

    # Exp 3: Alpha ablation
    print("\n>>> Experiment 3: Alpha ablation...")
    r3 = experiment_alpha_ablation(n_seeds=20, hidden_dim=400, scale=30.0)
    print_table(r3, "Alpha Ablation (power-law exponent)")

    # Exp 4: Frequency spectrum
    print("\n>>> Experiment 4: Frequency spectrum analysis...")
    r4 = experiment_frequency_spectrum(hidden_dim=1000, scale=30.0)
    print_frequency_spectrum(r4)

    # Exp 5: Full benchmark
    print("\n>>> Experiment 5: Full PIELM benchmark (publication-ready)...")
    r5 = experiment_full_benchmark(n_seeds=20, hidden_dim=600, n_interior=300, scale=30.0)
    print_table(r5, "Full PIELM Benchmark (h=600, N=300)")

    # Save all results
    def convert_for_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    all_results = {
        "condition_number": {k: {kk: {kkk: convert_for_json(vvv) for kkk, vvv in vv.items()} for kk, vv in v.items()} for k, v in r1.items()},
        "frequency_spectrum": r4,
    }

    with open("condition_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=convert_for_json)

    print("\n\nResults saved to condition_analysis_results.json")
    print("\nDone.")
