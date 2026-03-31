"""
Physics-Informed Extreme Learning Machine (PIELM) Solver
=========================================================

Key insight: standard ELM solves  H @ beta = u_true  (supervised).
PIELM instead enforces the PDE + boundary conditions as a linear system,
requiring NO known solution u_true for training.

For activation phi(x) = sin(w*x + b):
    phi'(x)  =  w * cos(w*x + b)
    phi''(x) = -w^2 * sin(w*x + b) = -w^2 * phi(x)

This closed-form derivative is the key property that keeps the system linear.

Mathematical framework
----------------------
Trial solution:  u(x) = sum_j  beta_j * sin(w_j * x + b_j)

Poisson  -u'' = f:
    sum_j beta_j * w_j^2 * sin(w_j * x_i + b_j) = f(x_i)       (PDE rows)
    sum_j beta_j * sin(w_j * (-1) + b_j) = 0                     (left BC)
    sum_j beta_j * sin(w_j * (+1) + b_j) = 0                     (right BC)

Helmholtz  -u'' - k^2 u = f:
    sum_j beta_j * (w_j^2 - k^2) * sin(w_j * x_i + b_j) = f(x_i)
    + same BC rows

Stacked system:  A @ beta = rhs,  solved via least-squares.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable, Dict, Any


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def random_hidden_params(
    input_dim: int,
    hidden_dim: int,
    init_type: str = "uniform",
    scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate frozen random weights and biases for the hidden layer.

    Parameters
    ----------
    input_dim : int
        Spatial dimension (1 for 1-D problems).
    hidden_dim : int
        Number of hidden neurons L.
    init_type : str
        'uniform' -> U(-scale, scale)
        'normal'  -> N(0, scale)
        'power'   -> 10^U(0, log10(scale))  (log-uniform, good for
                     multi-scale problems)
    scale : float
        Controls the spread of random frequencies.
    dtype : torch.dtype

    Returns
    -------
    W : (input_dim, hidden_dim)
    b : (1, hidden_dim)
    """
    if init_type == "uniform":
        W = (2 * torch.rand(input_dim, hidden_dim, dtype=dtype) - 1) * scale
        b = (2 * torch.rand(1, hidden_dim, dtype=dtype) - 1) * scale
    elif init_type == "normal":
        W = torch.randn(input_dim, hidden_dim, dtype=dtype) * scale
        b = torch.randn(1, hidden_dim, dtype=dtype) * scale
    elif init_type == "power":
        # Log-uniform frequencies: captures both low and high modes
        log_w = torch.rand(input_dim, hidden_dim, dtype=dtype) * torch.log10(
            torch.tensor(scale, dtype=dtype)
        )
        W = 10.0 ** log_w * (2 * torch.randint(0, 2, (input_dim, hidden_dim)).to(dtype) - 1)
        b = (2 * torch.rand(1, hidden_dim, dtype=dtype) - 1) * torch.pi
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    return W, b


def feature_matrix(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute the base feature matrix H[i,j] = sin(w_j * x_i + b_j).

    Parameters
    ----------
    x : (N, d) collocation points
    W : (d, L) random weights
    b : (1, L) random biases

    Returns
    -------
    H : (N, L)
    """
    return torch.sin(x @ W + b)


def feature_matrix_d2(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Second-derivative feature matrix: H_d2[i,j] = -w_j^2 * sin(w_j*x_i + b_j).

    Since  d^2/dx^2 sin(w*x+b) = -w^2 sin(w*x+b),  this is exact (no FD error).

    Returns
    -------
    H_d2 : (N, L)
    """
    H = feature_matrix(x, W, b)
    w_sq = (W ** 2).sum(dim=0, keepdim=True)  # (1, L) -- works for d>=1
    return -w_sq * H


# ---------------------------------------------------------------------------
# Linear system constructors for each PDE
# ---------------------------------------------------------------------------

def build_poisson_system(
    x_int: torch.Tensor,
    x_bc: torch.Tensor,
    f_int: torch.Tensor,
    u_bc: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    bc_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the PIELM linear system for 1-D Poisson:  -u''(x) = f(x).

    The operator  L[u] = -u''  applied to trial function gives:
        L[phi_j](x) = w_j^2 * sin(w_j*x + b_j)

    System layout (stacked):
        [ H_pde  ]         [ f_int  ]
        [ w * H_bc ] beta = [ w * u_bc ]

    where w = bc_weight (penalises BC violation).

    Parameters
    ----------
    x_int  : (N_int, 1)  interior collocation points
    x_bc   : (N_bc, 1)   boundary points, e.g. [[-1], [1]]
    f_int  : (N_int, 1)  source evaluated at interior points
    u_bc   : (N_bc, 1)   prescribed boundary values
    W, b   : frozen hidden-layer parameters
    bc_weight : float     weight multiplier for BC rows (default 1.0)

    Returns
    -------
    A   : (N_int + N_bc, L)
    rhs : (N_int + N_bc, 1)
    """
    # PDE rows:  -u'' = f  =>  sum_j beta_j * w_j^2 * sin(...) = f
    # feature_matrix_d2 returns -w^2 * sin, so negate it for the -u'' operator
    H_pde = -feature_matrix_d2(x_int, W, b)  # (N_int, L):  w_j^2 * sin(...)

    # BC rows:  u(x_bc) = u_bc  =>  sum_j beta_j * sin(w_j*x_bc + b_j) = u_bc
    H_bc = feature_matrix(x_bc, W, b)  # (N_bc, L)

    A = torch.cat([H_pde, bc_weight * H_bc], dim=0)
    rhs = torch.cat([f_int, bc_weight * u_bc], dim=0)

    return A, rhs


def build_helmholtz_system(
    x_int: torch.Tensor,
    x_bc: torch.Tensor,
    f_int: torch.Tensor,
    u_bc: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    k: float,
    bc_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the PIELM linear system for 1-D Helmholtz:  -u'' - k^2 u = f.

    Operator applied to each basis function:
        L[phi_j](x) = w_j^2 * sin(...) - k^2 * sin(...)
                     = (w_j^2 - k^2) * sin(w_j*x + b_j)

    Parameters
    ----------
    k : float   Helmholtz wave number

    Returns
    -------
    A, rhs : same layout as Poisson
    """
    H_base = feature_matrix(x_int, W, b)          # (N_int, L)
    w_sq = (W ** 2).sum(dim=0, keepdim=True)       # (1, L)
    H_pde = (w_sq - k ** 2) * H_base              # (N_int, L)

    H_bc = feature_matrix(x_bc, W, b)

    A = torch.cat([H_pde, bc_weight * H_bc], dim=0)
    rhs = torch.cat([f_int, bc_weight * u_bc], dim=0)

    return A, rhs


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_pielm(
    A: torch.Tensor,
    rhs: torch.Tensor,
    lambd: float = 1e-10,
) -> torch.Tensor:
    """Solve the over-determined system A @ beta = rhs via regularised least-squares.

    Solves:  (A^T A + lambda I) beta = A^T rhs

    Parameters
    ----------
    A     : (M, L)  stacked PDE + BC feature matrix
    rhs   : (M, 1)  stacked right-hand side
    lambd : float    Tikhonov regularisation

    Returns
    -------
    beta : (L, 1)
    """
    ATA = A.t() @ A                                        # (L, L)
    ATA += lambd * torch.eye(ATA.shape[0], dtype=A.dtype)  # regularise
    ATrhs = A.t() @ rhs                                    # (L, 1)
    beta = torch.linalg.solve(ATA, ATrhs)
    return beta


def predict(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the PIELM solution at arbitrary points.

    u(x) = H(x) @ beta = sin(x @ W + b) @ beta
    """
    H = feature_matrix(x, W, b)
    return H @ beta


# ---------------------------------------------------------------------------
# High-level convenience solvers
# ---------------------------------------------------------------------------

def solve_poisson_1d(
    f_fn: Callable[[torch.Tensor], torch.Tensor],
    n_interior: int = 200,
    hidden_dim: int = 300,
    domain: Tuple[float, float] = (-1.0, 1.0),
    bc_values: Tuple[float, float] = (0.0, 0.0),
    init_type: str = "power",
    scale: float = 30.0,
    lambd: float = 1e-10,
    bc_weight: float = 10.0,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Solve  -u''(x) = f(x)  on [a,b] with Dirichlet BCs.

    Parameters
    ----------
    f_fn         : callable, source term  f(x) -> (N,1)
    n_interior   : number of interior collocation points
    hidden_dim   : number of random neurons  L
    domain       : (a, b)
    bc_values    : (u(a), u(b))
    init_type    : weight initialisation strategy
    scale        : frequency scale for random weights
    lambd        : Tikhonov regularisation
    bc_weight    : penalty weight for boundary condition rows
    dtype        : torch dtype

    Returns
    -------
    dict with keys: 'beta', 'W', 'b', 'x_int', 'predict_fn'
    """
    a, b_dom = domain

    # Collocation points (interior, excluding boundary)
    x_int = torch.linspace(a, b_dom, n_interior + 2, dtype=dtype)[1:-1].unsqueeze(1)
    x_bc = torch.tensor([[a], [b_dom]], dtype=dtype)
    f_int = f_fn(x_int)
    u_bc = torch.tensor([[bc_values[0]], [bc_values[1]]], dtype=dtype)

    # Random hidden layer (frozen)
    W, bias = random_hidden_params(1, hidden_dim, init_type, scale, dtype)

    # Build and solve
    A, rhs = build_poisson_system(x_int, x_bc, f_int, u_bc, W, bias, bc_weight)
    beta = solve_pielm(A, rhs, lambd)

    return {
        "beta": beta,
        "W": W,
        "b": bias,
        "x_int": x_int,
        "predict_fn": lambda x: predict(x, W, bias, beta),
    }


def solve_helmholtz_1d(
    f_fn: Callable[[torch.Tensor], torch.Tensor],
    k: float,
    n_interior: int = 200,
    hidden_dim: int = 300,
    domain: Tuple[float, float] = (-1.0, 1.0),
    bc_values: Tuple[float, float] = (0.0, 0.0),
    init_type: str = "power",
    scale: float = 30.0,
    lambd: float = 1e-10,
    bc_weight: float = 10.0,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Solve  -u''(x) - k^2 u(x) = f(x)  on [a,b] with Dirichlet BCs."""
    a, b_dom = domain

    x_int = torch.linspace(a, b_dom, n_interior + 2, dtype=dtype)[1:-1].unsqueeze(1)
    x_bc = torch.tensor([[a], [b_dom]], dtype=dtype)
    f_int = f_fn(x_int)
    u_bc = torch.tensor([[bc_values[0]], [bc_values[1]]], dtype=dtype)

    W, bias = random_hidden_params(1, hidden_dim, init_type, scale, dtype)

    A, rhs = build_helmholtz_system(x_int, x_bc, f_int, u_bc, W, bias, k, bc_weight)
    beta = solve_pielm(A, rhs, lambd)

    return {
        "beta": beta,
        "W": W,
        "b": bias,
        "x_int": x_int,
        "predict_fn": lambda x: predict(x, W, bias, beta),
    }


# ---------------------------------------------------------------------------
# Demo / validation
# ---------------------------------------------------------------------------

def demo():
    """Run all three benchmarks and print errors."""
    import math

    dtype = torch.float64
    N_test = 1000
    x_test = torch.linspace(-1, 1, N_test, dtype=dtype).unsqueeze(1)

    print("=" * 70)
    print("PIELM Solver -- Benchmark Results")
    print("=" * 70)

    # --- Benchmark 1: Simple Poisson ---
    # Manufactured solution:  u(x) = sin(pi*x),  so  -u'' = pi^2 sin(pi*x)
    u_exact_1 = torch.sin(math.pi * x_test)
    f_fn_1 = lambda x: (math.pi ** 2) * torch.sin(math.pi * x)

    result_1 = solve_poisson_1d(f_fn_1, n_interior=200, hidden_dim=400,
                                 scale=30.0, bc_weight=100.0)
    u_pred_1 = result_1["predict_fn"](x_test)
    err_1 = torch.norm(u_pred_1 - u_exact_1) / torch.norm(u_exact_1)
    print(f"\n[1] Poisson  -u'' = pi^2 sin(pi x)")
    print(f"    L2 relative error: {err_1.item():.2e}")

    # --- Benchmark 2: Multi-frequency Poisson ---
    # u(x) = sin(pi*x) + 0.5*sin(5*pi*x)
    # -u'' = pi^2 sin(pi*x) + 0.5*(5*pi)^2 sin(5*pi*x)
    u_exact_2 = torch.sin(math.pi * x_test) + 0.5 * torch.sin(5 * math.pi * x_test)
    f_fn_2 = lambda x: (math.pi ** 2) * torch.sin(math.pi * x) + \
                        0.5 * (5 * math.pi) ** 2 * torch.sin(5 * math.pi * x)

    result_2 = solve_poisson_1d(f_fn_2, n_interior=300, hidden_dim=600,
                                 scale=50.0, bc_weight=100.0, init_type="power")
    u_pred_2 = result_2["predict_fn"](x_test)
    err_2 = torch.norm(u_pred_2 - u_exact_2) / torch.norm(u_exact_2)
    print(f"\n[2] Multi-freq Poisson  -u'' = pi^2 sin(pi x) + 12.5 pi^2 sin(5 pi x)")
    print(f"    L2 relative error: {err_2.item():.2e}")

    # --- Benchmark 3: Helmholtz ---
    # u(x) = sin(pi*x),  -u'' - k^2 u = (pi^2 - k^2) sin(pi*x)
    k = 3.0
    u_exact_3 = torch.sin(math.pi * x_test)
    f_fn_3 = lambda x: (math.pi ** 2 - k ** 2) * torch.sin(math.pi * x)

    result_3 = solve_helmholtz_1d(f_fn_3, k=k, n_interior=200, hidden_dim=400,
                                   scale=30.0, bc_weight=100.0)
    u_pred_3 = result_3["predict_fn"](x_test)
    err_3 = torch.norm(u_pred_3 - u_exact_3) / torch.norm(u_exact_3)
    print(f"\n[3] Helmholtz  -u'' - {k}^2 u = ({math.pi**2 - k**2:.4f}) sin(pi x)")
    print(f"    L2 relative error: {err_3.item():.2e}")

    print("\n" + "=" * 70)
    print("Burgers equation analysis: see docstring in burgers_discussion()")
    print("=" * 70)


def burgers_discussion() -> str:
    """
    Burgers equation:  u_t + u * u_x = nu * u_xx

    Why PIELM cannot handle it directly
    ------------------------------------
    The trial solution is:  u(x,t) = sum_j beta_j * phi_j(x,t)

    The nonlinear term  u * u_x  expands to:

        [sum_j beta_j phi_j] * [sum_k beta_k phi_k']

    This is QUADRATIC in beta (beta_j * beta_k products), so the PDE
    residual is no longer a linear function of beta. The core PIELM
    assumption -- that we can write A @ beta = rhs -- breaks down.

    Options for handling Burgers (or any nonlinear PDE)
    ---------------------------------------------------

    1. QUASILINEARISATION (Newton-like iteration):
       - Start with initial guess u^{(0)} (e.g., from initial condition)
       - At iteration n, linearise:  u^{(n)} * u_x^{(n+1)} + u^{(n+1)} * u_x^{(n)}
         This gives a linear PDE for u^{(n+1)}, solvable by PIELM
       - Repeat until convergence
       - Pro: preserves the PIELM framework at each step
       - Con: requires iteration, convergence not guaranteed for large Re

    2. TIME-STEPPING + PIELM:
       - Discretise in time (e.g., backward Euler):
         (u^{n+1} - u^n)/dt + u^n * u_x^{n+1} = nu * u_xx^{n+1}
       - At each time step, u^n is known from previous step, so the PDE
         for u^{n+1} is LINEAR:
         nu * u_xx^{n+1} - u^n * u_x^{n+1} - u^{n+1}/dt = -u^n/dt
       - This is a sequence of linear elliptic problems, each solvable by PIELM
       - Pro: robust, well-understood stability
       - Con: small dt needed, accumulates error over time steps

    3. HYBRID PIELM + GRADIENT DESCENT:
       - Use PIELM for the linear part (nu * u_xx)
       - Treat the nonlinear residual u*u_x as a correction
       - Iterate: solve linear part with PIELM, update nonlinear part
       - Similar to operator splitting

    4. FULL PINN (abandon PIELM for this equation):
       - Use gradient-based training with automatic differentiation
       - The standard approach for nonlinear PDEs
       - Loses the single-shot solve advantage of PIELM

    Recommendation: Option 2 (time-stepping) is the most practical for
    implementation. It reduces Burgers to a sequence of linear problems,
    each solved in one shot by PIELM, and inherits the stability theory
    of implicit time-stepping schemes.
    """
    return burgers_discussion.__doc__


if __name__ == "__main__":
    demo()
