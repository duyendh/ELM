"""
Systematic search for where power-law initialization has a genuine edge.
Supervised ELM: H @ beta = u_true (no PDE operator to help).
Compare against Gaussian ELM and a simple PINN baseline.
"""
import torch
import torch.nn as nn
import numpy as np
import time, math
torch.set_default_dtype(torch.float64)

def init_weights(dim_in, dim_h, init_type, scale):
    if init_type == 'normal':
        W = torch.randn(dim_in, dim_h) * scale
        b = torch.randn(1, dim_h) * scale
    elif init_type == 'uniform':
        W = (2*torch.rand(dim_in, dim_h)-1) * scale
        b = (2*torch.rand(1, dim_h)-1) * scale
    elif init_type == 'power':
        log_w = torch.rand(dim_in, dim_h) * torch.log10(torch.tensor(scale))
        sign = 2*torch.randint(0,2,(dim_in, dim_h)).double() - 1
        W = 10.0**log_w * sign
        b = (2*torch.rand(1, dim_h)-1) * torch.pi
    return W, b

def elm_solve(X, y, W, b, lambd=1e-8):
    H = torch.sin(X @ W + b)
    ATA = H.t() @ H + lambd * torch.eye(H.shape[1])
    beta = torch.linalg.solve(ATA, H.t() @ y)
    return beta

def elm_predict(X, W, b, beta):
    return torch.sin(X @ W + b) @ beta

# ── Simple PINN baseline ──
class SimplePINN(nn.Module):
    def __init__(self, dim_in, dim_h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_h), nn.Tanh(),
            nn.Linear(dim_h, 1)
        )
    def forward(self, x): return self.net(x)

def train_pinn_supervised(model, X, y, lr=1e-3, epochs=2000):
    """Train PINN as supervised regression (same data as ELM, fair comparison)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(X) - y)**2).mean()
        loss.backward()
        opt.step()
    return model

# ── Target functions ──
def make_target_1d(name):
    """Returns (fn, domain, description)"""
    targets = {
        'single': (lambda x: torch.sin(math.pi*x), (-1,1), 'sin(pi*x)'),
        'multi3': (lambda x: torch.sin(math.pi*x) + 0.3*torch.sin(3*math.pi*x) + 0.1*torch.sin(7*math.pi*x),
                   (-1,1), 'sum of 3 freqs'),
        'multi5': (lambda x: torch.sin(math.pi*x) + 0.5*torch.sin(5*math.pi*x) + 0.3*torch.sin(11*math.pi*x) + 0.1*torch.sin(23*math.pi*x) + 0.05*torch.sin(47*math.pi*x),
                   (-1,1), 'sum of 5 freqs (wideband)'),
        'sharp': (lambda x: torch.tanh(20*x), (-1,1), 'tanh(20x) sharp transition'),
        'localized': (lambda x: torch.sin(10*math.pi*x) * torch.exp(-5*x**2), (-1,1), 'localized oscillation'),
        'step_approx': (lambda x: torch.sigmoid(30*x) + 0.3*torch.sin(5*math.pi*x), (-1,1), 'step + oscillation'),
        'multiscale': (lambda x: torch.sin(x) + 0.5*torch.sin(10*x) + 0.2*torch.sin(100*x), (-1,1), 'freqs spanning 2 decades'),
    }
    fn, dom, desc = targets[name]
    return fn, dom, desc

def make_target_2d(name):
    targets = {
        'product': (lambda xy: torch.sin(math.pi*xy[:,0:1]) * torch.cos(3*math.pi*xy[:,1:2]),
                    'sin(pi*x)*cos(3*pi*y)'),
        'multi2d': (lambda xy: torch.sin(math.pi*xy[:,0:1])*torch.sin(2*math.pi*xy[:,1:2]) + 
                    0.3*torch.sin(5*math.pi*xy[:,0:1])*torch.cos(7*math.pi*xy[:,1:2]),
                    'multi-freq 2D'),
        'radial': (lambda xy: torch.sin(10*torch.sqrt(xy[:,0:1]**2 + xy[:,1:2]**2 + 0.01)),
                   'radial oscillation'),
    }
    fn, desc = targets[name]
    return fn, desc

# ── Run single experiment ──
def run_elm_trial(X_train, y_train, X_test, y_test, init_type, h, scale, seed):
    torch.manual_seed(seed)
    dim_in = X_train.shape[1]
    t0 = time.perf_counter()
    W, b = init_weights(dim_in, h, init_type, scale)
    beta = elm_solve(X_train, y_train, W, b)
    t_train = time.perf_counter() - t0
    y_pred = elm_predict(X_test, W, b, beta)
    err = (torch.norm(y_pred - y_test) / torch.norm(y_test)).item()
    return err, t_train

def run_pinn_trial(X_train, y_train, X_test, y_test, h, seed, epochs=2000):
    torch.manual_seed(seed)
    dim_in = X_train.shape[1]
    model = SimplePINN(dim_in, h).double()
    t0 = time.perf_counter()
    train_pinn_supervised(model, X_train, y_train, epochs=epochs)
    t_train = time.perf_counter() - t0
    with torch.no_grad():
        y_pred = model(X_test)
    err = (torch.norm(y_pred - y_test) / torch.norm(y_test)).item()
    return err, t_train

# ═══════════════════════════════════════
# MAIN: Systematic edge search
# ═══════════════════════════════════════
if __name__ == '__main__':
    NS = 10  # seeds
    
    # ── 1D targets ──
    print('='*100)
    print('1D SUPERVISED ELM: Power-Law vs Gaussian vs Uniform vs PINN')
    print('='*100)
    
    targets_1d = ['single', 'multi3', 'multi5', 'sharp', 'localized', 'step_approx', 'multiscale']
    h_vals = [10, 20, 40, 80, 200]
    scales = [1.0, 5.0, 10.0, 30.0, 50.0]
    inits = ['power', 'normal', 'uniform']
    
    N_train = 200
    N_test = 1000
    
    for tname in targets_1d:
        fn, (a,b), desc = make_target_1d(tname)
        X_train = torch.linspace(a, b, N_train).unsqueeze(1)
        y_train = fn(X_train)
        X_test = torch.linspace(a, b, N_test).unsqueeze(1)
        y_test = fn(X_test)
        
        print(f'\n{"─"*100}')
        print(f'  Target: {desc}')
        print(f'{"─"*100}')
        
        # Find best scale per init per h
        for h in h_vals:
            best = {}
            for init in inits:
                best_err = float('inf')
                best_sc = None
                for sc in scales:
                    errs = [run_elm_trial(X_train, y_train, X_test, y_test, init, h, sc, s)[0] for s in range(NS)]
                    m = np.mean(errs)
                    if m < best_err:
                        best_err = m
                        best_sc = sc
                        best_std = np.std(errs)
                best[init] = (best_err, best_std, best_sc)
            
            # PINN at same h
            pinn_errs = [run_pinn_trial(X_train, y_train, X_test, y_test, h, s, epochs=2000)[0] for s in range(min(NS, 5))]
            pinn_m = np.mean(pinn_errs)
            
            # Format
            pl = best['power']
            ga = best['normal']
            un = best['uniform']
            ratio_g = ga[0] / pl[0] if pl[0] > 0 else 0
            ratio_u = un[0] / pl[0] if pl[0] > 0 else 0
            
            winner = min(best.items(), key=lambda x: x[1][0])[0]
            marker = '***' if winner == 'power' and ratio_g > 2 else ''
            
            print(f'  h={h:<4d}  PL={pl[0]:.2e}(s={pl[2]:<4.0f})  G={ga[0]:.2e}(s={ga[2]:<4.0f})  U={un[0]:.2e}(s={un[2]:<4.0f})  PINN={pinn_m:.2e}  G/PL={ratio_g:>6.1f}x  {marker}')
    
    # ── 2D targets ──
    print(f'\n{"="*100}')
    print('2D SUPERVISED ELM: Power-Law vs Gaussian vs Uniform vs PINN')
    print('='*100)
    
    targets_2d = ['product', 'multi2d', 'radial']
    N2d = 50  # 50x50 grid = 2500 points
    
    xx = torch.linspace(-1, 1, N2d)
    yy = torch.linspace(-1, 1, N2d)
    grid_x, grid_y = torch.meshgrid(xx, yy, indexing='ij')
    X_train_2d = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
    xx_t = torch.linspace(-1, 1, 80)
    yy_t = torch.linspace(-1, 1, 80)
    gx_t, gy_t = torch.meshgrid(xx_t, yy_t, indexing='ij')
    X_test_2d = torch.stack([gx_t.flatten(), gy_t.flatten()], dim=1)
    
    h_vals_2d = [20, 50, 100, 200, 500]
    
    for tname in targets_2d:
        fn, desc = make_target_2d(tname)
        y_train = fn(X_train_2d)
        y_test = fn(X_test_2d)
        
        print(f'\n{"─"*100}')
        print(f'  Target: {desc}')
        print(f'{"─"*100}')
        
        for h in h_vals_2d:
            best = {}
            for init in inits:
                best_err = float('inf')
                best_sc = None
                for sc in scales:
                    errs = [run_elm_trial(X_train_2d, y_train, X_test_2d, y_test, init, h, sc, s)[0] for s in range(NS)]
                    m = np.mean(errs)
                    if m < best_err:
                        best_err = m
                        best_sc = sc
                        best_std = np.std(errs)
                best[init] = (best_err, best_std, best_sc)
            
            pinn_errs = [run_pinn_trial(X_train_2d, y_train, X_test_2d, y_test, h, s, epochs=3000)[0] for s in range(min(NS, 3))]
            pinn_m = np.mean(pinn_errs)
            
            pl = best['power']
            ga = best['normal']
            un = best['uniform']
            ratio_g = ga[0] / pl[0] if pl[0] > 0 else 0
            ratio_u = un[0] / pl[0] if pl[0] > 0 else 0
            
            winner = min(best.items(), key=lambda x: x[1][0])[0]
            marker = '***' if winner == 'power' and ratio_g > 2 else ''
            
            print(f'  h={h:<4d}  PL={pl[0]:.2e}(s={pl[2]:<4.0f})  G={ga[0]:.2e}(s={ga[2]:<4.0f})  U={un[0]:.2e}(s={un[2]:<4.0f})  PINN={pinn_m:.2e}  G/PL={ratio_g:>6.1f}x  {marker}')

