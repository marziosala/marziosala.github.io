---
layout: splash
permalink: /particle-calibration/
title: "Particle Calibration"
header:
  overlay_image: /assets/images/particle-calibration/particle-calibration-splash.png
excerpt: "A basic implementation of the particle calibration method."
---

# Particle Calibration of a Local Stochastic Volatility Model

The *particle method* calibrates a local stochastic volatility (LSV) model to a target smile
in a single forward pass.  Here we use the simplest possible target — a **flat Black-Scholes
smile** with constant volatility $\sigma_{\rm BS}$ — so that calibration errors are immediately
visible as any departure from a horizontal line.

**Calibration condition.** A Heston+local-vol hybrid
$$d\log S^i_t = \left(r - q - \tfrac{1}{2}\,a^2(t,S^i_t)\,V^i_t\right)dt
               + \sqrt{a^2(t,S^i_t)\,V^i_t}\;dW^{S,i}_t$$
reproduces any target smile exactly when the local-vol factor satisfies
$$a^2(t, x) = \frac{\sigma^2_{\mathrm{Dup}}(t, x)}{\mathbb{E}[V_t \mid S_t = x]}$$
For a flat smile,

$$
\sigma^2_{\mathrm{Dup}} = \sigma^2_{\mathrm{BS}}
$$

everywhere, so at steady
state

$$
\mathbb{E}[V]\approx\theta$ and $a^2\approx\sigma^2_{\mathrm{BS}}/\theta,
$$

giving an
effective volatility $\sqrt{a^2 V}\approx\sigma_{\mathrm{BS}}$ regardless of the Heston skew
induced by $\rho_h$.

The calibration is composed by two passes.
A first pass of $N_{\rm cal}$ interacting particles builds and stores the calibrated
$a^2(t,\cdot)$ surface.  A second pass of $M \gg N_{\rm cal}$ independent paths
prices options via linear interpolation into that surface.


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

Φ = norm.cdf
rng = np.random.default_rng(42)
```

## Black-Scholes flat smile target

The target is a flat implied volatility $\sigma_{\rm BS}$ at every strike and maturity.
Dupire's formula for a flat smile collapses to the trivial identity
$\sigma^2_{\rm Dup}(t, x) = \sigma^2_{\rm BS}$, so no term-structure interpolation is needed.
The Garman-Kohlhagen formula provides reference call prices for inverting Monte Carlo
payoffs back to implied volatilities.


```python
σ_BS = 0.10   # 10% flat implied volatility target


def bs_call(S0, K, T, r, q, σ):
    """Garman-Kohlhagen call price."""
    F  = S0 * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * σ**2 * T) / (σ * np.sqrt(T))
    return np.exp(-r * T) * (F * Φ(d1) - K * Φ(d1 - σ * np.sqrt(T)))


def dupire_local_var(Y, T):
    """Dupire local variance for a flat Black-Scholes smile: constant sigma_BS^2 everywhere."""
    return np.full_like(np.asarray(Y, float), σ_BS**2)
```

## The particle algorithm

The $N$ coupled particles $(S^i_t, V^i_t)$ evolve as

$$
d\log S^i_t = \left(r - q - \tfrac{1}{2}\,a^2(t,S^i_t)\,V^i_t\right)dt
  + \sqrt{a^2(t,S^i_t)\,V^i_t}\;dW^{S,i}_t
$$

$$
dV^i_t = \kappa(\theta - V^i_t)\,dt + \xi\sqrt{V^i_t}\,dW^{V,i}_t,
  \qquad \langle dW^S, dW^V\rangle = \rho_h\,dt
$$

At each time step, a fixed grid of $G$ log-spot values spanning the central
95\% of the particle cloud is constructed.
The conditional expectation is estimated at every grid node via Nadaraya-Watson
regression with a Gaussian kernel and Silverman bandwidth $h$:

$$
a^2(t, g_k) =
  \frac{\sigma^2_{\mathrm{BS}}}{\hat{E}^N[V_t \mid \log S_t = g_k]},
  \qquad
  \hat{E}^N[V \mid g_k] =
  \frac{\sum_j K_h(g_k - \log S^j)\,V^j}{\sum_j K_h(g_k - \log S^j)},
  \qquad K_h(u) = e^{-u^2/(2h^2)}
$$

$h = 1.06\,\hat\sigma_{\log S}\,N^{-1/5}$ (Silverman's rule).

The $a^2$ values on the grid are stored as a cubic spline; flat extrapolation is used
outside $[g_1, g_G]$.

This grid-based evaluation costs $O(G \times N)$ per step — not $O(N^2)$ — which is
why the method is fast.
The forward is $F(t) = S_0\,e^{(r-q)t}$.


```python
S_0 = 1
r   = 0.05
q   = 0.03

# Heston parameters: theta = sigma_BS^2 so the long-run vol matches the target
κ   = 2.0
θ_V = σ_BS**2   # = 0.01; long-run variance equals BS target variance
ξ   = 0.3
ρ_h = -0.7      # negative skew -- calibration must flatten it out
V_0 = θ_V
```


```python
N_cal      = 40_000  # calibration particles
G          = 51      # grid points for the a² surface
n_substeps = 4       # time-steps per trading day

snap_times = [0.25, 0.50, 1.0, 2.0]
T_max      = max(snap_times)
dt         = 1 / (252 * n_substeps)
Nt         = int(np.round(T_max / dt))

S_cal = np.full(N_cal, S_0, dtype=float)
V_cal = np.full(N_cal, V_0, dtype=float)

# a² surface: one CubicSpline per time step
a2_surface = []

for n in range(Nt):
    t     = n * dt
    log_S = np.log(S_cal)
    log_F = np.log(S_0) + (r - q) * t

    # --- Silverman bandwidth ---
    bw = max(1.06 * log_S.std() * N_cal**(-0.2), 1e-3)

    # --- Fixed grid within central 95% of particle cloud ---
    lo, hi = np.percentile(log_S, [2.5, 97.5])
    mid = 0.5 * (lo + hi)
    lo  = min(lo, mid - 3 * bw)
    hi  = max(hi, mid + 3 * bw)
    grid = np.linspace(lo, hi, G)

    # --- Nadaraya-Watson with Gaussian kernel: O(G × N_cal) ---
    u   = (grid[:, None] - log_S[None, :]) / bw
    Kg  = np.exp(-0.5 * u**2)
    EV  = (Kg @ V_cal) / Kg.sum(axis=1)
    EV  = np.maximum(EV, 1e-8)

    # --- a² on the grid; store as cubic spline (flat extrapolation via input clipping) ---
    a2_grid = np.maximum(dupire_local_var(grid - log_F, max(t, 1e-3)) / EV, 0.0)
    a2_cs   = CubicSpline(grid, a2_grid)
    a2_surface.append(a2_cs)

    # --- Each particle's a² via cubic spline lookup ---
    log_S_clipped = np.clip(log_S, grid[0], grid[-1])
    a2 = np.maximum(a2_cs(log_S_clipped), 0.0)

    # --- Correlated Brownian increments ---
    Z1, Z2  = rng.standard_normal((2, N_cal))
    sqrt_dt = np.sqrt(dt)
    dWS = Z1 * sqrt_dt
    dWV = (ρ_h * Z1 + np.sqrt(1 - ρ_h**2) * Z2) * sqrt_dt

    # --- Advance log-spot (Euler) and variance (full truncation Euler) ---
    log_S += (r - q - 0.5 * a2 * V_cal) * dt + np.sqrt(a2 * V_cal) * dWS
    S_cal  = np.exp(log_S)
    V_cal += κ * (θ_V - V_cal) * dt + ξ * np.sqrt(np.maximum(V_cal, 0.)) * dWV
    V_cal  = np.maximum(V_cal, 0.)

    t_next = (n + 1) * dt
    for t_s in snap_times:
        if abs(t_next - t_s) < 0.5 * dt:
            print(f't={t_s:.2f}  S mean={S_cal.mean():.4f}  std={S_cal.std():.4f}')
```

    t=0.25  S mean=1.0048  std=0.0503
    t=0.50  S mean=1.0098  std=0.0717
    t=1.00  S mean=1.0206  std=0.1025
    t=2.00  S mean=1.0411  std=0.1476
    Calibration done.


### Pricing pass

With `a2_surface` in hand we propagate two batches of $M/2$ **independent** paths in
lock-step: a *regular* batch and its *antithetic* counterpart (all Brownian increments
negated).  Pooling the two batches gives $M = 100{,}000$ effective paths while drawing
only $M/2$ random numbers per step.

For call options the payoff $\max(S_T - K, 0)$ is monotone in $S_T$, so a high
realization in the regular batch is paired with a low one in the antithetic batch,
giving a large variance reduction for deep out-of-the-money strikes.


```python
M_half = 50_000   # 50k regular + 50k antithetic = 100k effective paths

S_pos = np.full(M_half, S_0, dtype=float)
V_pos = np.full(M_half, V_0, dtype=float)
S_neg = np.full(M_half, S_0, dtype=float)   # antithetic batch
V_neg = np.full(M_half, V_0, dtype=float)

snapshots = {}
sqrt_dt = np.sqrt(dt)

for n in range(Nt):
    a2_cs   = a2_surface[n]
    grid_lo = a2_cs.x[0]
    grid_hi = a2_cs.x[-1]

    # look up a² for both batches (clip for flat extrapolation outside calibration grid)
    log_S_pos = np.log(S_pos)
    log_S_neg = np.log(S_neg)
    a2_pos = np.maximum(a2_cs(np.clip(log_S_pos, grid_lo, grid_hi)), 0.0)
    a2_neg = np.maximum(a2_cs(np.clip(log_S_neg, grid_lo, grid_hi)), 0.0)

    # one draw: antithetic batch uses the negated increments
    Z1, Z2 = rng.standard_normal((2, M_half))

    log_S_pos += (r - q - 0.5*a2_pos*V_pos)*dt + np.sqrt(a2_pos*V_pos) *  Z1 * sqrt_dt
    log_S_neg += (r - q - 0.5*a2_neg*V_neg)*dt + np.sqrt(a2_neg*V_neg) * (-Z1) * sqrt_dt
    S_pos = np.exp(log_S_pos)
    S_neg = np.exp(log_S_neg)

    dWV_pos = ( ρ_h*Z1 + np.sqrt(1 - ρ_h**2)*Z2) * sqrt_dt
    dWV_neg = (-ρ_h*Z1 - np.sqrt(1 - ρ_h**2)*Z2) * sqrt_dt
    V_pos += κ*(θ_V - V_pos)*dt + ξ*np.sqrt(np.maximum(V_pos, 0.))*dWV_pos
    V_neg += κ*(θ_V - V_neg)*dt + ξ*np.sqrt(np.maximum(V_neg, 0.))*dWV_neg
    V_pos = np.maximum(V_pos, 0.)
    V_neg = np.maximum(V_neg, 0.)

    t_next = (n + 1) * dt
    for t_s in snap_times:
        if abs(t_next - t_s) < 0.5 * dt:
            snapshots[t_s] = np.concatenate([S_pos, S_neg])
            S_all = snapshots[t_s]
            print(f't={t_s:.2f}  S mean={S_all.mean():.4f}  std={S_all.std():.4f}')

print('Pricing done.')
```

    t=0.25  S mean=1.0050  std=0.0506
    t=0.50  S mean=1.0100  std=0.0713
    t=1.00  S mean=1.0204  std=0.1023
    t=2.00  S mean=1.0414  std=0.1484
    Pricing done.


## Verification

We compute call prices from the particle distributions at each snapshot, invert them to
implied volatilities via Garman-Kohlhagen, and compare with the flat target $\sigma_{\rm BS}$.
Strikes are parameterised by their Black-Scholes call delta $\Delta = e^{-qT}\,\mathcal{N}(d_1)$,
computed using $\sigma_{\rm BS}$ itself; the range covers $0.01$ call delta to $0.01$ put delta
($\Delta = e^{-qT}-0.01$), with the put side on the left.
A well-calibrated model produces a horizontal line at $\sigma_{\rm BS}$ for all deltas.


```python
def implied_vol(S0, K, T, r, q, price):
    """Implied volatility from a Garman-Kohlhagen call price via Brent's method."""
    lb = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0) + 1e-10
    ub = S0 * np.exp(-q * T)
    if price <= lb or price >= ub:
        return np.nan
    try:
        return brentq(lambda σ: bs_call(S0, K, T, r, q, σ) - price, 1e-6, 5.0, xtol=1e-6)
    except ValueError:
        return np.nan


# market-convention delta ticks: negative = put delta, positive = call delta, 0 = ATM
# listed in left-to-right display order (put side left, call side right)
TICK_MARKET = [-0.10, -0.25, 0.0, 0.25, 0.10]
TICK_LABELS = ['-0.1', '-0.25', '0', '0.25', '0.1']

plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, t_s in zip(axes.flat, snap_times):
    S_snap = snapshots[t_s]
    F_snap = S_0 * np.exp((r - q) * t_s)
    disc_q = np.exp(-q * t_s)

    # delta grid: 0.01 call delta (OTM call) to 0.01 put delta (OTM put)
    delta_lo = 0.05
    delta_hi = 0.95
    delta_grid = np.linspace(delta_lo, delta_hi, 41)

    # call delta → strike via Black-Scholes (using σ_BS for the delta mapping)
    d1_grid = norm.ppf(delta_grid / disc_q)
    K_grid  = F_snap * np.exp(-d1_grid * σ_BS * np.sqrt(t_s) + 0.5 * σ_BS**2 * t_s)

    mc_prices = np.exp(-r * t_s) * np.maximum(S_snap[:, None] - K_grid[None, :], 0.0).mean(axis=0)

    σ_mc = np.array([
        implied_vol(S_0, K, t_s, r, q, price)
        for K, price in zip(K_grid, mc_prices)
    ])

    # convert market-convention ticks to call-delta axis positions
    tick_pos = [
        1.0 + d if d < 0 else (0.5 * disc_q if d == 0 else d)
        for d in TICK_MARKET
    ]

    ax.axhline(100 * σ_BS, color='k', lw=2, label='BS target')
    ax.plot(delta_grid, 100 * σ_mc, 'r--', lw=1.5, label='Particle MC')
    ax.set_xlim(delta_hi, delta_lo)   # reversed: put side on left, call side on right
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(TICK_LABELS)
    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel('Implied vol (%)')
    ax.set_title(f'T = {t_s}')
    ax.legend()

plt.tight_layout()
```


    
![png](/assets/images/particle-calibration/particle-calibration-1.png)
    


## Conclusion

With a flat Black-Scholes target the calibration condition simplifies to

$$
a^2 = \sigma^2_{\rm BS}/\hat{E}^N[V\mid S],
$$

and the output smile should be a horizontal
line at $\sigma_{\rm BS}$ for all strikes and maturities.  Despite the Heston correlation
$\rho_h = -0.7$ that would ordinarily generate a downward skew, the particle method corrects
for it on the fly.

The residual errors are the finite-particle noise of the Nadaraya-Watson estimator, which
scales as $N^{-2/5}$.  Using $N = 40{,}000$ calibration particles and cubic-spline
interpolation of the $a^2$ grid keeps errors below a few basis points across all maturities.
On the pricing side, $M = 100{,}000$ antithetic paths provide further variance reduction
for out-of-the-money strikes.
