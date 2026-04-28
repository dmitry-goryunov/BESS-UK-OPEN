"""
Schwartz-Smith two-factor commodity model — Kalman filter calibration.

Model:
    ln P_t = chi_t + xi_t + f(t)

    d chi_t = -kappa * chi_t * dt + sigma_chi * dW_chi   (mean-reverting short factor)
    d xi_t  =  mu_xi           * dt + sigma_xi  * dW_xi  (drifting long factor)
    corr(dW_chi, dW_xi) = rho

Calibration:
    Kalman filter on log-forward prices at maturities T_1, ..., T_N.
    State: x_t = [chi_t, xi_t]

References:
    Schwartz & Smith (2000), Management Science 46(7)
    Lucia & Schwartz (2002), Review of Derivatives Research
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class SSParams:
    kappa:      float
    mu_xi:      float
    sigma_chi:  float
    sigma_xi:   float
    rho:        float
    sigma_obs:  float = float('nan')
    log_likelihood: float = float('nan')
    n_obs:          int   = 0

    def to_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

def _forward_coefficients(kappa, mu_xi, sigma_chi, sigma_xi, rho, tau):
    """ln F(t, t+tau) = A(tau) + B_chi(tau)*chi + B_xi(tau)*xi"""
    B_chi = np.exp(-kappa * tau)
    B_xi  = np.ones_like(tau)
    A = (mu_xi * tau
         + 0.5 * (sigma_chi**2 / (2*kappa) * (1 - np.exp(-2*kappa*tau))
                  + sigma_xi**2 * tau
                  + 2*rho*sigma_chi*sigma_xi / kappa * (1 - np.exp(-kappa*tau))))
    return A, np.column_stack([B_chi, B_xi])


def _ss_kalman_filter(log_forwards, taus, dt, kappa, mu_xi, sigma_chi, sigma_xi, rho, sigma_obs):
    T, N = log_forwards.shape
    F = np.array([[np.exp(-kappa * dt), 0.0],
                  [0.0,                 1.0]])
    c = np.array([0.0, mu_xi * dt])
    v11 = sigma_chi**2 / (2*kappa) * (1 - np.exp(-2*kappa*dt))
    v22 = sigma_xi**2 * dt
    v12 = rho * sigma_chi * sigma_xi / kappa * (1 - np.exp(-kappa*dt))
    Q = np.array([[v11, v12], [v12, v22]])

    A, B = _forward_coefficients(kappa, mu_xi, sigma_chi, sigma_xi, rho, taus)
    H = B
    R = sigma_obs**2 * np.eye(N)

    x = np.zeros(2)
    P = np.diag([sigma_chi**2 / (2*kappa), 1.0])
    neg_ll = 0.0

    for t in range(T):
        y = log_forwards[t]
        mask = np.isfinite(y)
        if mask.sum() < 2:
            x = F @ x + c
            P = F @ P @ F.T + Q
            continue

        H_t, R_t, y_t, A_t = H[mask], R[np.ix_(mask, mask)], y[mask], A[mask]
        v = y_t - (A_t + H_t @ x)
        S = H_t @ P @ H_t.T + R_t
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return 1e12
        neg_ll += 0.5 * (logdet + v @ np.linalg.solve(S, v) + mask.sum() * np.log(2*np.pi))
        K = P @ H_t.T @ np.linalg.inv(S)
        x = x + K @ v
        P = (np.eye(2) - K @ H_t) @ P
        x = F @ x + c
        P = F @ P @ F.T + Q

    return neg_ll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate(log_forwards, taus, dt=1/52, init_params=None, n_restarts=5, seed=42):
    """
    Calibrate SS two-factor model to a panel of log-forward prices.

    Parameters
    ----------
    log_forwards : (T, N) — T observation dates, N maturity columns (NaN for missing)
    taus         : (N,) — time to maturity in years
    dt           : observation step in years (default 1/52 = weekly)
    """
    rng = np.random.default_rng(seed)
    # kappa: (0.01,20), mu_xi: (-.5,.5), sigma_chi: (0.01,1.5), sigma_xi: (0.005,1),
    # rho: (-.99,.99), sigma_obs: (0.001,0.50)
    # sigma_chi capped at 1.5 (annualised ~150% short-factor vol); the old ceiling of 3.5
    # caused the filter to hit the bound when short maturities (<6m) are absent from the
    # panel, because B_chi(tau)=exp(-kappa*tau)~0 for tau>1yr makes chi unidentifiable.
    bounds = [(0.01,20),(-.5,.5),(0.01,1.5),(0.005,1),(-.99,.99),(0.001,0.50)]

    def objective(params):
        return _ss_kalman_filter(log_forwards, taus, dt, *params)

    starts = []
    if init_params:
        starts.append([init_params.get(k, v) for k, v in
                       [('kappa',2.5),('mu_xi',0.01),('sigma_chi',0.45),
                        ('sigma_xi',0.18),('rho',-0.30),('sigma_obs',0.02)]])
    for _ in range(n_restarts):
        starts.append([rng.uniform(*b) for b in bounds])

    best_res, best_ll = None, np.inf
    for x0 in starts:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-12})
        if res.fun < best_ll:
            best_ll, best_res = res.fun, res

    kappa, mu_xi, sigma_chi, sigma_xi, rho, sigma_obs = best_res.x
    param_names = ['kappa', 'mu_xi', 'sigma_chi', 'sigma_xi', 'rho', 'sigma_obs']
    for name, val, (lo, hi) in zip(param_names, best_res.x, bounds):
        tol = 0.01 * (hi - lo)
        if val <= lo + tol or val >= hi - tol:
            import warnings
            warnings.warn(
                f"[SS] {name}={val:.4f} is within 1% of its bound [{lo}, {hi}] — "
                "parameter may be unidentified; consider adding shorter maturities.",
                stacklevel=2,
            )
    return SSParams(kappa=kappa, mu_xi=mu_xi, sigma_chi=sigma_chi,
                    sigma_xi=sigma_xi, rho=rho,
                    sigma_obs=sigma_obs,
                    log_likelihood=-best_ll,
                    n_obs=int(np.isfinite(log_forwards).sum()))


def simulate_paths(params, n_paths, n_steps, dt=1/(365*48), seed=42):
    """
    Simulate (chi, xi) paths under risk-neutral measure.
    Returns chi, xi each of shape (n_paths, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    p = params
    F = np.array([[np.exp(-p.kappa * dt), 0.0], [0.0, 1.0]])
    c = np.array([0.0, p.mu_xi * dt])
    v11 = p.sigma_chi**2 / (2*p.kappa) * (1 - np.exp(-2*p.kappa*dt))
    v22 = p.sigma_xi**2 * dt
    v12 = p.rho * p.sigma_chi * p.sigma_xi / p.kappa * (1 - np.exp(-p.kappa*dt))
    Q = np.array([[v11, v12], [v12, v22]])
    L = np.linalg.cholesky(Q)

    chi = np.zeros((n_paths, n_steps + 1))
    xi  = np.zeros((n_paths, n_steps + 1))
    for t in range(n_steps):
        z = rng.standard_normal((n_paths, 2))
        s = np.column_stack([chi[:, t], xi[:, t]]) @ F.T + c + z @ L.T
        chi[:, t+1] = s[:, 0]
        xi[:, t+1]  = s[:, 1]
    return chi, xi


def forward_curve_from_state(params, chi, xi, taus):
    """Compute log-forward curve ln F(t, t+tau) at a given state."""
    p = params
    A, B = _forward_coefficients(p.kappa, p.mu_xi, p.sigma_chi, p.sigma_xi, p.rho, taus)
    return A + B @ np.array([chi, xi])
