"""
Hourly Price Forward Curve (HPFC) — PCA shape decomposition.

Decomposes the within-day half-hourly shape into 3 PCA factors:
    lambda_1 (level), lambda_2 (slope/morning-evening), lambda_3 (curvature)

Each factor follows an OU process:
    d lambda_k = -alpha_k * lambda_k * dt + sigma_lk * dW_lk

The reconstructed log-price is:
    ln P_{h,t} = ln P_t + sum_{k=1}^3 lambda_k(t) * phi_k(h)

where phi_k are the eigenvectors (shape loadings) and P_t is the
baseload price from the Schwartz-Smith model.

Usage
-----
    from src.processes.hpfc import calibrate_pca, simulate_shape_paths
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from scipy.optimize import minimize


N_HH = 48   # half-hours per day


@dataclass
class HPFCParams:
    """PCA shape parameters."""
    n_factors:    int
    eigenvalues:  List[float]             # variance explained per factor
    loadings:     List[List[float]]       # (n_factors, N_HH) eigenvectors
    alpha:        List[float]             # OU mean-reversion speeds (1/day)
    sigma_lambda: List[float]             # OU vols
    explained_variance_ratio: List[float] = field(default_factory=list)

    def to_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def loadings_array(self):
        return np.array(self.loadings)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_shape_matrix(df_da) -> np.ndarray:
    """
    Build (T, N_HH) matrix of log within-day shape residuals.

    Parameters
    ----------
    df_da : DataFrame with columns settlement_date, settlement_period, price_gbp_mwh
            settlement_period 1..48

    Returns
    -------
    shape_matrix : (T, 48) — each row is log(P_h / P_daily_mean), zero-mean across HH
    dates        : (T,) array of dates
    """
    import pandas as pd

    df = df_da.copy()
    df = df[df['settlement_period'].between(1, N_HH)].copy()
    df['log_price'] = np.log(np.maximum(df['price_gbp_mwh'], 1.0))   # floor at £1 to avoid log issues
    daily_mean = df.groupby('settlement_date')['log_price'].mean().rename('daily_log_mean')
    df = df.join(daily_mean, on='settlement_date')
    df['shape_residual'] = df['log_price'] - df['daily_log_mean']

    pivot = df.pivot_table(index='settlement_date', columns='settlement_period',
                           values='shape_residual', aggfunc='first')
    pivot = pivot.reindex(columns=range(1, N_HH + 1))
    pivot = pivot.dropna()

    dates = pivot.index.values
    shape_matrix = pivot.values   # (T, 48)
    return shape_matrix, dates


def calibrate_pca(df_da, n_factors: int = 3, dt: float = 1.0) -> HPFCParams:
    """
    Calibrate PCA shape decomposition from DA price history.

    Parameters
    ----------
    df_da     : DA price DataFrame (settlement_date, settlement_period, price_gbp_mwh)
    n_factors : number of PCA factors to retain (default 3)
    dt        : time step in days for OU calibration

    Returns
    -------
    HPFCParams
    """
    shape_matrix, dates = build_shape_matrix(df_da)

    # ---- PCA ----
    cov = np.cov(shape_matrix, rowvar=False)   # (48, 48)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]   # columns are eigenvectors

    # Top-K factors
    phi = eigenvectors[:, :n_factors].T   # (n_factors, 48)
    ev_k = eigenvalues[:n_factors]
    evr  = ev_k / eigenvalues.sum()

    # ---- Project shape matrix onto factors → time series ----
    # lambda_k(t) = shape_matrix[t] @ phi_k
    lambdas = shape_matrix @ phi.T   # (T, n_factors)

    # ---- Calibrate OU for each factor via MLE ----
    alphas  = []
    sigmas  = []
    for k in range(n_factors):
        lam = lambdas[:, k]
        alpha_k, sigma_k = _calibrate_ou(lam, dt)
        alphas.append(float(alpha_k))
        sigmas.append(float(sigma_k))

    return HPFCParams(
        n_factors=n_factors,
        eigenvalues=ev_k.tolist(),
        loadings=phi.tolist(),
        alpha=alphas,
        sigma_lambda=sigmas,
        explained_variance_ratio=evr.tolist(),
    )


def _calibrate_ou(series, dt):
    """
    Calibrate OU mean-reversion speed alpha and vol sigma by OLS regression.
    lambda_{t+1} = exp(-alpha*dt) * lambda_t + epsilon
    """
    x = series[:-1]
    y = series[1:]
    # OLS for AR(1)
    phi = np.cov(x, y)[0, 1] / np.var(x)
    phi = np.clip(phi, 1e-6, 1 - 1e-6)
    alpha = -np.log(phi) / dt
    residuals = y - phi * x
    # Vol of continuous process
    sigma = float(np.std(residuals) / np.sqrt(dt * (1 - phi**2) / (2 * alpha) if alpha > 0.01 else dt))
    return max(alpha, 0.01), max(sigma, 1e-4)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_shape_paths(params: HPFCParams,
                         n_paths: int,
                         n_steps: int,
                         dt: float = 1.0,
                         seed: int = 42) -> np.ndarray:
    """
    Simulate lambda factor paths via discretised OU.

    Returns
    -------
    lambdas : (n_paths, n_steps+1, n_factors)
    """
    rng = np.random.default_rng(seed)
    p = params
    K = p.n_factors
    alphas = np.array(p.alpha)
    sigmas = np.array(p.sigma_lambda)

    phi = np.exp(-alphas * dt)
    vol = sigmas * np.sqrt((1 - phi**2) / (2 * alphas + 1e-12))

    lambdas = np.zeros((n_paths, n_steps + 1, K))
    for t in range(n_steps):
        z = rng.standard_normal((n_paths, K))
        lambdas[:, t+1, :] = lambdas[:, t, :] * phi + z * vol

    return lambdas


def reconstruct_shape(params: HPFCParams, lambda_vec: np.ndarray) -> np.ndarray:
    """
    Reconstruct log within-day shape from factor values.

    Parameters
    ----------
    lambda_vec : (n_factors,) or (n_paths, n_factors)

    Returns
    -------
    shape : (48,) or (n_paths, 48) — log shape residuals
    """
    phi = params.loadings_array()   # (n_factors, 48)
    return lambda_vec @ phi
