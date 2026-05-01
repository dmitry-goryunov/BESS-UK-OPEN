"""
Imbalance basis process — arithmetic OU + asymmetric compound Poisson jumps.

Model (arithmetic, not log-normal — GB had 53 negative-price hours Apr 2024):
    dDelta_t = -theta * Delta_t * dt + sigma * dW + J_t

    J_t ~ compound Poisson:
        - arrival intensity lambda_J (per half-hour)
        - jump size: positive (system short) ~ Exp(eta_pos)
                     negative (system long)  ~ Exp(eta_neg)
        - asymmetric: P(positive) = p_pos

Calibration:
    Two-step MLE on Elexon DA–SP settlement price pairs.
    Step 1: filter jumps via threshold (|Delta| > jump_threshold_sigma * sigma_base)
    Step 2: MLE on OU diffusion parameters from non-jump observations
    Step 3: MLE on jump size distribution from jump observations

References:
    Cartea & Figueroa (2005), Applied Mathematical Finance 12(4)
    Finnah, Gonsch & Ziel (2022), European Journal of OR 301
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import expon, norm


@dataclass
class ImbalanceParams:
    theta_delta:    float   # OU mean-reversion speed (per half-hour)
    sigma_delta:    float   # diffusion vol (£/MWh per sqrt(HH))
    lambda_jump:    float   # jump arrival intensity (per half-hour)
    jump_scale_pos: float   # exponential scale for positive jumps
    jump_scale_neg: float   # exponential scale for negative jumps
    p_pos:          float   # probability jump is positive (system short)
    mu_delta:       float = 0.0   # long-run mean of basis (usually close to 0)
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
# Jump filter
# ---------------------------------------------------------------------------

def _filter_jumps(delta: np.ndarray, threshold_sigma: float = 2.5) -> np.ndarray:
    """
    Classify observations as jumps or diffusion.
    Returns boolean mask: True = jump.
    Uses iterative filtering: estimate diffusion sigma on non-jumps, reapply.
    """
    mask_jump = np.zeros(len(delta), dtype=bool)
    for _ in range(3):
        sigma_est = np.std(delta[~mask_jump]) if (~mask_jump).sum() > 10 else np.std(delta)
        mask_jump = np.abs(delta) > threshold_sigma * sigma_est
    return mask_jump


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(df_da, df_sp, dt: float = 1.0, threshold_sigma: float = 2.5) -> ImbalanceParams:
    """
    Calibrate imbalance basis process from DA and SP price histories.

    Parameters
    ----------
    df_da : DataFrame with settlement_date, settlement_period, price_gbp_mwh
    df_sp : DataFrame with settlement_date, settlement_period, system_price
    dt    : time step (1 = one half-hour, the natural unit)

    Returns
    -------
    ImbalanceParams
    """
    import pandas as pd

    # ---- Compute basis Delta = SP - DA ----
    merged = df_da[['settlement_date', 'settlement_period', 'price_gbp_mwh']].merge(
        df_sp[['settlement_date', 'settlement_period', 'system_price']],
        on=['settlement_date', 'settlement_period'], how='inner'
    )
    delta = merged['system_price'].values - merged['price_gbp_mwh'].values
    delta = delta[np.isfinite(delta)]

    n_obs = len(delta)
    if n_obs < 100:
        raise ValueError(f"Too few valid observations: {n_obs}")

    mu_delta = float(np.mean(delta))

    # ---- Classify jumps ----
    mask_jump = _filter_jumps(delta, threshold_sigma)
    delta_diff = delta[~mask_jump]
    delta_jump = delta[mask_jump]

    # ---- Step 2: OU calibration on diffusion observations ----
    # OLS for AR(1): Delta_{t+1} = phi * Delta_t + eps
    # phi = exp(-theta * dt),  sigma = std(eps) / sqrt(...)
    d = delta_diff - mu_delta
    x, y = d[:-1], d[1:]
    # Simple OLS
    phi = float(np.cov(x, y)[0, 1] / max(np.var(x), 1e-8))
    phi = np.clip(phi, 1e-3, 1 - 1e-3)
    theta = float(-np.log(phi) / dt)
    residuals = y - phi * x
    sigma_delta = float(np.std(residuals) / np.sqrt(dt))

    # ---- Step 3: Jump distribution ----
    if len(delta_jump) < 5:
        lambda_jump = 0.01
        jump_scale_pos = 10.0
        jump_scale_neg = 10.0
        p_pos = 0.5
    else:
        lambda_jump = float(mask_jump.sum()) / n_obs
        pos_jumps = delta_jump[delta_jump > 0]
        neg_jumps = np.abs(delta_jump[delta_jump < 0])
        p_pos = float(len(pos_jumps)) / max(len(delta_jump), 1)
        jump_scale_pos = float(np.mean(pos_jumps)) if len(pos_jumps) > 0 else 15.0
        jump_scale_neg = float(np.mean(neg_jumps)) if len(neg_jumps) > 0 else 10.0

    # ---- Log-likelihood at final params ----
    ll = _log_likelihood(delta, mu_delta, theta, sigma_delta, lambda_jump,
                         jump_scale_pos, jump_scale_neg, p_pos, dt)

    return ImbalanceParams(
        theta_delta=theta,
        sigma_delta=sigma_delta,
        lambda_jump=lambda_jump,
        jump_scale_pos=jump_scale_pos,
        jump_scale_neg=jump_scale_neg,
        p_pos=p_pos,
        mu_delta=mu_delta,
        log_likelihood=float(ll),
        n_obs=n_obs,
    )


def _log_likelihood(delta, mu, theta, sigma, lambda_j, scale_pos, scale_neg, p_pos, dt):
    """
    Approximate log-likelihood: mixture of OU diffusion and jump components.
    Uses a Gaussian approximation for the diffusion transition and
    exponential tails for the jump component.
    """
    d = delta - mu
    n = len(d)

    # Transition density (Gaussian) for OU with no jump
    phi   = np.exp(-theta * dt)
    sig_t = sigma * np.sqrt(dt)
    mean_next = phi * d[:-1]
    res = d[1:] - mean_next

    ll_diff = norm.logpdf(res, scale=sig_t).sum()

    # Jump observations: rough contribution
    mask_j = np.abs(d) > 2 * sigma * np.sqrt(dt)
    if mask_j.sum() > 0:
        d_j = d[mask_j]
        pos = d_j[d_j > 0]
        neg = np.abs(d_j[d_j < 0])
        if len(pos) > 0:
            ll_diff += np.sum(np.log(p_pos) + expon.logpdf(pos, scale=scale_pos))
        if len(neg) > 0:
            ll_diff += np.sum(np.log(1 - p_pos) + expon.logpdf(neg, scale=scale_neg))

    return ll_diff


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_paths(params: ImbalanceParams,
                   n_paths: int,
                   n_steps: int,
                   dt: float = 1.0,
                   seed: int = 42) -> np.ndarray:
    """
    Simulate imbalance basis paths (arithmetic OU + jumps).
    dt = 1 corresponds to one half-hour period.

    Returns
    -------
    delta : (n_paths, n_steps+1) in £/MWh
    """
    rng = np.random.default_rng(seed)
    p = params

    phi   = np.exp(-p.theta_delta * dt)
    sig_t = p.sigma_delta * np.sqrt(dt)
    lam_t = p.lambda_jump * dt   # scaled intensity

    delta = np.zeros((n_paths, n_steps + 1))

    for t in range(n_steps):
        d = delta[:, t] - p.mu_delta
        diffusion = phi * d + sig_t * rng.standard_normal(n_paths)

        # Poisson number of jumps this step
        n_jumps = rng.poisson(lam_t, n_paths)
        jump_contribution = np.zeros(n_paths)
        for path_idx in np.where(n_jumps > 0)[0]:
            nj = n_jumps[path_idx]
            signs = rng.binomial(1, p.p_pos, nj) * 2 - 1
            sizes = np.where(signs > 0,
                             rng.exponential(p.jump_scale_pos, nj),
                             rng.exponential(p.jump_scale_neg, nj))
            jump_contribution[path_idx] = (signs * sizes).sum()

        delta[:, t+1] = p.mu_delta + diffusion + jump_contribution

    return delta
