"""
Ancillary service clearing process — AR(1) per EFA block + supply saturation curve.

For each product k (DC_Low, DC_High, DM_Low, DM_High, DR_Low, DR_High, QR_Pos, QR_Neg):

    Within-path EFA-block innovation:
        pi_{k,b,t+1} = phi_k * pi_{k,b,t} + eps_{k,b,t},   eps ~ N(0, sigma_k^2)

    Saturation-adjusted clearing price:
        pi_adj_{k,t} = p_res_k * max(0, 1 - Q_t/Q_req)^gamma

    where:
        Q_t   = current GB BESS fleet capacity (MW)
        Q_req = service volume requirement (MW, scenario input)
        gamma = saturation exponent (calibrated to observed DC collapse 2021-24)

    The AR(1) process captures short-run persistence within a contract week.
    The saturation curve captures the structural fleet-capacity effect.

References:
    NESO Dynamic Response tender data 2021-2024
    Modo Energy BESS GB Index
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm


PRODUCTS = ['DC_Low', 'DC_High', 'DM_Low', 'DM_High', 'DR_Low', 'DR_High', 'QR_Pos', 'QR_Neg']

# Service volume requirements (MW, approximate 2025)
SERVICE_VOLUME_MW = {
    'DC_Low':  500,
    'DC_High': 500,
    'DM_Low':  200,
    'DM_High': 200,
    'DR_Low':  400,
    'DR_High': 400,
    'QR_Pos':  800,
    'QR_Neg':  800,
}

# Peak pre-saturation clearing prices (£/MW/h) — DCL 2021 peak, used for saturation anchor
PEAK_PRICE = {
    'DC_Low':  17.0,
    'DC_High': 17.0,
    'DM_Low':  20.0,
    'DM_High': 20.0,
    'DR_Low':  20.0,
    'DR_High': 20.0,
    'QR_Pos':  12.0,
    'QR_Neg':   8.0,
}


@dataclass
class ProductParams:
    """AR(1) params for a single product."""
    product:    str
    phi:        float    # AR(1) coefficient
    sigma:      float    # innovation SD (£/MW/h)
    mu:         float    # unconditional mean (£/MW/h)
    n_obs:      int = 0


@dataclass
class AncillaryParams:
    """Ancillary calibration: per-product AR(1) + fleet saturation curve."""
    products:   Dict[str, ProductParams]    # keyed by product name
    gamma:      float = 2.1                 # saturation exponent
    fleet_mw:   float = 6000.0             # current fleet capacity used to anchor gamma
    q_req:      Dict[str, float] = field(default_factory=lambda: dict(SERVICE_VOLUME_MW))
    p_res:      Dict[str, float] = field(default_factory=lambda: dict(PEAK_PRICE))

    def to_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            'gamma': self.gamma,
            'fleet_mw': self.fleet_mw,
            'q_req': self.q_req,
            'p_res': self.p_res,
            'products': {k: asdict(v) for k, v in self.products.items()},
        }
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        products = {k: ProductParams(**v) for k, v in d.pop('products').items()}
        return cls(products=products, **d)


# ---------------------------------------------------------------------------
# Saturation curve
# ---------------------------------------------------------------------------

def saturation_price(fleet_mw: float,
                     q_req_mw: float,
                     p_res: float,
                     gamma: float) -> float:
    """
    Expected clearing price given fleet capacity.

    pi = p_res * max(0, 1 - fleet_mw / q_req)^gamma

    Saturates to zero when fleet_mw >= q_req.
    """
    ratio = max(0.0, 1.0 - fleet_mw / q_req_mw)
    return p_res * (ratio ** gamma)


def calibrate_gamma(fleet_mw_series: np.ndarray,
                    price_series: np.ndarray,
                    q_req: float,
                    p_res: float,
                    gamma_bounds: tuple = (0.5, 6.0)) -> float:
    """
    Calibrate saturation exponent gamma to observed (fleet, price) pairs.

    Minimises sum of squared residuals:
        pi_obs - p_res * max(0, 1 - Q/Q_req)^gamma
    """
    def residuals_ss(gamma):
        pred = np.array([saturation_price(q, q_req, p_res, gamma) for q in fleet_mw_series])
        return np.sum((price_series - pred)**2)

    res = minimize_scalar(residuals_ss, bounds=gamma_bounds, method='bounded')
    return float(res.x)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(df_anc,
              fleet_mw_history: Optional[np.ndarray] = None,
              gamma_prior: float = 2.1) -> AncillaryParams:
    """
    Calibrate ancillary AR(1) + saturation from NESO EAC clearing data.

    Parameters
    ----------
    df_anc : DataFrame with columns: date, efa_block, product, clearing_price_gbp_mwh
    fleet_mw_history : (T,) array of GB BESS fleet capacity over the same period
                       (used to calibrate gamma; if None, use prior 2.1)
    gamma_prior : saturation exponent prior (used if fleet_mw_history is None)

    Returns
    -------
    AncillaryParams
    """
    product_params = {}

    for product in PRODUCTS:
        df_p = df_anc[df_anc['product'] == product].copy() if not df_anc.empty else None

        if df_p is None or df_p.empty or len(df_p) < 20:
            # Use placeholder from config / prior knowledge
            product_params[product] = ProductParams(
                product=product,
                phi=0.85,
                sigma=1.5,
                mu=PEAK_PRICE.get(product, 5.0) * 0.3,   # assume ~30% of peak post-saturation
                n_obs=0,
            )
            continue

        # Mean clearing per EFA block-day
        series = df_p.groupby(['date', 'efa_block'])['clearing_price_gbp_mwh'].mean().values
        series = series[np.isfinite(series)]

        if len(series) < 10:
            product_params[product] = ProductParams(
                product=product, phi=0.85, sigma=1.5,
                mu=float(np.nanmean(df_p['clearing_price_gbp_mwh'])), n_obs=len(series))
            continue

        mu  = float(np.mean(series))
        d   = series - mu
        phi = float(np.clip(np.cov(d[:-1], d[1:])[0,1] / max(np.var(d[:-1]), 1e-8), 0.0, 0.999))
        residuals = d[1:] - phi * d[:-1]
        sigma = float(np.std(residuals))

        product_params[product] = ProductParams(
            product=product, phi=phi, sigma=sigma, mu=mu, n_obs=len(series))

    # ---- Calibrate gamma (DCL only — most data) ----
    gamma = gamma_prior
    if fleet_mw_history is not None and not df_anc.empty:
        dc_low = df_anc[df_anc['product'] == 'DC_Low']
        if len(dc_low) > 50:
            dc_price = dc_low.groupby('date')['clearing_price_gbp_mwh'].mean().values
            n_min = min(len(dc_price), len(fleet_mw_history))
            gamma = calibrate_gamma(
                fleet_mw_history[-n_min:],
                dc_price[-n_min:],
                q_req=SERVICE_VOLUME_MW['DC_Low'],
                p_res=PEAK_PRICE['DC_Low'],
            )

    return AncillaryParams(
        products=product_params,
        gamma=gamma,
        fleet_mw=fleet_mw_history[-1] if fleet_mw_history is not None else 6000.0,
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_paths(params: AncillaryParams,
                   product: str,
                   n_paths: int,
                   n_steps: int,
                   fleet_mw_scenario: Optional[np.ndarray] = None,
                   seed: int = 42) -> np.ndarray:
    """
    Simulate clearing price paths for a single product.

    Fleet saturation provides the path-conditional mean; AR(1) provides
    short-run variation around that mean within each EFA block.

    Parameters
    ----------
    params           : calibrated AncillaryParams
    product          : product name (e.g. 'DC_Low')
    n_paths          : number of paths
    n_steps          : number of half-hour steps
    fleet_mw_scenario: (n_steps+1,) deterministic fleet capacity scenario
                       If None, uses params.fleet_mw (flat)

    Returns
    -------
    pi : (n_paths, n_steps+1) — clearing prices £/MW/h, floor at 0
    """
    rng = np.random.default_rng(seed)
    pp = params.products.get(product)
    if pp is None:
        raise ValueError(f"Unknown product: {product}")

    q_req = params.q_req.get(product, 500.0)
    p_res = params.p_res.get(product, 17.0)

    if fleet_mw_scenario is None:
        fleet_mw_scenario = np.full(n_steps + 1, params.fleet_mw)

    # Path-conditional mean at each step from saturation curve
    sat_mean = np.array([saturation_price(fleet_mw_scenario[t], q_req, p_res, params.gamma)
                         for t in range(n_steps + 1)])

    pi = np.zeros((n_paths, n_steps + 1))
    # Initialise at saturation mean
    pi[:, 0] = sat_mean[0]

    for t in range(n_steps):
        mu_t = sat_mean[t + 1]
        deviation = pi[:, t] - sat_mean[t]
        pi[:, t+1] = mu_t + pp.phi * deviation + pp.sigma * rng.standard_normal(n_paths)
        pi[:, t+1] = np.maximum(pi[:, t+1], 0.0)   # price floor at 0

    return pi


def simulate_all_products(params: AncillaryParams,
                          n_paths: int,
                          n_steps: int,
                          fleet_mw_scenario: Optional[np.ndarray] = None,
                          seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Simulate all products. Returns dict of product -> (n_paths, n_steps+1) arrays.
    """
    results = {}
    for i, product in enumerate(PRODUCTS):
        results[product] = simulate_paths(
            params, product, n_paths, n_steps,
            fleet_mw_scenario=fleet_mw_scenario,
            seed=seed + i * 1000,
        )
    return results
