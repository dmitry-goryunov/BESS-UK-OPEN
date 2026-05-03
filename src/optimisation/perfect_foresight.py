"""
Perfect-foresight battery arbitrage benchmark.

Solves one deterministic LP over a known historical half-hourly price path.
This is an upper benchmark for energy arbitrage because the optimizer sees the
whole price path before choosing dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


@dataclass
class PerfectForesightResult:
    price: np.ndarray
    discharge_mw: np.ndarray
    charge_mw: np.ndarray
    soc_mwh: np.ndarray
    cashflow_gbp: np.ndarray
    objective_gbp: float
    cycles_equiv: float
    status: str


def solve_perfect_foresight(
    prices: np.ndarray,
    asset_cfg: dict,
    dt_h: float = 0.5,
    initial_soc_mwh: float | None = None,
    terminal_soc_mwh: float | None = None,
    vom_gbp_mwh: float | None = None,
) -> PerfectForesightResult:
    """
    Maximise historical arbitrage value over the full known price path.

    Decision variables are discharge MW, charge MW, and SoC MWh:
        x = [d_0..d_T-1, c_0..c_T-1, e_0..e_T]

    SoC transition:
        e[t+1] = e[t] + eta_c * c[t] * dt - d[t] / eta_d * dt

    The LP also imposes d[t] + c[t] <= power_mw to avoid simultaneous full
    charge/discharge in the same half-hour. If terminal_soc_mwh is provided,
    the final SoC is fixed to that value; otherwise it is left free within the
    battery SoC bounds.
    """
    price = np.asarray(prices, dtype=float)
    mask = np.isfinite(price)
    if not mask.all():
        price = price[mask]
    if price.size == 0:
        raise ValueError("prices is empty after removing NaNs")

    T = int(price.size)
    p_mw = float(asset_cfg["power_mw"])
    e_name = float(asset_cfg["energy_mwh"])
    eta_c = float(asset_cfg["eta_charge"])
    eta_d = float(asset_cfg["eta_discharge"])
    e_min = float(asset_cfg["soc_min_frac"]) * e_name
    e_max = float(asset_cfg["soc_max_frac"]) * e_name
    e_init = (
        float(initial_soc_mwh)
        if initial_soc_mwh is not None
        else float(asset_cfg.get("soc_init_frac", 0.5)) * e_name
    )
    vom = float(asset_cfg.get("vom_gbp_mwh", 0.0) if vom_gbp_mwh is None else vom_gbp_mwh)

    n_d = T
    n_c = T
    n_e = T + 1
    off_d = 0
    off_c = n_d
    off_e = n_d + n_c
    n_vars = n_d + n_c + n_e

    # linprog minimises. Negative terms are discharge revenue; positive terms
    # are charge cost. VOM is charged on both discharge and charge throughput.
    c_obj = np.zeros(n_vars)
    c_obj[off_d:off_d + T] = -(price * dt_h) + vom * dt_h
    c_obj[off_c:off_c + T] = +(price * dt_h) + vom * dt_h

    bounds = [(0.0, p_mw)] * (n_d + n_c) + [(e_min, e_max)] * n_e
    bounds[off_e] = (e_init, e_init)
    if terminal_soc_mwh is not None:
        e_terminal = float(terminal_soc_mwh)
        bounds[off_e + T] = (e_terminal, e_terminal)

    # Equality constraints: one initial condition row plus T transition rows.
    a_eq = lil_matrix((T + 1, n_vars))
    b_eq = np.zeros(T + 1)
    a_eq[0, off_e] = 1.0
    b_eq[0] = e_init
    for t in range(T):
        row = t + 1
        a_eq[row, off_e + t + 1] = 1.0
        a_eq[row, off_e + t] = -1.0
        a_eq[row, off_d + t] = dt_h / eta_d
        a_eq[row, off_c + t] = -dt_h * eta_c

    # Inequality: d[t] + c[t] <= p_mw.
    a_ub = lil_matrix((T, n_vars))
    b_ub = np.full(T, p_mw)
    for t in range(T):
        a_ub[t, off_d + t] = 1.0
        a_ub[t, off_c + t] = 1.0

    res = linprog(
        c_obj,
        A_ub=a_ub.tocsr(),
        b_ub=b_ub,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"disp": False},
    )
    if not res.success:
        raise RuntimeError(f"Perfect-foresight LP failed: {res.message}")

    x = res.x
    d = x[off_d:off_d + T]
    c = x[off_c:off_c + T]
    e = x[off_e:off_e + T + 1]
    cashflow = price * (d - c) * dt_h - vom * (d + c) * dt_h
    cycles = float((d.sum() * dt_h) / e_name)

    return PerfectForesightResult(
        price=price,
        discharge_mw=d,
        charge_mw=c,
        soc_mwh=e,
        cashflow_gbp=cashflow,
        objective_gbp=float(cashflow.sum()),
        cycles_equiv=cycles,
        status=res.message,
    )
