"""
Rolling intrinsic LP benchmark — Phase 4.

At each EFA gate (every 4h = 8 HH steps), solve a deterministic LP over the
next 48 HH (1 day) using the current DA price forward strip. Apply the first
period's dispatch decision; roll forward; repeat.

This gives the rolling intrinsic value V_RI — a conservative lower bound.
V_LSMC >= V_RI must hold; if not, the stochastic solver is broken.

The LP is:
    max  sum_t  P_da[t] * (d[t] - c[t]) * dt
    s.t. E[0]   = E_init
         E[t+1] = E[t] - d[t]/eta_d * dt + c[t]*eta_c * dt
         E_min  <= E[t] <= E_max
         0      <= d[t] <= P_bar
         0      <= c[t] <= P_bar
         d[t] * c[t] = 0  (linearised: d[t] + c[t] <= P_bar)

The LP ignores ancillary services (they require stochastic uncertainty).
This makes V_RI a clean DA-only benchmark.

References
----------
Boogert & de Jong (2008) — rolling intrinsic for gas storage
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from typing import Optional


def solve_daily_lp(
    prices:     np.ndarray,   # (n_hh,) DA prices for the optimisation window
    E_init:     float,        # starting SoC (MWh)
    E_min:      float,
    E_max:      float,
    P_bar:      float,
    eta_c:      float,
    eta_d:      float,
    dt_h:       float = 0.5,
) -> tuple:
    """
    Solve a deterministic LP dispatch over one optimisation window.

    Uses scipy.optimize.linprog (HiGHS backend — fast enough for 48 vars).

    Decision variables: x = [d[0]..d[T-1], c[0]..c[T-1]]  (2T vars)

    Returns
    -------
    d_opt : (T,) discharge MW
    c_opt : (T,) charge MW
    revenue : float £
    """
    from scipy.optimize import linprog

    T = len(prices)

    # Objective: maximise revenue = sum P[t]*(d[t]-c[t])*dt
    # linprog minimises, so negate
    c_obj = np.concatenate([
        -prices * dt_h,   # -revenue from discharge (d)
         prices * dt_h,   # +cost of charge (c)
    ])

    # Inequality constraints: A_ub @ x <= b_ub
    # 1. Power: d[t] + c[t] <= P_bar  (no simultaneous charge/discharge)
    n_vars = 2 * T
    rows_power = []
    for t in range(T):
        row = np.zeros(n_vars)
        row[t]     = 1.0   # d[t]
        row[T + t] = 1.0   # c[t]
        rows_power.append(row)
    A_power = np.array(rows_power)
    b_power = np.full(T, P_bar)

    # 2. Energy upper bound: E[t+1] <= E_max
    # E[t+1] = E[t] + c[t]*eta_c*dt - d[t]/eta_d*dt
    # Cumulative: E_init + sum_{s<=t} (c[s]*eta_c - d[s]/eta_d)*dt <= E_max
    rows_Eup = []
    for t in range(T):
        row = np.zeros(n_vars)
        for s in range(t + 1):
            row[s]     = -dt_h / eta_d    # discharge reduces E
            row[T + s] =  dt_h * eta_c    # charge increases E
        rows_Eup.append(row)
    A_Eup = np.array(rows_Eup)
    b_Eup = np.full(T, E_max - E_init)

    # 3. Energy lower bound: E[t+1] >= E_min  i.e. -E[t+1] <= -E_min
    rows_Elo = []
    for t in range(T):
        row = np.zeros(n_vars)
        for s in range(t + 1):
            row[s]     =  dt_h / eta_d   # negate: discharge increases -E
            row[T + s] = -dt_h * eta_c
        rows_Elo.append(row)
    A_Elo = np.array(rows_Elo)
    b_Elo = np.full(T, -(E_min - E_init))

    A_ub = np.vstack([A_power, A_Eup, A_Elo])
    b_ub = np.concatenate([b_power, b_Eup, b_Elo])

    # Bounds: 0 <= d[t], c[t] <= P_bar
    bounds = [(0.0, P_bar)] * n_vars

    result = linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
        method='highs',
        options={'disp': False},
    )

    if not result.success:
        # Fallback: idle
        return np.zeros(T), np.zeros(T), 0.0

    d_opt = result.x[:T]
    c_opt = result.x[T:]
    revenue = float(-result.fun)
    return d_opt, c_opt, revenue


def rolling_intrinsic(
    P_da_paths: np.ndarray,   # (N_paths, T) DA prices
    asset_cfg:  dict,
    lsmc_cfg:   dict,
    fin_cfg:    dict,
    E_init_frac: float = 0.5,
    window_hh:  int   = 48,   # re-optimise every 48 HH (daily)
    gate_hh:    int   = 8,    # re-solve every 8 HH (EFA gate, every 4h)
    verbose:    bool  = True,
) -> tuple:
    """
    Rolling intrinsic valuation over N_paths price scenarios.

    At each EFA gate (every `gate_hh` steps), solve the LP over the next
    `window_hh` half-hours. Apply the first `gate_hh` dispatch decisions;
    advance; repeat.

    Returns
    -------
    pv_paths : (N_paths,) discounted PV of rolling intrinsic cash flows
    soc_paths: (N_paths, T+1) SoC trajectories
    """
    N, T  = P_da_paths.shape
    dt_h  = float(lsmc_cfg.get('dt_hours', 0.5))
    P_bar = float(asset_cfg['power_mw'])
    E_nm  = float(asset_cfg['energy_mwh'])
    eta_c = float(asset_cfg['eta_charge'])
    eta_d = float(asset_cfg['eta_discharge'])
    E_min = float(asset_cfg['soc_min_frac']) * E_nm
    E_max = float(asset_cfg['soc_max_frac']) * E_nm
    r     = float(fin_cfg.get('wacc_merchant', 0.09))
    disc  = float(np.exp(-r * dt_h / 8760))

    E_init = E_init_frac * E_nm

    cf_store  = np.zeros((N, T), dtype=np.float32)
    soc_store = np.zeros((N, T + 1), dtype=np.float32)
    soc_store[:, 0] = E_init

    disc_factors = disc ** np.arange(T, dtype=np.float32)

    for n in range(N):
        if verbose and n % max(1, N // 10) == 0:
            print(f"  RI path {n}/{N} ...", end='\r')

        E_n  = E_init
        t    = 0
        while t < T:
            # DA price window
            t_end   = min(t + window_hh, T)
            prices  = P_da_paths[n, t:t_end]
            win_len = len(prices)

            # Solve LP
            d_opt, c_opt, _ = solve_daily_lp(
                prices, E_n, E_min, E_max, P_bar, eta_c, eta_d, dt_h,
            )

            # Apply first `gate_hh` decisions
            apply_len = min(gate_hh, T - t)
            for s in range(apply_len):
                d_s = float(d_opt[s]) if s < len(d_opt) else 0.0
                c_s = float(c_opt[s]) if s < len(c_opt) else 0.0

                cf_store[n, t + s] = float(
                    (P_da_paths[n, t + s] * (d_s - c_s)) * dt_h
                )
                dE = (-d_s / eta_d + c_s * eta_c) * dt_h
                E_n = float(np.clip(E_n + dE, E_min, E_max))
                soc_store[n, t + s + 1] = E_n

            t += apply_len

    if verbose:
        print()

    pv_paths = (cf_store * disc_factors[None, :]).sum(axis=1)
    return pv_paths, soc_store


def _rolling_intrinsic_one_path(args) -> tuple:
    (
        idx,
        prices_path,
        asset_cfg,
        lsmc_cfg,
        fin_cfg,
        E_init_frac,
        window_hh,
        gate_hh,
    ) = args

    T     = len(prices_path)
    dt_h  = float(lsmc_cfg.get('dt_hours', 0.5))
    P_bar = float(asset_cfg['power_mw'])
    E_nm  = float(asset_cfg['energy_mwh'])
    eta_c = float(asset_cfg['eta_charge'])
    eta_d = float(asset_cfg['eta_discharge'])
    E_min = float(asset_cfg['soc_min_frac']) * E_nm
    E_max = float(asset_cfg['soc_max_frac']) * E_nm
    r     = float(fin_cfg.get('wacc_merchant', 0.09))
    disc  = float(np.exp(-r * dt_h / 8760))

    E_n = float(E_init_frac * E_nm)
    soc = np.zeros(T + 1, dtype=np.float32)
    soc[0] = E_n
    pv = 0.0

    t = 0
    while t < T:
        t_end = min(t + window_hh, T)
        prices = prices_path[t:t_end]

        d_opt, c_opt, _ = solve_daily_lp(
            prices, E_n, E_min, E_max, P_bar, eta_c, eta_d, dt_h,
        )

        apply_len = min(gate_hh, T - t)
        for s in range(apply_len):
            d_s = float(d_opt[s]) if s < len(d_opt) else 0.0
            c_s = float(c_opt[s]) if s < len(c_opt) else 0.0
            step = t + s

            cashflow = float(prices_path[step] * (d_s - c_s) * dt_h)
            pv += cashflow * (disc ** step)

            dE = (-d_s / eta_d + c_s * eta_c) * dt_h
            E_n = float(np.clip(E_n + dE, E_min, E_max))
            soc[step + 1] = E_n

        t += apply_len

    return idx, float(pv), soc


def rolling_intrinsic_parallel(
    P_da_paths: np.ndarray,
    asset_cfg: dict,
    lsmc_cfg: dict,
    fin_cfg: dict,
    E_init_frac: float = 0.5,
    window_hh: int = 48,
    gate_hh: int = 8,
    max_workers: Optional[int] = None,
    backend: str = "thread",
    verbose: bool = True,
) -> tuple:
    """
    Parallel path-level rolling intrinsic valuation.

    Each price path is independent, so this distributes paths across workers.
    The default thread backend is notebook-friendly on Windows and works well
    here because the expensive LP solve runs in SciPy/HiGHS native code. Use
    backend="process" only from a normal Python script with spawn-safe entry.
    """
    P_da_paths = np.asarray(P_da_paths, dtype=np.float32)
    N, T = P_da_paths.shape
    if max_workers is None:
        max_workers = max(1, min(N, (os.cpu_count() or 2) - 1))

    pv_paths = np.zeros(N, dtype=np.float64)
    soc_paths = np.zeros((N, T + 1), dtype=np.float32)

    tasks = [
        (
            n,
            P_da_paths[n],
            asset_cfg,
            lsmc_cfg,
            fin_cfg,
            E_init_frac,
            window_hh,
            gate_hh,
        )
        for n in range(N)
    ]

    done = 0
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor

    with executor_cls(max_workers=max_workers) as executor:
        futures = [executor.submit(_rolling_intrinsic_one_path, task) for task in tasks]
        for future in as_completed(futures):
            idx, pv, soc = future.result()
            pv_paths[idx] = pv
            soc_paths[idx] = soc
            done += 1
            if verbose and (done == 1 or done == N or done % max(1, N // 10) == 0):
                print(f"  RI paths complete {done}/{N} ...", end='\r')

    if verbose:
        print()

    return pv_paths, soc_paths
