#!/usr/bin/env python
# coding: utf-8

# # 12 - Phase 4 Method Comparison
# 
# Single-duration comparison notebook for a 2h BESS under medium Phase 4 settings.
# 
# Methods included:
# - **Initial hourly intrinsic**: HPFC daily LP from notebook 11, renamed from HPFC perfect-foresight dispatch.
# - **DA rolling intrinsic**: HPFC-anchored simulated DA paths, 48 half-hour look-ahead and 48 half-hour roll step.
# - **WD rolling intrinsic**: HPFC-anchored simulated DA paths, 48 half-hour look-ahead and 8 half-hour roll step.
# - **Forward simulation**: LSMC backward policy plus out-of-sample forward application on HPFC-anchored simulated DA paths.
# - **Perfect foresight**: full-horizon perfect-foresight LP on sampled HPFC-anchored simulated DA paths.
# 
# Configured here as:
# ```python
# PHASE4_RUN_MODE_FOR_SWEEP = "medium"
# SWEEP_DURATIONS_H = [2.0]
# ```
# 
# The simulated methods use the HPFC hourly curve from the initial hourly intrinsic method as the deterministic anchor, then apply relative stochastic movements from the Phase 3 simulation bundle.
# 

# In[ ]:


from __future__ import annotations

import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.config import ASSET, DEGRADATION, FINANCE, LSMC as LSMC_CFG, SCHWARTZ_SMITH, configure_asset_duration
from src.optimisation.dispatch import enumerate_modes
from src.optimisation.lsmc import LSMCSolver
from src.optimisation.perfect_foresight import solve_perfect_foresight
from src.optimisation.rolling_intrinsic import rolling_intrinsic_parallel, rolling_intrinsic, solve_daily_lp
from src.processes.ancillary import AncillaryParams
from src.processes.hpfc import HPFCParams
from src.processes.imbalance import ImbalanceParams
from src.processes.schwartz_smith import SSParams
from src.processes.simulate import PathBundle, default_params_from_config
from src.validation import summarize_action_distribution, validate_path_bundle

RAW = PROJECT_ROOT / 'data' / 'raw'
PROCESSED = PROJECT_ROOT / 'data' / 'processed'
PROCESSED.mkdir(parents=True, exist_ok=True)

PHASE4_RUN_MODE_FOR_SWEEP = 'medium'
# Override via NB12_DURATION_H env var for batch execution from notebook 13.
# Interactive use: edit the default value in the os.environ.get() call below.
VALUATION_DURATION_H = float(os.environ.get('NB12_DURATION_H', '2.0'))
SWEEP_DURATIONS_H = [VALUATION_DURATION_H]
SEED = int(LSMC_CFG.get('seed', 42))

# Keep the notebook compatible with the duration-sweep convention.
config.VALID_ASSET_DURATIONS_H = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
ASSET_VAL = copy.deepcopy(ASSET)
configure_asset_duration(ASSET_VAL, VALUATION_DURATION_H)
DURATION_LABEL = f'{VALUATION_DURATION_H:g}h'

PHASE4_RUNS = {
    'medium': {
        'ri_paths': 500,
        'bwd_paths': 500,
        'bwd_steps': 4320,
        'fwd_paths': 500,
        'fwd_workers': 2,          # 2 threads keeps CPU cooler during forward pass
        # ridge_alpha=1.0: mild regularisation (shrinkage ~0.2% with N=500).
        # Previous value of 200 over-shrunk slope features, preventing the
        # continuation value from learning intertemporal DA spread signals, and
        # causing the LSMC to behave greedily (7% DA energy capture vs DA RI).
        'ridge_alpha': 1.0,
        # Use the production cap; low caps clip long-duration continuation values.
        'continuation_value_cap_gbp': float(LSMC_CFG.get('continuation_value_cap_gbp', 25_000_000)),
        # SoC grid scaling: 5 nodes/hour (min 9).
        # Gives 1h:9, 2h:10, 3h:15, 4h:20 — finer grid for longer durations
        # where the SoC range (MWh) is wider and interpolation errors are larger.
        'n_soc_nodes_per_hour': 5,
        'soh_nodes': [1.00, 0.90, 0.82],
        'net_levels': [-1.0, -0.5, 0.0, 0.5, 1.0],
        'dc_levels': [0.0, 0.5],
        'qr_levels': [0.0, 0.25],
    }
}
PHASE4_RUN = dict(PHASE4_RUNS[PHASE4_RUN_MODE_FOR_SWEEP])

# Apply per-run overrides from sweep parent (NB12_ env vars set by notebook 13)
for _k, _e in [('bwd_paths', 'NB12_BWD_PATHS'), ('ri_paths', 'NB12_RI_PATHS'),
               ('fwd_paths', 'NB12_FWD_PATHS'), ('bwd_steps', 'NB12_BWD_STEPS'),
               ('fwd_workers', 'NB12_FWD_WORKERS')]:
    _v = os.environ.get(_e, '')
    if _v:
        PHASE4_RUN[_k] = int(_v)
# Float overrides
for _k, _e in [('ridge_alpha', 'NB12_RIDGE_ALPHA')]:
    _v = os.environ.get(_e, '')
    if _v:
        PHASE4_RUN[_k] = float(_v)

print(f'Project root: {PROJECT_ROOT}')
print(f'Run mode: {PHASE4_RUN_MODE_FOR_SWEEP}')
print(f'Duration: {DURATION_LABEL} ({ASSET_VAL["power_mw"]:.0f} MW / {ASSET_VAL["energy_mwh"]:.0f} MWh)')
print(f'Seed: {SEED}')


# ## 1  Load Simulation Bundle and Parameters
# 

# In[ ]:


def _load_json_or_default(cls, default_obj, *fnames):
    for fname in fnames:
        path = PROCESSED / fname
        if path.exists():
            print(f'Loaded calibrated {fname}')
            return cls.from_json(path)
    print(f'Using config default for {" or ".join(fnames)}')
    return default_obj

ss_def, hpfc_def, imb_def, anc_def, bm_def = default_params_from_config()
ss_p = _load_json_or_default(SSParams, ss_def, 'ss_params.json')
hpfc_p = _load_json_or_default(HPFCParams, hpfc_def, 'pca_params.json', 'hpfc_params.json')
imb_p = _load_json_or_default(ImbalanceParams, imb_def, 'imbalance_params.json')
anc_p = _load_json_or_default(AncillaryParams, anc_def, 'ancillary_params.json')
bm_p = _load_json_or_default(BMParams, bm_def, 'bm_params.json')

SIM_BUNDLE_PATH = PROCESSED / 'sim_bundle.pkl'
if not SIM_BUNDLE_PATH.exists():
    raise FileNotFoundError(f'Missing simulation bundle: {SIM_BUNDLE_PATH}. Run notebooks/03_simulation.ipynb first.')

with SIM_BUNDLE_PATH.open('rb') as f:
    bundle = pickle.load(f)

bundle_check = validate_path_bundle(bundle, forward_anchor_gbp_mwh=SCHWARTZ_SMITH['forward_anchor_gbp_mwh'])
bundle_check.raise_if_failed()
print(bundle_check.summary())

RUN_STEPS = min(int(PHASE4_RUN['bwd_steps'] or bundle.n_steps), bundle.n_steps)
RUN_BWD_PATHS = min(int(PHASE4_RUN['bwd_paths'] or bundle.n_paths), bundle.n_paths)
RUN_FWD_PATHS = min(int(PHASE4_RUN['fwd_paths'] or bundle.n_paths), bundle.n_paths)
RI_N_PATHS = min(int(PHASE4_RUN['ri_paths']), max(bundle.n_paths - RUN_BWD_PATHS, 0))
if int(PHASE4_RUN['ri_paths']) > RI_N_PATHS:
    print(
        f'WARNING: requested ri_paths={int(PHASE4_RUN["ri_paths"])} but only '
        f'{RI_N_PATHS} paths remain after reserving {RUN_BWD_PATHS} backward paths.'
    )
DT_H = float(LSMC_CFG.get('dt_hours', 0.5))
valuation_horizon_years = RUN_STEPS * DT_H / 8760.0
annualization_factor = 1.0 / valuation_horizon_years

ln_P_bundle_mat = np.asarray(bundle.ln_P_base[:, :RUN_STEPS + 1], dtype=np.float32)
P_da_bundle_mat = np.exp(np.clip(ln_P_bundle_mat, -100.0, np.log(500.0))).astype(np.float32)

# Reserve backward paths first so the common evaluation set is strictly disjoint
# from the LSMC training set.  All three evaluation methods (RI, PF, LSMC forward)
# then run on identical paths, making per-path P5/P95 distributions directly comparable.
rng = np.random.default_rng(SEED)
bwd_idx    = rng.choice(bundle.n_paths, size=RUN_BWD_PATHS, replace=False)
_remaining = np.setdiff1d(np.arange(bundle.n_paths), bwd_idx)
common_idx = rng.choice(_remaining, size=RI_N_PATHS, replace=False)
ri_idx     = common_idx   # alias used by RI and PF cells

# Filled after the HPFC curve is loaded in the initial hourly intrinsic section.
P_da_anchored_mat = None
ln_P_anchored_mat = None
P_da_ri = None

print(f'Selected horizon: {RUN_STEPS:,} half-hours = {valuation_horizon_years:.3f} years')
print(f'Common evaluation paths: {RI_N_PATHS:,}  (RI, PF, LSMC forward — disjoint from LSMC backward)')
print(f'LSMC backward paths: {RUN_BWD_PATHS:,}')


# ## 2  Initial Hourly Intrinsic (HPFC Daily LP)
# 

# In[ ]:


# This method is the HPFC daily LP from notebook 11, renamed here as initial hourly intrinsic.
# It values deterministic hourly HPFC arbitrage over a 5-year forward-curve horizon.
DATA_PROC = PROCESSED
DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
HORIZON_YEARS = 5
APPLY_MARKET_DECAY = True
AS_OF_DATE = pd.Timestamp('2026-05-03')
HORIZON_END = AS_OF_DATE + pd.DateOffset(years=HORIZON_YEARS)

def calendar_weighted_mean(year: int, month: int, mult: pd.DataFrame) -> float:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    total, count = 0.0, 0
    for d in pd.date_range(start, end, freq='D'):
        total += float(mult[DOW_NAMES[d.dayofweek]].sum())
        count += 24
    return total / count

def load_or_build_hpfc_full() -> tuple[pd.DataFrame, pd.Timestamp]:
    hpfc_path = DATA_PROC / 'hpfc.parquet'
    shape_path = DATA_PROC / 'shape_multipliers.parquet'
    if not hpfc_path.exists() or not shape_path.exists():
        raise FileNotFoundError('Missing hpfc.parquet or shape_multipliers.parquet. Run notebooks/11_hourly_forward_curve.ipynb first.')
    hpfc = pd.read_parquet(hpfc_path)
    hpfc['delivery_date'] = pd.to_datetime(hpfc['delivery_date'])
    hpfc_fut = hpfc[hpfc['delivery_date'] >= AS_OF_DATE].copy()
    hpfc_cov = hpfc_fut['delivery_date'].max()
    shape = pd.read_parquet(shape_path)
    raw_mult = shape.pivot(index='hour', columns='dow_name', values='multiplier')[DOW_NAMES]
    gap_days = max(0, (HORIZON_END - hpfc_cov).days - 1)
    if gap_days <= 0:
        return hpfc_fut[hpfc_fut['delivery_date'] <= HORIZON_END].copy(), hpfc_cov
    trailing_start = hpfc_cov - pd.DateOffset(months=12)
    flat_px = float(hpfc_fut[hpfc_fut['delivery_date'] >= trailing_start]['monthly_fwd_gbp_mwh'].mean())
    ext_rows = []
    for d in pd.date_range(hpfc_cov + pd.Timedelta(days=1), HORIZON_END, freq='D'):
        cm = calendar_weighted_mean(d.year, d.month, raw_mult)
        col = DOW_NAMES[d.dayofweek]
        for h in range(24):
            m = float(raw_mult.loc[h, col]) / cm
            ext_rows.append({
                'delivery_date': d, 'hour': h, 'dow': d.dayofweek,
                'delivery_month': d.strftime('%Y-%m'),
                'monthly_fwd_gbp_mwh': flat_px,
                'multiplier': m, 'price_gbp_mwh': flat_px * m,
            })
    return pd.concat([hpfc_fut, pd.DataFrame(ext_rows)], ignore_index=True), hpfc_cov

def run_initial_hourly_intrinsic(asset_cfg: dict) -> dict:
    hpfc_full, hpfc_cov = load_or_build_hpfc_full()
    hpfc_full = hpfc_full.sort_values(['delivery_date', 'hour']).reset_index(drop=True)
    decay = float(FINANCE['revenue_decay_per_year'])
    p_bar = float(asset_cfg['power_mw'])
    e_nm = float(asset_cfg['energy_mwh'])
    eta_c = float(asset_cfg['eta_charge'])
    eta_d = float(asset_cfg['eta_discharge'])
    e_min = float(asset_cfg['soc_min_mwh'])
    e_max = float(asset_cfg['soc_max_mwh'])
    avail = float(asset_cfg['availability'])
    e_t = float(asset_cfg['soc_init_mwh'])
    daily_rows = []
    for d, grp in hpfc_full.groupby('delivery_date'):
        prices = grp.sort_values('hour')['price_gbp_mwh'].to_numpy()
        if len(prices) < 24:
            continue
        d_opt, c_opt, gross_rev = solve_daily_lp(prices, e_t, e_min, e_max, p_bar, eta_c, eta_d, dt_h=1.0)
        e_t = float(np.clip(e_t + (c_opt * eta_c - d_opt / eta_d).sum(), e_min, e_max))
        daily_rows.append({
            'date': pd.Timestamp(d),
            'gross_revenue_gbp': gross_rev * avail,
            'cycles_equiv': float(d_opt.sum() / e_nm) * avail,
        })
    daily = pd.DataFrame(daily_rows)
    daily['years_elapsed'] = (daily['date'] - AS_OF_DATE).dt.days / 365.25
    daily['decay_factor'] = (1 - decay) ** daily['years_elapsed'] if APPLY_MARKET_DECAY else 1.0
    daily['gross_adj_gbp'] = daily['gross_revenue_gbp'] * daily['decay_factor']
    annual = daily.groupby(((daily['date'] - AS_OF_DATE).dt.days // 365.25).astype(int)).agg(
        gross_gbp=('gross_adj_gbp', 'sum'), cycles=('cycles_equiv', 'sum')
    ).head(HORIZON_YEARS)
    annual_mean = float(annual['gross_gbp'].mean())
    return {
        'method': 'Initial hourly intrinsic',
        'value_gbp_horizon_mean': annual_mean,
        'value_gbp_annualized_mean': annual_mean,
        'gbp_per_mw_year': annual_mean / asset_cfg['power_mw'],
        'n_paths': 1, 'window_hh': 24, 'gate_hh': 24,
        'pv_paths': np.array([annual_mean]),
        'notes': f'HPFC hourly daily LP; 5-year annual avg gross revenue as-of {AS_OF_DATE.date()}; energy-only',
    }

initial_hourly_row = run_initial_hourly_intrinsic(ASSET_VAL)
initial_hourly_row


def build_hpfc_half_hour_anchor(hpfc_full: pd.DataFrame) -> np.ndarray:
    repeats_per_hour = int(round(1.0 / DT_H))
    if not np.isclose(repeats_per_hour * DT_H, 1.0):
        raise ValueError(f'DT_H={DT_H} does not divide one hour cleanly.')
    hourly = (
        hpfc_full.sort_values(['delivery_date', 'hour'])['price_gbp_mwh']
        .to_numpy(dtype=float)
    )
    hh = np.repeat(hourly, repeats_per_hour)
    if len(hh) < RUN_STEPS + 1:
        hh = np.pad(hh, (0, RUN_STEPS + 1 - len(hh)), mode='edge')
    return np.clip(hh[:RUN_STEPS + 1], 1e-3, 500.0).astype(np.float32)


def build_hpfc_anchored_simulation_prices(anchor_curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel_log = ln_P_bundle_mat - ln_P_bundle_mat[:, :1]
    relative_move = np.exp(np.clip(rel_log, -3.0, 3.0))
    prices = np.clip(anchor_curve[None, :] * relative_move, 1e-3, 500.0).astype(np.float32)
    return prices, np.log(prices).astype(np.float32)


hpfc_full_for_anchor, hpfc_anchor_coverage = load_or_build_hpfc_full()
hpfc_half_hour_anchor = build_hpfc_half_hour_anchor(hpfc_full_for_anchor)
P_da_anchored_mat, ln_P_anchored_mat = build_hpfc_anchored_simulation_prices(hpfc_half_hour_anchor)
P_da_ri = P_da_anchored_mat[ri_idx, :RUN_STEPS].astype(np.float32)

print(f'HPFC anchor: {len(hpfc_half_hour_anchor):,} half-hours; first price GBP{hpfc_half_hour_anchor[0]:.2f}/MWh')
print(f'Anchored DA paths: mean GBP{P_da_anchored_mat[:, :RUN_STEPS].mean():.1f}/MWh, std GBP{P_da_anchored_mat[:, :RUN_STEPS].std():.1f}/MWh')

# ── LSMC config ───────────────────────────────────────────────────────────────
# SoC grid: 6 nodes/hour (min 9) keeps MWh grid spacing roughly constant.
_n_soc = max(9, int(PHASE4_RUN.get('n_soc_nodes_per_hour', 6) * VALUATION_DURATION_H))

# Forward HPFC window: duration-aware.
# A battery needs to look far enough ahead to justify holding charge vs cycling now.
# Rule: 12 HH per hour of duration (min 16 HH = 8h).
#   1h → 16 HH (8h),  2h → 24 HH (12h),  3h → 36 HH (18h),  4h → 48 HH (24h)
# Feature is max(HPFC[t+1:t+W]) - HPFC[t]: best available sell price in the window.
_fwd_window = max(16, int(PHASE4_RUN.get('fwd_window_hh_per_hour', 12) * VALUATION_DURATION_H))

print(f'n_soc_nodes : {_n_soc}')
print(f'fwd_window  : {_fwd_window} HH = {_fwd_window * DT_H:.0f}h look-ahead  (max HPFC spread)')

# ── Prepare and launch LSMC backward in background thread ────────────────────
from concurrent.futures import ThreadPoolExecutor as _TPE

LSMC_FAST_CFG = dict(LSMC_CFG)
LSMC_FAST_CFG.update({
    'n_soc_nodes': _n_soc,
    'soh_nodes': PHASE4_RUN['soh_nodes'],
    'n_soh_nodes': len(PHASE4_RUN['soh_nodes']),
    'ridge_alpha': float(PHASE4_RUN['ridge_alpha']),
    'continuation_value_cap_gbp': float(PHASE4_RUN['continuation_value_cap_gbp']),
    'fwd_window_hh': _fwd_window,
})
FAST_MODES = enumerate_modes(
    net_levels=PHASE4_RUN['net_levels'],
    dc_levels=PHASE4_RUN['dc_levels'],
    qr_levels=PHASE4_RUN['qr_levels'],
    bm_levels=PHASE4_RUN.get('r_bm_levels', [0.0, 0.25, 0.5]),
)
bundle_bwd = PathBundle(
    chi=bundle.chi[bwd_idx, :RUN_STEPS + 1],
    xi=bundle.xi[bwd_idx, :RUN_STEPS + 1],
    ln_P_base=ln_P_anchored_mat[bwd_idx, :RUN_STEPS + 1],
    lam=bundle.lam[bwd_idx, :RUN_STEPS + 1, :],
    delta_imb=bundle.delta_imb[bwd_idx, :RUN_STEPS + 1],
    pi_bm=bundle.pi_bm[bwd_idx, :RUN_STEPS + 1],
    pi={k: v[bwd_idx, :RUN_STEPS + 1] for k, v in bundle.pi.items()},
    dt=bundle.dt,
    n_paths=RUN_BWD_PATHS,
    n_steps=RUN_STEPS,
)
solver = LSMCSolver(
    ASSET_VAL, LSMC_FAST_CFG, DEGRADATION, FINANCE,
    modes=FAST_MODES, verbose=False,
    hpfc_params=hpfc_p,
    hpfc_curve=hpfc_half_hour_anchor,
)
_bwd_executor = _TPE(max_workers=1, thread_name_prefix='lsmc_bwd')
_bwd_t0 = time.time()
_bwd_future = _bwd_executor.submit(solver.backward, bundle_bwd)
print(f'LSMC backward started in background ({RUN_BWD_PATHS:,} paths × {RUN_STEPS:,} steps; {len(FAST_MODES)} modes)')


# ## 3  Rolling Intrinsic Benchmarks
# 

# In[ ]:


from concurrent.futures import ThreadPoolExecutor as _TPE  # may already be imported; safe to repeat

# Intraday basis paths for WD — same path indices as P_da_ri
delta_imb_ri = bundle.delta_imb[ri_idx, :RUN_STEPS].astype(np.float32)
# Cap intraday visibility at a realistic ID-continuous spread (not full SP spike range).
# delta_imb includes extreme OU+jump values (±1000+ GBP/MWh) that are not visible at
# the WD gate; capping here models ID price visibility rather than SP settlement exposure.
_wd_cap = float(os.environ.get('NB12_WD_UPLIFT_CAP_GBP_MWH', '10'))
delta_imb_ri_wd = np.clip(delta_imb_ri, -_wd_cap, _wd_cap)
print(f'WD intraday visibility cap: ±GBP{_wd_cap:.0f}/MWh (raw delta_imb std={delta_imb_ri.std():.1f})')

def summarise_path_values(method: str, values: np.ndarray, n_paths: int, window_hh: int, gate_hh: int, notes: str) -> dict:
    mean_horizon = float(np.mean(values))
    ann = mean_horizon * annualization_factor
    return {
        'method': method,
        'value_gbp_horizon_mean': mean_horizon,
        'value_gbp_annualized_mean': ann,
        'gbp_per_mw_year': ann / ASSET_VAL['power_mw'],
        'n_paths': int(n_paths),
        'window_hh': int(window_hh),
        'gate_hh': int(gate_hh),
        'pv_paths': np.asarray(values) * annualization_factor,   # annualised per-path values
        'notes': notes,
    }

def run_rolling_intrinsic_case(
    name: str,
    window_hh: int,
    gate_hh: int,
    max_workers: int | None = None,
    delta_imb_paths: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    cpu = os.cpu_count() or 2
    workers = max(1, min(8, cpu - 1, RI_N_PATHS)) if max_workers is None else max(1, min(max_workers, RI_N_PATHS))
    label = 'WD (intraday prices)' if delta_imb_paths is not None else 'DA'
    print(f'Running {name} [{label}]: window_hh={window_hh}, gate_hh={gate_hh}, paths={RI_N_PATHS}, workers={workers}')
    t0 = time.time()
    try:
        pv, soc = rolling_intrinsic_parallel(
            P_da_ri,
            ASSET_VAL,
            LSMC_CFG,
            FINANCE,
            E_init_frac=0.5,
            window_hh=window_hh,
            gate_hh=gate_hh,
            max_workers=workers,
            backend='thread',
            verbose=True,
            delta_imb_paths=delta_imb_paths,
        )
    except Exception as exc:
        print(f'Parallel rolling intrinsic failed ({exc}); falling back to serial')
        pv, soc = rolling_intrinsic(
            P_da_ri,
            ASSET_VAL,
            LSMC_CFG,
            FINANCE,
            E_init_frac=0.5,
            window_hh=window_hh,
            gate_hh=gate_hh,
            verbose=True,
            delta_imb_paths=delta_imb_paths,
        )
    print(f'{name} complete in {time.time() - t0:.1f}s; mean horizon GBP{np.mean(pv):,.0f}')
    notes_suffix = 'intraday (DA + delta_imb) prices at gate' if delta_imb_paths is not None else 'DA prices only'
    return summarise_path_values(
        name,
        pv,
        RI_N_PATHS,
        window_hh,
        gate_hh,
        f'HPFC-anchored simulated paths ({RI_N_PATHS} paths); rolling LP; {notes_suffix}; energy-only, VOM=0',
    ), pv

# Run DA and WD concurrently — split available cores evenly between them.
# Progress lines from both will interleave in the output; that's expected.
# Both use gate_hh=8 (re-solve every EFA gate = 4h) so the only structural
# difference between DA RI and WD RI is the intraday price premium at the gate.
_ri_cpu = os.cpu_count() or 2
_ri_workers_each = max(1, (_ri_cpu - 1) // 2)
print(f'Launching DA and WD rolling intrinsic in parallel ({_ri_workers_each} workers each; {_ri_cpu} CPUs total)')
_ri_t0 = time.time()
with _TPE(max_workers=2) as _ri_exc:
    _da_fut = _ri_exc.submit(run_rolling_intrinsic_case, 'DA rolling intrinsic', 48,  8, _ri_workers_each, None)
    _wd_fut = _ri_exc.submit(run_rolling_intrinsic_case, 'WD rolling intrinsic', 48,  8, _ri_workers_each, delta_imb_ri_wd)
    da_rolling_row, da_rolling_pv = _da_fut.result()
    wd_rolling_row, wd_rolling_pv = _wd_fut.result()
print(f'Both RI complete in {time.time() - _ri_t0:.1f}s wall-clock')
[da_rolling_row, wd_rolling_row]


# ## 4  Forward Simulation (LSMC Policy)
# 

# In[ ]:


# LSMC_FAST_CFG, FAST_MODES, solver, bundle_bwd were set up in the initial hourly
# intrinsic cell, and the backward pass is already running in the background.
print('Waiting for LSMC backward to complete...')
policy = _bwd_future.result()
_bwd_executor.shutdown(wait=False)
print(f'Backward complete in {time.time() - _bwd_t0:.1f}s')
print(f'cont_beta shape: {policy.cont_beta.shape if policy.cont_beta is not None else "None (legacy)"}')
_clip_obs = float(policy.diagnostics.get('continuation_clip_observation_fraction', 0.0))
_clip_max = float(policy.diagnostics.get('continuation_clip_fraction_max', 0.0))
_clip_cap = float(policy.diagnostics.get('continuation_value_cap_gbp', 0.0))
print(f'continuation clip: obs={_clip_obs:.3%}, max_reg={_clip_max:.3%}, cap=GBP{_clip_cap:,.0f}')
if _clip_obs > 0.01 or _clip_max > 0.05:
    raise RuntimeError(
        'LSMC continuation clipping is material; increase continuation_value_cap_gbp before using this duration output.'
    )

# ── Build original and antithetic forward bundles ────────────────────────────
# Antithetic: reflect all stochastic innovations around their anchor/zero.
#   ln_P:      mirror around HPFC anchor  → ln_P_anti = 2*ln_hpfc - ln_P
#   chi/lam/delta_imb: zero-mean OU/PCA  → negate directly
#   xi/pi:     non-zero start             → reflect around initial value
fwd_idx    = common_idx
fwd_n_paths = len(fwd_idx)
ln_hpfc_1d = np.log(hpfc_half_hour_anchor[:RUN_STEPS + 1]).astype(np.float32)

def _make_fwd_bundle(idx, anti: bool) -> PathBundle:
    chi       = bundle.chi[idx, :RUN_STEPS + 1]
    xi        = bundle.xi[idx, :RUN_STEPS + 1]
    ln_P_base = ln_P_anchored_mat[idx, :RUN_STEPS + 1]
    lam       = bundle.lam[idx, :RUN_STEPS + 1, :]
    delta_imb = bundle.delta_imb[idx, :RUN_STEPS + 1]
    pi        = {k: v[idx, :RUN_STEPS + 1] for k, v in bundle.pi.items()}
    if anti:
        chi       = -chi
        xi        = 2 * xi[:, :1]            - xi
        ln_P_base = 2 * ln_hpfc_1d[None, :] - ln_P_base
        lam       = -lam
        delta_imb = -delta_imb
        pi        = {k: 2 * v[:, :1] - v for k, v in pi.items()}
    return PathBundle(
        chi=chi, xi=xi, ln_P_base=ln_P_base, lam=lam, delta_imb=delta_imb,
        pi=pi, dt=bundle.dt, n_paths=len(idx), n_steps=RUN_STEPS,
    )

bundle_fwd      = _make_fwd_bundle(fwd_idx, anti=False)
bundle_fwd_anti = _make_fwd_bundle(fwd_idx, anti=True)

# ── Run both forward passes and average pairwise ─────────────────────────────
workers = max(1, min(int(PHASE4_RUN['fwd_workers']), (os.cpu_count() or 2) - 1, fwd_n_paths))
print(f'Forward simulation: {fwd_n_paths:,} paths × 2 (+ antithetic); workers={workers}')
t0 = time.time()
try:
    result_orig = solver.forward_parallel(bundle_fwd,      policy, max_workers=workers)
    result_anti = solver.forward_parallel(bundle_fwd_anti, policy, max_workers=workers)
except Exception as exc:
    print(f'Parallel forward failed ({exc}); falling back to serial')
    result_orig = solver.forward(bundle_fwd,      policy)
    result_anti = solver.forward(bundle_fwd_anti, policy)
print(f'Forward complete in {time.time() - t0:.1f}s')

pv_av = (result_orig.pv_paths + result_anti.pv_paths) / 2.0
print(f'Std original: £{result_orig.pv_paths.std():,.0f}   '
      f'Std averaged: £{pv_av.std():,.0f}   '
      f'Variance reduction: {100*(1 - pv_av.std()/result_orig.pv_paths.std()):.1f}%')

lsmc_ann_pv = pv_av * annualization_factor
forward_row = {
    'method': 'Forward simulation (LSMC)',
    'value_gbp_horizon_mean': float(np.mean(pv_av)),
    'value_gbp_annualized_mean': float(np.mean(lsmc_ann_pv)),
    'gbp_per_mw_year': float(np.mean(lsmc_ann_pv)) / ASSET_VAL['power_mw'],
    'n_paths': fwd_n_paths,
    'window_hh': 0,
    'gate_hh': 0,
    'pv_paths': lsmc_ann_pv,
    'notes': (
        f'Non-anticipative LSMC policy on {fwd_n_paths} paths + antithetic variates; '
        f'full-stack: DA+ancillary, VOM={ASSET_VAL.get("vom_gbp_mwh", 1.2):.2f} £/MWh, degradation cost'
    ),
}

# ── Value attribution breakdown (antithetic-averaged, annualised) ─────────────
_BD_KEYS = ['da', 'imbalance', 'dc', 'qr', 'bm', 'costs']
cf_bd_ann = {
    k: (result_orig.cf_breakdown[k] + result_anti.cf_breakdown[k]) / 2.0 * annualization_factor
    for k in _BD_KEYS
}

print('\nLSMC value attribution (mean annualised, £m/year):')
_check = 0.0
for k in _BD_KEYS:
    sign = -1 if k == 'costs' else 1
    v = sign * float(np.mean(cf_bd_ann[k])) / 1e6
    _check += v
    print(f'  {k:12s}  {v:+.3f}m')
print(f'  {"─" * 20}')
print(f'  {"total":12s}  {_check:+.3f}m  (direct sum)')
print(f'  {"pv_av check":12s}  {float(np.mean(lsmc_ann_pv)) / 1e6:+.3f}m  (from pv_paths)')

forward_row


# ## 5  Perfect Foresight (Full-Horizon Simulated DA Paths)
# 

# In[ ]:


PF_N_PATHS = RI_N_PATHS
# Perfect-foresight is an energy-only DA upper benchmark: VOM=0, terminal SoC
# pinned to initial SoC so the optimizer can't harvest stranded inventory value
# that the other methods also lose at horizon end.
PF_TERMINAL_SOC = float(ASSET_VAL['soc_init_mwh'])

print(f'Running perfect foresight on {PF_N_PATHS} sampled HPFC-anchored DA paths x {RUN_STEPS} half-hours')
print(f'  VOM=0 (energy-only upper bound), terminal SoC pinned to {PF_TERMINAL_SOC:.1f} MWh')

pf_values = []
t0 = time.time()
for i, idx in enumerate(ri_idx[:PF_N_PATHS], start=1):
    prices = np.clip(P_da_anchored_mat[idx, :RUN_STEPS], -100.0, 500.0)
    res = solve_perfect_foresight(
        prices, ASSET_VAL, dt_h=DT_H,
        vom_gbp_mwh=0.0,
        terminal_soc_mwh=PF_TERMINAL_SOC,
    )
    pf_values.append(res.objective_gbp)
    if i == 1 or i % max(1, PF_N_PATHS // 5) == 0:
        print(f'  perfect foresight path {i}/{PF_N_PATHS}: GBP{res.objective_gbp:,.0f}')

pf_values = np.asarray(pf_values, dtype=float)
print(f'Perfect foresight complete in {time.time() - t0:.1f}s')

pf_ann = pf_values * annualization_factor
perfect_row = {
    'method': 'Perfect foresight (DA energy)',
    'value_gbp_horizon_mean': float(np.mean(pf_values)),
    'value_gbp_annualized_mean': float(np.mean(pf_ann)),
    'gbp_per_mw_year': float(np.mean(pf_ann)) / ASSET_VAL['power_mw'],
    'n_paths': PF_N_PATHS,
    'window_hh': RUN_STEPS,
    'gate_hh': RUN_STEPS,
    'pv_paths': pf_ann,
    'notes': (
        f'Full-horizon LP on {PF_N_PATHS} sampled HPFC-anchored DA paths; '
        f'energy-only (VOM=0), terminal SoC={PF_TERMINAL_SOC:.0f} MWh; '
        f'upper bound on DA energy arbitrage'
    ),
}
perfect_row


# ## 6  Summary
# 

# In[ ]:


comparison_rows = [
    initial_hourly_row,
    da_rolling_row,
    wd_rolling_row,
    forward_row,
    perfect_row,
]
comparison = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'pv_paths'}
    for r in comparison_rows
])
comparison.insert(0, 'duration_h', VALUATION_DURATION_H)
comparison.insert(1, 'run_mode', PHASE4_RUN_MODE_FOR_SWEEP)
comparison['value_gbp_annualized_m'] = comparison['value_gbp_annualized_mean'] / 1e6
comparison['gbp_per_mw_year_k'] = comparison['gbp_per_mw_year'] / 1e3

# ±1 std error bars — shows scenario spread across paths
for row_d in comparison_rows:
    pvp = np.asarray(row_d.get('pv_paths', []))
    method = row_d['method']
    mask = comparison['method'] == method
    if len(pvp) >= 2:
        std = np.std(pvp, ddof=1)
        comparison.loc[mask, 'p5_ann_m']  = (float(np.mean(pvp)) - std) / 1e6
        comparison.loc[mask, 'p95_ann_m'] = (float(np.mean(pvp)) + std) / 1e6
    else:
        comparison.loc[mask, 'p5_ann_m']  = comparison.loc[mask, 'value_gbp_annualized_m'].values[0]
        comparison.loc[mask, 'p95_ann_m'] = comparison.loc[mask, 'value_gbp_annualized_m'].values[0]

out_csv = PROCESSED / f'phase4_method_comparison_{DURATION_LABEL}.csv'
comparison.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')

out_json = PROCESSED / f'phase4_method_comparison_{DURATION_LABEL}.json'
comparison.to_json(out_json, orient='records', indent=2)
print(f'Saved: {out_json}')

print(comparison[[
    'method', 'value_gbp_annualized_m', 'p5_ann_m', 'p95_ann_m',
    'gbp_per_mw_year_k', 'n_paths', 'notes'
]].round(3).to_string(index=False))

# ── Single combined chart ─────────────────────────────────────────────────────
ENERGY_ONLY = ['Initial hourly intrinsic', 'DA rolling intrinsic', 'WD rolling intrinsic',
               'Perfect foresight (DA energy)']
FULL_STACK  = ['Forward simulation (LSMC)']

METHOD_COLOURS = {
    'Initial hourly intrinsic':      '#aec7e8',
    'Perfect foresight (DA energy)': '#4e79a7',
    'DA rolling intrinsic':          '#2c5f8a',
    'WD rolling intrinsic':          '#1a3a57',
    'Forward simulation (LSMC)':     '#f28e2b',
}

df_all  = comparison.sort_values('value_gbp_annualized_m').reset_index(drop=True)
colours = [METHOD_COLOURS.get(m, '#888888') for m in df_all['method']]
means   = df_all['value_gbp_annualized_m'].values
xerr_lo = np.maximum(means - df_all['p5_ann_m'].values,  0)
xerr_hi = np.maximum(df_all['p95_ann_m'].values - means, 0)

pv_paths_lookup = {
    row['method']: np.asarray(row['pv_paths'], dtype=float) / 1e6
    for row in comparison_rows
}

fig, (ax_bar, ax_hist) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(
    f'Phase 4 method comparison — {DURATION_LABEL} BESS ({PHASE4_RUN_MODE_FOR_SWEEP})\n'
    f'as-of {AS_OF_DATE.date()}',
    fontsize=11,
)

# ── Bar chart ─────────────────────────────────────────────────────────────────
ax_bar.barh(df_all['method'], means, color=colours, alpha=0.85, zorder=2)
ax_bar.errorbar(
    means, df_all['method'],
    xerr=[xerr_lo, xerr_hi],
    fmt='none', color='#333333', capsize=4, linewidth=1.2, zorder=3,
)
for i, (v, n, xhi) in enumerate(zip(means, df_all['n_paths'].values, xerr_hi)):
    ax_bar.text(v + max(0.02, xhi) * 1.1, i, f' {v:.2f}m  (n={n})', va='center', fontsize=8)

ax_bar.set_xlabel('GBP million/year (annualised)')
ax_bar.set_title('All methods  [error bars: ±1 std]', fontsize=10)
ax_bar.grid(axis='x', alpha=0.35, zorder=0)

lsmc_y = list(df_all['method']).index('Forward simulation (LSMC)')
ax_bar.axhline(lsmc_y - 0.5, color='#999999', linewidth=0.8, linestyle='--')

# ── Histogram ─────────────────────────────────────────────────────────────────
for row_d in comparison_rows:
    pvp = pv_paths_lookup[row_d['method']]
    if len(pvp) >= 2:
        ax_hist.hist(
            pvp, bins=25, alpha=0.45,
            color=METHOD_COLOURS.get(row_d['method'], '#888888'),
            label=row_d['method'],
        )
ax_hist.set_xlabel('GBP million/year (annualised)')
ax_hist.set_ylabel('Frequency')
ax_hist.set_title('Distribution across paths  [scenario spread]', fontsize=10)
ax_hist.grid(axis='x', alpha=0.25)
ax_hist.legend(fontsize=7, loc='upper right')

fig.tight_layout()
out_png = PROCESSED / f'phase4_method_comparison_{DURATION_LABEL}.png'
fig.savefig(out_png, dpi=140, bbox_inches='tight')
print(f'Saved: {out_png}')
plt.show()


# ## 7  LSMC Value Attribution

# In[ ]:


BD_LABELS  = {
    'hpfc':      'HPFC anchor',
    'da_surp':   'DA surprise',
    'imbalance': 'Imbalance proxy (BM/ID substitute)',
    'dc':        'DC ancillary',
    'qr':        'QR ancillary',
    'bm':        'Balancing Mechanism',
    'costs':     'Costs (deg+VOM)',
}
BD_COLOURS = {
    'hpfc':      '#aec7e8',
    'da_surp':   '#4e79a7',
    'imbalance': '#76b7b2',
    'dc':        '#59a14f',
    'qr':        '#edc948',
    'bm':        '#f28e2b',
    'costs':     '#e15759',
}

# ── Split da component into HPFC anchor vs DA surprise ───────────────────────
# Post-hoc from action_paths: no lsmc.py changes needed.
#   da = hpfc[t] × net × P_bar × dt  +  (P_da[t] − hpfc[t]) × net × P_bar × dt
#   (both sum to cf_bd_ann['da'] by construction)

_disc_factors = solver.disc ** np.arange(RUN_STEPS)      # (T,)
_net_fracs    = solver._net_fracs                         # (M,) one per mode
_hpfc_t       = hpfc_half_hour_anchor[:RUN_STEPS]         # (T,) deterministic anchor

def _hpfc_da_split(bundle_fwd_b, result_b):
    P_da  = np.exp(np.clip(bundle_fwd_b.ln_P_base[:, :RUN_STEPS],
                           -100.0, np.log(500.0))).astype(np.float64)   # (N, T)
    net   = _net_fracs[result_b.action_paths].astype(np.float64)        # (N, T)
    w     = (solver.P_bar * solver.dt_h) * _disc_factors[None, :]       # (1, T)
    pv_hpfc = ((_hpfc_t[None, :])            * net * w).sum(axis=1)     # (N,)
    pv_surp = ((P_da - _hpfc_t[None, :])     * net * w).sum(axis=1)     # (N,)
    return pv_hpfc, pv_surp

_hpfc_orig, _surp_orig = _hpfc_da_split(bundle_fwd,      result_orig)
_hpfc_anti, _surp_anti = _hpfc_da_split(bundle_fwd_anti, result_anti)

# Antithetic average then annualise
_ANN = annualization_factor
cf_bd_refined = {
    'hpfc':      (_hpfc_orig + _hpfc_anti) / 2.0 * _ANN,
    'da_surp':   (_surp_orig + _surp_anti) / 2.0 * _ANN,
    'imbalance': cf_bd_ann['imbalance'],
    'dc':        cf_bd_ann['dc'],
    'qr':        cf_bd_ann['qr'],
    'bm':        cf_bd_ann['bm'],
    'costs':     cf_bd_ann['costs'],
}

# Verify: hpfc + da_surp should equal the original da component
_da_recon = (cf_bd_refined['hpfc'] + cf_bd_refined['da_surp']) / 1e6
_da_orig   = cf_bd_ann['da'] / 1e6
print(f'DA reconciliation check: '
      f'hpfc+surp mean £{np.mean(_da_recon):.3f}m  vs  da mean £{np.mean(_da_orig):.3f}m  '
      f'(max abs diff: {np.abs(_da_recon - _da_orig).max():.1e})')

# Mean values per component (costs shown as negative)
_keys_ord = ['hpfc', 'da_surp', 'imbalance', 'dc', 'qr', 'bm', 'costs']
_sign = {k: (-1 if k == 'costs' else 1) for k in _keys_ord}
_bd_means = {k: _sign[k] * float(np.mean(cf_bd_refined[k])) / 1e6 for k in _keys_ord}
total_check = sum(_bd_means.values())

print('\nLSMC value attribution (mean annualised, £m/year):')
for k in _keys_ord:
    print(f'  {BD_LABELS[k]:18s}  {_bd_means[k]:+.3f}m')
print(f'  {"─" * 28}')
print(f'  {"total":18s}  {total_check:+.3f}m')

# ── Charts ────────────────────────────────────────────────────────────────────
fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(
    f'LSMC value attribution — {DURATION_LABEL} BESS ({PHASE4_RUN_MODE_FOR_SWEEP})\n'
    f'{fwd_n_paths} paths + antithetic  |  as-of {AS_OF_DATE.date()}',
    fontsize=11,
)

# ── Left: mean contribution per component ─────────────────────────────────────
_labels = [BD_LABELS[k] for k in _keys_ord]
_vals   = [_bd_means[k] for k in _keys_ord]
_cols   = [BD_COLOURS[k] for k in _keys_ord]

bars = ax_bar.bar(_labels, _vals, color=_cols, alpha=0.85, zorder=2)
ax_bar.axhline(0, color='#555555', linewidth=0.7)
ax_bar.bar_label(bars, labels=[f'£{v:.2f}m' for v in _vals], padding=4, fontsize=9)
ax_bar.axhline(total_check, color='#f28e2b', linewidth=1.5, linestyle='--',
               label=f'Total £{total_check:.2f}m')

ax_bar.axvline(2.5, color='#bbbbbb', linewidth=0.8, linestyle=':')
ax_bar.text(1.0, ax_bar.get_ylim()[1] if ax_bar.get_ylim()[1] > 0 else 0.5,
            'Price moves', ha='center', fontsize=8, color='#555555')
ax_bar.text(4.0, ax_bar.get_ylim()[1] if ax_bar.get_ylim()[1] > 0 else 0.5,
            'Ancillary / costs', ha='center', fontsize=8, color='#555555')

ax_bar.legend(fontsize=9)
ax_bar.set_ylabel('GBP million/year (annualised)')
ax_bar.set_title('Mean contribution per component', fontsize=10)
ax_bar.grid(axis='y', alpha=0.3, zorder=0)
ax_bar.tick_params(axis='x', labelsize=8, rotation=15)

# ── Right: box plot of distribution across paths ──────────────────────────────
_box_data = [_sign[k] * cf_bd_refined[k] / 1e6 for k in _keys_ord]

bp = ax_box.boxplot(
    _box_data, labels=_labels,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=1.5),
    whiskerprops=dict(linewidth=1.0),
    flierprops=dict(marker='.', markersize=3, alpha=0.4),
)
for patch, k in zip(bp['boxes'], _keys_ord):
    patch.set_facecolor(BD_COLOURS[k])
    patch.set_alpha(0.7)
ax_box.axhline(0, color='#555555', linewidth=0.7)
ax_box.axvline(2.5, color='#bbbbbb', linewidth=0.8, linestyle=':')
ax_box.set_ylabel('GBP million/year (annualised)')
ax_box.set_title('Distribution across paths (scenario spread)', fontsize=10)
ax_box.grid(axis='y', alpha=0.3)
ax_box.tick_params(axis='x', labelsize=8, rotation=15)

fig.tight_layout()
out_attr_png = PROCESSED / f'lsmc_attribution_{DURATION_LABEL}.png'
fig.savefig(out_attr_png, dpi=140, bbox_inches='tight')
print(f'Saved: {out_attr_png}')
plt.show()

# ── Attribution table + JSON export ──────────────────────────────────────────
_rows = []
for k in _keys_ord:
    arr = _sign[k] * cf_bd_refined[k] / 1e6
    _rows.append({
        'key':            k,
        'component':      BD_LABELS[k],
        'duration_h':     VALUATION_DURATION_H,
        'mean_m':         float(np.mean(arr)),
        'std_m':          float(np.std(arr, ddof=1)),
        'p25_m':          float(np.percentile(arr, 25)),
        'p75_m':          float(np.percentile(arr, 75)),
        'pct_of_gross':   100 * float(np.mean(arr)) / (total_check + _bd_means['costs']) if total_check else 0.0,
    })

out_attr_json = PROCESSED / f'lsmc_attribution_{DURATION_LABEL}.json'
with open(out_attr_json, 'w') as _fh:
    json.dump(_rows, _fh, indent=2)
print(f'Saved: {out_attr_json}')

pd.DataFrame(_rows).drop(columns=['key', 'duration_h']).round(3)

