"""
LSMC backward induction — Phase 4.

Implements the Longstaff-Schwartz Monte Carlo (LSMC) algorithm adapted for
BESS dispatch optimisation, following Boogert & de Jong (2008) gas storage.

Algorithm
---------
BACKWARD PASS (t = T-1 downto 0):
  For each SoC grid node E_j and SoH node k:
    1. Compute Q(n, m) = CF(n, m) + discount × Ê[V_{t+1}(E'_m, S_{t+1})]
       for all paths n and feasible dispatch modes m.
    2. Target:  Y_n = max_m Q(n, m)
    3. Regress: Y_n ~ β^T ψ(S_n(t), E_j)  →  β[t, j, k]
    4. Continuation approx: V_t(E_j, SoH_k, S_n) = β[t,j,k]^T ψ(S_n(t), E_j)

FORWARD PASS:
  Track each path with its own (E_n, SoH_n) state.
  At each step, interpolate continuation value from stored β.
  Choose mode maximising Q.  Accumulate discounted cashflows.

Basis functions (CLAUDE.md spec — 14 features):
  [1, P_da, P_da^2, P_da^3, P_id-P_da, delta_imb, pi_dc, pi_qr,
   E, E^2, E×P_da, sin(2π t/24), cos(2π t/24), EFA_block]

References
----------
Boogert & de Jong (2008) J. Derivatives 15(3)
Nadarajah, Margot & Secomandi (2017) EJOR 256 — regress-later tighter bounds
Longstaff & Schwartz (2001) Rev. Fin. Studies 14(1)
"""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.optimisation.dispatch import (
    DispatchMode, DEFAULT_MODES, cashflow_batch, cashflow_batch_components,
    next_soc_grid, feasibility_mask,
)
from src.processes.simulate import PathBundle
from src.validation import (
    validate_asset_config,
    validate_path_bundle,
    validate_policy,
    validate_valuation_result,
)


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

BASIS_NAMES = [
    "const",
    "P_da", "P_da_sq", "P_da_cu",
    "P_id_spread",
    "delta_imb",
    "pi_dc", "pi_qr",
    "E", "E_sq", "E_x_Pda",
    "P_da_x_delta",      # price × imbalance interaction
    "P_da_x_dc",         # price × ancillary interaction
    "dc_x_qr",           # joint ancillary signal
    "hpfc_fwd_spread",   # mean(HPFC[t+1:t+W]) - HPFC[t]: forward carry signal
]
N_BASIS = len(BASIS_NAMES)   # 15


def basis_matrix(
    P_da:           np.ndarray,   # (N,) £/MWh
    P_id_spr:       np.ndarray,   # (N,) intraday premium (£/MWh)
    delta:          np.ndarray,   # (N,) imbalance basis
    pi_dc:          np.ndarray,   # (N,) DC clearing
    pi_qr:          np.ndarray,   # (N,) QR clearing
    E:              float,        # scalar SoC (MWh) — same for all paths at one grid node
    t_hh:           int,          # half-hour of day (0..47)
    efa_block:      int,          # EFA block (0..5)
    hpfc_fwd_spr:   float = 0.0,  # mean(HPFC[t+1:t+W]) - HPFC[t], £/MWh
) -> np.ndarray:
    """
    Build (N_paths, 15) basis matrix for one SoC grid node.

    Inputs are clipped and scaled here so the polynomial regression remains
    numerically stable. Without this, P_da^3 and E^2 can dominate the normal
    equations and create explosive continuation values.

    Feature 14 (hpfc_fwd_spread) gives the regression a deterministic
    forward-carry signal: when future HPFC prices are higher than current,
    the policy should prefer to hold charge rather than discharge now.
    """
    N  = len(P_da)

    P = np.clip(P_da, -100.0, 500.0).astype(np.float32) / 100.0
    P_id = np.clip(P_id_spr, -200.0, 200.0).astype(np.float32) / 100.0
    dlt = np.clip(delta, -500.0, 500.0).astype(np.float32) / 100.0
    dc = np.clip(pi_dc, 0.0, 100.0).astype(np.float32) / 20.0
    qr = np.clip(pi_qr, 0.0, 100.0).astype(np.float32) / 20.0
    E_scaled = np.float32(E / 100.0)
    fwd = np.float32(np.clip(hpfc_fwd_spr, -100.0, 100.0) / 100.0)

    Phi = np.empty((N, N_BASIS), dtype=np.float32)
    Phi[:, 0]  = 1.0
    Phi[:, 1]  = P
    Phi[:, 2]  = P ** 2
    Phi[:, 3]  = P ** 3
    Phi[:, 4]  = P_id
    Phi[:, 5]  = dlt
    Phi[:, 6]  = dc
    Phi[:, 7]  = qr
    Phi[:, 8]  = E_scaled
    Phi[:, 9]  = E_scaled ** 2
    Phi[:, 10] = E_scaled * P
    Phi[:, 11] = P * dlt    # P_da × delta_imb — price×imbalance interaction
    Phi[:, 12] = P * dc     # P_da × pi_dc     — price×ancillary interaction
    Phi[:, 13] = dc * qr    # pi_dc × pi_qr    — joint ancillary signal
    Phi[:, 14] = fwd        # HPFC forward carry — same for all paths at time t

    return Phi


# ---------------------------------------------------------------------------
# LSMC output containers
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    """
    Regression coefficients from the backward pass.

    beta[t, j, k] is the (N_BASIS,) coefficient vector for time step t,
    SoC grid index j, and SoH node index k.  Used for V_curr propagation
    and (legacy) anticipative forward evaluation.

    cont_beta[t, j, k, m] is the (N_BASIS,) coefficient vector that
    approximates E[V(t+1, E_next_m) | S(t), SoC ≈ grid[j], SoH = k]
    (undiscounted next-step value, symmetric with beta).
    When present, forward() uses this for a non-anticipative action choice
    (dispatch at t sees only S(t), not S(t+1)).

    Shapes:
        beta      : (n_steps, n_soc_nodes, n_soh_nodes, N_BASIS)
        cont_beta : (n_steps, n_soc_nodes, n_soh_nodes, n_modes, N_BASIS) or None
        soc_grid  : (n_soc_nodes,) — MWh
        soh_nodes : (n_soh_nodes,) — fraction
        dt_h      : float — half-hour step in hours
    """
    beta:       np.ndarray             # (T, J, K, 14)
    soc_grid:   np.ndarray             # (J,)
    soh_nodes:  np.ndarray             # (K,)
    modes:      List[DispatchMode]
    dt_h:       float
    n_steps:    int
    n_paths:    int
    cont_beta:  Optional[np.ndarray] = None   # (T, J, K, M, 14); None → legacy
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValuationResult:
    """Output of the LSMC forward simulation."""
    pv_paths:       np.ndarray    # (N_paths,) — discounted PV per path
    cashflow_paths: np.ndarray    # (N_paths, T) — period cashflows
    soc_paths:      np.ndarray    # (N_paths, T+1) — SoC trajectory
    soh_paths:      np.ndarray    # (N_paths, T+1) — SoH trajectory
    action_paths:   np.ndarray    # (N_paths, T) — mode index chosen
    mtm_mean:       float
    mtm_std:        float
    mtm_p5:         float
    mtm_p95:        float
    efc_total:      float         # equivalent full cycles consumed
    action_diagnostics: Dict[str, object] = field(default_factory=dict)
    # Per-source discounted PV vectors (N_paths,) for revenue attribution.
    # Keys: 'da', 'imbalance', 'dc', 'qr', 'costs'.
    cf_breakdown: Optional[Dict[str, np.ndarray]] = None


# ---------------------------------------------------------------------------
# LSMC Solver
# ---------------------------------------------------------------------------

class LSMCSolver:
    """
    LSMC solver for BESS dispatch optimisation.

    Parameters
    ----------
    asset_cfg : dict  — from src.config.ASSET
    lsmc_cfg  : dict  — from src.config.LSMC
    deg_cfg   : dict  — from src.config.DEGRADATION
    fin_cfg   : dict  — from src.config.FINANCE
    modes     : list of DispatchMode (defaults to DEFAULT_MODES)
    verbose   : bool
    """

    def __init__(
        self,
        asset_cfg:  dict,
        lsmc_cfg:   dict,
        deg_cfg:    dict,
        fin_cfg:    dict,
        modes:      Optional[List[DispatchMode]] = None,
        verbose:    bool = True,
        hpfc_params = None,
        hpfc_curve: Optional[np.ndarray] = None,
    ) -> None:
        self.asset   = asset_cfg
        self.lsmc    = lsmc_cfg
        self.deg     = deg_cfg
        self.fin     = fin_cfg
        self.modes   = modes if modes is not None else DEFAULT_MODES
        self.verbose = verbose
        self.hpfc_params = hpfc_params
        # Half-hourly HPFC prices used to compute the forward-carry basis feature.
        self._hpfc_curve = (
            np.asarray(hpfc_curve, dtype=np.float32) if hpfc_curve is not None else None
        )
        self._fwd_window_hh = int(lsmc_cfg.get('fwd_window_hh', 16))

        # Derived constants
        self.P_bar    = float(asset_cfg['power_mw'])
        self.E_name   = float(asset_cfg['energy_mwh'])
        self.eta_c    = float(asset_cfg['eta_charge'])
        self.eta_d    = float(asset_cfg['eta_discharge'])
        self.E_min_fr = float(asset_cfg['soc_min_frac'])
        self.E_max_fr = float(asset_cfg['soc_max_frac'])
        self.dt_h     = float(lsmc_cfg.get('dt_hours', 0.5))

        # Grids
        n_soc = int(lsmc_cfg.get('n_soc_nodes', 21))
        self.soc_grid = np.linspace(
            self.E_min_fr * self.E_name,
            self.E_max_fr * self.E_name,
            n_soc, dtype=np.float32,
        )
        self.soh_nodes = np.array(
            lsmc_cfg.get('soh_nodes', [1.00, 0.95, 0.90, 0.85, 0.82]),
            dtype=np.float32,
        )
        self.n_soc = n_soc
        self.n_soh = len(self.soh_nodes)

        # Discount factor per half-hour
        r = float(fin_cfg.get('wacc_merchant', 0.09))
        self.disc = float(np.exp(-r * self.dt_h / 8760))

        # Degradation shadow price
        self.deg_cost = float(deg_cfg.get('lambda_deg_init_gbp_mwh', 6.0))
        self.vom      = float(asset_cfg.get('vom_gbp_mwh', 1.2))
        self.ridge_alpha = float(lsmc_cfg.get('ridge_alpha', 1e-3))
        self.continuation_cap = float(
            lsmc_cfg.get(
                'continuation_value_cap_gbp',
                max(10_000_000.0, self.P_bar * 250_000.0),
            )
        )

        # Pre-stack mode fractions as arrays (M,) for vectorised dispatch
        self._net_fracs = np.array([m.net_frac  for m in self.modes], np.float32)
        self._dc_fracs  = np.array([m.r_dc_frac for m in self.modes], np.float32)
        self._qr_fracs  = np.array([m.r_qr_frac for m in self.modes], np.float32)

        # EFA block lookup (0..5) given half-hour index 0..47
        self._efa = np.array([hh // 8 for hh in range(48)], dtype=np.int32)

        # Pre-compute next-SoC table for all (j, k, m) combinations.
        # Shape (n_soc, n_soh, M).  Used in forward() without a Python loop.
        M = len(self.modes)
        E_next_jkm = np.zeros((self.n_soc, self.n_soh, M), dtype=np.float32)
        for j, E_j in enumerate(self.soc_grid):
            for k_idx, soh_k in enumerate(self.soh_nodes):
                E_min_k = self.E_min_fr * self.E_name
                E_max_k = self.E_max_fr * self.E_name * float(soh_k)
                dE = np.where(
                    self._net_fracs > 0,
                    -self._net_fracs * self.P_bar / self.eta_d * self.dt_h,
                    -self._net_fracs * self.P_bar * self.eta_c * self.dt_h,
                )
                E_next_jkm[j, k_idx] = np.clip(
                    float(E_j) + dE, E_min_k, E_max_k,
                ).astype(np.float32)
        self._E_next_jkm = E_next_jkm   # (n_soc, n_soh, M)

        # Full feasibility also includes ancillary reserve sustain requirements.
        # The next-SoC table only depends on net charge/discharge, so reserve-only
        # modes need a separate mask to avoid earning infeasible DC/QR revenues.
        feasible_jkm = np.zeros((self.n_soc, self.n_soh, M), dtype=bool)
        for j, E_j in enumerate(self.soc_grid):
            for k_idx, soh_k in enumerate(self.soh_nodes):
                feasible_jkm[j, k_idx] = feasibility_mask(
                    self.modes,
                    E_curr=float(E_j),
                    SoH=float(soh_k),
                    P_bar_mw=self.P_bar,
                    E_min_frac=self.E_min_fr,
                    E_max_frac=self.E_max_fr,
                    energy_mwh=self.E_name,
                    eta_charge=self.eta_c,
                    eta_discharge=self.eta_d,
                    dt_h=self.dt_h,
                )
        self._feasible_jkm = feasible_jkm

    def _hpfc_forward_spread(self, T: int) -> np.ndarray:
        """
        Compute the HPFC forward-carry signal for each of the T timesteps.

        Returns spread[t] = max(HPFC[t+1 : t+1+W]) - HPFC[t]  in £/MWh,
        where W = self._fwd_window_hh.  Using the maximum rather than the mean
        directly encodes the best available sell price in the look-ahead window,
        which is what drives the hold-vs-dispatch decision.

        When no HPFC curve is provided, returns zeros so the feature is
        inactive without breaking the regression.
        """
        if self._hpfc_curve is None:
            return np.zeros(T, dtype=np.float32)
        W = self._fwd_window_hh
        needed = T + W + 1
        curve = self._hpfc_curve
        if len(curve) < needed:
            curve = np.pad(curve, (0, needed - len(curve)), mode='edge')
        curve = curve.astype(np.float32)
        # Rolling max via stride_tricks — no Python loop over T
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(curve[1:], window_shape=W)   # (T+..., W)
        fwd_max = windows[:T].max(axis=1)                          # (T,)
        return (fwd_max - curve[:T]).astype(np.float32)

    def _intraday_spread_matrix(self, bundle: PathBundle, P_da_all: np.ndarray) -> np.ndarray:
        """
        Build a peak-minus-trough intraday spread proxy for the LSMC basis.

        If calibrated HPFC parameters are not supplied, fall back to the default
        config loadings. This keeps the basis feature active while preserving
        backward-compatible caller signatures.
        """
        hpfc = self.hpfc_params
        if hpfc is None:
            try:
                from src.processes.simulate import default_params_from_config
                hpfc = default_params_from_config()[1]
            except Exception:
                return np.zeros_like(P_da_all, dtype=np.float32)

        loadings = np.asarray(getattr(hpfc, "loadings", []), dtype=np.float32)
        if loadings.ndim != 2 or bundle.lam.shape[2] < loadings.shape[0]:
            return np.zeros_like(P_da_all, dtype=np.float32)

        peak_hh = int(self.lsmc.get("intraday_peak_hh", 34))
        trough_hh = int(self.lsmc.get("intraday_trough_hh", 14))
        peak_hh = int(np.clip(peak_hh, 0, loadings.shape[1] - 1))
        trough_hh = int(np.clip(trough_hh, 0, loadings.shape[1] - 1))

        k = min(bundle.lam.shape[2], loadings.shape[0])
        loading_diff = loadings[:k, peak_hh] - loadings[:k, trough_hh]
        log_spread = np.tensordot(bundle.lam[:, :, :k], loading_diff, axes=([2], [0]))
        spread = P_da_all * np.expm1(np.clip(log_spread, -2.0, 2.0))
        return np.clip(spread, -200.0, 200.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Backward induction
    # ------------------------------------------------------------------

    def backward(
        self,
        bundle: PathBundle,
        E_init_frac: float = 0.5,
    ) -> Policy:
        """
        Run the backward induction over all time steps.

        Parameters
        ----------
        bundle       : PathBundle from Phase 3 simulation
        E_init_frac  : initial SoC fraction for all paths

        Returns
        -------
        Policy with regression coefficients β[t, j, k, 14]
        """
        T  = bundle.n_steps
        N  = bundle.n_paths
        J  = self.n_soc
        K  = self.n_soh
        M  = len(self.modes)
        dt = self.dt_h

        _bwd_t0 = time.time()
        _print_stride = max(1, T // 10)   # print ~10 progress lines

        if self.verbose:
            print(f"LSMC backward: T={T} steps, N={N} paths, "
                  f"J={J} SoC nodes, K={K} SoH nodes, M={M} modes", flush=True)

        # Coefficient store
        beta      = np.zeros((T, J, K,    N_BASIS), dtype=np.float32)
        cont_beta = np.zeros((T, J, K, M, N_BASIS), dtype=np.float32)
        diagnostics = {
            "regression_count": 0,
            "nonfinite_beta_count": 0,
            "beta_abs_max": 0.0,
            "target_std_min": float("inf"),
            "target_std_max": 0.0,
            "sampled_regression_count": 0,
            "sample_rank_deficient_count": 0,
            "sample_condition_max": 0.0,
            "active_feature_count_min": float("inf"),
            "active_feature_count_max": 0,
            "fallback_lstsq_count": 0,
            "fallback_zero_count": 0,
            "continuation_clip_fraction_max": 0.0,
            "continuation_clip_observation_count": 0,
            "continuation_clip_observation_total": 0,
            "continuation_clip_regression_count": 0,
            "continuation_value_cap_gbp": float(self.continuation_cap),
            "intraday_spread_std": 0.0,
        }

        # Continuation value store: V[j, k, n] = Ê[V_{t+1}(E_j, SoH_k, S_n)]
        # Updated at each backward step and then used at t-1.
        V_next = np.zeros((J, K, N), dtype=np.float32)

        # Pre-extract and cap market arrays. These are valuation stabilisers for
        # dev notebooks; production calibration should reduce the need for caps.
        P_da_all   = np.exp(np.clip(bundle.ln_P_base, -100.0, np.log(500.0))).astype(np.float32)
        delta_all  = np.clip(bundle.delta_imb, -500.0, 500.0).astype(np.float32)
        pi_dc_all  = np.clip(bundle.pi['DC_Low'], 0.0, 100.0).astype(np.float32)
        pi_qr_all  = np.clip(bundle.pi.get('QR_Pos', bundle.pi['DC_Low']), 0.0, 100.0).astype(np.float32)

        # Intraday spread proxy: use lambda_1 loading on peak vs trough HH
        # If HPFC params aren't passed, approximate as zero
        P_id_spr_all = self._intraday_spread_matrix(bundle, P_da_all)
        diagnostics["intraday_spread_std"] = float(np.std(P_id_spr_all[:, :T]))

        # HPFC forward-carry: deterministic signal, shape (T,)
        hpfc_fwd_spread = self._hpfc_forward_spread(T)

        # Pre-compute next-SoC table: E_next[j, m] — (J, M)
        # For each SoC node and each mode, deterministic E' (before SoH scaling)
        E_next_table = np.zeros((J, M), dtype=np.float32)
        for j, E_j in enumerate(self.soc_grid):
            E_next_table[j] = next_soc_grid(
                E_j, self._net_fracs,
                self.P_bar, self.eta_c, self.eta_d,
                dt, self.soc_grid[0], self.soc_grid[-1],
            )

        # Ridge regularisation coefficient. 1e-4 is large enough to prevent
        # float32 normal-equation rounding from producing explosive coefficients
        # (seen as ±1e27 betas), yet small relative to typical PhiT_Phi diagonals
        # of ~N*feature_scale² ≈ 250, so regression accuracy is barely affected.
        reg = self.ridge_alpha
        diag_stride = int(self.lsmc.get("diagnostics_stride", 500))
        diag_stride = max(1, diag_stride)

        # Backward loop
        for t in range(T - 1, -1, -1):
            if self.verbose and (t == T - 1 or (T - 1 - t) % _print_stride == 0):
                pct = 100 * (T - t) / T
                elapsed = time.time() - _bwd_t0
                print(f"  bwd {pct:5.1f}%  t={t:5d}  {elapsed:6.1f}s elapsed", flush=True)

            # Market state at step t
            P_da  = P_da_all[:, t]       # (N,)
            delta = delta_all[:, t]      # (N,)
            pi_dc = pi_dc_all[:, t]      # (N,)
            pi_qr = pi_qr_all[:, t]      # (N,)
            P_id  = P_id_spr_all[:, t]   # (N,)

            t_hh      = t % 48
            efa_block = int(self._efa[t_hh])
            V_curr = np.zeros_like(V_next)

            # Cashflow matrix — same for all SoC nodes (CF doesn't depend on E)
            # CF: (N, M)
            CF = cashflow_batch(
                self.modes, P_da, delta, pi_dc, pi_qr,
                self.P_bar, dt, self.deg_cost, self.vom,
            )

            # Loop over SoH nodes and SoC nodes
            for k_idx, SoH_k in enumerate(self.soh_nodes):
                E_min_k = self.E_min_fr * self.E_name
                E_max_k = self.E_max_fr * self.E_name * SoH_k

                for j, E_j in enumerate(self.soc_grid):
                    # SoC node may be infeasible for this SoH
                    if E_j > E_max_k + 1e-3:
                        beta[t, j, k_idx, :] = 0.0
                        continue

                    # E_next for each mode at this SoC node (re-clip to SoH-adjusted max)
                    E_next_jm = np.clip(
                        E_next_table[j],
                        E_min_k, E_max_k,
                    )   # (M,)

                    # Interpolate V_next to get continuation at E_next_jm
                    # V_next[j', k, n] is on the soc_grid
                    # Use linear interpolation between grid nodes
                    V_cont = self._interp_V(V_next, E_next_jm, k_idx)   # (N, M)

                    # Q values: (N, M)
                    Q = CF + self.disc * V_cont   # (N, M)

                    Q[:, ~self._feasible_jkm[j, k_idx]] = -1e9

                    # Max Q over modes: (N,)
                    Y = Q.max(axis=1)

                    # Regression
                    Phi = basis_matrix(P_da, P_id, delta, pi_dc, pi_qr,
                                       E_j, t_hh, efa_block,
                                       hpfc_fwd_spr=float(hpfc_fwd_spread[t]))   # (N, 15)

                    # OLS with feature standardisation and ridge regularisation.
                    # Work in float64 to avoid float32 rounding in the normal
                    # equations producing explosive coefficients.
                    Phi64 = Phi.astype(np.float64)
                    Y64 = Y.astype(np.float64)
                    y_std = float(np.std(Y64))
                    # Normalise target to O(1) before the ridge solve.  This
                    # keeps the normal-equation RHS at float64-safe magnitudes
                    # and ensures the ridge penalty is applied on a consistent
                    # scale regardless of how far back in time we are.  The
                    # betas are un-normalised after solving so V_curr is in £.
                    y_scale = max(y_std, 1.0)
                    Y_norm = Y64 / y_scale
                    diagnostics["regression_count"] += 1
                    diagnostics["target_std_min"] = min(diagnostics["target_std_min"], y_std)
                    diagnostics["target_std_max"] = max(diagnostics["target_std_max"], y_std)
                    mu = Phi64.mean(axis=0)
                    raw_sd = Phi64.std(axis=0)
                    # Drop constant columns (E and any cross-product features
                    # that happen to be zero-variance at this node/step).
                    active = np.ones(N_BASIS, dtype=bool)
                    active[1:] = raw_sd[1:] > 1e-8
                    active_count = int(np.sum(active))
                    diagnostics["active_feature_count_min"] = min(
                        diagnostics["active_feature_count_min"], active_count
                    )
                    diagnostics["active_feature_count_max"] = max(
                        diagnostics["active_feature_count_max"], active_count
                    )
                    sd = raw_sd.copy()
                    mu[0] = 0.0
                    sd[0] = 1.0
                    sd = np.where(sd > 1e-8, sd, 1.0)
                    Z = (Phi64[:, active] - mu[active]) / sd[active]

                    ridge = reg * np.eye(active_count)
                    ridge[0, 0] = 0.0
                    PhiT_Phi = Z.T @ Z + ridge
                    PhiT_Y = Z.T @ Y_norm
                    do_sample_diag = (diagnostics["regression_count"] % diag_stride) == 0
                    if do_sample_diag:
                        diagnostics["sampled_regression_count"] += 1
                        rank = int(np.linalg.matrix_rank(Z))
                        if rank < active_count:
                            diagnostics["sample_rank_deficient_count"] += 1
                        cond = float(np.linalg.cond(PhiT_Phi))
                        if np.isfinite(cond):
                            diagnostics["sample_condition_max"] = max(
                                diagnostics["sample_condition_max"], cond
                            )
                    try:
                        gamma_active = np.linalg.solve(PhiT_Phi, PhiT_Y)
                        if not np.all(np.isfinite(gamma_active)):
                            diagnostics["fallback_lstsq_count"] += 1
                            gamma_active, _, _, _ = np.linalg.lstsq(PhiT_Phi, PhiT_Y, rcond=None)
                    except np.linalg.LinAlgError:
                        diagnostics["fallback_zero_count"] += 1
                        gamma_active = np.zeros(active_count)

                    gamma = np.zeros(N_BASIS, dtype=np.float64)
                    gamma[active] = gamma_active
                    b = np.zeros(N_BASIS, dtype=np.float64)
                    # Undo feature standardisation then undo target normalisation
                    b[active] = gamma_active * y_scale / sd[active]
                    b[0] = (gamma[0] - np.sum(gamma[1:] * mu[1:] / sd[1:])) * y_scale
                    if not np.all(np.isfinite(b)):
                        diagnostics["nonfinite_beta_count"] += 1
                    beta_abs_max = float(np.nanmax(np.abs(b))) if b.size else 0.0
                    if np.isfinite(beta_abs_max):
                        diagnostics["beta_abs_max"] = max(diagnostics["beta_abs_max"], beta_abs_max)
                    beta[t, j, k_idx, :] = np.where(
                        np.isfinite(b), b, 0.0
                    ).astype(np.float32)

                    # Per-mode continuation regression on current-state features.
                    # Regress undiscounted V(t+1, E_next_m) on phi(S(t)) for every
                    # mode m simultaneously.  forward() then applies disc once via
                    # Q = CF + disc * cont, avoiding the double-discount that arose
                    # when this target was disc*V_cont (B1 fix).
                    V_disc64 = V_cont.astype(np.float64)                 # (N, M)
                    v_scales = np.maximum(V_disc64.std(axis=0), 1.0)     # (M,)
                    V_disc_norm = V_disc64 / v_scales[None, :]            # (N, M)
                    PhiT_V = Z.T @ V_disc_norm                            # (active, M)
                    try:
                        gamma_cont_a = np.linalg.solve(PhiT_Phi, PhiT_V)
                        if not np.all(np.isfinite(gamma_cont_a)):
                            gamma_cont_a = np.zeros((active_count, M))
                    except np.linalg.LinAlgError:
                        gamma_cont_a = np.zeros((active_count, M))
                    gamma_cont = np.zeros((N_BASIS, M), dtype=np.float64)
                    gamma_cont[active, :] = gamma_cont_a
                    b_cont = np.zeros((N_BASIS, M), dtype=np.float64)
                    b_cont[active, :] = (
                        gamma_cont_a * v_scales[None, :] / sd[active, None]
                    )
                    b_cont[0, :] = (
                        gamma_cont[0, :]
                        - np.sum(gamma_cont[1:, :] * (mu[1:] / sd[1:])[:, None], axis=0)
                    ) * v_scales
                    b_cont = np.where(np.isfinite(b_cont), b_cont, 0.0)
                    cont_beta[t, j, k_idx, :, :] = b_cont.T.astype(np.float32)

                    # Update V_next at this node for the NEXT backward step (t-1)
                    raw_cont = Phi @ b
                    clip_mask = (
                        (raw_cont < -self.continuation_cap)
                        | (raw_cont > self.continuation_cap)
                    )
                    clip_count = int(np.sum(clip_mask))
                    clip_frac = float(clip_count / max(len(raw_cont), 1))
                    diagnostics["continuation_clip_observation_count"] += clip_count
                    diagnostics["continuation_clip_observation_total"] += int(len(raw_cont))
                    if clip_count:
                        diagnostics["continuation_clip_regression_count"] += 1
                    diagnostics["continuation_clip_fraction_max"] = max(
                        diagnostics["continuation_clip_fraction_max"], clip_frac
                    )
                    V_curr[j, k_idx, :] = np.clip(
                        raw_cont,
                        -self.continuation_cap,
                        self.continuation_cap,
                    ).astype(np.float32)

            V_next = V_curr

        if self.verbose:
            print(f"\n  Backward pass complete. beta shape: {beta.shape}")
            print(
                "  LSMC diagnostics: "
                f"beta_abs_max={diagnostics['beta_abs_max']:.3g}, "
                f"sample_cond_max={diagnostics['sample_condition_max']:.3g}, "
                f"rank_def={int(diagnostics['sample_rank_deficient_count'])}/"
                f"{int(diagnostics['sampled_regression_count'])}"
            )

        if diagnostics["target_std_min"] == float("inf"):
            diagnostics["target_std_min"] = 0.0
        if diagnostics["active_feature_count_min"] == float("inf"):
            diagnostics["active_feature_count_min"] = 0
        clip_total = int(diagnostics.get("continuation_clip_observation_total", 0))
        clip_count = int(diagnostics.get("continuation_clip_observation_count", 0))
        reg_count = int(diagnostics.get("regression_count", 0))
        clip_reg_count = int(diagnostics.get("continuation_clip_regression_count", 0))
        diagnostics["continuation_clip_observation_fraction"] = (
            float(clip_count / clip_total) if clip_total else 0.0
        )
        diagnostics["continuation_clip_regression_fraction"] = (
            float(clip_reg_count / reg_count) if reg_count else 0.0
        )

        return Policy(
            beta      = beta,
            cont_beta = cont_beta,
            soc_grid  = self.soc_grid,
            soh_nodes = self.soh_nodes,
            modes     = self.modes,
            dt_h      = self.dt_h,
            n_steps   = T,
            n_paths   = N,
            diagnostics = diagnostics,
        )

    def _interp_V(
        self,
        V_next:    np.ndarray,   # (J, K, N)
        E_next_jm: np.ndarray,   # (M,)
        k_idx:     int,
    ) -> np.ndarray:
        """
        Linearly interpolate V_next[j, k, :] to E values in E_next_jm.

        Returns (N, M) array.
        """
        grid = self.soc_grid   # (J,) float32
        J    = len(grid)

        # Vectorised over all M modes at once
        j_lo = np.clip(
            np.searchsorted(grid, E_next_jm, side='right') - 1,
            0, J - 2,
        )   # (M,)
        j_hi = j_lo + 1   # (M,)

        denom = grid[j_hi] - grid[j_lo]   # (M,) float32
        alpha = np.where(
            denom > 1e-9,
            (E_next_jm - grid[j_lo]) / np.where(denom > 1e-9, denom, 1.0),
            0.0,
        )
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)   # (M,)

        # V_next[j_lo, k_idx, :] → (M, N); transpose to (N, M)
        V_interp = (
            (1.0 - alpha[:, None]) * V_next[j_lo, k_idx, :]
            + alpha[:, None]       * V_next[j_hi, k_idx, :]
        ).T.astype(np.float32)   # (N, M)

        return V_interp

    # ------------------------------------------------------------------
    # Forward simulation
    # ------------------------------------------------------------------

    def forward(
        self,
        bundle:  PathBundle,
        policy:  Policy,
        E_init_frac:  float = 0.5,
        SoH_init:     float = 1.0,
        annual_cycles: float = 520.0,
    ) -> ValuationResult:
        """
        Forward simulation: apply the learned policy to collect cashflows.

        Parameters
        ----------
        bundle        : PathBundle (can be same as backward or fresh paths)
        policy        : Policy from backward()
        E_init_frac   : starting SoC as fraction of nameplate
        SoH_init      : starting state of health (1.0 = fresh)
        annual_cycles : EFC budget / year for SoH fade (from config)

        Returns
        -------
        ValuationResult
        """
        T   = bundle.n_steps
        N   = bundle.n_paths
        dt  = self.dt_h

        # Initialise state
        E_n   = np.full(N, E_init_frac * self.E_name, dtype=np.float32)
        SoH_n = np.full(N, SoH_init, dtype=np.float32)

        # Storage
        cf_store  = np.zeros((N, T), dtype=np.float32)
        soc_store = np.zeros((N, T + 1), dtype=np.float32)
        soh_store = np.zeros((N, T + 1), dtype=np.float32)
        act_store = np.zeros((N, T), dtype=np.int16)
        soc_store[:, 0] = E_n
        soh_store[:, 0] = SoH_n

        # Per-source discounted PV accumulators (N,) updated each step.
        pv_da_paths    = np.zeros(N, dtype=np.float64)
        pv_imb_paths   = np.zeros(N, dtype=np.float64)
        pv_dc_paths    = np.zeros(N, dtype=np.float64)
        pv_qr_paths    = np.zeros(N, dtype=np.float64)
        pv_costs_paths = np.zeros(N, dtype=np.float64)

        # Pre-extract and cap prices consistently with backward().
        P_da_all  = np.exp(np.clip(bundle.ln_P_base, -100.0, np.log(500.0))).astype(np.float32)
        delta_all = np.clip(bundle.delta_imb, -500.0, 500.0).astype(np.float32)
        pi_dc_all = np.clip(bundle.pi['DC_Low'], 0.0, 100.0).astype(np.float32)
        pi_qr_all = np.clip(bundle.pi.get('QR_Pos', bundle.pi['DC_Low']), 0.0, 100.0).astype(np.float32)
        P_id_spr_all = self._intraday_spread_matrix(bundle, P_da_all)

        # HPFC forward-carry signal, shape (T,)
        hpfc_fwd_spread = self._hpfc_forward_spread(T)

        # SoH fade rate per HH step
        # Simplified: linear SoH degradation at cycle_rate EFCs/year
        # dSoH/dHH = (0.18 total SoH fade / 520 EFC) × (avg DoD) / (HH per cycle)
        # Here: rough approx — use 0 for forward sim SoH (track separately)
        n_hh_per_cycle = self.E_name / (self.P_bar * dt)   # ~400 HH per full cycle
        soh_per_efc    = 0.18 / max(annual_cycles, 1.0)    # SoH lost per EFC

        disc_factors = np.float64(self.disc) ** np.arange(T, dtype=np.float64)

        net_fracs = self._net_fracs   # (M,)
        M = len(self.modes)
        action_counts = np.zeros(M, dtype=np.int64)
        action_cf_sum = np.zeros(M, dtype=np.float64)
        action_cont_sum = np.zeros(M, dtype=np.float64)
        action_q_sum = np.zeros(M, dtype=np.float64)
        q_gap_sum = 0.0
        q_gap_count = 0
        q_gap_small_count = 0
        q_gap_min = float("inf")

        _fwd_t0 = time.time()
        _fwd_print_stride = max(1, T // 10)

        if self.verbose:
            print(f"LSMC forward:  T={T}, N={N}", flush=True)

        for t in range(T):
            if self.verbose and t % _fwd_print_stride == 0:
                pct = 100 * t / T
                elapsed = time.time() - _fwd_t0
                print(f"  fwd {pct:5.1f}%  t={t:5d}  {elapsed:6.1f}s elapsed", flush=True)
            P_da  = P_da_all[:, t]
            delta = delta_all[:, t]
            pi_dc = pi_dc_all[:, t]
            pi_qr = pi_qr_all[:, t]

            t_hh      = t % 48
            efa_block = int(self._efa[t_hh])

            # 1. SoC grid index (floor) for each path
            j_arr = np.clip(
                np.searchsorted(self.soc_grid, E_n, side='right') - 1,
                0, self.n_soc - 2,
            )   # (N,) ∈ {0, …, J-2}

            # 2. SoH grid index (floor) for each path.
            # soh_nodes is decreasing (e.g. [1.00, 0.90, 0.82]), so reverse for searchsorted.
            k_arr = np.clip(
                self.n_soh - np.searchsorted(self.soh_nodes[::-1], SoH_n, side='right'),
                0, self.n_soh - 1,
            )   # (N,) ∈ {0, …, K-1}

            # 3. Cashflow matrix — vectorised over all (N, M)
            CF = cashflow_batch(
                self.modes, P_da, delta, pi_dc, pi_qr,
                self.P_bar, dt, self.deg_cost, self.vom,
            )   # (N, M)

            # 4. Continuation value — fully vectorised, no Python loop over j/k/m.
            #
            # For action t, compare CF_t + discount * V_{t+1}(E_next, S_{t+1}).
            # The terminal step has zero continuation. Earlier versions used
            # beta[t] at the current SoC node, which can hallucinate large
            # same-time continuation values in forward policy evaluation.
            E_next_nm = self._E_next_jkm[j_arr, k_arr, :]   # (N, M)
            if t + 1 >= T:
                cont = np.zeros((N, M), dtype=np.float32)
            elif policy.cont_beta is not None:
                # Non-anticipative: evaluate per-mode continuation regressions
                # fitted on S(t) during backward induction.  No t+1 price data.
                P_c   = np.clip(P_da, -100.0, 500.0).astype(np.float32) / 100.0
                dlt_c = np.clip(delta, -500.0, 500.0).astype(np.float32) / 100.0
                dc_c  = np.clip(pi_dc,  0.0, 100.0).astype(np.float32) / 20.0
                qr_c  = np.clip(pi_qr,  0.0, 100.0).astype(np.float32) / 20.0
                p_id_c = np.clip(P_id_spr_all[:, t], -200.0, 200.0).astype(np.float32) / 100.0
                E_sc  = (E_n / 100.0).astype(np.float32)
                fwd_c = np.float32(
                    np.clip(hpfc_fwd_spread[t], -100.0, 100.0) / 100.0
                )
                phi_t = np.empty((N, N_BASIS), dtype=np.float32)
                phi_t[:, 0]  = 1.0
                phi_t[:, 1]  = P_c
                phi_t[:, 2]  = P_c ** 2
                phi_t[:, 3]  = P_c ** 3
                phi_t[:, 4]  = p_id_c
                phi_t[:, 5]  = dlt_c
                phi_t[:, 6]  = dc_c
                phi_t[:, 7]  = qr_c
                phi_t[:, 8]  = E_sc
                phi_t[:, 9]  = E_sc ** 2
                phi_t[:, 10] = P_c * E_sc
                phi_t[:, 11] = P_c * dlt_c
                phi_t[:, 12] = P_c * dc_c
                phi_t[:, 13] = dc_c * qr_c
                phi_t[:, 14] = fwd_c
                # cont_beta[t, j, k, m, :] · phi(S(t)) → (N, M)
                cb = policy.cont_beta[t, j_arr, k_arr, :, :]   # (N, M, 14)
                cont = np.clip(
                    np.einsum('nmb,nb->nm', cb, phi_t),
                    -self.continuation_cap,
                    self.continuation_cap,
                ).astype(np.float32)
            else:
                # Legacy anticipative fallback for policies saved without cont_beta.
                t_next = t + 1
                P_next = P_da_all[:, t_next]
                P_c_next = np.clip(P_next, -100.0, 500.0).astype(np.float32) / 100.0
                dlt_c_next = np.clip(delta_all[:, t_next], -500.0, 500.0).astype(np.float32) / 100.0
                dc_c_next = np.clip(pi_dc_all[:, t_next], 0.0, 100.0).astype(np.float32) / 20.0
                qr_c_next = np.clip(pi_qr_all[:, t_next], 0.0, 100.0).astype(np.float32) / 20.0
                p_id_c_next = np.clip(P_id_spr_all[:, t_next], -200.0, 200.0).astype(np.float32) / 100.0
                fwd_c_next = np.float32(np.clip(hpfc_fwd_spread[t_next], -100.0, 100.0) / 100.0)
                phi_next_base = np.empty((N, 12), dtype=np.float32)
                phi_next_base[:, 0] = 1.0
                phi_next_base[:, 1] = P_c_next
                phi_next_base[:, 2] = P_c_next ** 2
                phi_next_base[:, 3] = P_c_next ** 3
                phi_next_base[:, 4] = p_id_c_next
                phi_next_base[:, 5] = dlt_c_next
                phi_next_base[:, 6] = dc_c_next
                phi_next_base[:, 7] = qr_c_next
                phi_next_base[:, 8]  = P_c_next * dlt_c_next
                phi_next_base[:, 9]  = P_c_next * dc_c_next
                phi_next_base[:, 10] = dc_c_next * qr_c_next
                phi_next_base[:, 11] = fwd_c_next
                j_next_nm = np.clip(
                    np.searchsorted(self.soc_grid, E_next_nm, side='right') - 1,
                    0,
                    self.n_soc - 1,
                )
                k_next_nm = np.broadcast_to(k_arr[:, None], (N, M))
                b_nm = policy.beta[t_next, j_next_nm, k_next_nm, :]   # (N, M, 15)
                b_base_nm = np.concatenate(
                    [b_nm[:, :, :8], b_nm[:, :, 11:15]], axis=2
                )
                E_sc_nm = (E_next_nm / 100.0).astype(np.float32)
                cont = np.clip(
                    np.sum(phi_next_base[:, None, :] * b_base_nm, axis=2)
                    + b_nm[:, :, 8] * E_sc_nm
                    + b_nm[:, :, 9] * E_sc_nm ** 2
                    + b_nm[:, :, 10] * P_c_next[:, None] * E_sc_nm,
                    -self.continuation_cap,
                    self.continuation_cap,
                ).astype(np.float32)

            infeas = ~self._feasible_jkm[j_arr, k_arr, :]

            Q = CF + np.float32(self.disc) * cont   # (N, M)
            Q[infeas] = np.float32(-1e9)

            # 5. Choose optimal mode per path
            m_star = np.argmax(Q, axis=1)   # (N,)

            # 5. Apply chosen action
            net_chosen = net_fracs[m_star]   # (N,)

            # SoC update
            dE = np.where(
                net_chosen > 0,
                -net_chosen * self.P_bar / self.eta_d * dt,
                -net_chosen * self.P_bar * self.eta_c * dt,
            ).astype(np.float32)

            # SoH-adjusted E_max
            E_max_n = self.E_max_fr * self.E_name * SoH_n
            E_min_n = np.full(N, self.E_min_fr * self.E_name, dtype=np.float32)

            E_n_new = np.clip(E_n + dE, E_min_n, E_max_n).astype(np.float32)

            # SoH degradation: proportional to |net_power| * dt
            throughput_mwh = np.abs(net_chosen) * self.P_bar * dt   # MWh per path
            efc_n = throughput_mwh / (self.E_name * 2.0)            # EFC fraction
            SoH_n = np.clip(SoH_n - efc_n * soh_per_efc, 0.72, 1.0).astype(np.float32)

            # Cashflow at chosen mode
            cf_n = CF[np.arange(N), m_star]   # (N,)
            cont_n = cont[np.arange(N), m_star]
            q_n = Q[np.arange(N), m_star]
            if M > 1:
                top2 = np.partition(Q, -2, axis=1)[:, -2:]
                runner_up_q = top2[:, 0]
                valid_gap = runner_up_q > -1e8
                if np.any(valid_gap):
                    q_gap = q_n[valid_gap] - runner_up_q[valid_gap]
                    q_gap_sum += float(np.sum(q_gap))
                    q_gap_count += int(q_gap.size)
                    q_gap_small_count += int(np.sum(q_gap <= 1.0))
                    q_gap_min = min(q_gap_min, float(np.min(q_gap)))

            action_counts += np.bincount(m_star, minlength=M)
            action_cf_sum += np.bincount(m_star, weights=cf_n, minlength=M)
            action_cont_sum += np.bincount(m_star, weights=cont_n, minlength=M)
            action_q_sum += np.bincount(m_star, weights=q_n, minlength=M)

            # Store
            cf_store[:, t]      = cf_n
            soc_store[:, t + 1] = E_n_new
            soh_store[:, t + 1] = SoH_n
            act_store[:, t]     = m_star.astype(np.int16)
            E_n = E_n_new

            # Per-source components for the chosen mode only — (N,) vectors.
            # Cheaper than indexing into the full (N, M) component matrices.
            net_ch = net_fracs[m_star]                          # (N,)
            d_ch   = np.maximum(net_ch, 0.0) * self.P_bar      # (N,)
            c_ch   = np.maximum(-net_ch, 0.0) * self.P_bar     # (N,)
            dc_ch  = self._dc_fracs[m_star] * self.P_bar       # (N,)
            qr_ch  = self._qr_fracs[m_star] * self.P_bar       # (N,)

            disc_t = disc_factors[t]
            pv_da_paths    += (P_da  * net_ch * self.P_bar * dt).astype(np.float64) * disc_t
            pv_imb_paths   += (delta * d_ch   * dt).astype(np.float64)              * disc_t
            pv_dc_paths    += (pi_dc * dc_ch  * dt).astype(np.float64)              * disc_t
            pv_qr_paths    += (pi_qr * qr_ch  * dt).astype(np.float64)              * disc_t
            pv_costs_paths += ((self.deg_cost + self.vom) * (d_ch + c_ch) * dt
                               ).astype(np.float64) * disc_t

        # Discount cashflows
        pv_paths = (cf_store * disc_factors[None, :]).sum(axis=1)   # (N,)

        if self.verbose:
            print(f"  Forward pass complete. MTM P50 = £{np.median(pv_paths):,.0f}")

        total_decisions = int(action_counts.sum())
        action_diagnostics = {
            "total_decisions": total_decisions,
            "selected_cashflow_mean_gbp": float(action_cf_sum.sum() / total_decisions) if total_decisions else 0.0,
            "selected_continuation_mean_gbp": float(action_cont_sum.sum() / total_decisions) if total_decisions else 0.0,
            "selected_q_mean_gbp": float(action_q_sum.sum() / total_decisions) if total_decisions else 0.0,
            "selected_q_gap_mean_gbp": float(q_gap_sum / q_gap_count) if q_gap_count else None,
            "selected_q_gap_min_gbp": float(q_gap_min) if np.isfinite(q_gap_min) else None,
            "selected_q_gap_le_1gbp_fraction": float(q_gap_small_count / q_gap_count) if q_gap_count else None,
            "by_mode": [
                {
                    "index": int(i),
                    "net_frac": float(self.modes[i].net_frac),
                    "r_dc_frac": float(self.modes[i].r_dc_frac),
                    "r_qr_frac": float(self.modes[i].r_qr_frac),
                    "count": int(action_counts[i]),
                    "selected_cashflow_mean_gbp": (
                        float(action_cf_sum[i] / action_counts[i]) if action_counts[i] else None
                    ),
                    "selected_continuation_mean_gbp": (
                        float(action_cont_sum[i] / action_counts[i]) if action_counts[i] else None
                    ),
                    "selected_q_mean_gbp": (
                        float(action_q_sum[i] / action_counts[i]) if action_counts[i] else None
                    ),
                }
                for i in range(M)
            ],
        }

        return ValuationResult(
            pv_paths       = pv_paths,
            cashflow_paths = cf_store,
            soc_paths      = soc_store,
            soh_paths      = soh_store,
            action_paths   = act_store,
            mtm_mean       = float(np.mean(pv_paths)),
            mtm_std        = float(np.std(pv_paths)),
            mtm_p5         = float(np.percentile(pv_paths, 5)),
            mtm_p95        = float(np.percentile(pv_paths, 95)),
            efc_total      = float(np.mean(
                np.sum(np.maximum(-np.diff(soc_store, axis=1), 0.0), axis=1)
                / max(self.E_name, 1.0)
            )),
            action_diagnostics = action_diagnostics,
            cf_breakdown = {
                'da':        pv_da_paths.astype(np.float64),
                'imbalance': pv_imb_paths.astype(np.float64),
                'dc':        pv_dc_paths.astype(np.float64),
                'qr':        pv_qr_paths.astype(np.float64),
                'costs':     pv_costs_paths.astype(np.float64),
            },
        )

    def forward_parallel(
        self,
        bundle: PathBundle,
        policy: Policy,
        E_init_frac: float = 0.5,
        SoH_init: float = 1.0,
        annual_cycles: float = 520.0,
        max_workers: Optional[int] = None,
    ) -> ValuationResult:
        """
        Forward simulation split across path chunks.

        The policy is fixed, so paths are independent. Threads are used because
        they are notebook-friendly on Windows and avoid pickling large policies.
        """
        N = bundle.n_paths
        if max_workers is None:
            max_workers = max(1, min(N, (os.cpu_count() or 2) - 1))
        max_workers = max(1, min(max_workers, N))

        chunks = np.array_split(np.arange(N), max_workers)
        chunks = [chunk for chunk in chunks if len(chunk)]

        old_verbose = self.verbose
        self.verbose = False

        def run_chunk(pos: int, idx: np.ndarray) -> tuple:
            sub = PathBundle(
                chi=bundle.chi[idx],
                xi=bundle.xi[idx],
                ln_P_base=bundle.ln_P_base[idx],
                lam=bundle.lam[idx],
                delta_imb=bundle.delta_imb[idx],
                pi={k: v[idx] for k, v in bundle.pi.items()},
                dt=bundle.dt,
                n_paths=len(idx),
                n_steps=bundle.n_steps,
            )
            return pos, self.forward(
                sub,
                policy,
                E_init_frac=E_init_frac,
                SoH_init=SoH_init,
                annual_cycles=annual_cycles,
            )

        results = [None] * len(chunks)
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_chunk, i, chunk) for i, chunk in enumerate(chunks)]
                pending = set(futures)
                done = 0
                t0 = time.time()
                heartbeat_s = 10.0
                while pending:
                    finished, pending = wait(
                        pending,
                        timeout=heartbeat_s if old_verbose else None,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in finished:
                        pos, res = future.result()
                        results[pos] = res
                        done += 1
                    if old_verbose:
                        elapsed = time.time() - t0
                        print(
                            f"  Forward chunks complete {done}/{len(chunks)} "
                            f"after {elapsed:.0f}s ...",
                            flush=True,
                        )
        finally:
            self.verbose = old_verbose

        if old_verbose:
            print()

        pv_paths      = np.concatenate([r.pv_paths      for r in results])
        cashflow_paths = np.concatenate([r.cashflow_paths for r in results], axis=0)
        soc_paths     = np.concatenate([r.soc_paths     for r in results], axis=0)
        soh_paths     = np.concatenate([r.soh_paths     for r in results], axis=0)
        action_paths  = np.concatenate([r.action_paths  for r in results], axis=0)

        bd_keys = ['da', 'imbalance', 'dc', 'qr', 'costs']
        cf_breakdown = {
            k: np.concatenate([r.cf_breakdown[k] for r in results])
            for k in bd_keys
        } if results[0].cf_breakdown is not None else None

        efc_total = float(np.mean(
            np.sum(np.maximum(-np.diff(soc_paths, axis=1), 0.0), axis=1)
            / max(self.E_name, 1.0)
        ))

        # Merge action_diagnostics: sum counts, weight-average means.
        diag_list = [r.action_diagnostics for r in results if r.action_diagnostics]
        if diag_list:
            total_decisions = sum(d.get("total_decisions", 0) for d in diag_list)
            def _wt_mean(key: str) -> float:
                return (
                    sum(d.get(key, 0.0) * d.get("total_decisions", 0) for d in diag_list)
                    / max(total_decisions, 1)
                )
            merged_diag: Dict[str, object] = {
                "total_decisions": total_decisions,
                "selected_cashflow_mean_gbp": _wt_mean("selected_cashflow_mean_gbp"),
                "selected_continuation_mean_gbp": _wt_mean("selected_continuation_mean_gbp"),
                "selected_q_mean_gbp": _wt_mean("selected_q_mean_gbp"),
            }
            # Merge per-mode counts (each chunk has same mode ordering)
            if diag_list[0].get("by_mode"):
                n_modes = len(diag_list[0]["by_mode"])
                by_mode = []
                for m_idx in range(n_modes):
                    m_count = sum(
                        d["by_mode"][m_idx]["count"] for d in diag_list
                        if m_idx < len(d.get("by_mode", []))
                    )
                    first_entry = diag_list[0]["by_mode"][m_idx]
                    by_mode.append({
                        "index": first_entry["index"],
                        "net_frac": first_entry["net_frac"],
                        "r_dc_frac": first_entry["r_dc_frac"],
                        "r_qr_frac": first_entry["r_qr_frac"],
                        "count": m_count,
                    })
                merged_diag["by_mode"] = by_mode
        else:
            merged_diag = {}

        return ValuationResult(
            pv_paths=pv_paths,
            cashflow_paths=cashflow_paths,
            soc_paths=soc_paths,
            soh_paths=soh_paths,
            action_paths=action_paths,
            mtm_mean=float(np.mean(pv_paths)),
            mtm_std=float(np.std(pv_paths)),
            mtm_p5=float(np.percentile(pv_paths, 5)),
            mtm_p95=float(np.percentile(pv_paths, 95)),
            efc_total=efc_total,
            action_diagnostics=merged_diag,
            cf_breakdown=cf_breakdown,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_lsmc(
    bundle:       PathBundle,
    asset_cfg:    dict,
    lsmc_cfg:     dict,
    deg_cfg:      dict,
    fin_cfg:      dict,
    modes:        Optional[List[DispatchMode]] = None,
    verbose:      bool = True,
    fwd_bundle:   Optional[PathBundle] = None,
    hpfc_params = None,
    hpfc_curve:   Optional[np.ndarray] = None,
) -> Tuple[Policy, ValuationResult]:
    """
    Run full LSMC: backward induction then forward simulation.

    Parameters
    ----------
    bundle      : PathBundle for backward pass
    fwd_bundle  : optional separate PathBundle for forward pass (default: same as bundle)
    hpfc_curve  : (T+W,) half-hourly HPFC prices for the forward-carry basis feature

    Returns
    -------
    (policy, result)
    """
    if lsmc_cfg.get("run_validation", True):
        asset_check = validate_asset_config(asset_cfg)
        asset_check.raise_if_failed()
        bundle_check = validate_path_bundle(bundle, require_anchor=False)
        bundle_check.raise_if_failed()
        if fwd_bundle is not None:
            fwd_check = validate_path_bundle(fwd_bundle, require_anchor=False)
            fwd_check.raise_if_failed()

    solver = LSMCSolver(
        asset_cfg, lsmc_cfg, deg_cfg, fin_cfg, modes, verbose,
        hpfc_params=hpfc_params, hpfc_curve=hpfc_curve,
    )
    policy = solver.backward(bundle)
    fwd    = fwd_bundle if fwd_bundle is not None else bundle
    result = solver.forward(fwd, policy)

    if lsmc_cfg.get("run_validation", True):
        policy_check = validate_policy(policy)
        policy_check.raise_if_failed()
        result_check = validate_valuation_result(result, asset_cfg)
        result_check.raise_if_failed()

    return policy, result
