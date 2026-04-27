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

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.optimisation.dispatch import (
    DispatchMode, DEFAULT_MODES, cashflow_batch,
    next_soc_grid, feasibility_mask,
)
from src.processes.simulate import PathBundle


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
    "sin_h", "cos_h",
    "efa_block",
]
N_BASIS = len(BASIS_NAMES)   # 14


def basis_matrix(
    P_da:       np.ndarray,   # (N,) £/MWh
    P_id_spr:   np.ndarray,   # (N,) intraday premium (£/MWh)
    delta:      np.ndarray,   # (N,) imbalance basis
    pi_dc:      np.ndarray,   # (N,) DC clearing
    pi_qr:      np.ndarray,   # (N,) QR clearing
    E:          float,        # scalar SoC (MWh) — same for all paths at one grid node
    t_hh:       int,          # half-hour of day (0..47)
    efa_block:  int,          # EFA block (0..5)
) -> np.ndarray:
    """
    Build (N_paths, 14) basis matrix for one SoC grid node.

    Inputs are clipped and scaled here so the polynomial regression remains
    numerically stable. Without this, P_da^3 and E^2 can dominate the normal
    equations and create explosive continuation values.
    """
    N  = len(P_da)
    h  = t_hh / 2.0   # convert to hour-of-day

    P = np.clip(P_da, -100.0, 500.0).astype(np.float32) / 100.0
    P_id = np.clip(P_id_spr, -200.0, 200.0).astype(np.float32) / 100.0
    dlt = np.clip(delta, -500.0, 500.0).astype(np.float32) / 100.0
    dc = np.clip(pi_dc, 0.0, 100.0).astype(np.float32) / 20.0
    qr = np.clip(pi_qr, 0.0, 100.0).astype(np.float32) / 20.0
    E_scaled = np.float32(E / 100.0)

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
    Phi[:, 11] = np.float32(np.sin(2 * np.pi * h / 24))
    Phi[:, 12] = np.float32(np.cos(2 * np.pi * h / 24))
    Phi[:, 13] = float(efa_block)

    return Phi


# ---------------------------------------------------------------------------
# LSMC output containers
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    """
    Regression coefficients from the backward pass.

    beta[t, j, k] is the (N_BASIS,) coefficient vector for time step t,
    SoC grid index j, and SoH node index k.

    Shapes:
        beta      : (n_steps, n_soc_nodes, n_soh_nodes, N_BASIS)
        soc_grid  : (n_soc_nodes,) — MWh
        soh_nodes : (n_soh_nodes,) — fraction
        dt_h      : float — half-hour step in hours
    """
    beta:       np.ndarray    # (T, J, K, 14)
    soc_grid:   np.ndarray    # (J,)
    soh_nodes:  np.ndarray    # (K,)
    modes:      List[DispatchMode]
    dt_h:       float
    n_steps:    int
    n_paths:    int


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
    ) -> None:
        self.asset   = asset_cfg
        self.lsmc    = lsmc_cfg
        self.deg     = deg_cfg
        self.fin     = fin_cfg
        self.modes   = modes if modes is not None else DEFAULT_MODES
        self.verbose = verbose

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

        # Pre-stack mode fractions as arrays (M,) for vectorised dispatch
        self._net_fracs = np.array([m.net_frac  for m in self.modes], np.float32)
        self._dc_fracs  = np.array([m.r_dc_frac for m in self.modes], np.float32)
        self._qr_fracs  = np.array([m.r_qr_frac for m in self.modes], np.float32)

        # EFA block lookup (0..5) given half-hour index 0..47
        self._efa = np.array([hh // 8 for hh in range(48)], dtype=np.int32)

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

        if self.verbose:
            print(f"LSMC backward: T={T} steps, N={N} paths, "
                  f"J={J} SoC nodes, K={K} SoH nodes, M={M} modes")

        # Coefficient store
        beta = np.zeros((T, J, K, N_BASIS), dtype=np.float32)

        # Continuation value store: V[j, k, n] = Ê[V_{t+1}(E_j, SoH_k, S_n)]
        # Updated at each backward step and then used at t-1.
        V_next = np.zeros((J, K, N), dtype=np.float32)

        # Pre-extract and cap market arrays. These are valuation stabilisers for
        # dev notebooks; production calibration should reduce the need for caps.
        P_da_all   = np.clip(np.exp(bundle.ln_P_base), -100.0, 500.0).astype(np.float32)
        delta_all  = np.clip(bundle.delta_imb, -500.0, 500.0).astype(np.float32)
        pi_dc_all  = np.clip(bundle.pi['DC_Low'], 0.0, 100.0).astype(np.float32)
        pi_qr_all  = np.clip(bundle.pi.get('QR_Pos', bundle.pi['DC_Low']), 0.0, 100.0).astype(np.float32)

        # Intraday spread proxy: use lambda_1 loading on peak vs trough HH
        # If HPFC params aren't passed, approximate as zero
        P_id_spr_all = np.zeros_like(P_da_all)   # (N, T+1) — todo: fill from hpfc

        # Pre-compute next-SoC table: E_next[j, m] — (J, M)
        # For each SoC node and each mode, deterministic E' (before SoH scaling)
        E_next_table = np.zeros((J, M), dtype=np.float32)
        for j, E_j in enumerate(self.soc_grid):
            E_next_table[j] = next_soc_grid(
                E_j, self._net_fracs,
                self.P_bar, self.eta_c, self.eta_d,
                dt, self.soc_grid[0], self.soc_grid[-1],
            )

        # Regularisation for lstsq
        reg = 1e-6

        # Backward loop
        for t in range(T - 1, -1, -1):
            if self.verbose and t % 1000 == 0:
                print(f"  t={t:5d} / {T}  ...", end='\r')

            # Market state at step t
            P_da  = P_da_all[:, t]       # (N,)
            delta = delta_all[:, t]      # (N,)
            pi_dc = pi_dc_all[:, t]      # (N,)
            pi_qr = pi_qr_all[:, t]      # (N,)
            P_id  = P_id_spr_all[:, t]   # (N,)

            t_hh      = t % 48
            efa_block = int(self._efa[t_hh])

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

                    # Feasibility: zero out infeasible modes
                    # (simplified: only charge/discharge SoC bounds; reserve
                    #  sustain feasibility checked via E_next clipping above)
                    infeasible = (E_next_jm <= E_min_k + 1e-3) & (self._net_fracs > 0)
                    infeasible |= (E_next_jm >= E_max_k - 1e-3) & (self._net_fracs < 0)
                    Q[:, infeasible] = -1e9

                    # Max Q over modes: (N,)
                    Y = Q.max(axis=1)

                    # Regression
                    Phi = basis_matrix(P_da, P_id, delta, pi_dc, pi_qr,
                                       E_j, t_hh, efa_block)   # (N, 14)

                    # OLS with Tikhonov regularisation
                    PhiT_Phi = Phi.T @ Phi + reg * np.eye(N_BASIS, dtype=np.float32)
                    PhiT_Y   = Phi.T @ Y.astype(np.float32)
                    try:
                        b = np.linalg.solve(PhiT_Phi, PhiT_Y)
                    except np.linalg.LinAlgError:
                        b = np.zeros(N_BASIS, dtype=np.float32)
                    beta[t, j, k_idx, :] = b.astype(np.float32)

                    # Update V_next at this node for the NEXT backward step (t-1)
                    V_next[j, k_idx, :] = np.clip(Phi @ b, -1e8, 1e8).astype(np.float32)

        if self.verbose:
            print(f"\n  Backward pass complete. beta shape: {beta.shape}")

        return Policy(
            beta      = beta,
            soc_grid  = self.soc_grid,
            soh_nodes = self.soh_nodes,
            modes     = self.modes,
            dt_h      = self.dt_h,
            n_steps   = T,
            n_paths   = N,
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
        N = V_next.shape[2]
        M = len(E_next_jm)
        V_interp = np.empty((N, M), dtype=np.float32)

        grid = self.soc_grid   # (J,)
        J    = len(grid)

        for m_idx in range(M):
            E_target = float(E_next_jm[m_idx])
            # Find bounding grid indices
            j_lo = np.searchsorted(grid, E_target, side='right') - 1
            j_lo = max(0, min(j_lo, J - 2))
            j_hi = j_lo + 1

            denom = float(grid[j_hi] - grid[j_lo])
            if denom < 1e-9:
                alpha = 0.0
            else:
                alpha = (E_target - float(grid[j_lo])) / denom
            alpha = max(0.0, min(1.0, alpha))

            V_interp[:, m_idx] = (
                (1.0 - alpha) * V_next[j_lo, k_idx, :]
                + alpha       * V_next[j_hi, k_idx, :]
            ).astype(np.float32)

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

        # Pre-extract and cap prices consistently with backward().
        P_da_all  = np.clip(np.exp(bundle.ln_P_base), -100.0, 500.0).astype(np.float32)
        delta_all = np.clip(bundle.delta_imb, -500.0, 500.0).astype(np.float32)
        pi_dc_all = np.clip(bundle.pi['DC_Low'], 0.0, 100.0).astype(np.float32)
        pi_qr_all = np.clip(bundle.pi.get('QR_Pos', bundle.pi['DC_Low']), 0.0, 100.0).astype(np.float32)

        # SoH fade rate per HH step
        # Simplified: linear SoH degradation at cycle_rate EFCs/year
        # dSoH/dHH = (0.18 total SoH fade / 520 EFC) × (avg DoD) / (HH per cycle)
        # Here: rough approx — use 0 for forward sim SoH (track separately)
        n_hh_per_cycle = self.E_name / (self.P_bar * dt)   # ~400 HH per full cycle
        soh_per_efc    = 0.18 / max(annual_cycles, 1.0)    # SoH lost per EFC

        disc_factors = np.float32(self.disc) ** np.arange(T, dtype=np.float32)

        # Mode fracs
        net_fracs = self._net_fracs   # (M,)

        if self.verbose:
            print(f"LSMC forward:  T={T}, N={N}")

        for t in range(T):
            P_da  = P_da_all[:, t]
            delta = delta_all[:, t]
            pi_dc = pi_dc_all[:, t]
            pi_qr = pi_qr_all[:, t]

            t_hh      = t % 48
            efa_block = int(self._efa[t_hh])

            # For each path, determine the optimal mode
            # 1. Get SoC grid index for each path via interpolation
            j_arr = np.clip(
                np.searchsorted(self.soc_grid, E_n, side='right') - 1,
                0, self.n_soc - 2,
            )
            k_arr = np.clip(
                np.searchsorted(self.soh_nodes[::-1], SoH_n[::-1], side='left'),
                0, self.n_soh - 1,
            )

            # 2. Compute cashflow for all modes (N, M)
            CF = cashflow_batch(
                self.modes, P_da, delta, pi_dc, pi_qr,
                self.P_bar, dt, self.deg_cost, self.vom,
            )

            # 3. Evaluate continuation value for each mode and each path
            # For efficiency: pick one representative SoC node per path
            # and use its beta coefficients to evaluate Q
            # (more exact: interpolate beta between j_arr[n] and j_arr[n]+1)
            Q = np.full((N, len(self.modes)), -1e9, dtype=np.float32)

            for j in range(self.n_soc - 1):
                mask_j = (j_arr == j)
                if not mask_j.any():
                    continue
                for k_idx in range(self.n_soh):
                    mask = mask_j  # simplified: ignore SoH interpolation for speed
                    if not mask.any():
                        continue

                    # E next for each mode
                    SoH_rep  = float(self.soh_nodes[k_idx])
                    E_min_k  = self.E_min_fr * self.E_name
                    E_max_k  = self.E_max_fr * self.E_name * SoH_rep

                    E_next_jm = np.clip(
                        next_soc_grid(
                            float(self.soc_grid[j]),
                            net_fracs, self.P_bar,
                            self.eta_c, self.eta_d, dt,
                            E_min_k, E_max_k,
                        ),
                        E_min_k, E_max_k,
                    )   # (M,)

                    # Beta at this node
                    b = policy.beta[t, j, k_idx, :]   # (14,)

                    # For each mode, build ψ with E_next (LSML-style approx)
                    P_id_spr_n = np.zeros(mask.sum(), dtype=np.float32)
                    for m_idx, E_m in enumerate(E_next_jm):
                        Phi_m = basis_matrix(
                            P_da[mask], P_id_spr_n, delta[mask],
                            pi_dc[mask], pi_qr[mask],
                            E_m, t_hh, efa_block,
                        )   # (mask.sum(), 14)
                        cont = np.clip(Phi_m @ b, -1e8, 1e8).astype(np.float32)

                        # Feasibility: clip infeasible modes to -inf
                        feas = True
                        E_curr_j = self.soc_grid[j]
                        if net_fracs[m_idx] > 0 and E_next_jm[m_idx] <= E_min_k + 1e-3:
                            feas = False
                        if net_fracs[m_idx] < 0 and E_next_jm[m_idx] >= E_max_k - 1e-3:
                            feas = False

                        if feas:
                            Q[mask, m_idx] = CF[mask, m_idx] + self.disc * cont

            # 4. Choose optimal mode per path
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

            # Store
            cf_store[:, t]      = cf_n
            soc_store[:, t + 1] = E_n_new
            soh_store[:, t + 1] = SoH_n
            act_store[:, t]     = m_star.astype(np.int16)
            E_n = E_n_new

        # Discount cashflows
        pv_paths = (cf_store * disc_factors[None, :]).sum(axis=1)   # (N,)

        if self.verbose:
            print(f"  Forward pass complete. MTM P50 = £{np.median(pv_paths):,.0f}")

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
            efc_total      = float(
                np.mean(np.abs(act_store).astype(float))
            ),  # approx
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
) -> Tuple[Policy, ValuationResult]:
    """
    Run full LSMC: backward induction then forward simulation.

    Parameters
    ----------
    bundle      : PathBundle for backward pass
    fwd_bundle  : optional separate PathBundle for forward pass (default: same as bundle)

    Returns
    -------
    (policy, result)
    """
    solver = LSMCSolver(asset_cfg, lsmc_cfg, deg_cfg, fin_cfg, modes, verbose)
    policy = solver.backward(bundle)
    fwd    = fwd_bundle if fwd_bundle is not None else bundle
    result = solver.forward(fwd, policy)
    return policy, result
