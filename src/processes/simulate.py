"""
Joint path simulator — Phase 3.

Draws N_paths correlated half-hourly paths for all state variables
required by the LSMC backward induction:

    (chi_t, xi_t)           — Schwartz-Smith short/long factors → baseload price
    (lam1_t, lam2_t, lam3_t)— HPFC PCA shape factors → half-hourly price shape
    delta_t                 — imbalance basis (arithmetic OU + asymmetric jumps)
    pi_dc_t, pi_qr_t        — ancillary clearing prices (AR(1) per EFA block)

Joint correlation structure (6×6) among diffusive shocks:
    factors: [chi, xi, lam1, lam2, delta, pi_dc]
    (CLAUDE.md empirical calibration; pi_qr and lam3 independent)

Jumps in delta are additive and independent of the diffusion correlation.
pi_QR and all other ancillary products (DM, DR, BR) use independent AR(1).
lam3 (curvature) uses an independent OU draw.

Memory guidance
---------------
At 10,000 paths × 17,520 steps × float32 one state variable ≈ 700 MB.
Use n_steps=17520 (1 year) for LSMC. For a 15-year run, call simulate()
year-by-year, passing the terminal state of year t as the initial state
of year t+1.

References
----------
    Boogert & de Jong (2008) — LSMC for gas storage, J. Derivatives 15(3)
    Schwartz & Smith (2000) — two-factor commodity model, Mgmt Sci 46
    Cartea & Figueroa (2005) — MR jump-diffusion for power, AMF 12
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from src.processes.schwartz_smith import SSParams
from src.processes.hpfc import HPFCParams
from src.processes.imbalance import ImbalanceParams
from src.processes.ancillary import AncillaryParams, PRODUCTS, SERVICE_VOLUME_MW, PEAK_PRICE


# ---------------------------------------------------------------------------
# Joint correlation matrix — CLAUDE.md spec
# Order: [chi, xi, lam1, lam2, delta, pi_dc]
# ---------------------------------------------------------------------------

JOINT_CORR = np.array([
    #  chi    xi    lam1   lam2   delta  pi_dc
    [ 1.00,  0.30,  0.45,  0.20,  0.55, -0.25],  # chi
    [ 0.30,  1.00,  0.15,  0.05,  0.10, -0.10],  # xi
    [ 0.45,  0.15,  1.00,  0.20,  0.30, -0.20],  # lam1
    [ 0.20,  0.05,  0.20,  1.00,  0.15, -0.10],  # lam2
    [ 0.55,  0.10,  0.30,  0.15,  1.00, -0.30],  # delta
    [-0.25, -0.10, -0.20, -0.10, -0.30,  1.00],  # pi_dc
])

# Indices into the 6-vector for each factor
IDX_CHI   = 0
IDX_XI    = 1
IDX_LAM1  = 2
IDX_LAM2  = 3
IDX_DELTA = 4
IDX_PIDC  = 5


def _cholesky_joint(corr: np.ndarray = JOINT_CORR) -> np.ndarray:
    """
    Cholesky decomposition of the joint correlation matrix.
    Raises if matrix is not positive-definite.

    Returns
    -------
    L : (6, 6) lower-triangular matrix such that L @ L.T = corr
    """
    # Check positive-definiteness
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() <= 0:
        raise ValueError(
            f"Joint correlation matrix is not positive-definite. "
            f"Min eigenvalue: {eigvals.min():.6f}. "
            f"Check off-diagonal entries for consistency."
        )
    return np.linalg.cholesky(corr)


# ---------------------------------------------------------------------------
# PathBundle — output container
# ---------------------------------------------------------------------------

@dataclass
class PathBundle:
    """
    Container for all simulated state variable paths.

    All arrays are float32, shape (n_paths, n_steps + 1).

    Attributes
    ----------
    chi        : SS short-term factor
    xi         : SS long-term factor
    ln_P_base  : log baseload price = chi + xi (no seasonal shape)
    lam        : shape (n_paths, n_steps+1, n_factors) — HPFC PCA factor levels
    delta_imb  : imbalance basis (£/MWh)
    pi         : dict[product_name -> (n_paths, n_steps+1)] ancillary prices (£/MW/h)
    dt         : simulation timestep in years
    n_paths    : int
    n_steps    : int
    """
    chi:       np.ndarray          # (n_paths, n_steps+1)
    xi:        np.ndarray          # (n_paths, n_steps+1)
    ln_P_base: np.ndarray          # (n_paths, n_steps+1)
    lam:       np.ndarray          # (n_paths, n_steps+1, K)
    delta_imb: np.ndarray          # (n_paths, n_steps+1)
    pi:        Dict[str, np.ndarray]  # each (n_paths, n_steps+1)
    dt:        float
    n_paths:   int
    n_steps:   int

    def spot_price(self, step: int) -> np.ndarray:
        """Baseload spot price at a given step (exp of log price), shape (n_paths,)."""
        return np.exp(self.ln_P_base[:, step]).astype(np.float32)

    def half_hourly_prices(
        self,
        hpfc_params: HPFCParams,
        step: int,
        hh_block: int,   # 0..47 — which half-hour within the day
    ) -> np.ndarray:
        """
        Reconstruct the log-price for a specific half-hour block at `step`.

        Returns shape (n_paths,) log-price.
        """
        phi = np.array(hpfc_params.loadings)   # (K, 48)
        lam_t = self.lam[:, step, :]            # (n_paths, K)
        shape_offset = lam_t @ phi[:, hh_block] # (n_paths,)
        return self.ln_P_base[:, step] + shape_offset

    def intraday_spread(
        self,
        hpfc_params: HPFCParams,
        step: int,
        peak_hh: int = 34,    # ~17:00
        trough_hh: int = 14,  # ~07:00
    ) -> np.ndarray:
        """
        Peak-minus-trough intraday spread at `step`, shape (n_paths,).
        Useful as LSMC basis feature (P_id - P_da proxy).
        """
        ln_peak   = self.half_hourly_prices(hpfc_params, step, peak_hh)
        ln_trough = self.half_hourly_prices(hpfc_params, step, trough_hh)
        return np.exp(ln_peak) - np.exp(ln_trough)

    def saturation_adjusted_pi(
        self,
        product: str,
        fleet_mw: float,
        anc_params: AncillaryParams,
    ) -> np.ndarray:
        """
        Apply the saturation multiplier to the simulated ancillary path.
        Returns shape (n_paths, n_steps+1).
        """
        from src.processes.ancillary import saturation_price
        q_req  = anc_params.q_req.get(product, SERVICE_VOLUME_MW.get(product, 500.0))
        p_res  = PEAK_PRICE.get(product, 15.0)
        sat    = saturation_price(fleet_mw, q_req, p_res, anc_params.gamma)
        # sat is a scalar multiplier (0..1 range of the pre-saturation price)
        # The stored pi path is already in £/MW/h absolute terms; scale proportionally
        ratio  = sat / p_res if p_res > 0 else 1.0
        return self.pi[product] * ratio


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

def simulate(
    ss_params:  SSParams,
    hpfc_params: HPFCParams,
    imb_params:  ImbalanceParams,
    anc_params:  AncillaryParams,
    n_paths:     int   = 10_000,
    n_steps:     int   = 17_520,   # 1 year at half-hourly (365 * 48)
    dt:          float = 1 / (365 * 48),   # fraction of year per HH
    seed:        int   = 42,
    corr_matrix: np.ndarray = None,
    # Initial state (for multi-year chaining)
    chi_0:       Optional[np.ndarray] = None,   # (n_paths,)
    xi_0:        Optional[np.ndarray] = None,
    lam_0:       Optional[np.ndarray] = None,   # (n_paths, K)
    delta_0:     Optional[np.ndarray] = None,
    pi_0:        Optional[Dict[str, np.ndarray]] = None,
    dtype:       type = np.float32,
    allow_unanchored: bool = False,
) -> PathBundle:
    """
    Simulate joint half-hourly paths for all state variables.

    Parameters
    ----------
    ss_params    : Schwartz-Smith two-factor parameters
    hpfc_params  : PCA shape decomposition parameters
    imb_params   : Imbalance OU + jump parameters
    anc_params   : Ancillary AR(1) + saturation parameters
    n_paths      : number of Monte Carlo paths
    n_steps      : number of half-hourly steps to simulate
    dt           : timestep in years (default 1/(365*48))
    seed         : RNG seed
    corr_matrix  : (6,6) joint correlation override (uses JOINT_CORR if None)
    chi_0, xi_0  : initial SS states, shape (n_paths,)
                  xi_0 is required unless allow_unanchored=True
    lam_0        : initial HPFC factor levels, shape (n_paths, K); zero if None
    delta_0      : initial imbalance basis, shape (n_paths,); zero if None
    pi_0         : initial ancillary prices per product; uses mu if None
    dtype        : storage dtype (float32 recommended to save memory)
    allow_unanchored
                 : when True, permit xi_0=None and start prices near exp(0).
                   This is useful for process tests only, not valuation.

    Returns
    -------
    PathBundle with all paths stored as dtype arrays.
    """
    if corr_matrix is None:
        corr_matrix = JOINT_CORR

    L = _cholesky_joint(corr_matrix)   # (6, 6) lower-triangular

    rng = np.random.default_rng(seed)

    K = hpfc_params.n_factors
    N = n_paths
    T = n_steps

    if xi_0 is None and not allow_unanchored:
        raise ValueError(
            "simulate() requires xi_0 to anchor the initial price level. "
            "Pass xi_0=np.full(n_paths, np.log(forward_anchor_gbp_mwh)) "
            "or set allow_unanchored=True for process-only tests."
        )

    # ------------------------------------------------------------------
    # Allocate output arrays (float32 to save memory)
    # ------------------------------------------------------------------
    chi       = np.zeros((N, T + 1), dtype=dtype)
    xi        = np.zeros((N, T + 1), dtype=dtype)
    lam       = np.zeros((N, T + 1, K), dtype=dtype)
    delta_imb = np.zeros((N, T + 1), dtype=dtype)
    pi_arr    = {p: np.zeros((N, T + 1), dtype=dtype) for p in PRODUCTS}

    # ------------------------------------------------------------------
    # Set initial states
    # ------------------------------------------------------------------
    if chi_0 is not None:
        chi[:, 0] = chi_0
    if xi_0 is not None:
        xi[:, 0] = xi_0
    if lam_0 is not None:
        lam[:, 0, :] = lam_0
    if delta_0 is not None:
        delta_imb[:, 0] = delta_0

    # Initial ancillary prices: unconditional means from calibration
    for prod in PRODUCTS:
        if pi_0 is not None and prod in pi_0:
            pi_arr[prod][:, 0] = pi_0[prod]
        else:
            pp = anc_params.products.get(prod)
            if pp is not None:
                pi_arr[prod][:, 0] = float(pp.mu)
            else:
                pi_arr[prod][:, 0] = PEAK_PRICE.get(prod, 10.0)

    # ------------------------------------------------------------------
    # Pre-compute SS discrete-time parameters
    # ------------------------------------------------------------------
    ss = ss_params
    exp_kappa  = float(np.exp(-ss.kappa * dt))
    std_chi    = float(ss.sigma_chi * np.sqrt(dt))
    std_xi     = float(ss.sigma_xi  * np.sqrt(dt))
    drift_xi   = float(ss.mu_xi * dt)

    # ------------------------------------------------------------------
    # Pre-compute HPFC OU parameters (daily dt → convert to HH)
    # hpfc alpha is calibrated in units of 1/day; dt is in years
    # dt_hh_in_days = dt * 365
    # ------------------------------------------------------------------
    dt_in_days = dt * 365.0
    alpha_arr  = np.array(hpfc_params.alpha, dtype=float)    # (K,) 1/day
    sigma_lam  = np.array(hpfc_params.sigma_lambda, dtype=float)  # (K,)
    exp_alpha  = np.exp(-alpha_arr * dt_in_days)              # (K,)
    std_lam    = sigma_lam * np.sqrt(dt_in_days)              # (K,) per HH step

    # ------------------------------------------------------------------
    # Pre-compute imbalance OU parameters.
    # Imbalance calibration uses one half-hour as the natural unit, so convert
    # simulation dt from years to half-hours before scaling theta/sigma/lambda.
    # ------------------------------------------------------------------
    imb = imb_params
    dt_in_hh    = dt * 365.0 * 48.0
    exp_theta   = float(np.exp(-imb.theta_delta * dt_in_hh))
    std_delta   = float(imb.sigma_delta * np.sqrt(dt_in_hh))

    # ------------------------------------------------------------------
    # Pre-compute ancillary AR(1) parameters
    # phi calibrated as AR(1) per EFA block (4-hour block = 8 HH steps)
    # Scale to per-HH: phi_hh = phi^(1/8)
    # ------------------------------------------------------------------
    phi_hh   = {}
    sigma_hh = {}
    mu_prod  = {}
    for prod in PRODUCTS:
        pp = anc_params.products.get(prod)
        if pp is not None:
            phi_efa = float(pp.phi)
            sig_efa = float(pp.sigma)
            mu_v    = float(pp.mu)
            # Scale from EFA block (8 HH) to 1 HH
            phi_hh[prod]   = float(np.exp(np.log(max(phi_efa, 1e-9)) / 8.0))
            sigma_hh[prod] = float(sig_efa * np.sqrt(1 / 8.0))
            mu_prod[prod]  = mu_v
        else:
            phi_hh[prod]   = 0.85
            sigma_hh[prod] = 1.2
            mu_prod[prod]  = PEAK_PRICE.get(prod, 10.0)

    # ------------------------------------------------------------------
    # Simulation loop — chunked RNG pre-generation
    # ------------------------------------------------------------------
    # Drawing random numbers one step at a time makes T separate RNG calls.
    # Pre-generating a chunk of CHUNK steps at once reduces call overhead by
    # ~CHUNK× and enables a single batched Cholesky matmul per chunk.
    # Note: chunking changes the RNG interleaving vs. per-step draws, so paths
    # will differ from the old implementation for the same seed.
    n_joint = 6                           # chi, xi, lam1, lam2, delta, pi_dc
    n_indep_lam = max(0, K - 2)          # lam3, lam4, ...
    n_indep_anc = len(PRODUCTS) - 1       # all products except DC_Low (pi_dc)
    indep_prods = [p for p in PRODUCTS if p != 'DC_Low']

    # Constants that were previously recomputed inside the loop
    mu_d          = float(imb.mu_delta)
    mu_dc         = mu_prod['DC_Low']
    lam_j         = float(imb.lambda_jump * dt_in_hh)
    p_pos         = float(imb.p_pos)
    jump_scale_pos = float(imb.jump_scale_pos)
    jump_scale_neg = float(imb.jump_scale_neg)

    CHUNK = 1024

    for t_start in range(0, T, CHUNK):
        t_end = min(t_start + CHUNK, T)
        c = t_end - t_start

        # Pre-generate correlated joint normals: (c, N, 6) → apply L once
        z_raw  = rng.standard_normal((c, N, n_joint))
        z_joint = (z_raw @ L.T).astype(np.float64)   # (c, N, 6)

        if n_indep_lam > 0:
            z_lam_extra_c = rng.standard_normal((c, N, n_indep_lam))
        if n_indep_anc > 0:
            z_anc_indep_c = rng.standard_normal((c, N, n_indep_anc))

        # Pre-generate jump noise for the whole chunk
        n_jumps_c    = rng.poisson(lam_j, size=(c, N))
        jump_signs_c = rng.uniform(size=(c, N)) < p_pos
        jump_pos_c   = rng.exponential(jump_scale_pos, size=(c, N))
        jump_neg_c   = rng.exponential(jump_scale_neg, size=(c, N))
        jump_sizes_c = np.where(jump_signs_c, jump_pos_c, -jump_neg_c)
        jump_total_c = ((n_jumps_c > 0) * jump_sizes_c)   # (c, N)

        for i in range(c):
            t = t_start + i
            z_corr = z_joint[i]       # (N, 6) — view, no copy
            jump_total = jump_total_c[i]   # (N,) — view

            # === 1. Schwartz-Smith chi and xi ===
            chi[:, t+1] = (exp_kappa * chi[:, t]
                           + std_chi * z_corr[:, IDX_CHI]).astype(dtype)
            xi[:, t+1]  = (xi[:, t]
                           + drift_xi
                           + std_xi * z_corr[:, IDX_XI]).astype(dtype)

            # === 2. HPFC shape factors lambda ===
            for k in range(K):
                if k == 0:
                    z_k = z_corr[:, IDX_LAM1]
                elif k == 1:
                    z_k = z_corr[:, IDX_LAM2]
                else:
                    z_k = z_lam_extra_c[i, :, k - 2]
                lam[:, t+1, k] = (exp_alpha[k] * lam[:, t, k]
                                   + std_lam[k] * z_k).astype(dtype)

            # === 3. Imbalance basis (OU diffusion + jumps) ===
            delta_diff = (exp_theta * (delta_imb[:, t] - mu_d)
                          + mu_d
                          + std_delta * z_corr[:, IDX_DELTA])
            delta_imb[:, t+1] = (delta_diff + jump_total).astype(dtype)

            # === 4. Ancillary clearing prices ===
            # DC_Low: correlated via z_corr[:, IDX_PIDC]
            pi_arr['DC_Low'][:, t+1] = np.maximum(
                0.0,
                (phi_hh['DC_Low'] * (pi_arr['DC_Low'][:, t] - mu_dc)
                 + mu_dc
                 + sigma_hh['DC_Low'] * z_corr[:, IDX_PIDC])
            ).astype(dtype)

            # All other products: independent normals
            for ii, prod in enumerate(indep_prods):
                mu_p = mu_prod[prod]
                pi_arr[prod][:, t+1] = np.maximum(
                    0.0,
                    (phi_hh[prod] * (pi_arr[prod][:, t] - mu_p)
                     + mu_p
                     + sigma_hh[prod] * z_anc_indep_c[i, :, ii])
                ).astype(dtype)

    # ------------------------------------------------------------------
    # Build log baseload price series
    # ------------------------------------------------------------------
    ln_P_base = (chi + xi).astype(dtype)

    return PathBundle(
        chi       = chi,
        xi        = xi,
        ln_P_base = ln_P_base,
        lam       = lam,
        delta_imb = delta_imb,
        pi        = pi_arr,
        dt        = dt,
        n_paths   = N,
        n_steps   = T,
    )


# ---------------------------------------------------------------------------
# Convenience: build default params from config (no calibration data needed)
# ---------------------------------------------------------------------------

def default_params_from_config():
    """
    Build SSParams, HPFCParams, ImbalanceParams, AncillaryParams from
    the project config dicts (src.config).
    Used for quick tests and demos before real calibration data is available.

    Important: the returned SSParams has mu_xi=0 and no embedded price level.
    Callers must pass xi_0=np.full(n_paths, np.log(forward_anchor)) to simulate()
    simulate() fails loudly without xi_0 unless allow_unanchored=True is passed.
    so that exp(chi+xi) starts near the forward anchor (£76.7/MWh by default).
    """
    from src.config import (SCHWARTZ_SMITH as SS_CFG, PCA_SHAPE,
                             IMBALANCE as IMB_CFG, ANCILLARY as ANC_CFG)
    from src.processes.ancillary import AncillaryParams, ProductParams

    # --- Schwartz-Smith ---
    ss = SSParams(
        kappa     = SS_CFG['kappa'],
        mu_xi     = SS_CFG['mu_xi'],
        sigma_chi = SS_CFG['sigma_chi'],
        sigma_xi  = SS_CFG['sigma_xi'],
        rho       = SS_CFG['rho_chi_xi'],
    )

    # --- HPFC PCA shape ---
    # Synthetic loadings: level / slope / curvature orthonormal basis
    n_hh = 48
    h    = np.linspace(-1, 1, n_hh)
    phi1 = np.ones(n_hh) / n_hh
    phi2 = h / np.sqrt((h**2).sum())
    phi3_raw = h**2 - (h**2).mean()
    phi3 = phi3_raw / np.sqrt((phi3_raw**2).sum())

    hpfc = HPFCParams(
        n_factors    = PCA_SHAPE['n_factors'],
        eigenvalues  = [0.70, 0.20, 0.06],
        loadings     = [phi1.tolist(), phi2.tolist(), phi3.tolist()],
        alpha        = PCA_SHAPE['alpha'],
        sigma_lambda = PCA_SHAPE['sigma_lambda'],
        explained_variance_ratio = [0.70, 0.20, 0.06],
    )

    # --- Imbalance ---
    imb = ImbalanceParams(
        theta_delta    = IMB_CFG['theta'],
        sigma_delta    = IMB_CFG['sigma'],
        lambda_jump    = IMB_CFG['jump_intensity'],
        jump_scale_pos = IMB_CFG['jump_mean_pos'],
        jump_scale_neg = IMB_CFG['jump_mean_neg'],
        p_pos          = IMB_CFG.get('jump_frac_pos', 0.55),
        mu_delta       = 0.0,
    )

    # --- Ancillary ---
    # Map product names (DC_Low, DC_High, DM_Low, ...) to config groups (DC, DM, ...)
    def _group(prod):
        for g in ('DC', 'DM', 'DR', 'QR', 'BR'):
            if prod.startswith(g):
                return g
        return 'DC'

    ar1_phi   = ANC_CFG['ar1_phi']      # {'DC': 0.75, ...}
    p_res     = ANC_CFG['p_reservation']  # {'DC': 17.0, ...}

    prod_params = {}
    for prod in PRODUCTS:
        grp   = _group(prod)
        phi_v = ar1_phi.get(grp, 0.75)
        mu_v  = PEAK_PRICE.get(prod, p_res.get(grp, 10.0))
        # Approx innovation sigma: ~20% of mean per EFA block
        sig_v = mu_v * 0.20
        prod_params[prod] = ProductParams(
            product = prod,
            phi     = phi_v,
            sigma   = sig_v,
            mu      = mu_v,
        )

    anc = AncillaryParams(
        products  = prod_params,
        gamma     = ANC_CFG['gamma'],
        fleet_mw  = float(ANC_CFG['fleet_mw_current']),
        q_req     = dict(SERVICE_VOLUME_MW),
    )

    return ss, hpfc, imb, anc


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_marginals(
    bundle: PathBundle,
    ss_params: SSParams,
    imb_params: ImbalanceParams,
    dt: float,
    rtol: float = 0.15,
) -> dict:
    """
    Check that simulated marginals match theoretical moments.

    Returns a dict of {check_name: (passed, simulated_value, expected_value)}.

    Notes
    -----
    - xi_mean uses an absolute tolerance floor (sigma/sqrt(N)) to handle mu_xi ≈ 0.
    - delta_mean checks against the theoretical stationary mean that includes
      the net jump drift (jump process shifts the OU equilibrium upward when
      positive jumps dominate).
    - Cross-correlation chi vs delta is tested at the SINGLE-STEP increment
      level, not the terminal state.  Large jump variance in delta dilutes
      terminal-state correlations even when diffusion shocks are correctly
      correlated; increment-level correlation is the right diagnostic.
    """
    results = {}
    N, T = bundle.n_paths, bundle.n_steps

    # --- chi: mean-zero OU, check variance at horizon ---
    chi_T = bundle.chi[:, -1]
    chi_var_theory = (ss_params.sigma_chi**2 / (2 * ss_params.kappa)) * \
                     (1 - np.exp(-2 * ss_params.kappa * T * dt))
    chi_var_sim = float(np.var(chi_T))
    ok_chi = abs(chi_var_sim - chi_var_theory) / max(chi_var_theory, 1e-9) < rtol
    results['chi_variance'] = (ok_chi, chi_var_sim, chi_var_theory)

    # --- xi: BM with drift; allow non-zero initial anchor xi_0 ---
    xi_T = bundle.xi[:, -1]
    xi_0 = bundle.xi[:, 0]
    xi_mean_theory = float(np.mean(xi_0) + ss_params.mu_xi * T * dt)
    xi_var_theory  = float(np.var(xi_0) + ss_params.sigma_xi**2 * T * dt)
    xi_mean_sim = float(np.mean(xi_T))
    xi_var_sim  = float(np.var(xi_T))
    # Tolerance = max(rtol * |mean|, 3 * std / sqrt(N))  (CLT-based)
    xi_std_err = np.sqrt(xi_var_theory / N) if N > 0 else 1.0
    ok_xi_m = abs(xi_mean_sim - xi_mean_theory) < max(rtol * abs(xi_mean_theory), 3 * xi_std_err)
    ok_xi_v = abs(xi_var_sim - xi_var_theory) / max(xi_var_theory, 1e-9) < rtol
    results['xi_mean']     = (ok_xi_m, xi_mean_sim, xi_mean_theory)
    results['xi_variance'] = (ok_xi_v, xi_var_sim, xi_var_theory)

    # --- delta: stationary mean accounting for net jump drift ---
    # Theoretical: E[Delta_ss] ≈ jump_drift_per_step / theta_per_step
    # where theta_per_step = -ln(exp(-theta * dt_hh)) ~= theta * dt_hh
    dt_in_hh = dt * 365.0 * 48.0
    theta_per_step = imb_params.theta_delta * dt_in_hh
    jump_drift = (imb_params.lambda_jump
                  * dt_in_hh
                  * (imb_params.p_pos * imb_params.jump_scale_pos
                     - (1.0 - imb_params.p_pos) * imb_params.jump_scale_neg))
    delta_mean_theory = (imb_params.mu_delta
                         + jump_drift / max(theta_per_step, 1e-9))
    delta_T = bundle.delta_imb[:, -1]
    delta_mean_sim = float(np.mean(delta_T))
    delta_std_err  = float(np.std(delta_T) / np.sqrt(N))
    ok_delta = abs(delta_mean_sim - delta_mean_theory) < max(10.0, 5 * delta_std_err)
    results['delta_stationary_mean'] = (ok_delta, delta_mean_sim, delta_mean_theory)

    # --- pi_dc: non-negative and bounded ---
    pi_dc_max = float(bundle.pi['DC_Low'].max())
    pi_dc_min = float(bundle.pi['DC_Low'].min())
    results['pi_dc_non_negative'] = (pi_dc_min >= 0.0, pi_dc_min, 0.0)
    results['pi_dc_bounded']      = (pi_dc_max < 500.0, pi_dc_max, 500.0)

    # --- Cross-correlation chi vs delta: test at increment level ---
    # Terminal-state correlation is diluted by accumulated jump variance.
    # Instead, measure corr of single-step increments dchi vs d_delta_diffusion.
    # We use consecutive-step increments (mixing diffusion + jumps in delta);
    # even with jumps, with enough steps the diffusion signal should show.
    # Use a representative mid-section to avoid initial condition effects.
    mid = T // 2
    span = min(500, T // 4)
    d_chi   = np.diff(bundle.chi[:, mid:mid+span+1], axis=1).ravel()
    d_delta = np.diff(bundle.delta_imb[:, mid:mid+span+1], axis=1).ravel()
    rho_incr = float(np.corrcoef(d_chi, d_delta)[0, 1])
    # Expected correlation is attenuated by jump noise in delta increments.
    # Attenuation ≈ std_diffusion / std_total:
    std_diff_delta  = float(imb_params.sigma_delta * np.sqrt(dt_in_hh))
    # Approximate total std including jumps
    jump_var_per_step = (imb_params.lambda_jump
                         * dt_in_hh
                         * (imb_params.p_pos * 2 * imb_params.jump_scale_pos**2
                            + (1 - imb_params.p_pos) * 2 * imb_params.jump_scale_neg**2))
    std_total_delta = float(np.sqrt(std_diff_delta**2 + jump_var_per_step))
    attenuation = std_diff_delta / max(std_total_delta, 1e-9)
    target_rho_incr = JOINT_CORR[IDX_CHI, IDX_DELTA] * attenuation
    # Allow 50% relative tolerance (noisy statistic)
    ok_rho = abs(rho_incr - target_rho_incr) < max(0.05, 0.5 * abs(target_rho_incr))
    results['cross_corr_chi_delta_increment'] = (ok_rho, rho_incr, target_rho_incr)

    return results


def print_validation_report(results: dict) -> None:
    """Pretty-print validation results."""
    all_ok = True
    print("\n── Marginal Validation Report ─────────────────────────────────")
    for name, (ok, sim, exp) in results.items():
        status = "✓" if ok else "✗"
        if not ok:
            all_ok = False
        print(f"  {status}  {name:<35s}  sim={sim:+.4f}  exp={exp:+.4f}")
    print("────────────────────────────────────────────────────────────────")
    if all_ok:
        print("  All checks passed.")
    else:
        print("  Some checks failed — review parameter scaling or correlation spec.")
    print()
