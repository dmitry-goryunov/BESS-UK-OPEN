"""
config.py — Single source of truth for the BESS Stochastic MTM Valuation model.

All modules import from here. Do not hardcode parameters elsewhere.

Asset: 100 MW fast-cycle LFP, GB, configurable 1/2/4h duration, 2025 vintage.
Market: Great Britain — DA (N2EX/EPEX), ID (M7), BM (OBP), EAC ancillary.
Framework: LSMC on (E, SoH) grid; Schwartz-Smith two-factor baseload;
           PCA hourly shape; OU+jump imbalance; AR(1)+saturation ancillary.
"""

import numpy as np
import os

# ---------------------------------------------------------------------------
# 1. ASSET ENVELOPE
# ---------------------------------------------------------------------------

ASSET = {
    # --- Nameplate ---
    "power_mw":             100.0,    # MW nameplate discharge power
    "energy_mwh":           200.0,    # MWh nameplate capacity, updated from duration_h
    "duration_h":             2.0,    # default duration; choose 1.0, 2.0, or 4.0

    # --- Efficiency ---
    "rte":                   0.88,    # AC-AC round-trip efficiency
    "eta_charge":            0.9381,  # sqrt(0.88)
    "eta_discharge":         0.9381,  # sqrt(0.88)

    # --- SoC bounds ---
    "soc_min_frac":          0.10,    # 10% floor (warranty + DC sustain)
    "soc_max_frac":          0.90,    # 90% ceiling (calendar fade mitigation)
    "soc_init_frac":         0.50,

    # --- Power limits ---
    "c_rate_max":            1.0,     # 1C -- qualifies for DC (500-1000 ms response)
    "ramp_mw_per_min":      50.0,

    # --- Auxiliary ---
    "aux_load_mw":           0.70,    # HVAC + BMS (scaled from 0.35 at 50 MW)
    "availability":          0.96,

    # --- Degradation warranty ---
    "efc_per_year_warranty": 520,     # OEM EFC budget (conservative LFP)
    "efc_per_year_kyos":     730,     # KYOS benchmark (upper bound reference)
    "soh_augment_trigger":   0.82,    # augment when usable capacity < 82% of nameplate

    # --- Project lifecycle ---
    "life_years":            15,
    "augment_years":        [4, 8, 12],  # augmentation waves; see note below
    # Augmentation schedule note:
    #   CLAUDE.md: [4, 8, 12]; some sources use [3, 6, 9, 12].
    #   [4, 8, 12] chosen: 4-year first wave aligns with typical LFP fade onset;
    #   3-year intervals thereafter match OEM warranty cycle limits.
    #   At gamma=2.1 saturation and 520 EFC/yr, SoH hits 0.82 trigger around yr 4
    #   under central-case dispatch intensity.
    "augment_gbp_kwh":       60.0,    # GBP/kWh added per wave (range 40-80)

    # --- Capex & opex ---
    "capex_gbp_kwh":        220.0,    # GBP/kWh all-in (UK 2h LFP range: 170-280)
    "fom_gbp_kw_yr":          8.0,    # GBP/kW/yr fixed O&M (range 6-10)
    "vom_gbp_mwh":            1.2,    # GBP/MWh throughput variable O&M (range 0.5-2)
    "optimiser_fee_frac":    0.12,    # 12% of gross revenue (merchant)
}

VALID_ASSET_DURATIONS_H = (1.0, 2.0, 4.0)


def configure_asset_duration(asset_cfg=None, duration_h=None):
    """Update energy-linked asset fields for a selected duration."""
    if asset_cfg is None:
        asset_cfg = ASSET
    duration = float(asset_cfg.get("duration_h", 2.0) if duration_h is None else duration_h)
    if duration not in VALID_ASSET_DURATIONS_H:
        valid = ", ".join(f"{d:g}" for d in VALID_ASSET_DURATIONS_H)
        raise ValueError(f"duration_h must be one of {valid}; got {duration:g}")

    asset_cfg["duration_h"] = duration
    asset_cfg["energy_mwh"] = float(asset_cfg["power_mw"]) * duration
    asset_cfg["soc_min_mwh"] = asset_cfg["soc_min_frac"] * asset_cfg["energy_mwh"]
    asset_cfg["soc_max_mwh"] = asset_cfg["soc_max_frac"] * asset_cfg["energy_mwh"]
    asset_cfg["soc_init_mwh"] = asset_cfg["soc_init_frac"] * asset_cfg["energy_mwh"]
    asset_cfg["capex_gbp"] = asset_cfg["capex_gbp_kwh"] * asset_cfg["energy_mwh"] * 1000
    return asset_cfg


configure_asset_duration(ASSET, ASSET["duration_h"])


# ---------------------------------------------------------------------------
# 2. DEGRADATION MODEL
# ---------------------------------------------------------------------------

DEGRADATION = {
    # --- Calendar fade (Arrhenius) ---
    "A_cal":              4.14e-10,
    "Ea_cal":             2.47e4,      # activation energy J/mol
    "R_gas":              8.314,
    "T_ref_celsius":     20.0,
    "soc_stress_coeff":   0.5,
    "soc_stress_ref":     0.5,         # no penalty at 50% SoC

    # --- Cycle fade (Woehler / power law) ---
    "beta":               2.3,         # DoD exponent for LFP (NMC: 1.5-2.0; LFP: 2.0-2.5)
    "N_f_ref":         3000,           # reference cycle life at 100% DoD (illustrative)

    # --- Shadow price of degradation ---
    "lambda_deg_init_gbp_mwh": 6.0,   # GBP/MWh throughput (notebooks: 5-6; range 3-10)
}


# ---------------------------------------------------------------------------
# 3. STOCHASTIC PROCESS PARAMETERS
# ---------------------------------------------------------------------------

# 3.1 Schwartz-Smith two-factor baseload
#   ln P_t = chi_t + xi_t + f(t)
#   d(chi) = -kappa*chi*dt + sigma_chi*dW_chi   (short-term, mean-reverting)
#   d(xi)  = mu_xi*dt + sigma_xi*dW_xi          (long-term, drifting equilibrium)
SCHWARTZ_SMITH = {
    "kappa":             0.08,
    "sigma_chi":        12.0,
    "mu_xi":             0.00,   # zero under risk-neutral; real-world drift for MTM
    "sigma_xi":          8.0,
    "rho_chi_xi":        0.30,
    "seasonal_amp_gbp": 18.0,
    "daily_peak_amp":   38.0,
    "forward_anchor_gbp_mwh": 76.7,   # KYOS Feb 2026 GB 10yr baseload
}

# 3.2 Hourly shape -- PCA decomposition
#   ln P_{h,t} = ln P_t + sum_k lambda_k(t) * phi_k(h)
PCA_SHAPE = {
    "n_factors":     3,
    "alpha":        [0.20, 0.35, 0.50],    # mean-reversion speeds
    "sigma_lambda": [0.08, 0.05, 0.03],    # diffusion per factor
}

# 3.3 Imbalance basis (OU + asymmetric jumps)
#   P_IMB,t = P_DA,t + Delta_t
#   d(Delta) = -theta*Delta*dt + sigma*dW + J_t
# Note: arithmetic formulation required -- GB had 53 negative-price hours in Apr 2024.
IMBALANCE = {
    "theta":            0.40,
    "sigma":           35.0,
    "jump_intensity":   0.015,
    "jump_mean_pos":  150.0,    # GBP/MWh -- short NIV (system buying)
    "jump_mean_neg":   80.0,    # GBP/MWh -- long NIV (system selling)
    "jump_frac_pos":    0.55,
    "allow_negative":  True,
    "negative_price_floor_gbp_mwh": -150.0,
}

# 3.4 Ancillary clearing -- AR(1) per EFA block + saturation supply curve
#   pi_{k,b,t+1} = phi_k * pi_{k,b,t} + epsilon
#   pi_{k,t} = max(0, p_res_k * (1 - Q_BESS,t / Q_req,t)^gamma)
ANCILLARY = {
    "ar1_phi": {
        "DC": 0.75, "DM": 0.70, "DR": 0.65, "QR": 0.80, "BR": 0.60,
    },
    "gamma":             2.1,     # saturation exponent (calibrated to DC collapse 2021-24)
                                  # Notebooks used 1.8; DC empirical evidence supports >=2.0
    "fleet_mw_current": 6_000,
    "fleet_mw_end":    11_000,
    "service_req_mw":   2_500,
    "min_multiplier":    0.15,
    "p_reservation": {
        "DC": 17.0, "DM": 15.0, "DR": 12.0, "QR": 8.75, "BR": 6.0,
    },
    "current_clearing": {
        "DC": (1.0, 5.0), "DM": (3.0, 20.0), "DR": (3.0, 20.0),
        "QR": (5.12, 8.75), "BR": None,
    },
    "sustain_halfhours": {
        "DC": 1, "DM": 1, "DR": 2, "QR": 1, "BR": 2,
    },
}

# 3.5  Balancing Mechanism (BM) offer price process
# pi_bm: mean-reverting (OU), positively correlated with chi (spot scarcity)
BM = {
    "kappa_bm":   2.0,     # mean-reversion speed (1/year)
    "mu_bm":      25.0,    # long-run mean (£/MW/h)
    "sigma_bm":   15.0,    # volatility (£/MW/h)
    "activation_prob": 0.05,  # probability of BM activation per half-hour
    "sustain_h":  1.0,     # sustained headroom required for BM (hours)
}
# Corr(pi_bm, chi) ~ +0.40  (scarcity drives both spot and BM clearing)
# Corr(pi_bm, delta_imb) ~ +0.35  (positive NIV → system needs energy → high BM bids)
BM = {
    "mu_bm":              80.0,   # GBP/MWh long-run mean offer price
    "sigma_bm":           30.0,   # diffusion vol (GBP/MWh per sqrt(year))
    "kappa_bm":            1.5,   # mean-reversion speed (annual)
    "p_activation":        0.12,  # Bernoulli prob of BOA per HH (≈6 calls/day across fleet)
    "sustain_hh":          4,     # HH units a battery must sustain BM headroom (2h)
    "r_bm_levels": [0.0, 0.25, 0.5],  # BM headroom fractions
}

# 3.6 Joint correlation matrix
# Dims: [chi, xi, lambda1(level), lambda2(slope), delta_imb, pi_DC, pi_bm]
CORRELATION = np.array([
    [1.00,  0.30,  0.45,  0.20,  0.55, -0.25,  0.40],
    [0.30,  1.00,  0.15,  0.05,  0.10, -0.10,  0.10],
    [0.45,  0.15,  1.00,  0.20,  0.30, -0.20,  0.20],
    [0.20,  0.05,  0.20,  1.00,  0.15, -0.10,  0.05],
    [0.55,  0.10,  0.30,  0.15,  1.00, -0.30,  0.35],
    [-0.25,-0.10, -0.20, -0.10, -0.30,  1.00, -0.15],
    [0.40,  0.10,  0.20,  0.05,  0.35, -0.15,  1.00],
], dtype=float)

CORRELATION_LABELS = ["chi", "xi", "lambda1", "lambda2", "delta_imb", "pi_dc", "pi_bm"]


# ---------------------------------------------------------------------------
# 4. LSMC CONFIGURATION
# ---------------------------------------------------------------------------

LSMC = {
    "n_paths":           5_000,    # dev; use n_paths_prod for production MTM
    "n_paths_prod":     20_000,
    "dt_hours":           0.5,
    "seed":               42,
    "n_soc_nodes":        21,
    "n_soh_nodes":         5,
    "soh_nodes":    [1.00, 0.95, 0.90, 0.85, 0.82],
    "basis_degree":        3,
    "basis_include_soc_price_cross": True,
    "basis_include_hour_trig":       True,
    "basis_include_efa_block":       True,
    "continuation_value_cap_gbp": 25_000_000,
    # Dispatch sees a lagged imbalance signal; realised delta_imb is only used
    # for settlement cashflow. Set to 0 only for clairvoyant diagnostics.
    "imbalance_signal_lag_hh": 1,
    "delta_imb_cashflow_lag_hh": 1,
    "reserve_sustain_h": 1.0,
    # "summary" uses next-window DA max/min/mean/spread. "raw" appends the
    # ordered next-window DA strip as extra regression features for diagnostics.
    "da_forward_feature_mode": "summary",
    "da_forward_raw_count_hh": 0,
    "dual_gap_acceptable": 0.02,
    "dual_gap_refine":     0.05,
    "run_rolling_intrinsic": True,
    "p_activation": 0.05,  # BM activation probability
}

EFA_BLOCKS = {
    0: range(0,  8),   # 00:00-03:30
    1: range(8,  16),  # 04:00-07:30
    2: range(16, 24),  # 08:00-11:30
    3: range(24, 32),  # 12:00-15:30
    4: range(32, 40),  # 16:00-19:30
    5: range(40, 48),  # 20:00-23:30
}


# ---------------------------------------------------------------------------
# 5. CO-OPTIMISATION CONSTRAINTS
# ---------------------------------------------------------------------------

COOPT = {
    "services_up":   ["DC", "DM", "DR", "QR"],
    "services_down": ["DC", "DR", "BR"],
    "bm_offer": "discharge",
    "bm_bid":   "charge",
}


# ---------------------------------------------------------------------------
# 6. FINANCIAL PARAMETERS
# ---------------------------------------------------------------------------

FINANCE = {
    "wacc_merchant":       0.09,
    "wacc_contracted":     0.06,
    "risk_free":           0.045,
    "gearing":             0.55,
    "debt_tenor_years":   15,
    "dscr_covenant":       1.30,
    "dscr_downside":       1.10,
    "revenue_decay_per_year": 0.015,
    "cm_derating_2h":      0.20,
    "cm_clearing": {
        "T4_2028_29": 60.0,
        "T4_2029_30": 27.1,
        "T1_2026_27":  5.0,
    },
    "alpha_merchant":      1.0,
    "toll_anchor_gbp_mw_yr":   78_000,
    "floor_anchor_gbp_mw_yr":  44_000,
    "floor_share_owner":        0.55,
    "risk_premium_id":     0.10,
    "risk_premium_anc":    0.12,
    "risk_premium_bm":     0.08,
}


# ---------------------------------------------------------------------------
# 7. CALIBRATION ANCHORS
# ---------------------------------------------------------------------------

CALIBRATION_ANCHORS = {
    "modo_2025_avg_gbp_mw_yr":       72_000,
    "modo_feb_2026_gbp_mw_yr":       41_000,
    "modo_jan_2026_gbp_mw_yr":       52_000,
    "modo_dec_2025_gbp_mw_yr":       47_000,
    "modo_all_in_cm_2025_gbp_mw_yr": 85_000,
    "stack_fractions": {
        "wholesale_da_id":            0.35,
        "balancing_mechanism":        0.30,
        "frequency_response_dc_dm_dr":0.18,
        "reserve_qr_br":              0.10,
        "capacity_market":            0.07,
    },
    # KYOS covers DA+ID+passive imbalance only (no ancillary, no CM).
    # Correct Modo comparison: DA+ID+BM bucket = 65% x 72k = ~47k GBP (2025).
    # Do NOT compare KYOS totals to full Modo index.
    "kyos_2026_assessment_eur_mw_yr":  96_400,
    "kyos_2027_assessment_eur_mw_yr":  83_000,
    "kyos_eur_gbp_rate":                0.845,
    "capture_rate_da_p50":             0.60,
    "capture_rate_da_top_quartile":    0.75,
    "skip_rate_dec_2023":              0.90,
    "skip_rate_aug_2024":              0.76,
    "skip_rate_target_eoy_2026":       0.65,
    "da_spread_2024_gbp_mwh":        (50.0, 70.0),
    "da_spread_late_2025_gbp_mwh":   (57.0, 65.0),
}


# ---------------------------------------------------------------------------
# 8. TENOR & SIMULATION HORIZON
# ---------------------------------------------------------------------------

TENOR = {
    "t0":                    "2026-04-26",
    "life_years":            15,
    "merchant_horizon_years": 3,
    "dt_hours":               0.5,
    "halfhours_per_day":     48,
    "halfhours_per_year":    17_520,
}


# ---------------------------------------------------------------------------
# 9. DATA PATHS
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "data_raw":            os.path.join(_ROOT, "data", "raw"),
    "data_processed":      os.path.join(_ROOT, "data", "processed"),
    "elexon_da_prices":    os.path.join(_ROOT, "data", "raw", "elexon_da_prices.parquet"),
    "elexon_sp_prices":    os.path.join(_ROOT, "data", "raw", "elexon_sp_prices.parquet"),
    "neso_eac_clearing":   os.path.join(_ROOT, "data", "raw", "neso_eac_clearing.parquet"),
    "ice_eex_forwards":    os.path.join(_ROOT, "data", "raw", "ice_eex_forwards.parquet"),
    "hpfc":                os.path.join(_ROOT, "data", "processed", "hpfc.parquet"),
    "pca_factors":         os.path.join(_ROOT, "data", "processed", "pca_factors.npz"),
    "ss_params":           os.path.join(_ROOT, "data", "processed", "ss_params.json"),
    "imbalance_params":    os.path.join(_ROOT, "data", "processed", "imbalance_params.json"),
    "ancillary_params":    os.path.join(_ROOT, "data", "processed", "ancillary_params.json"),
}


# ---------------------------------------------------------------------------
# 10. VALIDATION TARGETS
# ---------------------------------------------------------------------------

VALIDATION = {
    "backtest_coverage_target":  0.80,
    "backtest_window_months":   24,
    "max_intraday_capture_rate": 0.90,
    "lender_ref_tolerance":      0.15,
    "pnl_residual_warning":      0.05,
    "pnl_residual_critical":     0.10,
}
