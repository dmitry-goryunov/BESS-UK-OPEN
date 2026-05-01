"""
src/valuation/mtm.py
====================
MTM aggregation for BESS stochastic valuation.

Full MTM formula (CLAUDE.md §MTM aggregation):

    MTM = alpha * PV_merchant
        + (1 - alpha) * PV_contracted_legs    [toll, floor, CM]
        + MTM_floor_optionality               [put on annual revenue floor]
        - PV_optimiser_fee                    [fraction of gross merchant revenue]
        - PV_opex_fixed                       [FOM discounted over life]
        - PV_augmentation                     [capex waves at yr 4/8/12]

Multi-year scaling:
    The LSMC is calibrated on a 1-year simulation.  The MTM covers the full
    asset life (life_years = 15) by multiplying year-1 PV by an annuity factor
    that accounts for revenue decay and WACC:

        annuity_factor = sum_{k=0}^{N-1} ((1-decay)/(1+wacc))^k

References
----------
Boogert & de Jong (2008) — LSMC for gas storage
CLAUDE.md §MTM aggregation, §Financial parameters
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContractSpec:
    """Single bilateral contract overlay (toll, floor or Capacity Market)."""
    kind:            str    # "toll" | "floor" | "cm"
    start_year:      float  # years from valuation date
    end_year:        float
    value_gbp_mw_yr: float  # annual value per MW of contracted capacity
    power_mw:        float  # contracted capacity (≤ nameplate)
    wacc:            float  # discount rate for this leg
    floor_share:     float = 1.0   # owner's share of floor shortfall payment (0-1)


@dataclass
class MtmComponents:
    """
    Full MTM decomposition in GBP (spot, life-time).

    All 'pv_*' fields are already discounted to t = 0.
    Negative values represent costs.
    """
    # --- Merchant (stochastic) ---
    pv_merchant_mean:      float
    pv_merchant_std:       float
    pv_merchant_p5:        float
    pv_merchant_p95:       float
    pv_merchant_paths:     np.ndarray   # (N_paths,) annual PV × annuity factor

    # --- Contract overlays (deterministic DCF) ---
    pv_toll:               float = 0.0
    pv_floor:              float = 0.0
    pv_cm:                 float = 0.0
    pv_floor_optionality:  float = 0.0  # stochastic: E[max(0, floor − merchant)]

    # --- Deductions (negative) ---
    pv_optimiser_fee:      float = 0.0  # −fee_frac × gross revenue
    pv_opex_fixed:         float = 0.0  # −FOM × annuity
    pv_augmentation:       float = 0.0  # −sum of capex waves

    # --- Totals ---
    mtm_mean:              float = 0.0
    mtm_std:               float = 0.0
    mtm_p5:                float = 0.0
    mtm_p95:               float = 0.0
    mtm_paths:             np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # --- Metadata ---
    alpha_merchant:        float = 1.0
    life_years:            int   = 15
    power_mw:              float = 100.0
    energy_mwh:            float = 200.0
    annuity_factor:        float = 1.0   # multi-year scaling

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-compatible dict (arrays → lists)."""
        out: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    def summary_gbp_mw_yr(self) -> Dict[str, float]:
        """
        Express all MTM components in GBP/MW/yr for comparability with
        Modo Energy / KYOS / Cornwall Insight benchmarks.
        """
        denom = self.power_mw * self.life_years
        return {
            "merchant":           self.pv_merchant_mean / denom,
            "toll":               self.pv_toll / denom,
            "floor_contracted":   self.pv_floor / denom,
            "cm":                 self.pv_cm / denom,
            "floor_optionality":  self.pv_floor_optionality / denom,
            "optimiser_fee":      self.pv_optimiser_fee / denom,
            "opex_fixed":         self.pv_opex_fixed / denom,
            "augmentation":       self.pv_augmentation / denom,
            "total_mean":         self.mtm_mean / denom,
            "total_std":          self.mtm_std / denom,
            "total_p5":           self.mtm_p5 / denom,
            "total_p95":          self.mtm_p95 / denom,
        }


# ---------------------------------------------------------------------------
# Helper: annuity with revenue decay
# ---------------------------------------------------------------------------

def annuity_factor_decay(wacc: float, decay: float, n_years: int) -> float:
    """
    Annuity factor for a year-1 cash stream that decays geometrically.

        F = sum_{k=0}^{N-1}  ((1 - decay) / (1 + wacc))^k

    Parameters
    ----------
    wacc    : annual discount rate (e.g. 0.09 for 9%)
    decay   : annual revenue decline rate (e.g. 0.015 for 1.5%)
    n_years : asset life in years

    Returns
    -------
    scalar annuity factor ≥ 1
    """
    r = (1.0 - decay) / (1.0 + wacc)
    if abs(r - 1.0) < 1e-12:
        return float(n_years)
    return (1.0 - r ** n_years) / (1.0 - r)


def discount_factor(wacc: float, year: float) -> float:
    """PV discount factor for a cash-flow at `year` years from today."""
    return 1.0 / (1.0 + wacc) ** year


# ---------------------------------------------------------------------------
# Merchant leg
# ---------------------------------------------------------------------------

def _pv_merchant_multiyear(
    pv_year1_paths: np.ndarray,   # (N,) year-1 LSMC PV per path (already discounted)
    wacc:           float,
    decay:          float,
    life_years:     int,
) -> np.ndarray:
    """
    Scale year-1 per-path PV to full life-time PV.

    The LSMC discounts within year 1 at wacc_merchant/HH.  This function
    multiplies by the annuity factor to extend the same expected cashflow
    profile over life_years (with geometric revenue decay each year).

    Returns (N,) lifetime PV paths.
    """
    af = annuity_factor_decay(wacc, decay, life_years)
    return pv_year1_paths * af


# ---------------------------------------------------------------------------
# Optimiser fee
# ---------------------------------------------------------------------------

def _pv_optimiser_fee(
    cashflow_paths: np.ndarray,   # (N, T) period net cashflows (GBP)
    dt_h:           float,        # half-hour in hours (0.5)
    wacc:           float,
    fee_frac:       float,
    annuity:        float,
) -> np.ndarray:
    """
    Compute per-path optimiser fee PV (negative contribution to MTM).

    Gross revenue proxy: sum of *positive* period cashflows, discounted.
    The optimiser fee applies only to positive merchant income (not to loss
    periods or cost components).

    Returns (N,) array — negative values.
    """
    N, T = cashflow_paths.shape
    # Discount per half-hour step
    t_years = np.arange(T, dtype=np.float64) * dt_h / 8760.0
    disc = np.exp(-wacc * t_years).astype(np.float32)   # (T,)

    # Discounted gross-revenue proxy per path
    positive_cf = np.maximum(cashflow_paths, 0.0)        # (N, T)
    pv_gross = (positive_cf * disc[np.newaxis, :]).sum(axis=1)  # (N,)

    # Extend over full life (same annuity scaling as merchant)
    fee_paths = -fee_frac * pv_gross * annuity           # negative
    return fee_paths


# ---------------------------------------------------------------------------
# Fixed opex (FOM)
# ---------------------------------------------------------------------------

def _pv_opex_fixed(
    asset_cfg: dict,
    fin_cfg:   dict,
    life_years: int,
) -> float:
    """
    PV of annual Fixed O&M cost over the asset life (deterministic).

        FOM_annual = fom_gbp_kw_yr × power_mw × 1000  [GBP/yr]
        PV_FOM = FOM_annual × annuity(wacc_contracted, life_years)

    Discounted at contracted WACC (lower risk — certain cost).
    """
    fom_annual = asset_cfg["fom_gbp_kw_yr"] * asset_cfg["power_mw"] * 1000.0  # GBP/yr
    wacc_c     = fin_cfg["wacc_contracted"]
    # Standard annuity (no decay on costs)
    af = (1.0 - (1.0 + wacc_c) ** (-life_years)) / wacc_c
    return -fom_annual * af   # negative


# ---------------------------------------------------------------------------
# Augmentation capex
# ---------------------------------------------------------------------------

def _pv_augmentation(
    asset_cfg:  dict,
    fin_cfg:    dict,
) -> float:
    """
    PV of augmentation capex waves at scheduled years [4, 8, 12].

        Capex_k = augment_gbp_kwh × energy_mwh × 1000  [GBP]
        PV = sum_k Capex_k × discount_factor(wacc_merchant, year_k)

    Discounted at merchant WACC (equity-funded capex event).
    """
    wacc_m    = fin_cfg["wacc_merchant"]
    capex_per = asset_cfg["augment_gbp_kwh"] * asset_cfg["energy_mwh"] * 1000.0
    pv = 0.0
    for yr in asset_cfg.get("augment_years", []):
        pv += capex_per * discount_factor(wacc_m, yr)
    return -pv   # negative


# ---------------------------------------------------------------------------
# Contract overlays (deterministic DCF)
# ---------------------------------------------------------------------------

def _pv_contracted_legs(
    contracts:   List[ContractSpec],
) -> Dict[str, float]:
    """
    Compute PV for each contracted leg type (toll, floor, CM).

    For 'floor' contracts, `value_gbp_mw_yr` is the guaranteed floor payment.
    The floor optionality (incremental stochastic value) is computed separately
    in `_pv_floor_optionality`.

    Returns dict with keys "toll", "floor", "cm".
    """
    result = {"toll": 0.0, "floor": 0.0, "cm": 0.0}
    for c in contracts:
        # Annuity over contract period at the contracted WACC
        n = c.end_year - c.start_year
        af = (1.0 - (1.0 + c.wacc) ** (-n)) / c.wacc if c.wacc > 0 else n
        # Discount to t=0 from start_year
        df_start = discount_factor(c.wacc, c.start_year)
        pv = c.value_gbp_mw_yr * c.power_mw * af * df_start
        if c.kind in result:
            result[c.kind] += pv
        else:
            result[c.kind] = result.get(c.kind, 0.0) + pv
    return result


def _pv_cm_from_config(
    fin_cfg:   dict,
    asset_cfg: dict,
) -> float:
    """
    Compute PV of Capacity Market payments from FINANCE config.

    Uses cm_clearing prices (GBP/kW/yr) × power_mw × 1000 × cm_derating_2h
    for the auction years defined in FINANCE['cm_clearing'].

    CM payments are discounted at wacc_contracted.
    """
    wacc_c    = fin_cfg["wacc_contracted"]
    derating  = fin_cfg["cm_derating_2h"]      # legacy central-case CM de-rating assumption
    power_mw  = asset_cfg["power_mw"]

    cm_years = {
        "T1_2026_27":  0.5,   # approximate delivery year from t=0
        "T4_2028_29":  2.5,
        "T4_2029_30":  3.5,
    }

    pv = 0.0
    for key, yr in cm_years.items():
        if key in fin_cfg.get("cm_clearing", {}):
            annual_gbp_mw = fin_cfg["cm_clearing"][key]   # GBP/kW/yr → GBP/MW/yr × 1000
            annual_gbp    = annual_gbp_mw * 1000.0 * power_mw * derating
            pv += annual_gbp * discount_factor(wacc_c, yr)
    return pv


# ---------------------------------------------------------------------------
# Revenue floor optionality
# ---------------------------------------------------------------------------

def _pv_floor_optionality(
    pv_merchant_paths: np.ndarray,  # (N,) lifetime merchant PV per path
    floor_gbp_mw_yr:   float,
    power_mw:          float,
    floor_share:       float,
    wacc:              float,
    life_years:        int,
) -> float:
    """
    Value of a revenue floor put option.

    The floor guarantees a minimum annual revenue.  Using the cross-sectional
    distribution of life-time PV paths, compute:

        floor_pv_total = floor_gbp_mw_yr × power_mw × annuity(wacc, N)
        shortfall_paths = max(0, floor_total − lifetime_merchant_path)
        optionality = floor_share × E[shortfall_paths]

    This is a simplified 'put on the portfolio PV' approach.  A more accurate
    'put on each year's cashflow' would require annual-resolution cashflow paths.
    """
    floor_total = (
        floor_gbp_mw_yr * power_mw
        * (1.0 - (1.0 + wacc) ** (-life_years)) / wacc
    )
    shortfall = np.maximum(0.0, floor_total - pv_merchant_paths)
    return floor_share * float(np.mean(shortfall))


# ---------------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------------

def aggregate_mtm(
    val_result,            # ValuationResult from src.optimisation.lsmc
    asset_cfg:   dict,
    fin_cfg:     dict,
    deg_cfg:     dict,
    contracts:   Optional[List[ContractSpec]] = None,
    verbose:     bool = True,
) -> MtmComponents:
    """
    Assemble full life-time MTM from LSMC year-1 result and config.

    Parameters
    ----------
    val_result   : ValuationResult — LSMC forward pass output (year 1)
    asset_cfg    : ASSET dict from src.config
    fin_cfg      : FINANCE dict from src.config
    deg_cfg      : DEGRADATION dict from src.config
    contracts    : list of ContractSpec (optional bilateral contracts)
    verbose      : print component summary

    Returns
    -------
    MtmComponents with full breakdown
    """
    if contracts is None:
        contracts = []

    # --- Parameters ---
    wacc_m    = fin_cfg["wacc_merchant"]
    wacc_c    = fin_cfg["wacc_contracted"]
    decay     = fin_cfg.get("revenue_decay_per_year", 0.015)
    alpha     = fin_cfg.get("alpha_merchant", 1.0)
    life_years = asset_cfg.get("life_years", 15)
    power_mw   = asset_cfg["power_mw"]
    energy_mwh = asset_cfg["energy_mwh"]
    fee_frac   = asset_cfg.get("optimiser_fee_frac", 0.12)
    dt_h       = 0.5   # half-hour

    # --- Annuity factor (year-1 → life-time) ---
    af = annuity_factor_decay(wacc_m, decay, life_years)

    # --- 1. Merchant PV (multi-year) ---
    pv_m_paths = _pv_merchant_multiyear(
        val_result.pv_paths, wacc_m, decay, life_years
    )

    # --- 2. Optimiser fee ---
    fee_paths = _pv_optimiser_fee(
        val_result.cashflow_paths,
        dt_h=dt_h, wacc=wacc_m,
        fee_frac=fee_frac, annuity=af,
    )

    # --- 3. Contract legs (deterministic) ---
    contract_pvs = _pv_contracted_legs(contracts)
    pv_toll  = contract_pvs.get("toll",  0.0)
    pv_floor = contract_pvs.get("floor", 0.0)
    pv_cm    = _pv_cm_from_config(fin_cfg, asset_cfg)

    # --- 4. Floor optionality ---
    floor_opt = 0.0
    if fin_cfg.get("floor_anchor_gbp_mw_yr", 0.0) > 0:
        floor_opt = _pv_floor_optionality(
            pv_m_paths,
            floor_gbp_mw_yr = fin_cfg["floor_anchor_gbp_mw_yr"],
            power_mw        = power_mw,
            floor_share     = fin_cfg.get("floor_share_owner", 0.55),
            wacc            = wacc_c,
            life_years      = life_years,
        )

    # --- 5. Fixed opex ---
    pv_opex = _pv_opex_fixed(asset_cfg, fin_cfg, life_years)

    # --- 6. Augmentation ---
    pv_aug = _pv_augmentation(asset_cfg, fin_cfg)

    # --- 7. Per-path MTM total ---
    # Stochastic components: merchant + optimiser fee
    # Deterministic components: contracts, floor opt, opex, augmentation
    det_components = (
        pv_toll + pv_floor + pv_cm
        + floor_opt + pv_opex + pv_aug
    )

    mtm_paths = (
        alpha * pv_m_paths
        + (1.0 - alpha) * (pv_toll + pv_floor + pv_cm)  # per-path if stochastic
        + fee_paths                                       # per-path
        + pv_opex + pv_aug                               # scalar broadcast
    )
    mtm_paths += floor_opt   # add expected floor value (scalar)

    # --- 8. Statistics ---
    result = MtmComponents(
        pv_merchant_mean   = float(np.mean(pv_m_paths)),
        pv_merchant_std    = float(np.std(pv_m_paths)),
        pv_merchant_p5     = float(np.percentile(pv_m_paths,  5)),
        pv_merchant_p95    = float(np.percentile(pv_m_paths, 95)),
        pv_merchant_paths  = pv_m_paths,

        pv_toll            = pv_toll,
        pv_floor           = pv_floor,
        pv_cm              = pv_cm,
        pv_floor_optionality = floor_opt,

        pv_optimiser_fee   = float(np.mean(fee_paths)),
        pv_opex_fixed      = pv_opex,
        pv_augmentation    = pv_aug,

        mtm_mean  = float(np.mean(mtm_paths)),
        mtm_std   = float(np.std(mtm_paths)),
        mtm_p5    = float(np.percentile(mtm_paths,  5)),
        mtm_p95   = float(np.percentile(mtm_paths, 95)),
        mtm_paths = mtm_paths,

        alpha_merchant  = alpha,
        life_years      = life_years,
        power_mw        = power_mw,
        energy_mwh      = energy_mwh,
        annuity_factor  = af,
    )

    if verbose:
        s = result.summary_gbp_mw_yr()
        print(f"\n{'='*55}")
        print(f"  MTM Summary — {power_mw:.0f} MW / {energy_mwh:.0f} MWh BESS")
        print(f"  Life: {life_years} yr  |  alpha_merchant: {alpha:.0%}")
        print(f"  Annuity factor (wacc={wacc_m:.0%}, decay={decay:.1%}): {af:.2f}x")
        print(f"{'='*55}")
        for label, val in s.items():
            sign = "+" if val >= 0 else ""
            print(f"  {label:<28s}  {sign}{val:>9,.0f}  GBP/MW/yr")
        print(f"{'='*55}\n")

    return result


# ---------------------------------------------------------------------------
# Convenience: build default CM contracts from config
# ---------------------------------------------------------------------------

def default_contracts_from_config(
    fin_cfg:   dict,
    asset_cfg: dict,
) -> List[ContractSpec]:
    """
    Build a minimal list of CM ContractSpec objects from FINANCE config.

    In base case (alpha_merchant = 1.0) these have zero weight in the MTM,
    but are retained so the framework supports partial contracting.
    """
    return []   # base case: fully merchant


# ---------------------------------------------------------------------------
# Scenario: bump price level and revalue MTM analytically
# ---------------------------------------------------------------------------

def bump_merchant_mtm(
    mtm: MtmComponents,
    bump_frac: float,
) -> MtmComponents:
    """
    Apply a fractional bump to the merchant PV (for quick sensitivity check).
    Returns a new MtmComponents with scaled merchant paths.

    NOT a substitute for full LSMC re-solve.  Used for first-order Greeks
    when the dispatch action is held fixed.
    """
    import copy
    bumped = copy.deepcopy(mtm)
    scale = 1.0 + bump_frac

    bumped.pv_merchant_paths  = mtm.pv_merchant_paths * scale
    bumped.pv_merchant_mean  *= scale
    bumped.pv_merchant_std   *= scale
    bumped.pv_merchant_p5    *= scale
    bumped.pv_merchant_p95   *= scale

    bumped.mtm_paths  = mtm.mtm_paths * scale   # approx
    bumped.mtm_mean  *= scale
    bumped.mtm_std   *= scale
    bumped.mtm_p5    *= scale
    bumped.mtm_p95   *= scale

    return bumped
