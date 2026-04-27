"""
src/attribution/pnl_explain.py
================================
Daily P&L decomposition for BESS LSMC MTM.

Decomposes ΔMTM into:

    ΔMTM  =  Θ (theta, time decay)
           +  Σ_k Greek_k × ΔFactor_k    (delta-explain)
           +  [Realised CF − E[CF]]       (execution surprise)
           +  ΔSoH_actual − ΔSoH_model   (degradation surprise)
           +  Calibration effect          (MTM change from re-calibration)
           +  Residual                    (< 5% of |ΔMTM| target)

Reference: CLAUDE.md §Daily P&L attribution

Factor changes for delta-explain
---------------------------------
    ΔF_baseload   : change in baseload forward (GBP/MWh)
    Δσ_da         : change in DA vol (fraction)
    ΔE[Δ_imb]    : change in imbalance mean (GBP/MWh)
    Δσ_imb        : change in imbalance vol (fraction)
    ΔE[π_DC]     : change in DC clearing (GBP/MW/h)
    ΔE[π_QR]     : change in QR clearing (GBP/MW/h)
    Δγ            : change in saturation exponent
    Δr_disc       : change in discount rate (fraction)
    Δavail        : change in availability (fraction)

Usage
-----
    explainer = PnlExplainer(greeks, mtm_base)
    result = explainer.explain(
        mtm_end, realised_cf, expected_cf,
        soh_actual, soh_model,
        factor_changes, calibration_effect
    )
    result.print_waterfall()
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FactorChanges:
    """Observed market factor changes for one P&L attribution period."""
    # Price factors
    delta_baseload_gbp_mwh:  float = 0.0   # ΔF_baseload
    delta_peak_twist_gbp_mwh: float = 0.0  # Δ(F_peak - F_off)
    delta_pc1_sigma:          float = 0.0  # ΔPC1 in units of σ
    delta_pc2_sigma:          float = 0.0  # ΔPC2 in units of σ

    # Vol factors
    delta_vega_da_frac:       float = 0.0  # Δσ_DA / σ_DA (fraction)
    delta_vega_id_frac:       float = 0.0  # Δσ_ID / σ_ID

    # Imbalance
    delta_imb_drift_gbp_mwh:  float = 0.0  # ΔE[Δ]
    delta_imb_vol_frac:       float = 0.0  # Δσ_Δ / σ_Δ

    # Ancillary
    delta_dc_gbp_mwh:         float = 0.0  # ΔE[π_DC]
    delta_qr_gbp_mwh:         float = 0.0  # ΔE[π_QR]

    # Structural
    delta_saturation:         float = 0.0  # Δγ
    delta_skip_rate_pp:       float = 0.0  # Δskip (pp)

    # Asset
    delta_soh_rate_frac:      float = 0.0  # Δdeg_rate / deg_rate
    delta_rte_pp:             float = 0.0  # ΔηRTE (pp)
    delta_avail_pp:           float = 0.0  # Δavail (pp)

    # Discount
    delta_rho_bps:            float = 0.0  # Δr (basis points)

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


@dataclass
class PnlExplainResult:
    """Full P&L attribution for one period."""
    # Inputs
    mtm_start:   float   # GBP
    mtm_end:     float   # GBP
    delta_mtm:   float   # = mtm_end - mtm_start

    # Attribution components (GBP)
    theta:                float = 0.0   # time decay
    delta_explain:        Dict[str, float] = field(default_factory=dict)
    total_delta_explain:  float = 0.0
    execution_surprise:   float = 0.0   # actual - expected CF
    degradation_surprise: float = 0.0   # ΔSoH_actual - ΔSoH_model (in GBP)
    calibration_effect:   float = 0.0   # MTM change from recalibration
    residual:             float = 0.0   # unexplained

    # Metadata
    period_h:    float = 24.0   # attribution period in hours
    residual_pct: float = 0.0   # |residual| / |delta_mtm|

    def waterfall_dict(self) -> Dict[str, float]:
        """Ordered dict for waterfall chart."""
        out = {"MTM start": self.mtm_start, "Theta": self.theta}
        out.update({f"Δ {k}": v for k, v in self.delta_explain.items()})
        out["Execution surprise"] = self.execution_surprise
        out["Degradation surprise"] = self.degradation_surprise
        out["Calibration"] = self.calibration_effect
        out["Residual"] = self.residual
        out["MTM end"] = self.mtm_end
        return out

    def print_waterfall(self) -> None:
        """Pretty-print P&L attribution waterfall."""
        print(f"\n{'='*62}")
        print(f"  P&L Attribution ({self.period_h:.0f}h period)")
        print(f"{'='*62}")
        print(f"  {'Component':<30}  {'GBP':>12}  {'% of ΔMTM':>10}")
        print(f"  {'-'*30}  {'-'*12}  {'-'*10}")

        total_dm = self.delta_mtm if self.delta_mtm != 0 else 1.0

        rows = [
            ("Theta (time decay)",         self.theta),
        ]
        for k, v in self.delta_explain.items():
            rows.append((f"  Δ {k}", v))
        rows += [
            ("Execution surprise",          self.execution_surprise),
            ("Degradation surprise",        self.degradation_surprise),
            ("Calibration effect",          self.calibration_effect),
            ("Residual",                    self.residual),
        ]

        for name, val in rows:
            sign = "+" if val >= 0 else ""
            pct  = val / total_dm * 100
            print(f"  {name:<30}  {sign}{val:>11,.0f}  {pct:>+9.1f}%")

        print(f"  {'─'*30}  {'─'*12}  {'─'*10}")
        dm = self.delta_mtm
        sign = "+" if dm >= 0 else ""
        print(f"  {'ΔMTM':<30}  {sign}{dm:>11,.0f}  {'100.0%':>10}")
        print(f"  Residual as % of |ΔMTM|: {self.residual_pct:.2%}")
        print(f"{'='*62}\n")

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "mtm_start":            self.mtm_start,
            "mtm_end":              self.mtm_end,
            "delta_mtm":            self.delta_mtm,
            "theta":                self.theta,
            "delta_explain":        self.delta_explain,
            "total_delta_explain":  self.total_delta_explain,
            "execution_surprise":   self.execution_surprise,
            "degradation_surprise": self.degradation_surprise,
            "calibration_effect":   self.calibration_effect,
            "residual":             self.residual,
            "residual_pct":         self.residual_pct,
        }
        return d


# ---------------------------------------------------------------------------
# Theta (time decay)
# ---------------------------------------------------------------------------

def compute_theta(
    mtm_mean:  float,
    wacc:      float,
    dt_h:      float = 24.0,   # period in hours (24h for daily theta)
) -> float:
    """
    Theta: MTM decay from the passage of time alone.

    For a discounted PV, dV/dt = -r × V (continuous time).
    Discrete approximation: Θ = -wacc × MTM × dt / 8760.
    """
    dt_yr = dt_h / 8760.0
    return -wacc * mtm_mean * dt_yr


# ---------------------------------------------------------------------------
# Delta-explain: Greek × ΔFactor
# ---------------------------------------------------------------------------

def compute_delta_explain(
    greeks:         Dict[str, Any],   # from greeks_to_dict()
    factor_changes: FactorChanges,
) -> Dict[str, float]:
    """
    Compute first-order P&L attribution from Greeks × factor changes.

    P&L_k = Greek_k × ΔFactor_k

    Mapping between FactorChanges fields and Greek names:
        delta_baseload_gbp_mwh  → delta_baseload  (GBP per GBP/MWh bump)
        delta_vega_da_frac      → vega_da          (GBP per 10% bump × frac/0.10)
        delta_dc_gbp_mwh        → delta_dc         (GBP per GBP/MW/h bump)
        delta_rho_bps           → rho              (GBP per 50bps × bps/50)
        delta_avail_pp          → delta_avail      (GBP per -2pp × pp/-2)
        ...
    """
    fc = factor_changes
    explain: Dict[str, float] = {}

    def _apply(greek_name: str, factor_change: float, bump_ref: float) -> None:
        """
        Compute explain contribution.
        greek = ΔMTM / bump_ref → ΔMTM_factor = greek × (factor_change / bump_ref) × bump_ref
                                              = greek × factor_change
        But greek is defined as (MTM_bumped - MTM_base) / bump_size,
        so: ΔMTM = greek × factor_change.
        """
        if greek_name not in greeks:
            return
        g = greeks[greek_name]
        if isinstance(g, dict):
            greek_val = g.get("greek", 0.0)
            bump_size = g.get("bump_size", bump_ref)
        else:
            greek_val = getattr(g, "greek", 0.0)
            bump_size = getattr(g, "bump_size", bump_ref)

        if bump_size == 0:
            return

        pnl = greek_val * factor_change
        if abs(pnl) > 1:   # only include non-trivial contributions
            explain[greek_name] = pnl

    # Price-level Greeks
    _apply("delta_baseload",  fc.delta_baseload_gbp_mwh,   1.0)
    _apply("delta_pc1_shape", fc.delta_pc1_sigma,            1.0)
    _apply("delta_pc2_shape", fc.delta_pc2_sigma,            1.0)

    # Vol Greeks
    # vega_da bump_ref = 0.10 (10%); factor_change in fraction
    _apply("vega_da",         fc.delta_vega_da_frac,         0.10)
    _apply("vega_id",         fc.delta_vega_id_frac,         0.10)

    # Imbalance
    _apply("delta_imb_drift", fc.delta_imb_drift_gbp_mwh,   5.0)
    _apply("delta_imb_vol",   fc.delta_imb_vol_frac,         0.10)

    # Ancillary
    _apply("delta_dc",        fc.delta_dc_gbp_mwh,           1.0)
    _apply("delta_qr",        fc.delta_qr_gbp_mwh,           1.0)

    # Structural
    _apply("delta_saturation", fc.delta_saturation,           0.5)

    # Asset
    _apply("delta_soh",       fc.delta_soh_rate_frac,        0.20)
    _apply("delta_rte",       fc.delta_rte_pp,               -2.0)
    _apply("delta_avail",     fc.delta_avail_pp,             -2.0)

    # Discount
    _apply("rho",             fc.delta_rho_bps,              50.0)

    return explain


# ---------------------------------------------------------------------------
# PnlExplainer
# ---------------------------------------------------------------------------

class PnlExplainer:
    """
    Daily P&L attribution engine.

    Parameters
    ----------
    greeks   : dict from greeks_to_dict() — serialised Greek ladder
    asset_cfg : ASSET dict
    fin_cfg  : FINANCE dict
    """

    def __init__(
        self,
        greeks:    Dict[str, Any],
        asset_cfg: dict,
        fin_cfg:   dict,
    ) -> None:
        self.greeks    = greeks
        self.asset     = asset_cfg
        self.fin       = fin_cfg
        self.wacc      = float(fin_cfg["wacc_merchant"])

    def explain(
        self,
        mtm_start:           float,
        mtm_end:             float,
        realised_cf:         float,
        expected_cf:         float,
        soh_delta_actual:    float,    # actual ΔSoH (negative = decline)
        soh_delta_model:     float,    # model-expected ΔSoH
        factor_changes:      FactorChanges,
        calibration_effect:  float = 0.0,
        period_h:            float = 24.0,
    ) -> PnlExplainResult:
        """
        Decompose ΔMTM over one attribution period.

        Parameters
        ----------
        mtm_start          : MTM at start of period (GBP)
        mtm_end            : MTM at end of period (GBP)
        realised_cf        : actual cashflow received (GBP)
        expected_cf        : model-expected cashflow (GBP)
        soh_delta_actual   : observed change in SoH (fraction, negative)
        soh_delta_model    : model-predicted SoH change
        factor_changes     : FactorChanges object
        calibration_effect : MTM change from recalibrating model params
        period_h           : period length in hours

        Returns
        -------
        PnlExplainResult
        """
        delta_mtm = mtm_end - mtm_start

        # 1. Theta
        theta = compute_theta(mtm_start, self.wacc, period_h)

        # 2. Delta-explain
        delta_expl = compute_delta_explain(self.greeks, factor_changes)
        total_de   = sum(delta_expl.values())

        # 3. Execution surprise
        exec_surprise = realised_cf - expected_cf

        # 4. Degradation surprise
        # Convert ΔSoH difference to GBP via augmentation capex
        # degradation_surprise = (ΔSoH_actual - ΔSoH_model) × E_nameplate × augment_capex
        capex_gbp_mwh = self.asset.get("augment_gbp_kwh", 60.0) * 1000.0   # GBP/MWh
        e_name        = self.asset.get("energy_mwh", 200.0)
        deg_surprise  = (soh_delta_actual - soh_delta_model) * e_name * capex_gbp_mwh

        # 5. Residual
        attributed = theta + total_de + exec_surprise + deg_surprise + calibration_effect
        residual   = delta_mtm - attributed

        residual_pct = abs(residual) / abs(delta_mtm) if delta_mtm != 0 else float("nan")

        result = PnlExplainResult(
            mtm_start            = mtm_start,
            mtm_end              = mtm_end,
            delta_mtm            = delta_mtm,
            theta                = theta,
            delta_explain        = delta_expl,
            total_delta_explain  = total_de,
            execution_surprise   = exec_surprise,
            degradation_surprise = deg_surprise,
            calibration_effect   = calibration_effect,
            residual             = residual,
            period_h             = period_h,
            residual_pct         = residual_pct,
        )
        return result

    def explain_series(
        self,
        mtm_series:       np.ndarray,    # (D+1,) daily MTM values
        cf_realised:      np.ndarray,    # (D,) actual cashflows
        cf_expected:      np.ndarray,    # (D,) expected cashflows
        soh_actual:       np.ndarray,    # (D+1,) actual SoH
        soh_model:        np.ndarray,    # (D+1,) model SoH
        factor_changes:   List[FactorChanges],  # (D,) one per day
        calib_effects:    Optional[np.ndarray] = None,  # (D,)
        period_h:         float = 24.0,
    ) -> List[PnlExplainResult]:
        """
        Run P&L attribution over a multi-day series.

        Returns list of PnlExplainResult, one per period.
        """
        D = len(mtm_series) - 1
        if calib_effects is None:
            calib_effects = np.zeros(D)

        results = []
        for d in range(D):
            soh_d_actual = soh_actual[d+1] - soh_actual[d]
            soh_d_model  = soh_model[d+1]  - soh_model[d]

            r = self.explain(
                mtm_start          = float(mtm_series[d]),
                mtm_end            = float(mtm_series[d+1]),
                realised_cf        = float(cf_realised[d]),
                expected_cf        = float(cf_expected[d]),
                soh_delta_actual   = float(soh_d_actual),
                soh_delta_model    = float(soh_d_model),
                factor_changes     = factor_changes[d],
                calibration_effect = float(calib_effects[d]),
                period_h           = period_h,
            )
            results.append(r)

        return results


# ---------------------------------------------------------------------------
# Backtest report
# ---------------------------------------------------------------------------

def backtest_summary(
    results: List[PnlExplainResult],
    validation_cfg: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Aggregate P&L attribution results over a backtest period.

    Returns summary statistics: total ΔMTM, component contributions,
    residual statistics, and pass/fail vs validation targets.
    """
    if not results:
        return {}

    total_dm        = sum(r.delta_mtm          for r in results)
    total_theta     = sum(r.theta              for r in results)
    total_de        = sum(r.total_delta_explain for r in results)
    total_exec      = sum(r.execution_surprise for r in results)
    total_deg       = sum(r.degradation_surprise for r in results)
    total_calib     = sum(r.calibration_effect for r in results)
    total_residual  = sum(r.residual           for r in results)

    residual_pct_total = abs(total_residual) / abs(total_dm) if total_dm != 0 else float("nan")
    daily_resid_pcts   = [r.residual_pct for r in results if not np.isnan(r.residual_pct)]

    target_pct = 0.05
    if validation_cfg:
        target_pct = validation_cfg.get("pnl_residual_warning", 0.05)

    summary = {
        "n_periods":          len(results),
        "total_delta_mtm":    total_dm,
        "total_theta":        total_theta,
        "total_delta_explain": total_de,
        "total_exec_surprise": total_exec,
        "total_deg_surprise":  total_deg,
        "total_calib_effect":  total_calib,
        "total_residual":      total_residual,
        "residual_pct_total":  residual_pct_total,
        "mean_daily_residual_pct": np.mean(daily_resid_pcts) if daily_resid_pcts else float("nan"),
        "p95_daily_residual_pct":  np.percentile(daily_resid_pcts, 95) if daily_resid_pcts else float("nan"),
        "pass_residual_target": residual_pct_total < target_pct,
        "target_residual_pct":  target_pct,
    }
    return summary


def print_backtest_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print backtest attribution summary."""
    print(f"\n{'='*60}")
    print(f"  Backtest P&L Attribution Summary ({summary.get('n_periods', 0)} periods)")
    print(f"{'='*60}")
    dm = summary.get("total_delta_mtm", 0)
    print(f"  {'Total ΔMTM':<35}  GBP {dm:>12,.0f}")

    comps = [
        ("Theta",             "total_theta"),
        ("Delta-explain",     "total_delta_explain"),
        ("Execution surprise","total_exec_surprise"),
        ("Degradation surprise","total_deg_surprise"),
        ("Calibration effect","total_calib_effect"),
        ("Residual",          "total_residual"),
    ]
    for label, key in comps:
        v = summary.get(key, 0.0)
        pct = v / dm * 100 if dm else float("nan")
        sign = "+" if v >= 0 else ""
        print(f"  {label:<35}  GBP {sign}{v:>11,.0f}  ({pct:>+.1f}%)")

    print(f"\n  Residual vs target ({summary.get('target_residual_pct', 0.05):.0%}):")
    print(f"    Total residual %:     {summary.get('residual_pct_total', 0):.2%}")
    print(f"    Daily mean residual:  {summary.get('mean_daily_residual_pct', 0):.2%}")
    print(f"    Daily P95 residual:   {summary.get('p95_daily_residual_pct', 0):.2%}")
    status = "PASS ✓" if summary.get("pass_residual_target") else "FAIL ✗"
    print(f"    Status:               {status}")
    print(f"{'='*60}\n")
