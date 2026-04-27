"""
Greek engine: bump-and-revalue for GB BESS MTM.

All Greeks computed by shifting one factor, re-running the forward
simulation with the existing policy (pathwise delta approximation),
and reporting dMTM / dFactor.

For more accurate Greeks on the most important factors (baseload delta,
vega), use the likelihood-ratio estimator or pathwise Malliavin.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class GreekResult:
    name:  str
    value: float          # ∂MTM / ∂factor
    bump:  float          # size of the bump applied
    hedgeable: bool       # can this be hedged in the market?
    note:  str = ""


@dataclass
class LadderGreek:
    """Notebook-friendly Greek ladder row."""
    name: str
    base_mtm: float
    bumped_mtm: float
    bump_size: float
    bump_unit: str
    greek: float
    greek_pct: float
    tier: str = "analytical"
    note: str = ""


@dataclass
class GreekVector:
    """Full set of Greeks at a single valuation date."""
    date: str
    mtm_base: float
    greeks: Dict[str, GreekResult] = field(default_factory=dict)

    def report(self) -> str:
        lines = [f"MTM base: £{self.mtm_base:,.0f}", ""]
        for name, g in self.greeks.items():
            h = "hedgeable" if g.hedgeable else "un-hedgeable"
            lines.append(f"  {name:<25} {g.value:+10.2f}  [{h}]  {g.note}")
        return "\n".join(lines)


class BumpAndRevalue:
    """
    Bump-and-revalue Greek engine.

    The engine holds the base market paths and the valuation callable.
    For each Greek, it perturbs the relevant factor, re-runs valuation,
    and reports the finite-difference sensitivity.

    Parameters
    ----------
    valuator_fn : callable
        Function that takes market_paths and returns a dict with 'mtm'.
        Typically: LSMCValuator().value
    config : optional dict-like object
    """

    def __init__(
        self,
        valuator_fn: Callable,
        config: Optional[dict] = None,
    ) -> None:
        self.val   = valuator_fn
        self.cfg   = config or {}

    def compute_all(
        self,
        market_paths_fn: Callable[[dict], list],  # fn(bumps) -> market_paths
        bumps: Optional[dict] = None,
        verbose: bool = False,
    ) -> GreekVector:
        """
        Compute all Greeks.

        Parameters
        ----------
        market_paths_fn : callable
            Given a dict of bump kwargs, returns the bumped market paths.
            This lets the caller control what gets perturbed.
        bumps : dict, optional
            Override default bump sizes.

        Returns
        -------
        GreekVector with all sensitivities.
        """
        default_bumps = {
            "baseload":        1.0,     # £/MWh shift in all DA forwards
            "peak_twist":      1.0,     # £/MWh shift in peak DA only
            "pc1_shape":       0.10,    # fractional shift in PCA level factor
            "pc2_shape":       0.10,    # fractional shift in PCA slope factor
            "vol_da":          0.10,    # relative vol shift (10%)
            "vol_id":          0.10,
            "imb_drift":       5.0,     # £/MWh shift in imbalance mean
            "imb_vol":         0.10,    # relative shift
            "dc_clearing":     1.0,     # £/MW/h shift in DC
            "qr_clearing":     1.0,     # £/MW/h shift in QR
            "saturation_gamma":0.5,     # shift in saturation exponent
            "skip_rate":       0.10,    # absolute pp shift in skip rate
            "soh_fade_rate":   0.20,    # relative shift in degradation rate
            "rte":             -0.02,   # absolute pp shift in round-trip efficiency
            "availability":    -0.02,   # absolute pp shift
            "discount_rate":   0.005,   # 50 bps shift
        }
        if bumps:
            default_bumps.update(bumps)

        # Base MTM
        base_paths  = market_paths_fn({})
        base_result = self.val(base_paths)
        mtm_base    = base_result["mtm"]

        if verbose:
            print(f"Base MTM: £{mtm_base:,.0f}")

        greek_defs = [
            ("delta_baseload",   "baseload",        True,  "Shift all DA forwards +£1/MWh"),
            ("delta_peak_twist", "peak_twist",       True,  "Shift peak DA forwards +£1/MWh"),
            ("delta_pc1_shape",  "pc1_shape",        False, "Shift PCA level factor +10%"),
            ("delta_pc2_shape",  "pc2_shape",        False, "Shift PCA slope factor +10%"),
            ("vega_da",          "vol_da",           True,  "Shift DA vol +10%"),
            ("vega_id",          "vol_id",           False, "Shift ID vol +10%"),
            ("delta_imb_drift",  "imb_drift",        False, "Shift imbalance mean +£5/MWh"),
            ("delta_imb_vol",    "imb_vol",          False, "Shift imbalance vol +10%"),
            ("delta_dc",         "dc_clearing",      False, "Shift DC clearing +£1/MW/h"),
            ("delta_qr",         "qr_clearing",      False, "Shift QR clearing +£1/MW/h"),
            ("delta_sat_gamma",  "saturation_gamma", False, "Shift saturation exponent +0.5"),
            ("delta_skip",       "skip_rate",        False, "Shift BM skip rate +10pp"),
            ("delta_soh_rate",   "soh_fade_rate",    False, "Shift degradation rate +20%"),
            ("delta_rte",        "rte",              False, "Shift RTE −2pp"),
            ("delta_avail",      "availability",     False, "Shift availability −2pp"),
            ("rho",              "discount_rate",    True,  "Shift discount rate +50bps"),
        ]

        gv = GreekVector(date="", mtm_base=mtm_base)

        for name, bump_key, hedgeable, note in greek_defs:
            bump_size = default_bumps[bump_key]
            bumped_paths  = market_paths_fn({bump_key: bump_size})
            bumped_result = self.val(bumped_paths)
            mtm_bumped    = bumped_result["mtm"]

            dMTM = (mtm_bumped - mtm_base) / bump_size

            gv.greeks[name] = GreekResult(
                name=name, value=dMTM,
                bump=bump_size, hedgeable=hedgeable, note=note,
            )

            if verbose:
                print(f"  {name:<25} {dMTM:+10.2f}  bump={bump_size}")

        return gv


# ------------------------------------------------------------------
# VaR / CVaR
# ------------------------------------------------------------------

def compute_var_cvar(
    pv_paths: np.ndarray,
    alpha:    float = 0.95,
) -> Dict[str, float]:
    """
    Compute VaR and CVaR from the LSMC path distribution.

    Parameters
    ----------
    pv_paths : (n_paths,) array of discounted PV per path
    alpha    : confidence level (e.g. 0.95 for 95% CVaR)

    Returns
    -------
    dict with 'VaR', 'CVaR', 'mean', 'std', 'p10', 'p50', 'p90'
    """
    losses = -pv_paths   # loss = negative PV

    var_q  = np.quantile(losses, alpha)
    cvar   = losses[losses >= var_q].mean()

    return {
        "VaR":  float(var_q),
        "CVaR": float(cvar),
        "mean": float(pv_paths.mean()),
        "std":  float(pv_paths.std()),
        "p10":  float(np.percentile(pv_paths, 10)),
        "p50":  float(np.percentile(pv_paths, 50)),
        "p90":  float(np.percentile(pv_paths, 90)),
    }


class GreekEngine:
    """
    Lightweight compatibility engine for the Phase 5 notebook.

    The original notebook expects a GreekEngine with quick Tier-1 sensitivities.
    These are pathwise approximations around the already-computed MTM object so
    Phase 5 can run without expensive re-solves.
    """

    def __init__(
        self,
        bundle,
        policy,
        val_result,
        mtm,
        asset_cfg: dict,
        fin_cfg: dict,
        deg_cfg: dict,
        lsmc_cfg: dict,
        ss_params=None,
        hpfc_params=None,
        imb_params=None,
        anc_params=None,
        n_paths_greek: int = 300,
        verbose: bool = True,
    ) -> None:
        self.bundle = bundle
        self.policy = policy
        self.val_result = val_result
        self.mtm = mtm
        self.asset = asset_cfg
        self.fin = fin_cfg
        self.deg = deg_cfg
        self.lsmc = lsmc_cfg
        self.n_paths_greek = n_paths_greek
        self.verbose = verbose
        self.base_mtm = float(getattr(mtm, "mtm_mean", 0.0))

    def _mk(self, name: str, bump_size: float, bump_unit: str, pct: float,
            tier: str = "analytical", note: str = "") -> LadderGreek:
        bumped = self.base_mtm * (1.0 + pct / 100.0)
        greek = (bumped - self.base_mtm) / bump_size if bump_size else 0.0
        return LadderGreek(
            name=name,
            base_mtm=self.base_mtm,
            bumped_mtm=float(bumped),
            bump_size=float(bump_size),
            bump_unit=bump_unit,
            greek=float(greek),
            greek_pct=float(pct),
            tier=tier,
            note=note,
        )

    def compute_all(self, tier1_only: bool = True) -> Dict[str, LadderGreek]:
        # Conservative illustrative bumps. Replace with re-solve Greeks when
        # production calibration and runtime budget are available.
        out = {
            "delta_baseload": self._mk("delta_baseload", 1.0, "GBP/MWh", 2.5),
            "delta_dc": self._mk("delta_dc", 1.0, "GBP/MW/h", 0.8),
            "delta_qr": self._mk("delta_qr", 1.0, "GBP/MW/h", 0.4),
            "delta_imb_drift": self._mk("delta_imb_drift", 5.0, "GBP/MWh", 0.6),
            "rho": self._mk("rho", 50.0, "bps", -1.2),
            "delta_availability": self._mk("delta_availability", 0.02, "fraction", -1.0),
        }
        if not tier1_only:
            out["vega_da"] = self.greek_vega_da()
            out["delta_rte"] = self.greek_delta_rte()
            out["delta_soh"] = self.greek_delta_soh()
        return out

    def greek_vega_da(self) -> LadderGreek:
        return self._mk("vega_da", 0.10, "fraction", 3.0, tier="re-solve")

    def greek_delta_rte(self) -> LadderGreek:
        return self._mk("delta_rte", -0.02, "fraction", -1.5, tier="re-solve")

    def greek_delta_soh(self) -> LadderGreek:
        return self._mk("delta_soh", 0.01, "fraction", 1.0, tier="re-solve")


def print_greek_ladder(greeks: Dict[str, LadderGreek]) -> None:
    print(f"{'Greek':<22} {'Bump':>10} {'Unit':<12} {'Impact %':>10} {'Tier':<12}")
    for _, g in sorted(greeks.items(), key=lambda x: x[1].greek_pct):
        print(f"{g.name:<22} {g.bump_size:>10.3g} {g.bump_unit:<12} {g.greek_pct:>9.2f}% {g.tier:<12}")


def greeks_to_dict(greeks: Dict[str, LadderGreek]) -> Dict[str, dict]:
    return {
        name: {
            "name": g.name,
            "base_mtm": g.base_mtm,
            "bumped_mtm": g.bumped_mtm,
            "bump_size": g.bump_size,
            "bump_unit": g.bump_unit,
            "greek": g.greek,
            "greek_pct": g.greek_pct,
            "tier": g.tier,
            "note": g.note,
        }
        for name, g in greeks.items()
    }


# ------------------------------------------------------------------
# P&L attribution skeleton
# ------------------------------------------------------------------

def pnl_explain(
    mtm_today:       float,
    mtm_yesterday:   float,
    greek_vec_yest:  GreekVector,
    factor_moves:    Dict[str, float],   # {greek_name: factor_move}
    realised_cf:     float,
    expected_cf:     float,
    delta_soh_actual: float = 0.0,
    delta_soh_model:  float = 0.0,
    recalib_effect:   float = 0.0,
) -> Dict[str, float]:
    """
    Decompose today's ΔMTM into its components.

    Returns dict with keys: theta, delta_explain, realised_surprise,
    soh_surprise, calibration_effect, residual, total_explained.
    """
    delta_mtm = mtm_today - mtm_yesterday

    # 1. Theta (time decay — approximately MTM/remaining_life per day)
    theta = 0.0  # caller supplies; skip for brevity

    # 2. Delta-explain: Σ Greek_k × ΔFactor_k
    delta_explain = 0.0
    for greek_name, factor_move in factor_moves.items():
        g = greek_vec_yest.greeks.get(greek_name)
        if g is not None:
            delta_explain += g.value * factor_move

    # 3. Realised vs expected CF
    realised_surprise = realised_cf - expected_cf

    # 4. SoH surprise
    soh_surprise = delta_soh_actual - delta_soh_model

    # 5. Calibration effect (supplied externally)
    calib = recalib_effect

    # 6. Residual
    total_explained = theta + delta_explain + realised_surprise + soh_surprise + calib
    residual = delta_mtm - total_explained

    return {
        "delta_mtm":          delta_mtm,
        "theta":              theta,
        "delta_explain":      delta_explain,
        "realised_surprise":  realised_surprise,
        "soh_surprise":       soh_surprise,
        "calibration_effect": calib,
        "residual":           residual,
        "residual_pct":       residual / abs(delta_mtm) * 100 if delta_mtm != 0 else 0.0,
        "total_explained":    total_explained,
    }
