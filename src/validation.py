"""Validation helpers for the BESS stochastic valuation workflow.

The checks in this module are intentionally small and explicit. They are meant
to catch broken assumptions at phase boundaries before a notebook or valuation
summary turns them into plausible-looking numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class ValidationResult:
    """Container for validation errors, warnings, and numeric diagnostics."""

    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.metrics.update({f"{other.name}.{k}": v for k, v in other.metrics.items()})
        return self

    def raise_if_failed(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {msg}" for msg in self.errors)
            raise ValueError(f"{self.name} validation failed:\n{joined}")

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"{self.name}: {status}"]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - {msg}" for msg in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {msg}" for msg in self.warnings)
        return "\n".join(lines)


def _require_keys(cfg: dict[str, Any], keys: Iterable[str], result: ValidationResult) -> None:
    for key in keys:
        if key not in cfg:
            result.add_error(f"missing required config key: {key}")


def validate_asset_config(asset_cfg: dict[str, Any]) -> ValidationResult:
    """Validate the physical battery envelope."""
    result = ValidationResult("asset_config")
    _require_keys(
        asset_cfg,
        ["power_mw", "energy_mwh", "eta_charge", "eta_discharge", "soc_min_frac", "soc_max_frac"],
        result,
    )
    if result.errors:
        return result

    power = float(asset_cfg["power_mw"])
    energy = float(asset_cfg["energy_mwh"])
    eta_c = float(asset_cfg["eta_charge"])
    eta_d = float(asset_cfg["eta_discharge"])
    soc_min = float(asset_cfg["soc_min_frac"])
    soc_max = float(asset_cfg["soc_max_frac"])

    result.metrics.update(
        power_mw=power,
        energy_mwh=energy,
        duration_h=energy / power if power else float("nan"),
        soc_min_frac=soc_min,
        soc_max_frac=soc_max,
    )

    if power <= 0:
        result.add_error("power_mw must be positive")
    if energy <= 0:
        result.add_error("energy_mwh must be positive")
    if not (0.0 < eta_c <= 1.0):
        result.add_error("eta_charge must be in (0, 1]")
    if not (0.0 < eta_d <= 1.0):
        result.add_error("eta_discharge must be in (0, 1]")
    if not (0.0 <= soc_min < soc_max <= 1.0):
        result.add_error("SoC bounds must satisfy 0 <= soc_min_frac < soc_max_frac <= 1")
    if power > 0 and energy / power < 0.25:
        result.add_warning("duration is below 15 minutes; check energy_mwh/power_mw")
    return result


def validate_path_bundle(
    bundle: Any,
    *,
    forward_anchor_gbp_mwh: float | None = None,
    require_anchor: bool = True,
    price_min_gbp_mwh: float = 0.0,
    price_max_gbp_mwh: float = 500.0,
) -> ValidationResult:
    """Validate a simulated PathBundle-like object."""
    result = ValidationResult("path_bundle")

    arrays = {
        "chi": bundle.chi,
        "xi": bundle.xi,
        "ln_P_base": bundle.ln_P_base,
        "lam": bundle.lam,
        "delta_imb": bundle.delta_imb,
    }
    expected_2d = (int(bundle.n_paths), int(bundle.n_steps) + 1)

    for name, arr in arrays.items():
        arr_np = np.asarray(arr)
        if name != "lam" and arr_np.shape != expected_2d:
            result.add_error(f"{name} shape {arr_np.shape} != {expected_2d}")
        if not np.all(np.isfinite(arr_np)):
            result.add_error(f"{name} contains non-finite values")

    if "DC_Low" not in bundle.pi:
        result.add_error("bundle.pi is missing DC_Low")
    for product, arr in bundle.pi.items():
        arr_np = np.asarray(arr)
        if arr_np.shape != expected_2d:
            result.add_error(f"pi[{product}] shape {arr_np.shape} != {expected_2d}")
        if not np.all(np.isfinite(arr_np)):
            result.add_error(f"pi[{product}] contains non-finite values")
        if np.nanmin(arr_np) < 0.0:
            result.add_error(f"pi[{product}] contains negative prices")

    if not hasattr(bundle, 'pi_bm'):
        result.add_error("bundle is missing pi_bm")
    else:
        pi_bm_arr = np.asarray(bundle.pi_bm)
        if pi_bm_arr.shape != expected_2d:
            result.add_error(f"pi_bm shape {pi_bm_arr.shape} != {expected_2d}")
        if not np.all(np.isfinite(pi_bm_arr)):
            result.add_error("pi_bm contains non-finite values")
        if np.nanmin(pi_bm_arr) < 0.0:
            result.add_error("pi_bm contains negative prices")

    if not result.errors:
        p_da = np.exp(np.clip(np.asarray(bundle.ln_P_base), -100.0, np.log(price_max_gbp_mwh)))
        result.metrics["spot_mean_gbp_mwh"] = float(np.mean(p_da))
        result.metrics["spot_std_gbp_mwh"] = float(np.std(p_da))
        result.metrics["spot_min_gbp_mwh"] = float(np.min(p_da))
        result.metrics["spot_max_gbp_mwh"] = float(np.max(p_da))

        if result.metrics["spot_min_gbp_mwh"] < price_min_gbp_mwh:
            result.add_error("spot prices below configured minimum")
        if result.metrics["spot_max_gbp_mwh"] > price_max_gbp_mwh + 1e-5:
            result.add_error("spot prices above configured maximum")

        if forward_anchor_gbp_mwh is not None and require_anchor:
            p0 = np.exp(np.clip(np.asarray(bundle.ln_P_base)[:, 0], -100.0, np.log(price_max_gbp_mwh)))
            p0_mean = float(np.mean(p0))
            result.metrics["spot_initial_mean_gbp_mwh"] = p0_mean
            if abs(p0_mean - float(forward_anchor_gbp_mwh)) > max(10.0, 0.25 * forward_anchor_gbp_mwh):
                result.add_error(
                    "initial spot level is not anchored near forward_anchor_gbp_mwh; "
                    "check xi_0"
                )

    return result


def validate_policy(policy: Any) -> ValidationResult:
    """Validate a Policy-like object and its LSMC diagnostics."""
    result = ValidationResult("policy")
    beta = np.asarray(policy.beta)
    if beta.ndim != 4:
        result.add_error(f"policy.beta must be 4D, got shape {beta.shape}")
        return result
    if not np.all(np.isfinite(beta)):
        result.add_error("policy.beta contains non-finite coefficients")

    result.metrics["beta_abs_max"] = float(np.max(np.abs(beta))) if beta.size else 0.0
    if result.metrics["beta_abs_max"] > 1e8:
        result.add_error("policy.beta has explosive coefficients above 1e8")

    diagnostics = getattr(policy, "diagnostics", {}) or {}
    for key, value in diagnostics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            result.metrics[f"diagnostics_{key}"] = float(value)

    if diagnostics.get("nonfinite_beta_count", 0):
        result.add_error("LSMC regression produced non-finite coefficients")
    if diagnostics.get("regression_count", 0) <= 0:
        result.add_error("LSMC policy contains no fitted regressions")
    if diagnostics.get("fallback_zero_count", 0):
        result.add_warning("LSMC regression used zero-coefficient fallback at least once")
    if diagnostics.get("fallback_lstsq_count", 0):
        result.add_warning("LSMC regression used least-squares fallback at least once")
    if diagnostics.get("continuation_clip_fraction_max", 0.0) > 0.01:
        result.add_warning("LSMC continuation values were clipped for more than 1% of paths")
    if diagnostics.get("sample_condition_max", 0.0) > 1e10:
        result.add_warning("sampled LSMC regressions are very ill-conditioned")
    if diagnostics.get("sample_rank_deficient_count", 0):
        result.add_warning("sampled LSMC regressions include rank-deficient basis matrices")
    return result


def validate_valuation_result(result_obj: Any, asset_cfg: dict[str, Any]) -> ValidationResult:
    """Validate LSMC forward valuation outputs."""
    result = ValidationResult("valuation_result")
    pv = np.asarray(result_obj.pv_paths)
    cf = np.asarray(result_obj.cashflow_paths)
    soc = np.asarray(result_obj.soc_paths)
    soh = np.asarray(result_obj.soh_paths)
    actions = np.asarray(result_obj.action_paths)

    for name, arr in [("pv_paths", pv), ("cashflow_paths", cf), ("soc_paths", soc), ("soh_paths", soh)]:
        if not np.all(np.isfinite(arr)):
            result.add_error(f"{name} contains non-finite values")
    if actions.ndim != 2:
        result.add_error("action_paths must be 2D")
    elif actions.size:
        if not np.issubdtype(actions.dtype, np.integer):
            result.add_error("action_paths must contain integer mode indexes")
        if np.min(actions) < 0:
            result.add_error("action_paths contains negative mode indexes")
        unique_actions, counts = np.unique(actions, return_counts=True)
        dominant_fraction = float(np.max(counts) / actions.size)
        result.metrics["unique_action_count"] = float(len(unique_actions))
        result.metrics["dominant_action_fraction"] = dominant_fraction
        result.metrics["action_min"] = float(np.min(actions))
        result.metrics["action_max"] = float(np.max(actions))
        if len(unique_actions) <= 1:
            result.add_warning("forward policy selected only one action across all paths and steps")
        if dominant_fraction > 0.98:
            result.add_warning("forward policy is dominated by one action across more than 98% of decisions")

    result.metrics["pv_mean"] = float(np.mean(pv)) if pv.size else float("nan")
    result.metrics["pv_std"] = float(np.std(pv)) if pv.size else float("nan")
    if pv.size and float(np.std(pv)) <= 0.0:
        result.add_warning("pv_paths have zero cross-path variance")

    e_min = float(asset_cfg["soc_min_frac"]) * float(asset_cfg["energy_mwh"])
    e_max = float(asset_cfg["soc_max_frac"]) * float(asset_cfg["energy_mwh"])
    if soc.size:
        result.metrics["soc_min_mwh"] = float(np.min(soc))
        result.metrics["soc_max_mwh"] = float(np.max(soc))
        if np.min(soc) < e_min - 1e-3:
            result.add_error("SoC path goes below physical lower bound")
        if np.max(soc) > e_max + 1e-3:
            result.add_error("SoC path exceeds nameplate upper bound")
    if soh.size:
        result.metrics["soh_min"] = float(np.min(soh))
        result.metrics["soh_max"] = float(np.max(soh))
        if np.min(soh) < 0.0 or np.max(soh) > 1.0 + 1e-6:
            result.add_error("SoH path outside [0, 1]")
        soh_diff = np.diff(soh, axis=1)
        if np.max(soh_diff) > 1e-6:
            result.add_warning("SoH increases in at least one step; check augmentation/degradation logic")
    return result


def summarize_action_distribution(
    action_paths: Any,
    modes: Iterable[Any],
    cashflow_paths: Any | None = None,
) -> dict[str, Any]:
    """Summarise chosen dispatch modes for saved LSMC diagnostics."""
    actions = np.asarray(action_paths)
    cashflows = None if cashflow_paths is None else np.asarray(cashflow_paths)
    modes_list = list(modes)
    if actions.ndim != 2:
        raise ValueError(f"action_paths must be 2D, got shape {actions.shape}")
    if cashflows is not None and cashflows.shape != actions.shape:
        raise ValueError(
            f"cashflow_paths shape {cashflows.shape} must match action_paths shape {actions.shape}"
        )
    if not modes_list:
        raise ValueError("modes must contain at least one dispatch mode")

    flat = actions.ravel()
    if flat.size == 0:
        return {
            "total_decisions": 0,
            "unique_action_count": 0,
            "dominant_action_index": None,
            "dominant_action_fraction": 0.0,
            "cashflow_mean_gbp": 0.0,
            "by_mode": [],
            "by_net_frac": {},
        }
    if not np.issubdtype(flat.dtype, np.integer):
        raise ValueError("action_paths must contain integer mode indexes")
    if int(np.min(flat)) < 0 or int(np.max(flat)) >= len(modes_list):
        raise ValueError("action_paths contains mode indexes outside the supplied mode list")

    counts = np.bincount(flat.astype(int), minlength=len(modes_list))
    total = int(counts.sum())
    dominant_idx = int(np.argmax(counts))
    by_net_frac: dict[str, dict[str, float | int]] = {}
    by_mode = []
    charge_count = 0
    discharge_count = 0
    idle_count = 0

    for idx, (mode, count_raw) in enumerate(zip(modes_list, counts)):
        count = int(count_raw)
        fraction = float(count / total) if total else 0.0
        net_frac = float(getattr(mode, "net_frac"))
        dc_frac = float(getattr(mode, "r_dc_frac"))
        qr_frac = float(getattr(mode, "r_qr_frac"))
        by_mode.append(
            {
                "index": idx,
                "net_frac": net_frac,
                "r_dc_frac": dc_frac,
                "r_qr_frac": qr_frac,
                "count": count,
                "fraction": fraction,
                "cashflow_mean_gbp": (
                    float(np.mean(cashflows[actions == idx]))
                    if cashflows is not None and count
                    else None
                ),
            }
        )

        key = f"{net_frac:+.2f}"
        bucket = by_net_frac.setdefault(key, {"count": 0, "fraction": 0.0})
        bucket["count"] = int(bucket["count"]) + count
        if net_frac < 0.0:
            charge_count += count
        elif net_frac > 0.0:
            discharge_count += count
        else:
            idle_count += count

    for bucket in by_net_frac.values():
        bucket["fraction"] = float(int(bucket["count"]) / total) if total else 0.0

    return {
        "total_decisions": total,
        "mode_count": len(modes_list),
        "unique_action_count": int(np.count_nonzero(counts)),
        "dominant_action_index": dominant_idx,
        "dominant_action_fraction": float(counts[dominant_idx] / total),
        "charge_fraction": float(charge_count / total),
        "discharge_fraction": float(discharge_count / total),
        "idle_fraction": float(idle_count / total),
        "cashflow_mean_gbp": float(np.mean(cashflows)) if cashflows is not None else None,
        "by_mode": by_mode,
        "by_net_frac": by_net_frac,
    }
