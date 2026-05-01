"""Model status summaries for cached BESS valuation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _add(rows: list[dict[str, str]], area: str, status: str, evidence: str, action: str) -> None:
    rows.append(
        {
            "area": area,
            "status": status,
            "evidence": evidence,
            "next_action": action,
        }
    )


def build_model_status(processed_dir: str | Path) -> list[dict[str, str]]:
    """Return a compact status table from processed JSON artifacts."""
    processed = Path(processed_dir)
    rows: list[dict[str, str]] = []

    ss = _read_json(processed / "ss_params.json")
    sigma_obs = ss.get("sigma_obs")
    n_obs = ss.get("n_obs", 0)
    if not ss:
        _add(rows, "Schwartz-Smith calibration", "missing", "ss_params.json not found", "Run Phase 2 calibration.")
    elif isinstance(sigma_obs, (int, float)) and sigma_obs <= 0.0011:
        _add(
            rows,
            "Schwartz-Smith calibration",
            "synthetic/prior-driven",
            f"sigma_obs={sigma_obs:g}, n_obs={n_obs}",
            "Replace synthetic forward curve with multi-date ICE/EEX panels.",
        )
    else:
        _add(rows, "Schwartz-Smith calibration", "calibrated", f"n_obs={n_obs}", "Monitor parameter drift.")

    anc = _read_json(processed / "ancillary_params.json")
    products = anc.get("products", {}) if isinstance(anc.get("products"), dict) else {}
    product_count = len(products)
    products_with_obs = sum(
        1 for params in products.values()
        if isinstance(params, dict) and int(params.get("n_obs", 0) or 0) > 0
    )
    if not anc:
        _add(rows, "Ancillary calibration", "missing", "ancillary_params.json not found", "Run Phase 2 calibration.")
    elif product_count and products_with_obs == 0:
        _add(
            rows,
            "Ancillary calibration",
            "prior-driven",
            f"0/{product_count} products have observations",
            "Refresh NESO resource IDs or add manual CSV upload.",
        )
    else:
        _add(
            rows,
            "Ancillary calibration",
            "calibrated",
            f"{products_with_obs}/{product_count} products have observations",
            "Refit saturation by product and regime.",
        )

    sim = _read_json(processed / "sim_summary.json")
    spot = sim.get("spot_price_gbp_mwh", {}) if isinstance(sim.get("spot_price_gbp_mwh"), dict) else {}
    validation = sim.get("validation", {}) if isinstance(sim.get("validation"), dict) else {}
    failed_checks = [name for name, ok in validation.items() if not ok]
    p50 = spot.get("p50")
    if not sim:
        _add(rows, "Simulation", "missing", "sim_summary.json not found", "Run Phase 3 simulation.")
    elif failed_checks:
        _add(rows, "Simulation", "needs review", f"failed checks: {', '.join(failed_checks)}", "Inspect Phase 3 validation plots.")
    elif isinstance(p50, (int, float)) and p50 < 20:
        _add(rows, "Simulation", "stale output", f"spot P50=GBP {p50:.2f}/MWh", "Regenerate with explicit xi_0 anchor.")
    else:
        _add(rows, "Simulation", "passes sanity checks", f"spot P50=GBP {p50:.2f}/MWh", "Add negative-price regime before relying on tails.")

    lsmc = _read_json(processed / "lsmc_valuation_summary.json")
    diag = lsmc.get("lsmc_diagnostics", {}) if isinstance(lsmc.get("lsmc_diagnostics"), dict) else {}
    mtm_src = lsmc.get("mtm_gbp_annualized") or lsmc.get("mtm_gbp")
    mtm = mtm_src if isinstance(mtm_src, dict) else {}
    clip = float(diag.get("continuation_clip_fraction_max", 0.0) or 0.0)
    rank_def = int(float(diag.get("sample_rank_deficient_count", 0.0) or 0.0))
    sampled = int(float(diag.get("sampled_regression_count", 0.0) or 0.0))
    beta_abs_max = float(diag.get("beta_abs_max", 0.0) or 0.0)
    action_dist = lsmc.get("action_distribution", {})
    action_dist = action_dist if isinstance(action_dist, dict) else {}
    action_q = lsmc.get("action_q_diagnostics", {})
    action_q = action_q if isinstance(action_q, dict) else {}
    unique_actions = action_dist.get("unique_action_count")
    dominant_action_fraction = action_dist.get("dominant_action_fraction")
    charge_fraction = action_dist.get("charge_fraction")
    discharge_fraction = action_dist.get("discharge_fraction")
    ratio = lsmc.get("lsmc_ri_ratio")
    v_lsmc_gte_v_ri = lsmc.get("v_lsmc_gte_v_ri")
    ri_mean = lsmc.get("ri_mean_gbp_annualized", lsmc.get("ri_mean_gbp"))
    lsmc_mean = mtm.get("mean")
    if not lsmc:
        _add(rows, "LSMC valuation", "missing", "lsmc_valuation_summary.json not found", "Run Phase 4 valuation.")
    elif v_lsmc_gte_v_ri is False or (isinstance(ratio, (int, float)) and ratio < 1.0):
        evidence = []
        if isinstance(lsmc_mean, (int, float)):
            evidence.append(f"V_LSMC annualized mean=GBP {lsmc_mean:,.0f}/yr")
        if isinstance(ri_mean, (int, float)):
            evidence.append(f"V_RI annualized mean=GBP {ri_mean:,.0f}/yr")
        if isinstance(ratio, (int, float)):
            evidence.append(f"V_LSMC/V_RI={ratio:.2f}x")
        _add(
            rows,
            "LSMC valuation",
            "coherence warning",
            "; ".join(evidence) or "LSMC is below rolling-intrinsic benchmark",
            "Keep Phase 4 in partial mode and reconcile policy/RI artefacts before any full-mode run.",
        )
    elif isinstance(ratio, (int, float)) and ratio > 10.0:
        evidence = []
        if isinstance(lsmc_mean, (int, float)):
            evidence.append(f"V_LSMC annualized mean=GBP {lsmc_mean:,.0f}/yr")
        if isinstance(ri_mean, (int, float)):
            evidence.append(f"V_RI annualized mean=GBP {ri_mean:,.0f}/yr")
        evidence.append(f"V_LSMC/V_RI={ratio:.2f}x")
        _add(
            rows,
            "LSMC valuation",
            "benchmark warning",
            "; ".join(evidence),
            "Check RI comparability and ancillary/continuation contribution before full-mode run.",
        )
    elif beta_abs_max > 1e8:
        _add(
            rows,
            "LSMC valuation",
            "diagnostic warning",
            f"beta_abs_max={beta_abs_max:.3g}; rank_def={rank_def}/{sampled}",
            "Review basis scaling and ridge regularisation before refreshing headline economics.",
        )
    elif clip > 0.01:
        _add(
            rows,
            "LSMC valuation",
            "diagnostic warning",
            f"continuation_clip_fraction_max={clip:.1%}; rank_def={rank_def}/{sampled}",
            "Review basis scaling, SoC grid, and continuation-value diagnostics.",
        )
    elif rank_def:
        _add(
            rows,
            "LSMC valuation",
            "diagnostic warning",
            f"rank_def={rank_def}/{sampled}",
            "Increase paths or simplify/regularise basis functions.",
        )
    else:
        _add(
            rows,
            "LSMC valuation",
            "passes diagnostics",
            "no clipping or sampled rank deficiency",
            "Run out-of-sample stability checks.",
        )

    if lsmc:
        if not action_dist:
            _add(
                rows,
                "LSMC dispatch",
                "missing",
                "action_distribution missing from lsmc_valuation_summary.json",
                "Rerun Phase 4 in partial mode to persist dispatch action diagnostics.",
            )
        elif (
            isinstance(unique_actions, (int, float))
            and isinstance(dominant_action_fraction, (int, float))
            and (unique_actions <= 1 or dominant_action_fraction > 0.98)
        ):
            _add(
                rows,
                "LSMC dispatch",
                "diagnostic warning",
                f"unique_actions={int(unique_actions)}, dominant_action={dominant_action_fraction:.1%}",
                "Inspect dispatch economics and basis functions; policy is nearly degenerate.",
            )
        elif (
            isinstance(charge_fraction, (int, float))
            and isinstance(discharge_fraction, (int, float))
            and (charge_fraction < 0.01 or discharge_fraction < 0.01)
        ):
            _add(
                rows,
                "LSMC dispatch",
                "dispatch warning",
                f"charge={charge_fraction:.1%}, discharge={discharge_fraction:.1%}",
                "Inspect action Q-values and SoC feasibility; policy is one-sided.",
            )
        else:
            _add(
                rows,
                "LSMC dispatch",
                "passes diagnostics",
                (
                    f"unique_actions={int(unique_actions)}, "
                    f"dominant_action={dominant_action_fraction:.1%}, "
                    f"charge={charge_fraction:.1%}, discharge={discharge_fraction:.1%}"
                ),
                "Monitor action distribution after each Phase 4 refresh.",
            )

        selected_cf = action_q.get("selected_cashflow_mean_gbp")
        selected_cont = action_q.get("selected_continuation_mean_gbp")
        selected_gap = action_q.get("selected_q_gap_mean_gbp")
        if not action_q:
            _add(
                rows,
                "LSMC Q-values",
                "missing",
                "action_q_diagnostics missing from lsmc_valuation_summary.json",
                "Rerun Phase 4 in partial mode to persist selected Q diagnostics.",
            )
        elif isinstance(selected_cf, (int, float)) and isinstance(selected_cont, (int, float)):
            if abs(selected_cont) > 1_000_000 and abs(selected_cf) < 100:
                _add(
                    rows,
                    "LSMC Q-values",
                    "continuation warning",
                    (
                        f"selected CF=GBP {selected_cf:,.2f}; "
                        f"selected continuation=GBP {selected_cont:,.0f}"
                    ),
                    "Inspect continuation regression scale and terminal unwind in partial mode.",
                )
            else:
                gap_note = (
                    f"; Q gap=GBP {selected_gap:,.0f}"
                    if isinstance(selected_gap, (int, float))
                    else ""
                )
                _add(
                    rows,
                    "LSMC Q-values",
                    "passes diagnostics",
                    (
                        f"selected CF=GBP {selected_cf:,.2f}; "
                        f"selected continuation=GBP {selected_cont:,.0f}"
                        f"{gap_note}"
                    ),
                    "Monitor Q composition after each Phase 4 refresh.",
                )

    phase6 = _read_json(processed / "phase6_summary.json")
    dual = phase6.get("dual_bound", {}) if isinstance(phase6.get("dual_bound"), dict) else {}
    backtest = phase6.get("backtest", {}) if isinstance(phase6.get("backtest"), dict) else {}
    if not phase6:
        _add(rows, "Phase 6 validation", "missing", "phase6_summary.json not found", "Run Phase 6 backtest.")
    else:
        gap = dual.get("gap_pct")
        dual_ok = bool(dual.get("dual_ok", False))
        residual = backtest.get("residual_pct_total")
        residual_ok = bool(backtest.get("pass_residual_target", False))
        _add(
            rows,
            "Upper benchmark",
            "benchmark-only" if not dual_ok else "passes threshold",
            f"clairvoyant gap={gap:.1%}" if isinstance(gap, (int, float)) else "gap unavailable",
            "Implement true martingale-penalty Andersen-Broadie dual bound.",
        )
        _add(
            rows,
            "Backtest attribution",
            "fails target" if not residual_ok else "passes target",
            f"residual={residual:.1%}" if isinstance(residual, (int, float)) else "residual unavailable",
            "Replace synthetic cashflows with execution/market-realised validation.",
        )

    pf = _read_json(processed / "perfect_foresight_summary.json")
    if pf:
        results = pf.get("results", {}) if isinstance(pf.get("results"), dict) else {}
        _add(
            rows,
            "Perfect foresight",
            "benchmark-only",
            f"{', '.join(results.keys()) or 'no'} market runs available",
            "Compare beside LSMC as an upper benchmark, not tradable value.",
        )

    return rows
