"""
Sanity tests for the BESS stochastic valuation model.

Test A: price simulation produces finite, plausible GB prices.
Test C: MTM component signs are economically correct.

Run from the project root:
    pytest tests/test_sanity.py -v
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import pytest

# Make project root importable regardless of how pytest is launched
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processes.simulate import PathBundle, simulate, default_params_from_config
from src.processes.imbalance import ImbalanceParams
from src.optimisation.dispatch import DEFAULT_MODES, enumerate_modes
from src.optimisation.lsmc import LSMCSolver, N_BASIS, Policy, run_lsmc
from src.optimisation.dual_bound import compute_dual_bound
from src.valuation.mtm import aggregate_mtm
from src.config import (
    ASSET, LSMC as LSMC_CFG, DEGRADATION, FINANCE, SCHWARTZ_SMITH,
    configure_asset_duration,
)
from src.model_status import build_model_status
from src.utils import find_project_root
from src.validation import (
    summarize_action_distribution,
    validate_asset_config,
    validate_path_bundle,
    validate_policy,
    validate_valuation_result,
)


# ---------------------------------------------------------------------------
# Shared fixture: small PathBundle for fast tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_bundle():
    """50 paths × 96 half-hourly steps (2 days).  Runs in < 2 s."""
    ss, hpfc, imb, anc = default_params_from_config()
    # xi_0 must be set to log(forward_anchor) so exp(chi+xi) starts near £76.7/MWh.
    # default_params_from_config() leaves xi_0=None (→ 0), giving exp(0)=£1/MWh.
    xi_init = np.full(50, np.log(SCHWARTZ_SMITH["forward_anchor_gbp_mwh"]))
    return simulate(ss, hpfc, imb, anc, n_paths=50, n_steps=96, seed=0, xi_0=xi_init)


@pytest.fixture(scope="module")
def small_lsmc_result(small_bundle):
    """Small policy/result pair reused by MTM and benchmark sanity checks."""
    lsmc_cfg = dict(LSMC_CFG)
    lsmc_cfg["n_soc_nodes"] = 5
    lsmc_cfg["soh_nodes"] = [1.0, 0.82]
    return run_lsmc(
        bundle=small_bundle,
        asset_cfg=ASSET,
        lsmc_cfg=lsmc_cfg,
        deg_cfg=DEGRADATION,
        fin_cfg=FINANCE,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Test A — Price simulation sanity
# ---------------------------------------------------------------------------

class TestPriceSimulation:
    """Test A: simulated log-prices and spot prices are finite and plausible."""

    def test_ln_P_base_no_inf(self, small_bundle):
        """ln_P_base must contain no inf or nan before taking exp."""
        assert np.all(np.isfinite(small_bundle.ln_P_base)), (
            f"ln_P_base contains non-finite values; "
            f"n_inf={np.sum(~np.isfinite(small_bundle.ln_P_base))}"
        )

    def test_spot_price_no_overflow(self, small_bundle):
        """exp(ln_P_base) must not overflow to inf (the pre-fix bug)."""
        P_da = np.exp(small_bundle.ln_P_base)
        assert np.all(np.isfinite(P_da)), (
            f"P_da overflows after exp; "
            f"max ln_P_base={small_bundle.ln_P_base.max():.2f}, "
            f"n_inf={np.sum(~np.isfinite(P_da))}"
        )

    def test_spot_price_non_negative(self, small_bundle):
        """Spot prices from log-normal simulation must be strictly positive."""
        P_da = np.exp(np.clip(small_bundle.ln_P_base, -100.0, np.log(500.0)))
        assert np.all(P_da >= 0.0), "Spot prices must be non-negative"

    def test_spot_price_plausible_mean(self, small_bundle):
        """Mean spot price should be in the plausible GB range £30–200/MWh."""
        P_da = np.exp(np.clip(small_bundle.ln_P_base, -100.0, np.log(500.0)))
        mean_price = float(P_da.mean())
        assert 30.0 <= mean_price <= 200.0, (
            f"Mean spot price £{mean_price:.1f}/MWh is outside the plausible GB "
            f"range £30–200/MWh — check SS calibration parameters"
        )

    def test_spot_price_plausible_std(self, small_bundle):
        """Price std should not exceed £200/MWh (would indicate parameter explosion)."""
        P_da = np.exp(np.clip(small_bundle.ln_P_base, -100.0, np.log(500.0)))
        std_price = float(P_da.std())
        assert std_price <= 200.0, (
            f"Spot price std £{std_price:.1f}/MWh exceeds £200 — "
            f"possible parameter explosion or missing clip"
        )

    def test_ancillary_prices_non_negative(self, small_bundle):
        """All ancillary clearing prices must be ≥ 0 (prices are floored in simulator)."""
        for product, arr in small_bundle.pi.items():
            assert np.all(arr >= 0.0), (
                f"Ancillary product {product} has negative prices "
                f"(min={arr.min():.2f})"
            )

    def test_delta_imb_finite(self, small_bundle):
        """Imbalance basis (delta) must be finite everywhere."""
        assert np.all(np.isfinite(small_bundle.delta_imb)), (
            f"delta_imb contains non-finite values; "
            f"n_nonfinite={np.sum(~np.isfinite(small_bundle.delta_imb))}"
        )

    def test_delta_imb_uses_half_hour_units(self):
        """One HH step must scale theta/sigma/lambda in half-hour units, not days."""
        ss, hpfc, _, anc = default_params_from_config()
        imb = ImbalanceParams(
            theta_delta=np.log(2.0),   # half-life = 1 half-hour
            sigma_delta=0.0,
            lambda_jump=0.0,
            jump_scale_pos=0.0,
            jump_scale_neg=0.0,
            p_pos=0.5,
            mu_delta=10.0,
        )
        delta_0 = np.full(4, 50.0)
        xi_0 = np.full(4, np.log(SCHWARTZ_SMITH["forward_anchor_gbp_mwh"]))
        bundle = simulate(
            ss, hpfc, imb, anc,
            n_paths=4,
            n_steps=1,
            dt=1 / (365 * 48),
            seed=123,
            xi_0=xi_0,
            delta_0=delta_0,
        )
        expected = 10.0 + 0.5 * (50.0 - 10.0)
        assert np.allclose(bundle.delta_imb[:, 1], expected, atol=1e-6)

    def test_path_bundle_validation_passes_for_anchored_bundle(self, small_bundle):
        result = validate_path_bundle(
            small_bundle,
            forward_anchor_gbp_mwh=SCHWARTZ_SMITH["forward_anchor_gbp_mwh"],
        )
        assert result.ok, result.summary()

    def test_path_bundle_validation_catches_missing_xi_anchor(self):
        ss, hpfc, imb, anc = default_params_from_config()
        bundle = simulate(
            ss,
            hpfc,
            imb,
            anc,
            n_paths=10,
            n_steps=4,
            seed=99,
            allow_unanchored=True,
        )
        result = validate_path_bundle(
            bundle,
            forward_anchor_gbp_mwh=SCHWARTZ_SMITH["forward_anchor_gbp_mwh"],
        )
        assert not result.ok
        assert any("xi_0" in err for err in result.errors)

    def test_simulate_requires_xi_anchor_by_default(self):
        ss, hpfc, imb, anc = default_params_from_config()
        with pytest.raises(ValueError, match="requires xi_0"):
            simulate(ss, hpfc, imb, anc, n_paths=10, n_steps=4, seed=99)


class TestValidationHelpers:
    """Validation helpers should catch bad configs and inspect LSMC outputs."""

    def test_asset_config_validation_passes_for_default_asset(self):
        result = validate_asset_config(ASSET)
        assert result.ok, result.summary()

    def test_asset_config_validation_catches_bad_soc_bounds(self):
        bad = dict(ASSET)
        bad["soc_min_frac"] = 0.95
        bad["soc_max_frac"] = 0.10
        result = validate_asset_config(bad)
        assert not result.ok
        assert any("SoC bounds" in err for err in result.errors)

    def test_policy_and_valuation_validation_pass(self, small_lsmc_result):
        policy, val_result = small_lsmc_result
        policy_check = validate_policy(policy)
        valuation_check = validate_valuation_result(val_result, ASSET)
        assert policy_check.ok, policy_check.summary()
        assert valuation_check.ok, valuation_check.summary()
        assert policy.diagnostics["regression_count"] > 0
        assert policy.diagnostics["intraday_spread_std"] > 0.0
        assert policy.diagnostics["active_feature_count_min"] >= 1
        assert policy.diagnostics["active_feature_count_max"] < len(policy.beta[0, 0, 0])
        assert policy.diagnostics["continuation_clip_fraction_max"] == 0.0
        assert valuation_check.metrics["unique_action_count"] > 1
        assert valuation_check.metrics["dominant_action_fraction"] < 0.98
        action_diag = val_result.action_diagnostics
        assert action_diag["total_decisions"] == int(val_result.action_paths.size)
        assert np.isfinite(action_diag["selected_cashflow_mean_gbp"])
        assert np.isfinite(action_diag["selected_continuation_mean_gbp"])
        assert np.isfinite(action_diag["selected_q_mean_gbp"])
        assert len(action_diag["by_mode"]) == len(policy.modes)

    def test_valuation_validation_flags_degenerate_actions(self, small_lsmc_result):
        _, val_result = small_lsmc_result
        degenerate = type("DegenerateValuation", (), {})()
        degenerate.pv_paths = val_result.pv_paths
        degenerate.cashflow_paths = val_result.cashflow_paths
        degenerate.soc_paths = val_result.soc_paths
        degenerate.soh_paths = val_result.soh_paths
        degenerate.action_paths = np.zeros_like(val_result.action_paths)

        result = validate_valuation_result(degenerate, ASSET)

        assert result.ok, result.summary()
        assert result.metrics["unique_action_count"] == 1
        assert any("only one action" in warning for warning in result.warnings)

    def test_action_distribution_summary_counts_modes_and_net_buckets(self, small_lsmc_result):
        _, val_result = small_lsmc_result
        summary = summarize_action_distribution(
            val_result.action_paths,
            DEFAULT_MODES,
            val_result.cashflow_paths,
        )

        assert summary["total_decisions"] == int(val_result.action_paths.size)
        assert summary["mode_count"] == len(DEFAULT_MODES)
        assert summary["unique_action_count"] > 1
        assert 0.0 < summary["dominant_action_fraction"] < 0.98
        assert summary["charge_fraction"] > 0.0
        assert summary["discharge_fraction"] > 0.0
        assert summary["cashflow_mean_gbp"] is not None
        assert any(item["cashflow_mean_gbp"] is not None for item in summary["by_mode"])
        assert sum(item["count"] for item in summary["by_mode"]) == summary["total_decisions"]
        assert sum(item["count"] for item in summary["by_net_frac"].values()) == summary["total_decisions"]


# ---------------------------------------------------------------------------
# Test C — MTM component signs
# ---------------------------------------------------------------------------

class TestLSMCForwardStateHandling:
    """Forward policy evaluation should use actual path SoC, not the lower grid node."""

    @pytest.fixture()
    def coarse_4h_solver(self):
        asset = dict(ASSET)
        configure_asset_duration(asset, 4.0)
        cfg = dict(LSMC_CFG)
        cfg.update({
            "n_soc_nodes": 5,
            "soh_nodes": [1.0],
            "run_validation": False,
        })
        modes = enumerate_modes(
            net_levels=[0.0, 0.5],
            dc_levels=[0.0],
            qr_levels=[0.0],
        )
        return LSMCSolver(asset, cfg, DEGRADATION, FINANCE, modes=modes, verbose=False)

    def test_forward_feasibility_uses_actual_soc_between_grid_nodes(self, coarse_4h_solver):
        solver = coarse_4h_solver
        e_actual = np.array([95.0], dtype=np.float32)
        soh = np.array([1.0], dtype=np.float32)

        j_floor = np.searchsorted(solver.soc_grid, e_actual, side="right") - 1
        discharge_idx = 1

        assert not solver._feasible_jkm[j_floor[0], 0, discharge_idx]
        assert solver._feasibility_mask_for_states(e_actual, soh)[0, discharge_idx]

        e_next = solver._next_soc_for_states(e_actual, soh)[0, discharge_idx]
        expected = e_actual[0] - 0.5 * solver.P_bar / solver.eta_d * solver.dt_h
        assert np.isclose(e_next, expected)

    def test_forward_can_discharge_when_actual_soc_is_feasible(self, coarse_4h_solver):
        solver = coarse_4h_solver
        T = 1
        N = 1
        policy = Policy(
            beta=np.zeros((T, solver.n_soc, solver.n_soh, N_BASIS), dtype=np.float32),
            cont_beta=np.zeros(
                (T, solver.n_soc, solver.n_soh, len(solver.modes), N_BASIS),
                dtype=np.float32,
            ),
            soc_grid=solver.soc_grid,
            soh_nodes=solver.soh_nodes,
            modes=solver.modes,
            dt_h=solver.dt_h,
            n_steps=T,
            n_paths=N,
        )
        bundle = PathBundle(
            chi=np.zeros((N, T + 1), dtype=np.float32),
            xi=np.zeros((N, T + 1), dtype=np.float32),
            ln_P_base=np.log(np.full((N, T + 1), 1.0, dtype=np.float32)),
            lam=np.zeros((N, T + 1, 3), dtype=np.float32),
            delta_imb=np.full((N, T + 1), 500.0, dtype=np.float32),
            pi={
                "DC_Low": np.zeros((N, T + 1), dtype=np.float32),
                "QR_Pos": np.zeros((N, T + 1), dtype=np.float32),
            },
            dt=solver.dt_h / 8760.0,
            n_paths=N,
            n_steps=T,
        )

        result = solver.forward(bundle, policy, E_init_frac=95.0 / solver.E_name)

        assert result.action_paths[0, 0] == 1
        assert result.cashflow_paths[0, 0] > 0.0
        assert result.soc_paths[0, 1] < result.soc_paths[0, 0]


class TestMtmComponentSigns:
    """Test C: each MTM component has the correct economic sign."""

    @pytest.fixture(scope="class")
    def mtm_result(self, small_lsmc_result):
        """Aggregate MTM from the shared minimal LSMC run."""
        _, val_result = small_lsmc_result
        return aggregate_mtm(val_result, ASSET, FINANCE, DEGRADATION, verbose=False)

    def test_opex_is_negative(self, mtm_result):
        """Fixed O&M is a cost — must be negative."""
        assert mtm_result.pv_opex_fixed < 0, (
            f"pv_opex_fixed={mtm_result.pv_opex_fixed:,.0f} should be negative"
        )

    def test_augmentation_is_negative(self, mtm_result):
        """Augmentation capex is a cost — must be negative."""
        assert mtm_result.pv_augmentation < 0, (
            f"pv_augmentation={mtm_result.pv_augmentation:,.0f} should be negative"
        )

    def test_optimiser_fee_is_negative(self, mtm_result):
        """Optimiser fee is deducted from merchant revenue — must be negative."""
        assert mtm_result.pv_optimiser_fee < 0, (
            f"pv_optimiser_fee={mtm_result.pv_optimiser_fee:,.0f} should be negative"
        )

    def test_floor_optionality_non_negative(self, mtm_result):
        """Revenue floor put option has non-negative value by definition."""
        assert mtm_result.pv_floor_optionality >= 0.0, (
            f"pv_floor_optionality={mtm_result.pv_floor_optionality:,.0f} "
            f"should be ≥ 0 (put cannot have negative value)"
        )

    def test_merchant_mean_non_negative(self, mtm_result):
        """Merchant PV should be non-negative: a battery with positive arbitrage."""
        assert mtm_result.pv_merchant_mean >= 0.0, (
            f"pv_merchant_mean={mtm_result.pv_merchant_mean:,.0f} is negative — "
            f"battery is generating net losses before deductions; "
            f"check degradation cost vs. price spread"
        )

    def test_mtm_std_positive(self, mtm_result):
        """MTM must have cross-path variance (zero std = degenerate simulation)."""
        assert mtm_result.mtm_std > 0.0, (
            "mtm_std is zero — all paths produced identical MTM, "
            "suggesting degenerate price or dispatch simulation"
        )

    def test_annuity_factor_plausible(self, mtm_result):
        """Annuity factor should be between 1 and life_years (geometric series bounds)."""
        af = mtm_result.annuity_factor
        assert 1.0 <= af <= ASSET["life_years"], (
            f"annuity_factor={af:.3f} is outside [1, {ASSET['life_years']}]"
        )


class TestInformationRelaxationBenchmark:
    """The upper-benchmark diagnostic should be finite and never clamped to pass."""

    def test_clairvoyant_benchmark_is_not_forced_to_pass(self, small_bundle, small_lsmc_result):
        policy, val_result = small_lsmc_result
        result = compute_dual_bound(
            small_bundle,
            policy,
            val_result,
            ASSET,
            LSMC_CFG,
            DEGRADATION,
            FINANCE,
            n_dual_paths=5,
            threshold=LSMC_CFG["dual_gap_acceptable"],
            verbose=False,
        )
        assert result.n_paths == 5
        assert np.isfinite(result.v_lsmc)
        assert np.isfinite(result.v_dual)
        assert np.isfinite(result.gap_pct)
        if result.gap_abs < 0:
            assert not result.dual_ok


class TestProjectUtilities:
    """Shared utility helpers should work from nested project paths."""

    def test_find_project_root_from_notebooks_dir(self):
        root = find_project_root(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
        assert (root / "src").is_dir()
        assert (root / "data").is_dir()

    def test_phase4_notebook_uses_material_continuation_cap(self):
        """Notebook 12 must not silently clip long-duration continuation values."""
        nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "12_phase4_method_comparison.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", ""))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        )

        assert (
            "'continuation_value_cap_gbp': float(LSMC_CFG.get('continuation_value_cap_gbp', 25_000_000))"
            in source
        )
        assert "LSMC continuation clipping is material" in source


class TestModelStatus:
    """Model status summary should surface the major interpretation caveats."""

    def test_build_model_status_flags_prior_driven_and_benchmark_outputs(self, tmp_path):
        def write_json(name, data):
            (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")

        write_json("ss_params.json", {"sigma_obs": 0.001, "n_obs": 936})
        write_json(
            "ancillary_params.json",
            {"products": {"DC_Low": {"n_obs": 0}, "QR_Pos": {"n_obs": 0}}},
        )
        write_json(
            "sim_summary.json",
            {
                "spot_price_gbp_mwh": {"p50": 76.5},
                "validation": {"chi_variance": True, "xi_mean": True},
            },
        )
        write_json(
            "lsmc_valuation_summary.json",
            {
                "lsmc_diagnostics": {
                    "continuation_clip_fraction_max": 0.724,
                    "sample_rank_deficient_count": 10,
                    "sampled_regression_count": 10,
                }
            },
        )
        write_json(
            "phase6_summary.json",
            {
                "dual_bound": {"gap_pct": 4.12, "dual_ok": False},
                "backtest": {"residual_pct_total": 0.38, "pass_residual_target": False},
            },
        )
        write_json("perfect_foresight_summary.json", {"results": {"DA": {}, "SP": {}}})

        rows = build_model_status(tmp_path)
        by_area = {row["area"]: row for row in rows}

        assert by_area["Schwartz-Smith calibration"]["status"] == "synthetic/prior-driven"
        assert by_area["Ancillary calibration"]["status"] == "prior-driven"
        assert by_area["Simulation"]["status"] == "passes sanity checks"
        assert by_area["LSMC valuation"]["status"] == "diagnostic warning"
        assert by_area["Upper benchmark"]["status"] == "benchmark-only"
        assert by_area["Backtest attribution"]["status"] == "fails target"
        assert by_area["Perfect foresight"]["status"] == "benchmark-only"

    def test_build_model_status_flags_lsmc_below_rolling_intrinsic(self, tmp_path):
        def write_json(name, data):
            (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")

        write_json(
            "lsmc_valuation_summary.json",
            {
                "mtm_gbp": {"mean": 71},
                "ri_mean_gbp": 170985,
                "lsmc_ri_ratio": 0.0,
                "v_lsmc_gte_v_ri": False,
                "lsmc_diagnostics": {
                    "continuation_clip_fraction_max": 0.0,
                    "sample_rank_deficient_count": 0,
                    "sampled_regression_count": 10,
                    "beta_abs_max": 1.0e6,
                },
            },
        )

        rows = build_model_status(tmp_path)
        by_area = {row["area"]: row for row in rows}

        assert by_area["LSMC valuation"]["status"] == "coherence warning"
        assert "V_LSMC/V_RI=0.00x" in by_area["LSMC valuation"]["evidence"]
        assert "partial mode" in by_area["LSMC valuation"]["next_action"]

    def test_build_model_status_flags_high_lsmc_ri_ratio(self, tmp_path):
        def write_json(name, data):
            (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")

        write_json(
            "lsmc_valuation_summary.json",
            {
                "mtm_gbp": {"mean": 124760},
                "ri_mean_gbp": 7772,
                "lsmc_ri_ratio": 16.052,
                "v_lsmc_gte_v_ri": True,
                "action_distribution": {
                    "unique_action_count": 4,
                    "dominant_action_fraction": 0.70,
                    "charge_fraction": 0.20,
                    "discharge_fraction": 0.20,
                },
                "action_q_diagnostics": {
                    "selected_cashflow_mean_gbp": 520.0,
                    "selected_continuation_mean_gbp": 123627.0,
                    "selected_q_gap_mean_gbp": 251.0,
                },
                "lsmc_diagnostics": {
                    "continuation_clip_fraction_max": 0.0,
                    "sample_rank_deficient_count": 0,
                    "sampled_regression_count": 10,
                    "beta_abs_max": 1.0e6,
                },
            },
        )

        rows = build_model_status(tmp_path)
        by_area = {row["area"]: row for row in rows}

        assert by_area["LSMC valuation"]["status"] == "benchmark warning"
        assert "V_LSMC/V_RI=16.05x" in by_area["LSMC valuation"]["evidence"]

    def test_build_model_status_flags_one_sided_dispatch(self, tmp_path):
        def write_json(name, data):
            (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")

        write_json(
            "lsmc_valuation_summary.json",
            {
                "mtm_gbp": {"mean": 100000},
                "ri_mean_gbp": 50000,
                "lsmc_ri_ratio": 2.0,
                "v_lsmc_gte_v_ri": True,
                "action_distribution": {
                    "unique_action_count": 4,
                    "dominant_action_fraction": 0.70,
                    "charge_fraction": 1.0,
                    "discharge_fraction": 0.0,
                },
                "action_q_diagnostics": {
                    "selected_cashflow_mean_gbp": 50.0,
                    "selected_continuation_mean_gbp": 2000.0,
                    "selected_q_gap_mean_gbp": 100.0,
                },
                "lsmc_diagnostics": {
                    "continuation_clip_fraction_max": 0.0,
                    "sample_rank_deficient_count": 0,
                    "sampled_regression_count": 10,
                    "beta_abs_max": 1.0e6,
                },
            },
        )

        rows = build_model_status(tmp_path)
        by_area = {row["area"]: row for row in rows}

        assert by_area["LSMC dispatch"]["status"] == "dispatch warning"
        assert "discharge=0.0%" in by_area["LSMC dispatch"]["evidence"]

    def test_build_model_status_flags_continuation_scale_warning(self, tmp_path):
        def write_json(name, data):
            (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")

        write_json(
            "lsmc_valuation_summary.json",
            {
                "mtm_gbp": {"mean": 100000},
                "ri_mean_gbp": 50000,
                "lsmc_ri_ratio": 2.0,
                "v_lsmc_gte_v_ri": True,
                "action_distribution": {
                    "unique_action_count": 4,
                    "dominant_action_fraction": 0.70,
                    "charge_fraction": 0.20,
                    "discharge_fraction": 0.20,
                },
                "action_q_diagnostics": {
                    "selected_cashflow_mean_gbp": 0.31,
                    "selected_continuation_mean_gbp": 3058139.0,
                    "selected_q_gap_mean_gbp": 85891.0,
                },
                "lsmc_diagnostics": {
                    "continuation_clip_fraction_max": 0.0,
                    "sample_rank_deficient_count": 0,
                    "sampled_regression_count": 10,
                    "beta_abs_max": 1.0e6,
                },
            },
        )

        rows = build_model_status(tmp_path)
        by_area = {row["area"]: row for row in rows}

        assert by_area["LSMC Q-values"]["status"] == "continuation warning"
        assert "selected continuation=GBP 3,058,139" in by_area["LSMC Q-values"]["evidence"]
