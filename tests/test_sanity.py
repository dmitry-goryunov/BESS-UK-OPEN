"""
Sanity tests for the BESS stochastic valuation model.

Test A: price simulation produces finite, plausible GB prices.
Test C: MTM component signs are economically correct.

Run from the project root:
    pytest tests/test_sanity.py -v
"""

import sys
import os

import numpy as np
import pytest

# Make project root importable regardless of how pytest is launched
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processes.simulate import simulate, default_params_from_config
from src.processes.imbalance import ImbalanceParams
from src.optimisation.lsmc import run_lsmc
from src.valuation.mtm import aggregate_mtm
from src.config import ASSET, LSMC as LSMC_CFG, DEGRADATION, FINANCE, SCHWARTZ_SMITH


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


# ---------------------------------------------------------------------------
# Test C — MTM component signs
# ---------------------------------------------------------------------------

class TestMtmComponentSigns:
    """Test C: each MTM component has the correct economic sign."""

    @pytest.fixture(scope="class")
    def mtm_result(self, small_bundle):
        """Run a minimal LSMC and aggregate MTM.  Uses 5 SoC nodes to stay fast."""
        lsmc_cfg = dict(LSMC_CFG)
        lsmc_cfg["n_soc_nodes"] = 5
        lsmc_cfg["soh_nodes"]   = [1.0, 0.82]

        _, val_result = run_lsmc(
            bundle   = small_bundle,
            asset_cfg = ASSET,
            lsmc_cfg  = lsmc_cfg,
            deg_cfg   = DEGRADATION,
            fin_cfg   = FINANCE,
            verbose   = False,
        )
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
