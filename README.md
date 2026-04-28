# BESS-UK - Stochastic MTM Valuation

Research framework for stochastic mark-to-market valuation of a Great Britain
fast-cycle battery energy storage system (BESS), using Least-Squares Monte
Carlo (LSMC) adapted from the Boogert & de Jong (2008) gas storage approach.

**Asset**: 100 MW / 200 MWh LFP, 2h duration, 88% round-trip efficiency, GB grid.

---

## What this does

The model values a BESS as a real option: at each half-hour the operator chooses
how much power to charge, discharge, and reserve for ancillary services. LSMC
learns the optimal continuation value by backward induction, then rolls the
policy forward to produce an MTM distribution.

| Phase | Module | What it produces |
|---|---|---|
| 1 - Data | `src/data/` | Elexon DA/SP prices, NESO ancillary clearing, forward curve |
| 2 - Calibration | `src/processes/` | Schwartz-Smith two-factor, PCA hourly shape, imbalance OU+jump, ancillary AR(1) |
| 3 - Simulation | `src/processes/simulate.py` | 5,000-path x 17,520-step correlated `PathBundle` |
| 4 - LSMC | `src/optimisation/lsmc.py` | Regression coefficients, dispatch policy, MTM distribution |
| 5 - MTM / Greeks / VaR | `src/valuation/` | Lifetime MTM, Greek ladder, VaR/CVaR, scenario stresses |
| 6 - Backtest | `src/attribution/` | Dual bound gap, 30-day P&L attribution |
| 7 - Historical benchmark | `src/optimisation/perfect_foresight.py` | Full-horizon perfect-foresight DA/SP arbitrage benchmark |

Everything runs end-to-end in `notebooks/bess_valuation_full.ipynb`.

---

## Current Outputs

Latest published run: 28 April 2026.

### Phase 1 - Market Data

| Dataset | Rows | Period | Notes |
|---|---:|---|---|
| Elexon DA prices | 36,240 | Apr 2024-Apr 2026 | 2025 mean GBP 79.8/MWh, std GBP 40.3 |
| Elexon system prices | 36,240 | Apr 2024-Apr 2026 | Imbalance basis mean +GBP 0.2/MWh, std GBP 39.6 |
| NESO ancillary clearing | 757 | Apr 2024-Apr 2026 | DC GBP 1.97, DM GBP 6.24, QR GBP 4.62 /MW/h mean |
| Forward curve | 9 contracts | 2027-2030 | Synthetic, anchored to GBP 76.7/MWh |

1,032 half-hours had negative DA prices, confirming that a log-normal spot model
needs an arithmetic floor or a separate negative-price regime.

### Phase 2 - Calibration

| Process | Key parameters |
|---|---|
| Schwartz-Smith | kappa=4.02, mu_xi=0.0069, sigma_chi=2.00, sigma_xi=0.0113, rho=0.39, n=936 |
| PCA hourly shape | 3 factors, 76.7% variance explained |
| Imbalance OU+jump | theta=0.83/HH, sigma=18.65 GBP/MWh, lambda_J=0.043/HH, n=36,240 |
| Ancillary AR(1) | phi=0.85, gamma_sat=2.10, fitted to priors because current product observations are unavailable |

The Schwartz-Smith volatility split remains distorted by the synthetic forward
curve. Ancillary products still revert to priors because NESO resource IDs have
changed for the current date range.

### Phase 3 - Simulation

| Metric | Value |
|---|---:|
| Paths | 5,000 |
| Half-hour steps | 17,520 |
| Seed | 42 |
| Spot P5 / P50 / P95 | GBP 0.31 / GBP 1.00 / GBP 3.27 per MWh |
| Imbalance P5 / P50 / P95 | GBP -237.38 / GBP -22.56 / GBP 219.44 per MWh |
| DC low P5 / P50 / P95 | GBP 1.15 / GBP 5.15 / GBP 9.50 per MW/h |
| Validation checks | 7 / 7 pass |

The Phase 3 standalone spot path remains too low because `xi_0` defaults to 0 in
that cell. The LSMC valuation run sets the forward anchor explicitly.

### Phase 4 - LSMC Valuation

| Metric | Value |
|---|---:|
| Paths | 1,000 |
| Half-hour steps | 17,520 |
| Asset | 100 MW / 200 MWh |
| V_LSMC mean | GBP 3.03M |
| V_LSMC mean per MW | GBP 30.3k/MW |
| V_LSMC P5 / P50 / P95 | GBP 2.89M / GBP 3.03M / GBP 3.16M |
| Backward / forward pass | 42.6 s / 21.3 s |

The DA-only rolling-intrinsic lower-bound diagnostic is intentionally excluded
from the headline outputs because it is not comparable with historical all-in
GB BESS revenue benchmarks.

### Phase 5 - MTM, Greeks & VaR

| Component | GBP/MW/yr |
|---|---:|
| Merchant LSMC | +814 |
| Capacity Market | +1,051 |
| Floor optionality | +15,221 |
| Optimiser fee | -98 |
| Fixed O&M | -5,180 |
| Augmentation capex | -12,527 |
| Total mean | -1,769 |

| Risk metric | Value |
|---|---:|
| Lifetime MTM mean | GBP -2.65M |
| Lifetime MTM std | GBP 180k |
| VaR 95% | GBP 2.88M |
| CVaR 95% | GBP 2.93M |
| VaR 99% | GBP 2.95M |
| CVaR 99% | GBP 2.98M |

Largest current sensitivities:

| Greek | Bump | Sensitivity |
|---|---:|---:|
| delta_soh | +1 pp | GBP -2.65M per fraction |
| delta_rte | -2 pp | GBP -1.99M per fraction |
| delta_availability | +2 pp | GBP +1.33M per fraction |
| vega_da | +10 pp | GBP -796k per fraction |
| delta_baseload | +GBP 1/MWh | GBP -66k |

Scenario stresses:

| Scenario | Delta |
|---|---:|
| High price | GBP -398k |
| Low price | GBP +398k |
| High volatility | GBP -212k |
| Low ancillary | GBP +186k |
| High discount | GBP +133k |

### Phase 6 - Dual Bound & Backtest

| Metric | Value |
|---|---:|
| V_LSMC | GBP 64,471 |
| V_dual | GBP 64,471 |
| Dual gap | 0.00% |
| 30-day delta MTM | GBP 201,769 |
| 30-day residual | GBP 885,119 |
| Residual / total | 4.39% |
| Mean daily residual | 1.09% |
| P95 daily residual | 2.06% |
| Residual target | 5.00% |
| Target passed | No |
| Base SOH at year 15 | 68.9% |

The dual gap is still degenerate and should not be interpreted as a robust upper
bound. The backtest residual improved materially versus the earlier synthetic
run, but still narrowly misses the target flag because the implementation checks
a strict fractional threshold.

### Phase 7 - Historical Perfect-Foresight Benchmark

This is a deterministic upper benchmark for a 100 MW / 200 MWh 2h battery with
88% RTE, 10-90% SoC limits, VOM, and terminal SoC returned to the initial level.
It solves one LP over the full historical price path, so it is not a tradable
strategy.

| Market price series | Total value | GBP/MW/yr | Equivalent cycles |
|---|---:|---:|---:|
| Day-ahead | GBP 7.47M | GBP 36.1k | 1,063 |
| System price | GBP 22.03M | GBP 106.5k | 2,215 |

The DA and SP benchmarks are **not additive**. They are two alternative
perfect-foresight valuations using the same physical battery over the same
2.068-year historical horizon.

### Published Artifacts

| Artifact | Path |
|---|---|
| End-to-end notebook | `notebooks/bess_valuation_full.ipynb` |
| Phase notebooks | `notebooks/01_data_pipeline.ipynb` ... `notebooks/07_historical_perfect_foresight.ipynb` |
| LSMC valuation chart | `data/processed/lsmc_valuation.png` |
| LSMC summary | `data/processed/lsmc_valuation_summary.json` |
| MTM / Greeks / VaR summary | `data/processed/mtm_summary.json` |
| Dual bound / backtest summary | `data/processed/phase6_summary.json` |
| Perfect-foresight summary | `data/processed/perfect_foresight_summary.json` |
| Perfect-foresight dispatch | `data/processed/perfect_foresight_da_dispatch.parquet`, `data/processed/perfect_foresight_sp_dispatch.parquet` |
| Notebook output charts | `notebooks/mtm_components.png`, `notebooks/mtm_distribution.png`, `notebooks/greek_ladder.png`, `notebooks/var_cvar.png`, `notebooks/scenario_stress.png`, `notebooks/dual_bound.png`, `notebooks/pnl_attribution.png` |

---

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# Full end-to-end notebook
jupyter nbconvert --to notebook --execute notebooks/bess_valuation_full.ipynb

# Pre-generate the simulation bundle
python scripts/generate_sim_bundle.py --paths 1000 --steps 17520

# Quick dev bundle
python scripts/generate_sim_bundle.py --paths 200 --steps 240

# Historical perfect-foresight benchmark
python notebooks/run_historical_perfect_foresight.py

# Run sanity tests
pytest tests/test_sanity.py -v

# Streamlit dashboard
streamlit run streamlit_app.py
```

---

## Project Structure

```text
bess_project/
|-- notebooks/
|   |-- bess_valuation_full.ipynb
|   |-- 01_data_pipeline.ipynb
|   |-- 02_calibration.ipynb
|   |-- 03_simulation.ipynb
|   |-- 04_lsmc_valuation.ipynb
|   |-- 05_mtm_risk_greeks.ipynb
|   |-- 06_backtest_pnl.ipynb
|   |-- 07_historical_perfect_foresight.ipynb
|   `-- run_historical_perfect_foresight.py
|-- src/
|   |-- processes/
|   |-- optimisation/
|   |-- valuation/
|   `-- attribution/
|-- tests/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|-- scripts/
|-- streamlit_app.py
`-- README.md
```

---

## Known Issues

| # | Issue | Impact | Status |
|---:|---|---|---|
| 1 | Synthetic forward curve | Schwartz-Smith volatility split is distorted | Data gap |
| 2 | NESO API resource drift | Ancillary AR(1) reverts to priors | API change |
| 3 | `xi_0` not enforced in standalone simulation | Phase 3 spot P50 is about GBP 1/MWh | Footgun |
| 4 | LSMC regression diagnostics need hardening | V_LSMC / V_RI ratio remains inflated | Bug |
| 5 | Dual bound is degenerate | 0% gap is not a useful validation | Bug |
| 6 | Backtest still uses synthetic cashflows | Attribution is illustrative, not execution validation | Design |
| 7 | Intraday spread not simulated | Intraday premium is omitted | Feature gap |
| 8 | Negative prices require arithmetic treatment | Log-normal model clips negative-price behavior | Model gap |
| 9 | Perfect foresight is an upper benchmark | It assumes complete future price knowledge | Interpretation |

---

## Potential Improvements

1. Pull real GB baseload/peak monthly settlements from ICE WebICE or EEX and
   recalibrate Schwartz-Smith.
2. Update NESO ancillary data resource IDs or ingest CSV exports.
3. Add NaN and rank-deficiency guards in the LSMC regression loop.
4. Require a forward anchor or default `xi_0 = log(forward_anchor)` in simulation.
5. Add an arithmetic OU intraday spread process.
6. Add a negative-price regime or switch baseload to an arithmetic two-factor model.
7. Debug the Andersen-Broadie oracle path calculation and target a dual gap below 2%.
8. Connect P&L attribution to real BMU dispatch and cashflow data.
9. Fit a regime-switching ancillary saturation curve.
10. Replace year-by-year chaining with a single long-horizon simulation.
11. Replace flat Capacity Market revenue with delivery-year clearing prices.
12. Parallelise bump-and-revalue Greeks.

---

## Data Sources

| Source | What to pull | API / access |
|---|---|---|
| Elexon BMRS | DA MID prices, system prices, NIV, BOA | `api.elexon.co.uk` |
| NESO Data Portal | EAC clearing for DC/DM/DR/QR/BR | `api.nationalgrideso.com` |
| ICE / EEX | GB baseload and peak monthly forwards | ICE WebICE export or EEX transparency |
| Modo Energy | ME BESS GB Revenue Index | Subscription |
| Aurora / Baringa | Battery revenue forecasts | Subscription |

---

## References

- Boogert & de Jong (2008), LSMC for gas storage, *Journal of Derivatives* 15(3)
- Schwartz & Smith (2000), two-factor commodity model, *Management Science* 46
- Lucia & Schwartz (2002), electricity seasonality
- Nadarajah, Margot & Secomandi (2017), LSMC dual bounds, *EJOR* 256
- Finnah, Goensch & Ziel (2022), GB imbalance modelling, *EJOR* 301
- Shi, Xu & Baldick (2019), convex cycle-based degradation cost, *IEEE T-SG*
