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

Latest local refresh: 29 April 2026. Phase 4-6 are currently saved from the
`partial` Phase 4 mode: 250 paths over 240 half-hours for fast development.
Set `PHASE4_RUN_MODE = 'full'` in `notebooks/04_lsmc_valuation.ipynb` before
refreshing headline economics.

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
| Paths | 1,000 |
| Half-hour steps | 17,520 |
| Seed | 42 |
| Spot P5 / P50 / P95 | GBP 26.57 / GBP 78.55 / GBP 231.79 per MWh |
| Imbalance P5 / P50 / P95 | GBP -239.61 / GBP -24.10 / GBP 215.74 per MWh |
| DC low P5 / P50 / P95 | GBP 1.17 / GBP 5.17 / GBP 9.49 per MW/h |
| Validation checks | 7 / 7 pass |

Phase 3 now uses an explicit `xi_0` price anchor from the forward curve, avoiding
the older unanchored GBP 1/MWh spot-path artefact.

### Phase 4 - LSMC Valuation

| Metric | Value |
|---|---:|
| Paths | 1,000 |
| Half-hour steps | 17,520 |
| Asset | 100 MW / 200 MWh |
| V_LSMC mean | GBP 124.8k |
| V_LSMC mean per MW | GBP 1.25k/MW |
| V_LSMC P5 / P50 / P95 | GBP -17k / GBP 44k / GBP 515k |
| V_LSMC / V_RI | 16.05x |
| Backward / forward pass | 1.4 s / 0.1 s |

The partial run uses the 9-node SoC grid, 3 SoH nodes, and 12 dispatch modes.
It now applies a partial-mode ridge override of 1.0 and fixes the backward
`V_next` update so each time step is fitted from a read-only `V_{t+1}` surface.
Beta scale is lower (`beta_abs_max` about 1.16e6) with zero continuation
clipping in the smoke run. The high LSMC/RI ratio remains a benchmark warning,
not a production valuation.

### Phase 5 - MTM, Greeks & VaR

| Component | GBP/MW/yr |
|---|---:|
| Merchant LSMC | +145 |
| Capacity Market | +1,051 |
| Floor optionality | +15,589 |
| Optimiser fee | -62 |
| Fixed O&M | -5,180 |
| Augmentation capex | -12,527 |
| Total mean | -2,034 |

| Risk metric | Value |
|---|---:|
| Lifetime MTM mean | GBP -3.05M |
| Lifetime MTM std | GBP 431k |
| Lifetime P5 / P50 / P95 | GBP -3.59M / GBP -3.12M / GBP -2.28M |
| VaR 95% | GBP 3.59M |
| CVaR 95% | GBP 3.68M |

Largest current sensitivities:

| Greek | Bump | Sensitivity |
|---|---:|---:|
| delta_soh | +1 pp | GBP -3.05M per fraction |
| delta_rte | -2 pp | GBP -2.29M per fraction |
| delta_availability | +2 pp | GBP +1.53M per fraction |
| vega_da | +10 pp | GBP -915k per fraction |
| delta_baseload | +GBP 1/MWh | GBP -76k |

Scenario stresses:

| Scenario | Delta |
|---|---:|
| High price | GBP -458k |
| Low price | GBP +458k |
| High volatility | GBP -244k |
| Low ancillary | GBP +214k |
| High discount | GBP +153k |

### Phase 6 - Dual Bound & Backtest

| Metric | Value |
|---|---:|
| V_LSMC | GBP 33,044 |
| V_upper | GBP 187,879 |
| Upper gap | 468.57% |
| 30-day delta MTM | GBP 201,769 |
| 30-day residual | GBP 928,608 |
| Residual / total | 460.23% |
| Mean daily residual | 112.67% |
| P95 daily residual | 221.25% |
| Residual target | 5.00% |
| Target passed | No |
| Base SOH at year 15 | 68.9% |

Phase 6 reports a clairvoyant information-relaxation upper benchmark rather than
a martingale-penalty Andersen-Broadie proof. The synthetic 30-day backtest is
useful as an attribution plumbing check, but it is not an execution validation.
The residual table displays ratio fields as percentages; the raw stored
`residual_pct_total` value is 4.6023.

Phase 5-6 have not yet been refreshed after the latest partial Phase 4
continuation-surface fix, so treat them as stale relative to the current Phase 4
summary.

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
# Run the phase notebooks in order
jupyter nbconvert --to notebook --execute notebooks/00_run_all_phases.ipynb

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

For iterative work, run the phase notebooks individually from `notebooks/`.
Phase 1 creates `data/raw/`, Phase 2 writes calibrated parameter JSON files,
Phase 3 writes `sim_summary.json` and `sim_bundle.pkl`, and Phases 4-7 consume
those artefacts from `data/processed/`.

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
| 3 | Phase 4 uses a reduced LSMC grid/mode set | Full-grid production solve is still pending | Model approximation |
| 4 | LSMC diagnostics need hardening | Cached status now flags LSMC/RI coherence failures and dispatch concentration; continuation monotonicity and out-of-sample stability still need fuller reporting | Partial |
| 5 | True Andersen-Broadie dual bound not implemented | Current Phase 6 reports a clairvoyant upper benchmark, not a martingale-penalty dual proof | Feature gap |
| 6 | Backtest still uses synthetic cashflows | Attribution is illustrative, not execution validation | Design |
| 7 | Partial-mode backtest residual is large | 30-day attribution is not calibrated for execution validation | Design |
| 8 | Intraday spread uses HPFC shape proxy | No calibrated intraday market process yet | Feature gap |
| 9 | Negative prices require arithmetic treatment | Log-normal model clips negative-price behavior | Model gap |
| 10 | Perfect foresight is an upper benchmark | It assumes complete future price knowledge | Interpretation |

---

## Potential Improvements

### Already implemented

1. Shared notebook bootstrap via `src/utils.find_project_root()`.
2. Standalone historical perfect-foresight DA/SP benchmark.
3. Explicit Phase 3 `xi_0` anchor to avoid unanchored GBP 1/MWh spot paths.
4. Basic LSMC diagnostics in `lsmc_valuation_summary.json`.
5. Cached model-status checks now flag LSMC outputs that fall below the
   rolling-intrinsic benchmark before full-mode refresh.
6. Action-distribution and selected-action cashflow diagnostics are persisted
   in `lsmc_valuation_summary.json` and surfaced in model status.
7. `data/processed/sim_bundle*.pkl` excluded from Git by default.

### Tier 1 - Correctness

1. Replace the synthetic forward curve with historical ICE/EEX forward panels
   containing multiple `as_of_date`s and short maturities: front month, quarter,
   season, and year. The Kalman filter currently runs on a panel generated from
   its own prior parameters, so `sigma_obs` hits its lower bound (0.001) and
   the calibrated parameters are essentially the priors rather than
   market-implied values.
2. Reduce LSMC continuation clipping by regularising/scaling basis functions,
   widening continuation bounds, or increasing grid/mode fidelity, then re-run
   Phase 4-6 and compare stability.

### Tier 2 - Model gaps (significant impact on outputs)

3. Update NESO ancillary data ingestion by refreshing resource IDs or supporting
   manual CSV uploads for DC/DM/DR/QR/BR clearing prices. All eight products
   currently show `n_obs = 0` in Phase 2 output; the AR(1) parameters are pure
   calibration priors.
4. Model negative day-ahead prices directly with an arithmetic two-factor model,
   shifted lognormal process, or explicit negative-price regime. 1,032 of 36,240
   half-hours (2.8%) are negative in the Phase 1 data; the log-normal model
   clips these, missing charge-on-negative-price revenue.
5. Replace year-by-year chaining with a single long-horizon simulation for
   lifecycle MTM, so that SoH degradation, augmentation timing, and price
   dynamics interact correctly across the full 15-year horizon rather than being
   joined post-hoc by an annuity factor.

### Tier 3 - Model completeness

6. Add a combined market optimiser where DA, imbalance/system price, and
   ancillary decisions share one physical battery dispatch constraint.
7. Replace the current HPFC peak-minus-trough intraday proxy with a calibrated
   intraday market spread process or historical intraday benchmark.
8. Fit a regime-switching ancillary saturation curve rather than one static
   exponent. The static gamma = 2.1 does not capture product-mix shifts between
   DC/DM/DR/QR or seasonal patterns.
9. Replace flat Capacity Market revenue with delivery-year clearing prices and
   derating-specific assumptions.
10. Extend LSMC diagnostics beyond the current regression, clipping,
   LSMC-vs-rolling-intrinsic coherence, and action-distribution checks to
   include continuation-value monotonicity and out-of-sample forward stability.
11. Add a historical spot-derived baseload fallback, clearly labelled as a spot
   proxy rather than a risk-neutral forward calibration.
12. Promote the historical perfect-foresight benchmark in the phase workflow and
   compare LSMC, rolling intrinsic, DA perfect foresight, and SP perfect
   foresight side by side.

### Tier 4 - Engineering

13. Parallelise bump-and-revalue Greeks and cache shared simulation inputs.
   Tier-2 Greeks each re-run the full LSMC; only the forward pass needs to
   re-run per bump once the base beta coefficients are cached.
14. Replace the large pickle for `sim_bundle` with memory-mapped arrays
   (NumPy `.npy` memmap, zarr, or HDF5) to allow lazy loading, partial reads
   for Greek re-solves, and a smaller peak RAM footprint.
15. Keep large generated artefacts out of Git where possible; commit code and
   small summaries, and publish heavy plots/parquet outputs via releases or
   external storage.

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
