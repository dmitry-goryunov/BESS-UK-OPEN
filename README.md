# BESS-UK — Stochastic MTM Valuation

Research framework for stochastic mark-to-market valuation of a Great Britain
fast-cycle battery energy storage system (BESS), using Least-Squares Monte
Carlo (LSMC) adapted from the Boogert & de Jong (2008) gas storage approach.

**Asset**: 100 MW / 200 MWh LFP · 2h duration · 88% round-trip efficiency · GB grid

---

## What this does

The model values a BESS as a real option: at each half-hour the operator chooses
how much power to charge, discharge, and reserve for ancillary services. LSMC
learns the optimal continuation value by backward induction, then rolls the
policy forward to produce a MTM distribution. Six sequential phases:

| Phase | Module | What it produces |
|---|---|---|
| 1 · Data | `src/data/` | Elexon DA/SP prices, NESO ancillary clearing, forward curve |
| 2 · Calibration | `src/processes/` | SS two-factor, PCA hourly shape, imbalance OU+jump, ancillary AR(1) |
| 3 · Simulation | `src/processes/simulate.py` | 5,000-path × 17,520-step correlated PathBundle |
| 4 · LSMC | `src/optimisation/lsmc.py` | β coefficients, dispatch policy, MTM distribution |
| 5 · MTM / Greeks / VaR | `src/valuation/` | Life-time MTM, 15-factor Greek ladder, VaR/CVaR, scenarios |
| 6 · Backtest | `src/attribution/` | Dual bound gap, 30-day P&L attribution |

Everything runs end-to-end in `notebooks/bess_valuation_full.ipynb`.

---

## Key outputs (current run)

### Phase 1 — Market data
| Dataset | Rows | Period | Notes |
|---|---|---|---|
| Elexon DA prices | 36,240 | Apr 2024 – Apr 2026 | Mean £79.8/MWh, std £40.3 in 2025 |
| Elexon system prices | 36,240 | Apr 2024 – Apr 2026 | Imbalance basis mean +£0.2, std £39.6 |
| NESO ancillary clearing | 757 | Apr 2024 – Apr 2026 | DC £1.97, DM £6.24, QR £4.62 /MW/h mean |
| Forward curve | 9 contracts | 2027–2030 | **Synthetic**, anchored to £76.7/MWh |

1,032 half-hours had negative DA prices, confirming that a log-normal (Schwartz-Smith)
spot model needs an arithmetic floor or a separate negative-price regime.

### Phase 2 — Calibration
| Process | Key parameters |
|---|---|
| **Schwartz-Smith** | κ=4.02 (half-life 2.1 months), μ_ξ=0.007, σ_χ=2.00⚠, σ_ξ=0.011⚠, ρ=0.39 |
| **PCA hourly shape** | 3 factors, 76.7% variance; PC1 HL 0.6d, PC2 HL 0.2d, PC3 HL 0.3d |
| **Imbalance OU+jump** | θ=0.83/HH (HL 0.4h), σ=18.6 £/MWh, λ_J=0.043/HH (≈750 jumps/yr) |
| **Ancillary AR(1)** | φ=0.85, γ_sat=2.10 (DCL collapse calibrated); fitted to priors only⚠ |

⚠ σ_χ=2.0 is above the expected range (0.1–0.9) because the forward panel is
synthetic with limited cross-sectional variation. σ_ξ=0.011 is correspondingly low.
Both will normalise with real ICE/EEX forward data.

⚠ Ancillary products show n_obs=0 for the current date range — NESO reorganised
API resource IDs in late 2024. AR(1) parameters revert to calibration priors.

### Phase 3 — Simulation
- 5,000 paths × 17,520 half-hours (1 year), 40.9 s, 4.9 GB in-memory
- All 7 marginal moment checks pass (χ variance, ξ mean/variance, imbalance stationarity, ancillary bounds, cross-correlations)
- 3-year chaining validated (no SoC discontinuity at year boundary)
- Spot P5/P50/P95: £0.3/£1.0/£3.3 — **prices are too low** because `xi_0` defaults to 0 instead of `log(76.7)` in the Phase 3 cell; the LSMC phase corrects this in its own simulate call

### Phase 4 — LSMC valuation (1,000 paths, 1 yr)
| Metric | Value |
|---|---|
| V_LSMC mean | **£13.94M** (£139,356/MW) |
| V_LSMC std | £2.85M |
| V_LSMC P5/P95 | £9.0M / £17.8M |
| V_RI (rolling intrinsic) | £767k (£7,668/MW) |
| V_LSMC / V_RI | **18.2×** |
| Backward pass | 63.4 s · β shape (17520, 9, 3, 14) |
| Forward pass | 28.3 s |

V_LSMC ≥ V_RI confirmed. The 18× ratio is higher than typical (3–8×) and likely
reflects the NaN issue in the β coefficients (see known issues).

### Phase 5 — MTM, Greeks & VaR (mini run, 200 paths, 5 days)
| Component | £/MW/yr |
|---|---|
| Merchant LSMC | +1,000 |
| Capacity Market | +1,051 |
| Floor optionality (put) | +15,119 |
| Optimiser fee | −128 |
| Fixed O&M | −5,180 |
| Augmentation capex | −12,527 |
| **Total mean** | **−1,716** |

Life-time MTM mean **−£2.57M** (negative because the mini LSMC run underestimates
merchant revenue; the Phase 4 full run gives a positive V_LSMC before cost deductions).

| Risk metric | Value |
|---|---|
| VaR 90% | £3.08M/yr |
| VaR 95% | £3.15M/yr |
| CVaR 95% | £3.22M/yr |
| Degradation shadow cost | £5.99/MWh throughput |

### Phase 6 — Dual bound & backtest
- Dual bound gap: **0.00%** — degenerate result (upper bound std = £28B); needs more dual paths
- Backtest residual: **434%** against a <5% target — backtest uses synthetic cashflows, not real dispatch data

---

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# Full end-to-end notebook
jupyter nbconvert --to notebook --execute notebooks/bess_valuation_full.ipynb

# Pre-generate the simulation bundle (skips Phases 1-3 on subsequent runs)
python scripts/generate_sim_bundle.py --paths 1000 --steps 17520

# Quick dev bundle (200 paths, 5 days)
python scripts/generate_sim_bundle.py --paths 200 --steps 240

# Run sanity tests
pytest tests/test_sanity.py -v

# Streamlit dashboard (reads cached JSON/parquet outputs)
streamlit run streamlit_app.py
```

---

## Project structure

```
bess_project/
├── notebooks/bess_valuation_full.ipynb   ← single end-to-end notebook
├── src/
│   ├── processes/
│   │   ├── schwartz_smith.py             ← Kalman filter calibration
│   │   ├── hpfc.py                       ← PCA hourly shape
│   │   ├── imbalance.py                  ← OU+jump calibration & simulation
│   │   ├── ancillary.py                  ← AR(1) + saturation curve
│   │   └── simulate.py                   ← joint PathBundle generator
│   ├── optimisation/
│   │   ├── lsmc.py                       ← LSMC backward induction + forward pass
│   │   ├── rolling_intrinsic.py          ← LP benchmark
│   │   └── dual_bound.py                 ← Andersen-Broadie upper bound
│   ├── valuation/
│   │   ├── mtm.py                        ← MTM aggregation + contract overlays
│   │   ├── greeks.py                     ← bump-and-revalue Greek engine
│   │   └── var_cvar.py                   ← VaR / CVaR / scenario stress
│   └── attribution/
│       └── pnl_explain.py                ← daily P&L decomposition
├── tests/
│   └── test_sanity.py                    ← Test A (price sim) + Test C (MTM signs)
├── data/
│   ├── raw/                              ← parquet files from APIs
│   └── processed/                        ← calibration JSON + simulation bundle
└── docs/
    ├── stochastic_plan.md                ← 10-phase methodology
    └── pricing_items.md                  ← full revenue stack description
```

---

## Known issues

| # | Issue | Impact | Status |
|---|---|---|---|
| 1 | **Synthetic forward curve** (9 contracts, not real ICE/EEX) | SS σ_χ=2.0 is 2–10× too large; σ_ξ is too small; mean-reversion half-life correct | Data gap |
| 2 | **NESO API URLs broken** — ancillary n_obs=0 for all products post-2024 | Ancillary AR(1) reverts to hard-coded priors; saturation curve not updated | API change |
| 3 | **xi_0 not enforced in simulate()** — defaults to 0, giving exp(0)=£1/MWh instead of £76.7 | Phase 3 spot P50 = £1/MWh; LSMC phase sets xi_0 correctly but it's a footgun | Footgun |
| 4 | **β NaN in backward pass** — some basis function evaluations produce NaN (likely from overflow or rank-deficient regression on early steps) | LSMC policy is partially degenerate; V_LSMC/V_RI ratio of 18× is inflated | Bug |
| 5 | **Dual bound std = £28B** — oracle is computing identical cashflows for lower and upper path, producing zero variance across dual paths | 0% gap is meaningless; dual bound not validating LSMC quality | Bug |
| 6 | **Backtest is synthetic** — P&L attribution uses simulated, not real, realised cashflows | Residual 434% vs <5% target is expected; backtest cannot validate execution | Design |
| 7 | **Intraday spread not simulated** — `P_id` is set to `P_da` throughout; ID premium is zero | All ID revenue opportunity lost; underestimates BESS value | Feature gap |
| 8 | **Negative prices require arithmetic OU** — 1,032 negative-price half-hours break the log-normal assumption | Spot model clips negative prices upward; tail risk understated | Model gap |

---

## Potential improvements

### High priority (model correctness)

1. **Real forward curve** — pull GB baseload/peak monthly settlements from ICE WebICE or EEX
   transparency platform and re-calibrate SS. Expected effect: σ_χ drops to 0.2–0.5, σ_ξ rises to 0.05–0.15.

2. **Fix NESO ancillary data** — update resource IDs in `src/data/fetch_neso.py` after the late-2024
   API reorganisation, or download CSV and convert. Ancillary revenues are the largest source of BESS
   value in GB (£6–12k/MW/yr DC+DM) and are currently calibrated to stale priors.

3. **Fix β NaN in LSMC backward pass** — add a NaN guard in the regression loop; fall back to
   the unconditional mean when design matrix is rank-deficient. This will also normalise the
   V_LSMC/V_RI ratio to a plausible 3–8×.

4. **Enforce xi_0 in simulate()** — either accept `forward_anchor` as a required parameter
   or set `xi_0 = log(forward_anchor)` by default, so the footgun cannot fire silently.

### Medium priority (model completeness)

5. **Intraday price simulation** — add an arithmetic OU spread `P_id − P_da` to the PathBundle.
   GB intraday has mean spread ≈ £2–5/MWh with intraday momentum; a fast battery cycles this 2–4×/day.

6. **Negative price regime** — switch baseload simulation from log-normal SS to an arithmetic
   two-factor model (Lucia & Schwartz 2002 variant), or add a floor absorbing state at £0 to
   preserve log-normal convenience where prices are positive.

7. **Meaningful dual bound** — debug the oracle cashflow computation so that upper-bound paths
   genuinely relax the non-anticipativity constraint; target gap < 2% (Nadarajah et al. 2017).

8. **Real backtest** — connect `pnl_explain.py` to actual Elexon BOA / BMRS dispatch outturn
   data for a BMU registered to the asset, then re-run the 30-day P&L attribution.

### Lower priority (calibration & infrastructure)

9. **Regime-switching for ancillary saturation** — fit a time-varying γ or a two-regime model
   (pre/post 6 GW fleet) rather than a single exponent, to better capture the DCL price collapse.

10. **Multi-year simulation efficiency** — the 3-year path chaining proof-of-concept uses a fresh
    simulate call per year; a single long-horizon simulation would preserve state correlations across
    augmentation events.

11. **Capacity Market forward curve** — current CM revenue is a flat £1,051/MW/yr proxy.
    Replace with the actual T-4 and T-1 clearing prices for the relevant delivery years and apply
    the correct derated capacity factor.

12. **Parallel Greek engine** — bump-and-revalue currently runs Greeks sequentially; parallelise
    across factors to bring runtime from ~15 min to ~2 min for the full 15-Greek ladder.

---

## Data sources

| Source | What to pull | API / access |
|---|---|---|
| Elexon BMRS | DA MID prices, system prices, NIV, BOA | `api.elexon.co.uk` — no credentials |
| NESO Data Portal | EAC clearing (DC/DM/DR/QR/BR) | `api.nationalgrideso.com` — resource IDs drift |
| ICE / EEX | GB baseload + peak monthly forwards | ICE WebICE export or EEX transparency |
| Modo Energy | ME BESS GB Revenue Index | Subscription |
| Aurora / Baringa | Battery revenue forecasts | Subscription |

---

## References

- Boogert & de Jong (2008) — LSMC for gas storage, *J. Derivatives* 15(3)
- Schwartz & Smith (2000) — two-factor commodity model, *Management Science* 46
- Lucia & Schwartz (2002) — electricity seasonality
- Nadarajah, Margot & Secomandi (2017) — LSMC dual bounds, *EJOR* 256
- Finnah, Gönsch & Ziel (2022) — GB imbalance modelling, *EJOR* 301
- Shi, Xu & Baldick (2019) — convex cycle-based degradation cost, *IEEE T-SG*
