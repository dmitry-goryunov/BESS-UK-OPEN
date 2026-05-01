# BESS LSMC Project — TODO

_Last updated: 2026-05-01_

---

## Immediate next steps

- [ ] **Smoke test the 7 fixes** — open `notebooks/08_lsmc_duration_sweep.ipynb`, temporarily set:
  ```python
  PHASE4_RUN_MODE_FOR_SWEEP = "partial"
  SWEEP_SEEDS = [42]
  ```
  Run all cells. Expect 16 runs (8 durations × 2 scenarios × 1 seed), ~10 min total.
  Target: all LSMC/RI ratios < 10x. Check the diagnostics plot for sane beta magnitudes.

- [ ] **Run the full medium sweep** — once partial results are clean, restore:
  ```python
  PHASE4_RUN_MODE_FOR_SWEEP = "medium"
  SWEEP_SEEDS = [42, 123, 456]
  ```
  Set a fresh `OUTPUT_DIR` (or let it auto-timestamp). Expect 48 runs, 2–4 hours.
  Old results (`lsmc_duration_sweep_20260501_*`) are pre-fix and should not be reused.

- [ ] **Validate sweep outputs** — check `data/processed/lsmc_duration_sweep.csv` after the
  collect cell runs:
  - Curve should be monotonically increasing (or peak at 2h) with no extreme outliers
  - No `ratio_flagged` rows
  - `beta_abs_max` should be O(10k–100k), not O(1M+)
  - `continuation_clip_fraction_max` should be > 0 for some runs (confirms cap is firing)

---

## Fixes implemented in this session (all verified in source)

Seven improvements were made to address the non-monotonic duration curve and 67× LSMC/RI ratios:

1. **Ridge alpha 1.0 → 200.0** (`notebooks/04_lsmc_valuation.ipynb`, all three run modes)
   — Old ridge provided ~0.1% shrinkage with N=500 paths; new value is ~30%.

2. **Target Y normalisation** (`src/optimisation/lsmc.py`, backward pass ~L471)
   — Divides regression target by `y_std` before solving, then un-normalises betas.
   Keeps RHS at float64-safe magnitude and makes ridge penalty scale-consistent.

3. **Continuation value cap 25M → 3M** (`notebooks/04_lsmc_valuation.ipynb`, medium mode)
   — Old cap never fired (clip_fraction_max=0.0). New cap should clip explosive values
   during backward propagation.

4. **Disjoint forward paths** (`notebooks/04_lsmc_valuation.ipynb`, cell `a32e7c95`)
   — Uses `np.setdiff1d` so forward paths never overlap backward paths.
   Eliminates in-sample bias that inflated V_LSMC.

5. **Replaced always-zero basis features** (`src/optimisation/lsmc.py`, `BASIS_NAMES` + `basis_matrix()`)
   — `sin_h`, `cos_h`, `efa_block` were functions of `t` only (constant across paths),
   so raw_sd=0 and they were always dropped. Replaced with cross-sectional interactions:
   `P_da × delta_imb`, `P_da × pi_dc`, `pi_dc × pi_qr`.
   Forward pass updated to match (`phi_next_base` columns 8–10).

6. **Multi-seed sweep** (`notebooks/08_lsmc_duration_sweep.ipynb`)
   — `SWEEP_SEEDS = [42, 123, 456]`; each (scenario, duration) runs 3 seeds.
   Collect cell aggregates to mean ± std; plot shows ±1σ shading and flags high-ratio runs.

7. **Seed injection into parameterised notebooks** (`notebooks/08_lsmc_duration_sweep.ipynb`)
   — `parameterized_notebook()` now injects `SEED = {seed}` into the parameter cell.
   File naming includes seed suffix: `lsmc_valuation_summary_{scenario}_{duration}h_seed{seed}.json`.

---

## Known diagnostics from old (pre-fix) run — for comparison

| Metric | Old value | Expected post-fix |
|---|---|---|
| LSMC/RI ratio (2h, energy_only) | 67x | < 10x |
| beta_abs_max | £5,064,553 | < £500k |
| continuation_value_cap_gbp (actual) | £25,000,000 | £3,000,000 |
| continuation_clip_fraction_max | 0.0 | > 0.0 |
| active_feature_count_max | 9 / 14 | 13–14 / 14 |
| ridge_alpha | 1.0 | 200.0 |
| forward paths | 1000 (overlapping) | 500 (disjoint) |

---

## Longer-term items

- [ ] Full-mode sweep (yearly horizon, all paths) — set `PHASE4_RUN_MODE_FOR_SWEEP = "full"` once medium is stable
- [ ] Andersen-Broadie upper bound (`src/optimisation/dual_bound.py`) to bracket true value
- [ ] Greeks bump engine (`src/valuation/greeks.py`) — delta_baseload, vega_da at minimum
- [ ] Degrade SoH properly across the horizon (currently SoH nodes are static)
- [ ] Calibrate Schwartz-Smith to `data/raw/forward_uk.xlsx` (currently using config defaults)
