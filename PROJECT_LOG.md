# Project Log

## 2026-04-29

### Context

Worked on the BESS stochastic MTM valuation project, focusing on Phase 4 LSMC
valuation and keeping Phase 5-6 artefacts consistent with the selected Phase 4
run mode.

### Main Changes

1. Added a Phase 4 run-mode switch in `notebooks/04_lsmc_valuation.ipynb`.
   - `PHASE4_RUN_MODE = 'partial'` is the default for fast development.
   - `PHASE4_RUN_MODE = 'full'` uses the cached full yearly simulation bundle.

2. Restored partial-mode behaviour for development.
   - Partial mode uses 250 backward paths, 240 half-hour steps, 250 forward
     paths, and 25 rolling-intrinsic paths.
   - Full mode uses all cached paths and steps.

3. Fixed the partial-mode rolling-intrinsic comparison.
   - Rolling intrinsic now uses the same selected horizon as the LSMC run.
   - This avoids comparing a 240-step LSMC result with a full-year RI benchmark.

4. Added LSMC continuation-value cap configuration.
   - Added `continuation_value_cap_gbp` to `src/config.py`.
   - `src/optimisation/lsmc.py` now uses this configurable cap in backward and
     forward continuation calculations.

5. Added aggregate continuation clipping diagnostics.
   - `continuation_clip_observation_count`
   - `continuation_clip_observation_total`
   - `continuation_clip_regression_count`
   - `continuation_clip_observation_fraction`
   - `continuation_clip_regression_fraction`

6. Tuned partial-mode ridge regularisation.
   - Partial mode now applies `ridge_alpha = 1.0`.
   - This reduced beta scale and made the partial smoke test pass the RI check.

7. Refreshed Phase 4-6 processed outputs from partial mode.
   - `data/processed/lsmc_valuation_summary.json`
   - `data/processed/mtm_summary.json`
   - `data/processed/phase6_summary.json`
   - Related notebooks were executed and saved.

8. Updated `README.md`.
   - Current checked-in outputs are clearly labelled as partial-mode
     development artefacts.
   - README now instructs switching to `PHASE4_RUN_MODE = 'full'` before
     refreshing headline economics.

### Current Phase 4 Partial Results

From the latest partial run:

- Paths: 250
- Steps: 240
- `V_LSMC mean`: GBP 26.8k
- `V_RI mean`: GBP 7.8k
- `V_LSMC / V_RI`: 3.45x
- `beta_abs_max`: 5.93e6
- `sample_condition_max`: 1.06e3
- `continuation_clip_observation_fraction`: 0.0
- `continuation_clip_regression_fraction`: 0.0

### Current Phase 5 Partial Results

From the latest partial artefacts:

- Merchant LSMC: GBP +145/MW/yr
- Capacity Market: GBP +1,051/MW/yr
- Floor optionality: GBP +15,589/MW/yr
- Optimiser fee: GBP -62/MW/yr
- Fixed O&M: GBP -5,180/MW/yr
- Augmentation capex: GBP -12,527/MW/yr
- Total mean: GBP -2,034/MW/yr
- Lifetime MTM mean: GBP -3.05M

### Current Phase 6 Partial Results

From the latest partial artefacts:

- `V_LSMC`: GBP 33,044
- `V_upper`: GBP 187,879
- Upper gap: 468.57%
- 30-day residual: GBP 928,608
- Residual / total: 460.23%
- Target passed: No

Phase 6 remains a synthetic attribution/plumbing check, not execution
validation. The upper benchmark is a clairvoyant information-relaxation
benchmark, not a true Andersen-Broadie martingale dual proof.

### Verification

Ran:

```bash
pytest tests/test_sanity.py -q
```

Result:

```text
25 passed
```

### Important State Note

`data/processed` currently reflects the partial development run. It does not
reflect full yearly headline economics.

Do **not** run full mode yet. The immediate priority is to keep Phase 4 in
partial mode and reconcile the current partial-mode artefacts and diagnostics
before spending time on a full-year refresh.

Before publishing or relying on headline economics, after the partial-mode
state has been reconciled:

1. Set `PHASE4_RUN_MODE = 'full'` in `notebooks/04_lsmc_valuation.ipynb`.
2. Run Phase 4.
3. Run Phase 5.
4. Run Phase 6.
5. Update `README.md` with the refreshed full-mode outputs.

### Next Steps

1. Keep using partial mode for fast LSMC development.
2. Reconcile the current partial-mode artefacts against the recorded partial
   results before making any full-mode run.
3. Monitor the corrected partial dispatch diagnostic:
   - cached action distribution has 60,000 decisions
   - 9 unique action modes
   - dominant mode fraction 80.96%
   - idle/reserve fraction 83.30%
   - charge fraction 9.58%
   - discharge fraction 7.12%
4. Investigate whether the partial `ridge_alpha = 1.0` should also be tested in
   full mode.
5. Do not run a full-mode Phase 4-6 refresh until the partial-mode diagnostics
   are coherent.
6. When full mode is eventually run, compare diagnostics against the previous
   full run:
   - beta max
   - condition number
   - continuation clipping observation fraction
   - `V_LSMC / V_RI`
   - action distribution
7. Continue model improvements:
   - Replace synthetic forward curve with real ICE/EEX panels.
   - Repair NESO ancillary ingestion.
   - Add direct negative-price modelling.
   - Replace the clairvoyant upper benchmark with a true dual-bound method if
     needed.

### 2026-04-29 Progress Update - LSMC Diagnostics

Added action-distribution diagnostics without running full mode.

- Added `summarize_action_distribution()` in `src/validation.py`.
- Phase 4 summary notebooks now persist `action_distribution` into
  `data/processed/lsmc_valuation_summary.json`.
- `src/model_status.py` now reports `LSMC dispatch` separately from `LSMC
  valuation`, so dispatch pathologies are visible even when valuation
  coherence already fails.
- Fixed the action-distribution decoding to use `policy.modes` rather than
  `DEFAULT_MODES`; partial mode uses 12 reduced modes, so the earlier decode
  mislabeled action index 7.
- Backfilled the current cached partial summary from
  `data/processed/lsmc_policy.pkl` and
  `data/processed/lsmc_valuation_result.pkl`.
- Corrected current cached partial dispatch: idle/reserve 83.30%, charge
  9.58%, discharge 7.12%. The dominant action is `net=0.0`, `r_dc=0.5`,
  `r_qr=0.25`, not a charging action.
- Added selected-action cashflow diagnostics to the action summary. Current
  partial average cashflow is only GBP 0.31 per path-step because reserve
  revenues are largely offset by charge costs and discharge revenues over the
  240-step smoke horizon.
- Current dominant action cashflow: `net=0.0`, `r_dc=0.5`, `r_qr=0.25` earns
  about GBP 180.71 per selected half-hour.

Verification:

```bash
pytest tests/test_sanity.py -q
```

Result:

```text
28 passed
```

### 2026-04-29 Progress Update - Partial Phase 4 Continuation Fix

Fixed two Phase 4 policy-evaluation issues without running full mode.

1. Forward policy evaluation now uses `V_{t+1}(E_next, S_{t+1})` for
   continuation and zero continuation at the terminal step, rather than using
   same-time beta coefficients at the current SoC node.
2. Backward induction now builds a separate `V_curr` surface for each time step
   and only swaps it into `V_next` after all SoC/SoH nodes are fitted. This
   avoids contaminating same-step regressions with a mixture of `V_t` and
   `V_{t+1}` values.

Reran only the partial Phase 4 smoke path:

- Backward paths: 250
- Forward paths: 250
- Steps: 240 half-hours
- RI paths: 25
- `V_LSMC mean`: GBP 124,760
- `V_RI mean`: GBP 7,772
- `V_LSMC / V_RI`: 16.05x
- `beta_abs_max`: 1.16e6
- `sample_condition_max`: 1.06e3
- selected cashflow mean: GBP 520.22 per path-step
- selected continuation mean: GBP 123,627
- selected Q-gap mean: GBP 251

Current model status:

- `LSMC valuation`: benchmark warning because `V_LSMC / V_RI` remains high.
- `LSMC dispatch`: passes diagnostics.
- `LSMC Q-values`: passes diagnostics.

Phase 5-6 have not been rerun against this refreshed Phase 4 partial output.
They should be treated as stale until Phase 4 benchmark comparability is
understood.

Verification:

```bash
pytest tests/test_sanity.py -q
```

Result:

```text
30 passed
```
