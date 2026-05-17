# BESS Stochastic Valuation — Master Reference

_Last updated: 2026-05-17. Read this file first, every session._

---

## Asset

| Parameter | Value |
|---|---|
| Market | Great Britain (GB) |
| Rated power | 100 MW |
| Duration sweep | 1h / 2h / 3h / 4h |
| Energy (2h central) | 200 MWh |
| AC-AC RTE | 88% (η_c = η_d = 0.9381) |
| SoC bounds | 10–90% of nameplate |
| VOM | £1.2/MWh on throughput |
| Degradation shadow cost | £6.0/MWh (λ_deg_init) |
| Availability | 96% |
| As-of date | 2026-05-03 (pinned in nb12: `AS_OF_DATE`) |

---

## Revenue Stack

### Modelled
| Stream | Status | Location |
|---|---|---|
| DA energy (HPFC-anchored) | ✅ live | `src/optimisation/dispatch.py` |
| Imbalance basis (SP−DA) | ✅ live, calibrated post-fix | `src/optimisation/dispatch.py` |
| Dynamic Containment (DC) | ✅ live | mode grid `dc_levels` |
| Quick Reserve (QR) | ✅ live | mode grid `qr_levels` |
| Balancing Mechanism (BM) | ✅ Phase 3 (validated) | mode grid `bm_levels` |
| VOM + degradation costs | ✅ live | dispatch cashflow |
| Capacity Market (CM) | ✅ deterministic overlay | `scripts/apply_capacity_market_overlay.py`, £6k/MW/yr central |
| Dynamic Moderation/Regulation | ❌ not modelled | in Modo index, not calibrated |
| Slow Reserve | ❌ not modelled | added to Modo Apr 2026 |

### Excluded (by design, consistent with Modo)
- Mandatory Frequency Response, TNUoS/Triads, DUoS
- Tolling/floor arrangements, bilateral PPAs
- Voltage/reactive power Pathfinder

---

## Price Processes

| Process | Model | Key calibrated params |
|---|---|---|
| DA baseload | Schwartz-Smith two-factor | κ, σ_χ, μ_ξ, σ_ξ, ρ |
| Intraday shape | HPFC × SS relative move | Hourly shape multipliers 0.71–1.46 |
| Imbalance basis (SP−DA) | OU + asymmetric jumps | See `imbalance_params.json` |
| DC clearing | AR(1) per EFA block + saturation curve | DC_Low.mu = 5.1 £/MW/h |
| QR clearing | AR(1) per EFA block | QR_Pos.mu = 3.6 £/MW/h |
| BM offer price | Constant mean | π_bm mean ≈ £79.9/MWh, p_activation = 0.12 |

### Imbalance calibration (post-fix, `data/processed/imbalance_params.json`)
```
theta_delta   = 0.8281
sigma_delta   = 18.6510
lambda_jump   = 0.0429/HH
jump_scale_pos= 109.76
jump_scale_neg= 74.10
p_pos         = 0.3580    ← was 0.993 (bug), fixed 2026-05-08
mu_delta      = 0.2207    ← implied mean ≈ −£0.41/MWh (near-zero, correct)
```

**DO NOT revert p_pos to 0.993.** The old value was caused by duplicate zero-volume N2EX DA rows
in `data/raw/elexon_da_prices.parquet` that doubled the SP-DA observations and created a spurious
positive imbalance mean of £48.55/MWh. Fixed in `src/data/fetch_elexon.py` and
`src/processes/imbalance.py` (Action 10).

Signal lag: `delta_signal[t] = delta_realized[t-1]` — imbalance dispatch uses the
**previous** half-hour's realized spread, not the current one (prevents clairvoyant dispatch).

---

## LSMC Architecture

### Co-optimisation constraints (per HH)
```
|net_mw| + r_dc_mw + r_qr_mw + r_bm_mw  ≤  P_bar            (power)
E_t − (r_dc + r_qr + r_bm) × sustain_h / η_d  ≥  E_min      (discharge headroom)
E_t + (r_dc + r_qr + r_bm) × η_c × sustain_h  ≤  E_max      (charge headroom)
```

`reserve_sustain_h = 1.0` is the central realism setting (pending lock-in to nb12 defaults).

### Current mode grid (nb12 / nb13 as of 2026-05-14)
```python
dc_levels  = [0.0, 0.5]          # Step 1 of LSMC_CLEANUP: add allow_ancillary_stacking=False
qr_levels  = [0.0, 0.25]
bm_levels  = [0.0, 0.25, 0.5]    # Phase 3
net_levels = [-1, -0.5, 0, 0.5, 1]
```

### Key LSMC config (`src/config.py`)
```python
continuation_value_cap_gbp = 25_000_000   # DO NOT lower — was 3m, clipped 4h policy badly
n_soc_nodes = max(9, 3 * duration_h)
```

### Basis features (efa_blocks mode, N_BASIS = 38)
15 base features + 12 EFA-block DA forward-strip features (6 block means + 6 spreads) +
1 soc_x_da_max feature (`E_scaled × da_fwd_max`) + 10 ancillary features.

### Non-anticipativeness
`cont_beta` branch in `forward()` is non-anticipative: builds `φ(S(t))` from current-step
prices/SoC only. Legacy fallback using `policy.beta[t+1]·φ(S_{t+1})` reads t+1 prices
(clairvoyant) but is preserved for backward compatibility with pre-change pickled policies.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| nb00–nb08 | Early development (data fetching, calibration, process fitting) |
| nb11 | ? (missing from summary) |
| **nb12** | Single-duration LSMC run; source of truth for config; run before nb13 |
| **nb13** | Phase 4 duration sweep (1h–4h); produces all comparison CSVs/PNGs |
| nb14 | Phase 4 diagnostics |
| **nb15** | WD rolling vs WD-like LSMC comparison |
| **nb16** | Structural LSMC counterfactuals (no_imbalance, no_ancillary, energy_only) |
| nb17 | ? |
| **nb19** | Historical BESS index backtest; locked base case vs Modo |
| (missing) | nb09, nb10, nb18 not in repo |

**Target:** `notebooks/20_historical_lsmc.ipynb` — LSMC on actual 2024–26 prices (Option B).

---

## Key Source Files

| File | Role |
|---|---|
| `src/optimisation/lsmc.py` | LSMC core: `Policy`, `backward()`, `forward()`, `forward_parallel()` |
| `src/optimisation/dispatch.py` | Mode enumeration (`enumerate_modes()`), cashflow formula |
| `src/config.py` | Default LSMC and process config |
| `src/processes/imbalance.py` | Imbalance OU+jump calibration and simulation |
| `src/data/fetch_elexon.py` | DA price fetch (N2EX volume-weighted, de-duped) |
| `src/data/fetch_elexon_bm.py` | BM volume fetch (Python312, not .venv_test; checkpoints every 30 days) |
| `src/backtest/historical_index.py` | Rolling LP + WD + ancillary; `WD_CAP_DEFAULT = 60.0` |
| `src/backtest/ancillary_revenue.py` | `DEFAULT_HEADROOM` — DC=0.35 locked |
| `src/backtest/bm_revenue.py` | BM index from BOA volumes × system price |
| `streamlit_phase4_sweep.py` | Main Streamlit app (3 sections) |

### Key data files
| File | Contents | Status |
|---|---|---|
| `data/processed/imbalance_params.json` | Calibrated OU+jump params | ✅ post-fix (p_pos=0.358) |
| `data/processed/ancillary_params.json` | DC/QR AR(1) params | ✅ calibrated |
| `data/processed/sim_bundle.pkl` | 1000-path, 17520-step simulation bundle | ✅ rebuilt post-fix |
| `data/processed/phase4_all_durations_comparison.csv` | Main method-comparison results | ⚠️ STALE — pre-imbalance-fix values (1h LSMC = £75.9k). Do not use until nb13 rerun. |
| `data/raw/elexon_da_prices.parquet` | DA prices (APXMIDP only, positive-volume filtered) | ✅ |
| `data/raw/elexon_bm_volumes.parquet` | BM volumes, 952k rows, Jan 2024–Apr 2026 | ✅ |
| `MODO 1H.csv` | Modo monthly index Jan 2020–May 2026 | ✅ source of truth |
| `MODO 2H.csv` | Modo monthly index Jan 2023–May 2026 | ✅ source of truth |

---

## Units and Scale

**CRITICAL:** LSMC attribution values (e.g. in `lsmc_attribution_1h.json`) are in **£m/year
for the 100 MW asset**. To convert to £k/MW/yr: `mean_m × 10`.

Example: `imbalance mean_m = 1.092` → `1.092 × 10 = £10.92k/MW/yr`.

All Modo and KYOS benchmarks are quoted in £k/MW/yr — always convert before comparing.

---

## Current Valuation State

### Locked backtest base case (nb19, `historical_index.py`)
DC=35%, WD cap=£60/MWh, gross (costs excluded for like-for-like vs Modo):

| Stream | 1H | 2H |
|---|---|---|
| DA | £2.1k | £11.3k |
| WD (SP−DA, cap £60) | £32.4k | £40.4k |
| Ancillary (DC/DM/DR/QR) | £11.2k | £11.2k |
| BM fleet avg | £7.2k | £7.9k |
| **Total model** | **£52.9k** | **£70.8k** |
| Modo (Apr 2024–Apr 2026) | £50.8k | £71.9k |
| Gap | −£2.0k ✓ | +£1.1k ✓ |

**This gap is closed. Do not re-open the headroom calibration.**

### LSMC_CLEANUP result (2026-05-17, run_lsmc_sweep.py — 250 paths, 2160 steps, no-stacking, B1 fixed)
| Duration | LSMC £k/MW/yr | Top action | Clip | Stacked |
|---|---|---|---|---|
| 1h | 16.6k | QR=0.25 (66.5%) | 0% | 0% |
| 2h | 34.7k | DC=0.50 (60.7%) | 0% | 0% |
| 3h | 42.9k | DC=0.50, BM=0.25 (61.3%) | 0% | 0% |
| 4h | 43.3k | DC=0.50, BM=0.25 (60.5%) | 0% | 0% |

**2h/1h = 2.08× > Modo 1.52× gate. ALL VALIDATION CHECKS PASSED.**

Note: 1h QR-dominant (no DC) is **physically correct** — a 1h battery cannot sustain 50 MW DC
commitment for 1h in both directions (requires 100.2 MWh swing vs 80 MWh usable range).
Values use raw unanchored bundle prices (no HPFC), so absolute level is lower than HPFC-anchored;
ratios and structural checks are valid.

Remaining gap to Modo (£47.7k/£72.5k): HPFC anchoring ~£10–15k + BM/historical basis refinement.

---

## External Benchmarks

| Benchmark | 1H | 2H | Notes |
|---|---|---|---|
| Modo Energy (May25–May26) | £47.7k | £72.5k | Realized BMU fleet; includes DA, ID, BM, DC/DM/DR, QR/BR, CM |
| Modo Energy (Apr24–Apr26) | £50.8k | £71.9k | Period matching nb19 backtest |
| KYOS GB index (Jan–Dec 2025) | £63k | — | ID + imbalance only; no separate DA; LSMC methodology |
| KYOS GB forecast 2027 | £83.0k avg | — | P10 £72.3k; includes passive imbalance capped at 30% |

Modo index is an **optimal dispatch model on actual prices, BMU fleet only** — not actual fleet revenues.
KYOS index uses intraday perfect foresight for the realized index (not forward). Both are model-to-model comparisons.

**Modo methodology key points (read April 2026):**
- Imbalance = `(Metered − PN − ABSVD) × System price` (passive, ABSVD-adjusted) — near-zero mean
- Wholesale: hourly PN shape → avg(N2EX, EPEX) DA; sub-hourly PN shape → EPEX RPD HH continuous
- DC performance penalties NOT included (no public data) — consistent with project
- Slow Reserve added Apr 2026 — not yet in project
- 2H/1H = 1.52× driven partly by sub-hourly PN volume priced at intraday (continuous market)

---

## Known Bugs and Design Issues

### B1 — Double discount in `cont_beta` forward branch (OUTSTANDING)
**File:** `src/optimisation/lsmc.py`, `forward()` ~line 849

`backward()` regresses on `disc·V(t+1)`, so `cont_beta` ≈ `E[disc·V(t+1)]`.
`forward()` then multiplies by `disc` again: `Q = CF + disc * cont` → effectively `CF + disc²·V(t+1)`.

Per-step bias is tiny (disc ≈ 0.999995/HH) but systematically tilts toward cashflow-now.

**Preferred fix (option 2):** Regress `V_cont` (undiscounted) in `backward()` at ~line 547.
Leave `forward()` unchanged — both branches then predict undiscounted next-step value.
Fix before any production/final-reporting run.

### C1 — `cont_beta` fitted at constant SoC per grid node (design note, not a bug)
Within-cell SoC sensitivity is absent — the coefficients reflect a constant-E surface.
Forward pass uses actual path SoC but the fitted surface doesn't vary within a cell.
Worth bilinear interpolation across `j` if dispatch shows visible plateau artefacts.

---

## Architecture Gotchas

### 1. Backtest headroom ≠ LSMC mode grid — CRITICAL
These are two completely separate systems. Changing one has **zero effect** on the other.

**Backtest** (`src/backtest/ancillary_revenue.py`):
- Fixed fractions: `DEFAULT_HEADROOM = {DC: 0.35, DM: 0.10, DR: 0.05, QR: 0.15}`
- Used by nb19 (`historical_index.py`) only
- DA power = `P_bar × (1 − total_headroom)` passed to rolling LP

**LSMC mode grid** (`src/optimisation/dispatch.py`):
- `dc_levels`, `qr_levels`, `bm_levels` → endogenous decision variables
- Used by nb12, nb13 only
- Changing `DEFAULT_HEADROOM` has zero effect on nb13

### 2. `phase4_all_durations_comparison.csv` is STALE
Contains pre-imbalance-fix values (1h LSMC = £75.9k, imbalance label = "WD/intraday").
Do not cite these figures. They will be overwritten once nb13 is rerun (LSMC_CLEANUP step 3).

### 3. Streamlit / Python version
```bash
C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run streamlit_phase4_sweep.py
```
BM data fetch also uses Python312, not `.venv_test`.

### 4. Git: BESS-UK-OPEN remote
```bash
git push open main --force    # always force — histories have diverged
```
`.vscode` directory has Google Drive permission issue → blocks `git checkout`.
Use `git restore --source=main --worktree -- <files>` instead of full checkout.

### 5. Windows zmq deadlock (fixed)
`~/.ipython/profile_default/startup/00_win_selector_loop.py` — sets SelectorEventLoop.
Already applied. Do not remove.

---

## Pending Work (priority order)

1. ~~**Fix B1**~~ — **DONE** (commit a004651): `cont_beta` now regresses on undiscounted V(t+1); forward applies `disc` once. No double-discount.
2. ~~**LSMC_CLEANUP step 2**~~ — **DONE**: `reserve_sustain_h=1.0` already in `src/config.py` line 235, inherited by nb12.
3. ~~**LSMC_CLEANUP step 1**~~ — **DONE**: `allow_ancillary_stacking=False` added to `enumerate_modes()` in `src/optimisation/dispatch.py`; wired into nb12 `PHASE4_RUNS['medium']`.
4. ~~**LSMC_CLEANUP step 3**~~ — **DONE** (2026-05-17): `run_lsmc_sweep.py` ran 4 durations (250 paths, 2160 steps). All validation checks PASS. Outputs: `lsmc_attribution_{1h..4h}.json`, `phase4_all_durations_comparison.{csv,json}`.
5. ~~**Rerun nb16**~~ — **DONE** (2026-05-17): structural counterfactuals rerun with cleaned basis/no-stacking/BM modes. Outputs: `lsmc_counterfactual_results.{json,png}`.
6. ~~**Capacity Market overlay**~~ — **DONE** (2026-05-17): deterministic £6k/MW/yr overlay added via `scripts/apply_capacity_market_overlay.py`; outputs `capacity_market_overlay.{csv,json}` and appends `Forward simulation (LSMC + CM overlay)` rows to `phase4_all_durations_comparison.{csv,json}`.
7. ~~**Option B1**~~ — **DONE** (2026-05-17): `notebooks/20_historical_lsmc.py` / `.ipynb` runs perfect-foresight LP on actual 2024–26 DA, SP, and WD60 price paths. Notebook is self-contained with visible inputs/cells. Outputs `historical_lsmc_b1_summary.{csv,json,png}`, `historical_lsmc_b1_prices.parquet`, `historical_lsmc_b1_vs_nb13.{csv,png}`, and 1h/2h stacked nr20-vs-nr13 tables `historical_lsmc_b1_nr13_stacked_{table,wide}.csv`.
8. **Slow Reserve** — add to Phase 3 mode grid once QR is stable (SR ≈ QR structure)
9. **Rebuild 5000-path production bundle:**
   ```bash
   python scripts/generate_sim_bundle.py --paths 5000 --steps 17520 --seed 42
   ```

### LSMC_CLEANUP validation checklist — ALL PASSED (2026-05-17)
- ✅ 2h/1h LSMC ratio ≥ 1.30× → 2.08×
- ✅ Continuation clipping = 0.0% across all 4 durations
- ✅ Imbalance attribution 1h: £1.531m/yr (within £0.5–2.5m range)
- ✅ Dominant mode is NOT stacked DC+QR (stacked% = 0.0% all durations)
- ✅ `phase4_all_durations_comparison.csv` updated (2026-05-17)

### nb16 structural counterfactuals — DONE (2026-05-17)
Settings: 250 backward paths, 250 forward paths, 2160 HH, HPFC-anchored prices,
`ridge_alpha=1.0`, EFA-block basis, duration-aware forward window, no DC+QR stacking,
BM modes active except energy-only.

| Case | 1h | 2h | 2h/1h |
|---|---:|---:|---:|
| Baseline full stack (nb13) | £1.66m | £3.47m | 2.08x |
| No imbalance (delta=0) | £1.02m | £2.82m | 2.76x |
| No ancillary (DA+imb+BM) | £1.49m | £2.84m | 1.90x |
| Energy only (DA) | £0.43m | £0.97m | 2.25x |

Implementation note: `delta_imb_scale` is applied by scaling `PathBundle.delta_imb`
for both backward and forward passes; it is not an LSMC core config key.

### Capacity Market overlay — DONE (2026-05-17)
Central assumption: £6.0k/MW/yr deterministic CM revenue, added outside LSMC
dispatch/training. For the 100 MW asset this is £0.6m/year.

| Duration | LSMC | CM | LSMC + CM |
|---|---:|---:|---:|
| 1h | £16.6k | £6.0k | £22.6k |
| 2h | £34.7k | £6.0k | £40.7k |
| 3h | £42.9k | £6.0k | £48.9k |
| 4h | £43.3k | £6.0k | £49.3k |

Replace the flat £6k/MW/yr placeholder with CM register-derived duration-specific
values once the EMRS/CM register extract is added.

### Option B1 historical perfect-foresight LP — DONE (2026-05-17)
Runner: `notebooks/20_historical_lsmc.py`; wrapper notebook:
`notebooks/20_historical_lsmc.ipynb`.

Coverage uses corrected raw files: 2024-04-01 to 2026-04-25, 36,229 aligned
half-hours after inner-joining DA and SP by settlement date/period.

| Stream | 1h | 2h |
|---|---:|---:|
| DA | £19.8k | £36.0k |
| SP perfect foresight | £71.7k | £106.5k |
| WD60 perfect foresight | £67.3k | £100.5k |

Interpretation: these are clairvoyant upper benchmarks on actual prices, not a
realized dispatch strategy. The old `perfect_foresight_summary_{1h,2h}.json`
DA rows had duplicate-era coverage (~72k DA rows); use the new
`historical_lsmc_b1_summary.*` outputs for Option B1.

Additional notebook/comparison work completed (2026-05-17):
- Rebuilt `notebooks/20_historical_lsmc.ipynb` as a self-contained notebook;
  inputs, assumptions, load/align logic, LP runs, output writes, and comparison
  cells are visible directly in the notebook (no `%run` wrapper).
- Added unit conversion from `GBPk/MW/year` to `GBPm/year for 100 MW` via
  `value_gbp_annualized_m = gbp_per_mw_year_k * 100 / 1000`.
- Added direct nr20-vs-nr13 comparison for DA perfect-foresight:
  `historical_lsmc_b1_vs_nb13.{csv,png}`.
- Added stacked nr20/nr13 valuation tables for 1h and 2h only, sorted by
  duration first (all 1h valuations, then all 2h valuations):
  `historical_lsmc_b1_nr13_stacked_table.csv`,
  `historical_lsmc_b1_nr13_stacked_wide.csv`, and
  `historical_lsmc_b1_nr13_stacked.png`.

---

## Streamlit App

Entry point: `streamlit_phase4_sweep.py`

| Section | Content |
|---|---|
| Phase 4: Duration Sweep | nb13 method comparison 1h–4h; LSMC attribution |
| Historical Index vs Modo | nb19 backtest vs Modo Energy (4 pages) |
| Forward vs Realized | Three-way: nb13 WD / nb19 realized / nb13 LSMC (4 pages) |

Public repo: `https://github.com/dmitry-goryunov/BESS-UK-OPEN.git` (remote: `open`)
Live app: `https://bess-uk-open.streamlit.app`
