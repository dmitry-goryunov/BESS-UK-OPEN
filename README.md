# BESS UK â€” Stochastic Valuation

Live app: https://bess-uk-open-aw2qfunhxjrab9mr8bmhsh.streamlit.app/

A stochastic mark-to-market valuation framework for a 100 MW GB fast-cycle Battery
Energy Storage System, valued across 1h / 2h / 3h / 4h durations using LSMC
(Longstaff-Schwartz Monte Carlo) co-optimisation, calibrated against the Modo
Energy public GB BESS index.

## What this project does

Four things, exposed via the Streamlit app:

1. **Forward stochastic valuation (LSMC)** â€” Non-anticipative policy co-optimising
   DA energy, imbalance basis, DC, QR, and BM under a Schwartz-Smith two-factor
   price process with OU+jump imbalance and AR(1) ancillary clearing.
2. **Historical backtest (nb19)** â€” Realised-price rolling LP + fixed-headroom
   ancillary + fleet-average BM, validated against the Modo public index.
3. **Historical LSMC B1 (nb20)** â€” actual 2024â€“26 DA / SP / WD60
   perfect-foresight LP, converted to GBPm/year and stacked against nr13 1h/2h
   forward benchmarks.
4. **Forward vs realised comparison** â€” three-way view of nb13 forward (WD rolling,
   LSMC) against nb19 realised.

## Headline numbers

### Historical backtest vs Modo (Apr 2024 â€“ Apr 2026, gross basis)

| Stream | 1H | 2H |
|---|---|---|
| DA | Â£2.1k | Â£11.3k |
| WD (SPâˆ’DA, cap Â£60) | Â£32.4k | Â£40.4k |
| Ancillary (DC/DM/DR/QR) | Â£11.2k | Â£11.2k |
| BM fleet avg | Â£7.2k | Â£7.9k |
| **Model total** | **Â£52.9k** | **Â£70.8k** |
| Modo index | Â£50.8k | Â£71.9k |
| Gap | âˆ’Â£2.0k âœ“ | +Â£1.1k âœ“ |

Backtest base case is locked: DC headroom 35%, WD cap Â£60/MWh, costs excluded
for like-for-like vs Modo. See `CLAUDE.md` for full asset config.

### Forward LSMC (2026-05-17 LSMC_CLEANUP sweep, B1 fixed, no-stacking, reserve_sustain_h=1.0)

| Duration | LSMC Â£k/MW/yr | Top action |
|---|---|---|
| 1h | 16.6 | QR=0.25 (66.5%) â€” DC=0 by physical infeasibility |
| 2h | 34.7 | DC=0.50 (60.7%) |
| 3h | 42.9 | DC=0.50, BM=0.25 (61.3%) |
| 4h | 43.3 | DC=0.50, BM=0.25 (60.5%) |

**2h/1h = 2.08Ã—** â€” clears Modo 1.52Ã— gate. All validation checks PASSED
(continuation clipping 0% across all durations, no stacked dominant mode).

The forward LSMC sits below the Modo all-in level because the sweep used raw
unanchored bundle prices (no HPFC anchor in this run). Capacity Market
(~Â£6k/MW/yr) is now added as a deterministic overlay. Remaining gap to
Modo (Â£47.7k / Â£72.5k): HPFC anchoring ~Â£10â€“15k + BM / historical-basis refinement.
See `CLAUDE.md` for current status.

## Run locally

```bash
pip install -r requirements.txt
# Windows: use Python 3.12 explicitly for Streamlit / BM fetch
python -m streamlit run streamlit_phase4_sweep.py
```

## Repository layout

| Area | Where |
|---|---|
| LSMC core, dispatch, processes | `src/optimisation/`, `src/processes/` |
| Backtest engine | `src/backtest/` |
| Data fetchers (Elexon, NESO) | `src/data/` |
| Notebooks (data fetch â†’ calibration â†’ LSMC â†’ backtest) | `notebooks/` |
| Streamlit dashboard | `streamlit_phase4_sweep.py` |
| Reference docs | `CLAUDE.md` (master), `docs/`, `CHANGELOG.md` |

## Methodology in one screen

- **DA baseload:** Schwartz-Smith two-factor, HPFC-anchored
- **Intraday shape:** HPFC Ã— SS relative move (hourly multipliers 0.71â€“1.46)
- **Imbalance basis (SPâˆ’DA):** OU + asymmetric jumps; post-calibration p_pos = 0.358,
  implied stationary mean â‰ˆ âˆ’Â£0.41/MWh. Signal lag: dispatch uses delta[tâˆ’1] to
  avoid clairvoyant settlement.
- **DC / QR clearing:** AR(1) per EFA block with saturation curve
- **BM:** Mode-grid `bm_levels = [0.0, 0.25, 0.5]` with offer price Ï€_bm â‰ˆ Â£79.9/MWh,
  activation probability 0.12
- **Co-optimisation:** |net| + r_dc + r_qr + r_bm â‰¤ P_bar, with energy headroom
  `r Ã— sustain_h / Î·_d â‰¤ E âˆ’ E_min` (sustain_h = 1.0h central, no DC+QR stacking)

## Data files in `data/`

| File | Description |
|---|---|
| `data/processed/phase4_all_durations_comparison.csv` | LSMC method comparison Ã— duration (refreshed by `run_lsmc_sweep.py` 2026-05-17) |
| `data/processed/capacity_market_overlay.csv` | Deterministic CM overlay audit rows (GBP6k/MW/yr central) |
| `data/processed/phase4_all_durations_attribution.csv` | Per-stream LSMC attribution Ã— duration |
| `data/processed/historical_lsmc_b1_summary.csv` | nb20 historical B1 values on actual DA / SP / WD60 prices |
| `data/processed/historical_lsmc_b1_vs_nb13.csv` | Direct nr20 DA perfect-foresight vs nr13 DA perfect-foresight comparison |
| `data/processed/historical_lsmc_b1_nr13_stacked_table.csv` | 1h/2h stacked nr20 + nr13 comparison table, sorted by duration |
| `data/processed/historical_index_with_bm.png` | nb19 realised index vs Modo |
| `MODO 1H.csv` / `MODO 2H.csv` | Modo public index (source of truth for benchmarking) |

## Status notes

The forward LSMC numbers above are from the 2026-05-17 LSMC_CLEANUP sweep
(`run_lsmc_sweep.py`). The imbalance calibration was corrected in May 2026
(see `CHANGELOG.md`); any earlier write-up citing 1h LSMC â‰ˆ Â£75.9k/MW/yr or
imbalance attribution â‰ˆ Â£7â€“9m/yr predates that fix and should not be cited.

Public repo: https://github.com/dmitry-goryunov/BESS-UK-OPEN
