# BESS Valuation — Value Items Reference

## Revenue sources

| Source | Formula (per HH) | Driver |
|---|---|---|
| DA energy | `P_da × net_mw × dt` | Intraday buy-low/sell-high spread |
| Imbalance uplift | `Δ_imb × d_mw × dt` | System-short periods; discharge only |
| DC reserve | `π_DC × r_dc_mw × dt` | EFA-block DC clearing price |
| QR reserve | `π_QR × r_qr_mw × dt` | Quick Reserve clearing price |
| Degradation cost | `−λ_deg × (d_mw + c_mw) × dt` | Shadow price on throughput |
| Variable O&M | `−VoM × (d_mw + c_mw) × dt` | £1.2/MWh on throughput |

`net_mw` is positive when discharging, negative when charging.
Imbalance uplift only accrues on discharge (`d_mw = max(net_mw, 0)`).

---

## Valuation benchmarks

### Rolling Intrinsic (V_RI)
Deterministic LP solved daily over a rolling 48-HH window using the DA forward
strip. Re-solved at each EFA gate (every 8 HH = 4h); first `gate_hh` decisions
applied; repeat.

- DA-only: no ancillary, no imbalance
- 100% of capacity available for DA cycling
- Optimal timing of charge/discharge given known prices for the next 24h
- A conservative **lower bound** on V_LSMC

**Reference case (2h, 100 MW / 200 MWh, HPFC-anchored prices): ~£34k/MW/yr**

### LSMC (V_LSMC)
Stochastic backward induction over (SoC, SoH) grid. Co-optimises DA, imbalance,
DC and QR simultaneously. Accounts for uncertainty in all price processes.

**Reference case: ~£72–77k/MW/yr**

### Perfect foresight (DA energy only)
Daily LP with full knowledge of realised DA prices. Sets the DA-energy ceiling
(no stochastic uncertainty, no ancillary). Slightly above V_RI because it sees
the full day rather than rolling 24h windows.

**Reference case: ~£35k/MW/yr**

---

## Revenue attribution: two distinct concepts

### 1. Realised component cashflow (implemented in `LSMCResult.cf_breakdown`)

Records, for the joint-optimal policy, how much cash came from each source:

```

---

## Session minutes - 2026-05-05

### Context

The session reviewed Phase 4 BESS duration-sweep results, especially the
relationship between WD rolling intrinsic, base LSMC, and a new WD-like LSMC
experiment. The working question was whether LSMC can be parameterised to mimic
WD rolling intrinsic for 1h, 2h, 3h and 4h batteries.

### Notebook 15: WD rolling intrinsic vs WD-like LSMC

Created `notebooks/15_wd_rolling_vs_wd_like_lsmc.ipynb`.

The notebook runs notebook 12 in a WD-like LSMC configuration and compares it
against the existing WD rolling intrinsic outputs.

WD-like LSMC changes relative to base Phase 4 LSMC:

| Parameter | Base Phase 4 LSMC | WD-like LSMC | Value impact vs base |
|---|---:|---:|---|
| `dc_levels` | `[0.0, 0.5]` originally | `[0.0]` | Decrease |
| `qr_levels` | `[0.0, 0.25]` originally | `[0.0]` | Decrease |
| `vom_gbp_mwh` | `1.2` | `0.0` | Increase |
| `lambda_deg_init_gbp_mwh` | `6.0` | `0.0` | Increase |
| `imbalance_cashflow_mode` | `discharge_only` | `net` | Ambiguous; lower in this run |
| `net_levels` | `[-1, -0.5, 0, 0.5, 1]` | same | No change |
| DA and imbalance paths | HPFC-anchored simulated paths | same | No change |
| Monte Carlo paths | `500` | `500` | No change |
| `fwd_window_hh` | `16` HH | same | No change |
| `n_soc_nodes` | `max(9, 3 * duration_h)` | same | No change |

Outputs saved:

- `data/processed/phase4_wd_rolling_vs_wd_like_lsmc.csv`
- `data/processed/phase4_wd_rolling_vs_wd_like_lsmc.json`
- `data/processed/phase4_wd_rolling_vs_wd_like_lsmc.png`

Result:

| Duration | WD rolling intrinsic, GBPm/year | WD-like LSMC, GBPm/year | Difference, GBPm/year |
|---:|---:|---:|---:|
| 1h | 2.974 | 0.635 | -2.339 |
| 2h | 4.394 | 1.576 | -2.818 |
| 3h | 5.667 | 2.770 | -2.897 |
| 4h | 6.859 | 3.496 | -3.363 |

Interpretation: the WD-like LSMC has the same broad value stack as WD rolling
intrinsic, but it is still not the same optimisation problem. WD rolling
intrinsic solves a rolling deterministic LP over visible gate-window prices.
LSMC uses a discrete action grid and continuation-value regression. Therefore
removing ancillary and costs, and switching imbalance to net cashflow, is not
sufficient to make LSMC reproduce WD rolling intrinsic.

### Code change: imbalance cashflow mode

Added optional `imbalance_cashflow_mode` support in:

- `src/optimisation/dispatch.py`
- `src/optimisation/lsmc.py`

Default remains:

```python
imbalance_cashflow_mode = "discharge_only"
```

This preserves the base Phase 4 LSMC behaviour unless a notebook explicitly
sets:

```python
imbalance_cashflow_mode = "net"
```

Notebook 15 uses `"net"` to mimic the WD rolling intrinsic convention where
the price is effectively `DA + delta_imb` on net dispatch.

Targeted verification passed:

```text
2 passed, 32 deselected
```

### Ancillary interpretation

`dc_levels = [0.0, 0.5]` means the LSMC action grid can choose no Dynamic
Containment or reserve 50% of rated power for DC. For a 100 MW battery, this is
0 MW or 50 MW.

`qr_levels = [0.0, 0.25]` means no Quick Reserve or reserve 25% of rated power.
For a 100 MW battery, this is 0 MW or 25 MW.

DC and QR differ as follows:

| Item | DC | QR |
|---|---|---|
| Full name | Dynamic Containment | Quick Reserve |
| Main purpose | Very fast frequency response | Fast reserve / balancing support |
| Typical response | Seconds | Minutes / short notice |
| Model role | Capacity-like reserve revenue | Reserve availability revenue |
| Opportunity cost | Uses power headroom/footroom | Uses power availability |

Realistic DC commitment depends on market liquidity, auction saturation,
cleared volume, DC prices, competition from other batteries, and SoC
headroom/footroom. A fixed `dc_level = 0.5` is plausible technically, but may
be optimistic economically if the model assumes the whole quantity is always
accepted at the simulated price.

### Base-case ancillary update

The base Phase 4 LSMC action grid in `notebooks/12_phase4_method_comparison.ipynb`
was changed to use half of the previous maximum ancillary levels:

| Product level | Previous base case | New base case | Meaning for 100 MW |
|---|---:|---:|---:|
| `dc_levels` | `[0.0, 0.5]` | `[0.0, 0.25]` | max DC now 25 MW |
| `qr_levels` | `[0.0, 0.25]` | `[0.0, 0.125]` | max QR now 12.5 MW |

There is no separate executable `dr_levels` action grid in Phase 4. The
available ancillary action levels are currently `dc_levels` and `qr_levels`.

Because this changes the base LSMC action space, notebook 13 must be rerun to
refresh the Phase 4 duration-sweep outputs.

### Deleted outputs before rerun

The old notebook 13 outputs were deleted so notebook 13 will regenerate them
from the updated base case instead of reusing stale results.

Deleted output groups:

- `data/processed/phase4_method_comparison_1h..4h.*`
- `data/processed/lsmc_attribution_1h..4h.*`
- `data/processed/phase4_all_durations_comparison.*`
- `data/processed/phase4_all_durations_attribution.*`
- `data/processed/phase4_sweep_run_log.json`

Files from notebook 14 and notebook 15 were left intact.

### Rerun completed

`notebooks/13_phase4_duration_sweep.ipynb` was rerun and the base Phase 4
duration-sweep outputs were regenerated using:

```python
dc_levels = [0.0, 0.25]
qr_levels = [0.0, 0.125]
```

Updated base LSMC annualised values:

| Duration | Base LSMC, GBPm/year |
|---:|---:|
| 1h | 7.586 |
| 2h | 8.172 |
| 3h | 8.669 |
| 4h | 8.951 |
da        = P_da × net_mw × dt          (can be negative — see below)
imbalance = Δ_imb × d_mw × dt
dc        = π_DC × r_dc_mw × dt
qr        = π_QR × r_qr_mw × dt
costs     = (λ_deg + VoM) × throughput
```

Useful for: "where did the cash actually come from?"
**Not** suitable for measuring the marginal value of each revenue stream.

### 2. Marginal (counterfactual) value

Zero out one stream at a time, re-run the full LSMC, diff the totals:

```
V_DA_marginal  = V_LSMC(full) − V_LSMC(Δ_imb=0, π_DC=0, π_QR=0)
V_imbalance    = V_LSMC(full) − V_LSMC(Δ_imb=0)
V_ancillary    = V_LSMC(full) − V_LSMC(π_DC=0, π_QR=0)
```

Useful for: "what is each revenue stream worth to this asset?"
Expensive: requires a full backward + forward pass per counterfactual.

---

## Why the LSMC DA component ≠ V_RI

V_RI and the LSMC DA cashflow component measure fundamentally different things:

| | V_RI | LSMC DA component |
|---|---|---|
| Capacity available | 100% for DA | Shared with DC/QR headroom |
| Dispatch timing | Optimised purely for DA spread | Driven by imbalance + ancillary signals |
| Price uncertainty | None (deterministic LP) | Full stochastic |

When the policy co-optimises all sources, power committed to DC/QR reserve is
not available for net dispatch. Charge/discharge timing is driven by the
imbalance and ancillary signals rather than the DA intraday peak/trough.
Efficiency losses (88% RTE means buying 13.6% more energy than sold) then
outweigh the incidental DA spread captured, making the DA component slightly
**negative** even with correct HPFC-anchored prices.

This is rational: the policy chose  
`+£68k imbalance + £15k ancillary − £4k DA drag = £72k total`  
over the alternative of `+£34k DA only`.

The gap between V_RI (~£34k) and the LSMC DA component (~−£4k) represents the
**capacity opportunity cost** — cycles that V_RI would use for DA arbitrage are
redeployed to higher-value imbalance and ancillary dispatch.

---

## Co-optimisation constraints (per HH)

```
|net_mw| + r_dc_mw + r_qr_mw  <=  P_bar           (power headroom)
E_t − (r_dc + r_qr) × dt / η_d  >=  E_min          (energy headroom — discharge)
E_t + (r_dc + r_qr) × η_c × dt  <=  E_max(SoH)     (energy headroom — charge)
```

---

## Price process summary

| Process | Model | Key parameters |
|---|---|---|
| DA baseload | Schwartz-Smith two-factor | κ, σ_χ, μ_ξ, σ_ξ, ρ |
| Intraday shape | HPFC × SS relative move | Hourly multipliers 0.71–1.46 |
| Imbalance basis | OU + asymmetric jumps | θ_Δ, σ_Δ, λ_J, jump asymmetry |
| DC clearing | AR(1) per EFA block + saturation | φ_DC, saturation exponent γ ≈ 2.1 |
| QR clearing | AR(1) per EFA block | φ_QR |

HPFC anchoring: `P_t = hpfc_anchor[t] × exp(Δ ln P_SS[t])` rescales each
simulated path to start from the HPFC level while preserving relative SS dynamics.

---

## Phase 4 duration sweep findings (2026-05-04)

### Setup

100 MW GB fast-cycle BESS valued across four durations (1h / 2h / 3h / 4h) using
five methods: initial hourly intrinsic, DA rolling intrinsic, WD rolling
intrinsic, forward simulation (LSMC), and perfect foresight (DA energy).
500 HPFC-anchored Monte Carlo paths, 4,320 half-hour simulation horizon.

### Result

| Duration | Hourly intrinsic | WD rolling intrinsic | LSMC | PF (DA energy) |
|---|---|---|---|---|
| 1h | £0.97m | £2.97m | **£8.51m** | £1.79m |
| 2h | £1.86m | £4.39m | **£8.36m** | £3.37m |
| 3h | £2.60m | £5.67m | **£7.53m** | £4.69m |
| 4h | £3.17m | £6.86m | **£7.29m** | £5.71m |

All values are annualised GBPm/year for a 100 MW asset.

2026-05-05 correction: the declining LSMC duration curve should be treated as
a pre-fix numerical artifact, not a stable economic conclusion. The forward
policy evaluation was snapping each path's actual SoC down to the lower grid
node for feasibility and continuation. That is increasingly punitive as MWh
duration grows and grid spacing widens. The solver now evaluates feasibility
and next SoC from the actual path state, and interpolates continuation between
SoC grid nodes; duration-sweep artefacts need to be regenerated before this
table is used as evidence.

2026-05-05 follow-up: the remaining 1h-above-4h result was traced to the
Phase 13 notebook override `continuation_value_cap_gbp = 3_000_000`. On the
4,320 half-hour horizon this cap materially clipped continuation regressions,
especially for 4h assets, making the long-duration policy idle too often.
A controlled full-horizon diagnostic with 120 backward / 120 forward paths
showed:

| Cap | 1h LSMC | 4h LSMC | 1h clip obs | 4h clip obs |
|---|---:|---:|---:|---:|
| GBP3m | GBP7.63m | GBP5.01m | 11.4% | 50.9% |
| GBP10m+ | GBP7.74m | GBP9.79m | 0.0% | 0.0% |

Notebook 12 now uses the production cap from `LSMC_CFG`
(`continuation_value_cap_gbp = 25_000_000`) and fails fast if continuation
clipping becomes material. Existing Phase 13 files generated under the low cap
should be discarded and regenerated.

Post-cap rerun (2026-05-05 14:41) produced the mechanically coherent duration
curve below. The level is higher than the clipped run because the 2h-4h policies
are no longer forced to discard continuation value.

| Duration | WD rolling | LSMC | LSMC option premium |
|---|---:|---:|---:|
| 1h | GBP2.97m | GBP9.04m | +GBP6.06m |
| 2h | GBP4.39m | GBP10.15m | +GBP5.75m |
| 3h | GBP5.67m | GBP10.62m | +GBP4.95m |
| 4h | GBP6.86m | GBP10.92m | +GBP4.06m |

Diagnostics on the executed notebooks: continuation clipping was 0.0% for all
durations, sampled regression condition number was about 12.7, and sampled rank
deficiency was 0. The high level is therefore not a cap/clipping explosion.

Important interpretation caveat: current LSMC treats the simulated imbalance
basis as an actionable half-hourly signal in the dispatch decision. That makes
the post-cap LSMC an optimistic full-stack operational value. A more conservative
central case should add an imbalance information lag or forecast haircut before
treating the LSMC level as bankable market value.

### Diverging sensitivities

Rolling intrinsic and perfect foresight **correctly increase** with duration:
energy arbitrage (overnight trough → morning peak) scales with stored MWh.

LSMC **decreases** with duration. The LSMC "option value" above the rolling
intrinsic collapses from +£5.54m (1h) to +£0.43m (4h):

| Duration | WD rolling | LSMC | LSMC option premium |
|---|---|---|---|
| 1h | £2.97m | £8.51m | +£5.54m |
| 2h | £4.39m | £8.36m | +£3.97m |
| 3h | £5.67m | £7.53m | +£1.86m |
| 4h | £6.86m | £7.29m | +£0.43m |

### Root cause: MW-based revenue dominance + SoC grid resolution

The dominant revenue stream is **WD/intraday (imbalance)**, which is MW-based:
`CF_imb = δ_imb × net × P_bar × dt`. This revenue does not scale with battery
energy capacity — both a 1h and a 4h battery earn the same imbalance cashflow
per half-hour if they apply the same dispatch action.

As duration grows, the LSMC continuation value regression resolves the SoC
landscape less accurately. With a fixed 9 SoC nodes, grid spacing is:

| Duration | Usable MWh | Spacing (9 nodes) | Spacing (12 nodes) |
|---|---|---|---|
| 1h | 80 MWh | 10.0 MWh | - |
| 4h | 320 MWh | 40.0 MWh | 29.1 MWh |

The coarse grid for a 4h battery makes the continuation value regression
over-smooth, causing the policy to favour fast cycling (imbalance mode)
over multi-hour energy arbitrage. To achieve equivalent resolution for 4h
would require ~36 SoC nodes, making the backward pass ~4× slower.

### Attribution

```
Component          1h        2h        3h        4h
─────────────────────────────────────────────────────
HPFC anchor      -0.68m    -0.47m    -0.28m    -0.19m
DA surprise      -0.05m    -0.01m    +0.02m    +0.05m
WD/intraday      +9.39m    +8.41m    +7.76m    +7.21m
DC ancillary     +0.51m    +0.61m    +0.61m    +0.58m
QR ancillary     +0.32m    +0.72m    +0.28m    +0.44m
Costs (deg+VOM)  -0.98m    -0.91m    -0.86m    -0.80m
─────────────────────────────────────────────────────
Total             8.51m     8.36m     7.53m     7.29m
```

The HPFC anchor component becomes *less negative* with duration (the policy
holds charge longer, capturing more energy arbitrage), but the gain (+£0.5m
from 1h to 4h) is outweighed by the imbalance revenue loss (−£2.2m).

### Interpretation

The result is partly genuine: in a GB market where imbalance and ancillary
revenues dominate and are MW-based, a larger energy capacity does not
proportionally increase total value. It is also partly numerical: the LSMC
policy is suboptimal for 4h batteries given the current SoC grid resolution
and 4,320 HH training horizon.

The rolling intrinsic remains the correct lower bound for energy-only
comparisons across durations. LSMC captures the full-stack option value for
1–2h batteries but becomes increasingly conservative for 3–4h batteries.

---

## Intrinsic vs extrinsic split

```
V_LSMC  =  V_RI  +  (V_LSMC − V_RI)
         = intrinsic  +  extrinsic

Intrinsic  ≈ £34k/MW/yr   — deterministic DA arbitrage value
Extrinsic  ≈ £38k/MW/yr   — option value from uncertainty:
                             imbalance spikes, ancillary, flexible re-dispatch
```

---

## Phase 3 results — BM revenue + duration sensitivity (2026-05-14)

### Setup

Phase 3 adds Balancing Mechanism (BM) revenue to the full-stack LSMC. The mode
grid gains a `r_bm_frac` dimension: `bm_levels = [0.0, 0.25, 0.5]`. The BM
offer price (`pi_bm`, mean ≈ £79.9/MWh) is activated with probability
`p_activation = 0.12`.

Run config: `medium` (500 paths, 4 320 HH horizon, `dc_levels=[0.0, 0.5]`,
`qr_levels=[0.0, 0.25]`). Ancillary prices: calibrated (`DC_Low.mu = 5.1`,
`QR_Pos.mu = 3.6 £/MW/h`) — see `data/processed/ancillary_params.json`.

### Results

| Duration | LSMC (£m/yr) | £k/MW/yr | DC | QR | BM | Imbalance | Costs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1h | 1.414 | 14.1 | 0.000 | +0.377 | +0.146 | +1.092 | −0.326 |
| 2h | 3.238 | 32.4 | +0.999 | +0.323 | +0.355 | +1.574 | −0.527 |

**2h/1h ratio = 2.29×** — Phase 3 gate (≥ 1.52×) **PASSED**.

*(Pre-calibration run with PEAK_PRICE defaults (DC=17, QR=12 £/MW/h) gave
1h=£14.7k, 2h=£44.5k, ratio=3.03×. Calibrated ancillary prices reduce DC revenue
3.1× for 2h, leaving the gate comfortably passed.)*

### Why DC = 0 for 1h (physically correct, not a bug)

`reserve_sustain_h = 1.0` hour (LSMC config). DC ancillary requires the battery
to sustain its committed headroom for one full hour in **both** up and down
directions. For DC 50% (50 MW commitment):

| Constraint | Calculation | Value |
|---|---|---|
| Downward energy needed | 50 MW × 1.0 h / η_d | 53.3 MWh |
| Upward energy needed | 50 MW × 1.0 h × η_c | 46.9 MWh |
| **Total swing required** | | **100.2 MWh** |
| Usable range (SoC 10–90%) | | **80 MWh** |

100.2 MWh > 80 MWh → DC 50% is **geometrically impossible** for a 1 h battery.
The feasibility window [63.3, 43.1] MWh is inverted: there is no SoC at which
both the downward and upward headroom requirements can be satisfied simultaneously.

DC feasibility by duration (P_bar = 100 MW, `dc_levels = [0.0, 0.5]`):

| Duration | Usable (MWh) | Swing needed (MWh) | Feasible window | Width |
|---:|---:|---:|---|---:|
| 1h | 80 | 100.2 | — impossible — | 0 MWh |
| 2h | 160 | 100.2 | [73.3, 133.1] MWh | 59.8 MWh (37%) |
| 3h | 240 | 100.2 | [83.3, 223.1] MWh | 139.8 MWh (58%) |
| 4h | 320 | 100.2 | [93.3, 313.1] MWh | 219.8 MWh (69%) |

This is the correct physical constraint: short-duration batteries genuinely
cannot provide DC service under a 1-hour sustain requirement.

**Footnote — DC 25%**: with `r_dc_frac = 0.25` (25 MW), the swing requirement
drops to 50.1 MWh and the feasibility window is [36.6, 66.5] MWh (37% of usable
range). The current `dc_levels = [0.0, 0.5]` excludes this option. Adding
`dc=0.25` to the mode grid would give the 1h battery real DC revenue and reduce
the 2h/1h ratio (still well above the 1.52× gate).

### Comparison: LSMC vs other methods (1h and 2h)

| Method | 1h (£k/MW/yr) | 2h (£k/MW/yr) |
|---|---:|---:|
| Initial hourly intrinsic | 9.7 | 18.6 |
| DA rolling intrinsic | 16.8 | 31.9 |
| WD rolling intrinsic | 56.5 | 82.0 |
| **LSMC (Phase 3)** | **14.1** | **32.4** |
| Perfect foresight (DA) | 16.8 | 31.9 |

WD rolling >> LSMC because WD has a 4-hour look-ahead (rolling LP sees gate
prices), while LSMC is non-anticipative. With the corrected mean-zero imbalance
calibration (p_pos = 0.358), LSMC cannot exploit intraday spreads without
predictive features. WD rolling intrinsic represents an upper bound on what a
forecast-equipped intraday strategy could earn.

Notable: 2h LSMC (£32.4k) ≈ DA rolling (£31.9k) ≈ Perfect Foresight (£31.9k).
The full-stack LSMC policy captures essentially all available DA-plus-ancillary
value for a 2h battery with calibrated prices.

---

## Phase 4 method comparison — interpretation guide

The notebook 13 comparison table (`83225d9b`) ranks six methods in increasing
information / complexity order. This section explains what each method captures
and how to read the gaps between them.

### What each method is

| Method | Pricing information used | Revenue streams | Role |
|---|---|---|---|
| **Initial hourly intrinsic** | Day-ahead hourly prices (deterministic HPFC LP) | DA energy only | Simplest lower bound; no uncertainty, no rolling |
| **DA rolling intrinsic** | DA prices at EFA gate (48-HH window, 8-HH roll) | DA energy only | Conservative floor; best a DA-only automated strategy can do |
| **WD rolling intrinsic** | Within-day gate prices (8-HH window, 8-HH roll) | DA + intraday | Oracle benchmark; uses actual gate prices the battery cannot yet see |
| **MODO style forward look** | Within-day gate prices, capped at ±£60/MWh | DA + intraday | Third-party calibration point (MODO Energy model equivalent) |
| **Forward simulation (LSMC)** | No look-ahead (non-anticipative) | Full stack: DA, imbalance, DC, QR, costs | The non-anticipative reference; what a real battery earns with optimal policy |
| **Perfect foresight (DA energy)** | All future DA prices perfectly known | DA energy only | Ceiling for DA-only arbitrage; excludes ancillary and imbalance entirely |

### Why WD rolling intrinsic > LSMC (not a model deficiency)

WD rolling intrinsic feeds the LP the *actual* within-day gate prices as inputs.
A real battery cannot see those prices at the time it must commit. LSMC must act
before the gate price is revealed and form an expectation from the simulated
path state. The gap WD − LSMC is therefore the value of perfect within-day
price information, not a flaw in the LSMC. At longer durations this gap narrows
because a larger energy reservoir reduces the urgency of timing individual
half-hours precisely.

### Why perfect foresight (DA) < LSMC

Perfect foresight sees all future DA prices but earns only DA energy revenue.
LSMC adds imbalance, DC, and QR simultaneously. The stochastic ancillary and
imbalance streams more than compensate for DA uncertainty: LSMC beats perfect
foresight on total value even though it does not know future prices.

### MODO style forward look

Re-runs WD rolling intrinsic with the intraday price signal capped at ±£60/MWh.
MODO Energy's published valuations are understood to use a similar cap to limit
exposure to extreme intraday spikes. Comparing MODO style to the base WD rolling
(cap ≈ £10/MWh) shows how much WD value depends on uncapped spike events.

### How to read the P5–P95 range

The range shows path-to-path variability across Monte Carlo scenarios. A narrow
band (e.g. initial hourly intrinsic, n=1) means fully deterministic. A wide band
means the revenue is lumpy: good in some market regimes, poor in others. LSMC
bands are narrower than perfect foresight because the non-anticipative policy
stabilises dispatch across scenarios rather than chasing each path's peak.

### How to read % of LSMC

The ratio-to-LSMC table scales every method to LSMC = 100%.

| Ratio range | Interpretation |
|---|---|
| > 100% | Method uses look-ahead information unavailable to a real battery |
| ≈ 100% | Captures equivalent total value by a different route |
| 80–99% | Slightly below LSMC; missing some stochastic or ancillary value |
| < 80% | Material undercount — missing revenue streams or significant information disadvantage |

DA rolling intrinsic runs at roughly 75–85% of LSMC across durations; it
captures most DA energy value but is blind to imbalance and ancillary. WD rolling
runs at 110–140% because of the gate-price oracle. Perfect foresight (DA only)
runs at 70–82% because the certain DA gains do not offset the missing ancillary
stack.

### What the table is and is not

The table uses 125 debug-mode Monte Carlo paths (1 080 half-hour horizon) and is
suitable for comparing relative method ranking across durations. The absolute
£k/MW/yr levels are calibrated-debug numbers and should be compared against the
full 500-path medium run (4 320 HH) before using as bankable figures.
