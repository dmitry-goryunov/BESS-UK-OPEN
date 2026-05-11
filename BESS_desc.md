# BESS Valuation — Model Reference

_Last updated: 2026-05-10 (anti-stall). All values are for a 100 MW GB LFP asset unless noted._

---

## 1. Revenue sources modelled

| Stream | Formula (per HH) | Status |
|---|---|---|
| DA energy | `P_da × net_mw × dt` | Fully modelled |
| Imbalance proxy | `Δ_imb × d_mw × dt` | Modelled as BM/ID substitute (lagged signal) |
| DC reserve | `π_DC × r_dc_mw × dt` | Fully modelled, co-optimised |
| QR reserve | `π_QR × r_qr_mw × dt` | Fully modelled, co-optimised |
| BM headroom | `π_bm × r_bm_mw × p_act × dt` | Process + dispatch modelled; contribution currently small |
| Degradation cost | `−λ_deg × (d_mw + c_mw) × dt` | Shadow price £6/MWh throughput |
| Variable O&M | `−£1.2/MWh × throughput` | Fully modelled |
| Balancing Mechanism (BOA) | — | Headroom in dispatch; cashflow stochastic via pi_bm |
| Capacity Market | — | **Not modelled**; deterministic contracted overlay |
| Dynamic Moderation / Regulation | — | **Not modelled** separately; headroom only |

`net_mw` is positive on discharge. Imbalance accrues on discharge only
(`d_mw = max(net_mw, 0)`). The dispatch signal uses `delta_imb[t−1]`
(1 HH lag); settlement cashflow uses `delta_imb[t]`.

---

## 2. Price processes

### 2.1 Schwartz-Smith two-factor baseload

```
ln P_t = χ_t + ξ_t + f(t)         f(t) = seasonal + EFA peak shape

dχ = −κ·χ·dt + σ_χ·dW_χ           κ = 0.08, σ_χ = 12 £/MWh
dξ = μ_ξ·dt  + σ_ξ·dW_ξ           μ_ξ = 0 (risk-neutral), σ_ξ = 8
corr(dW_χ, dW_ξ) = 0.30

Forward anchor: 76.7 £/MWh (KYOS Feb 2026 GB 10yr baseload)
```

### 2.2 Hourly shape — PCA

```
ln P_{h,t} = ln P_t + Σ_k λ_k(t) · φ_k(h)
dλ_k = −α_k·λ_k·dt + σ_λk·dW_λk      3 factors; α = [0.20, 0.35, 0.50]
```

### 2.3 Imbalance basis — OU + asymmetric jumps

```
P_IMB,t = P_DA,t + Δ_t
dΔ = −θ·Δ·dt + σ·dW + J_t

Config fallback (src/config.py):
  θ = 0.40, σ = 35, λ_J = 0.015/HH, μ_pos = 150, μ_neg = 80, p_pos = 0.55

Calibrated artifact (data/processed/imbalance_params.json) — corrected May 2026:
  θ = 0.83, σ = 18.65, λ_J = 0.043/HH, μ_pos = 109.8, μ_neg = 74.1, p_pos = 0.358
  stationary mean ≈ −0.41 £/MWh

Bug history: the original calibration used provider-level DA rows with
72,258 duplicate/zero-price N2EX rows. Filtering to volume-weighted MID
per SP reduced the stationary mean from £48.55/MWh to −0.41/MWh.
```

### 2.4 Ancillary clearing — AR(1) per EFA block

```
π_{k,b,t+1} = φ_k · π_{k,b,t} + ε         φ: DC 0.75, QR 0.80
Saturation: π_k,t ~ p_res_k · (1 − Q_fleet/Q_req)^γ    γ = 2.1
```

### 2.5 Balancing Mechanism — OU

```
dπ_bm = −κ_bm·(π_bm − μ_bm)·dt + σ_bm·dW_bm
  κ_bm = 1.5/yr, μ_bm = 80 £/MWh, σ_bm = 30 £/MWh
  p_activation = 0.12 per HH (≈6 BOA calls/day across fleet)
  sustain_hh = 4 (2h sustained headroom required)

Implemented in: src/processes/bm.py, src/processes/simulate.py
```

### 2.6 Joint correlation matrix

```
           χ      ξ      λ₁     λ₂     Δ      π_DC   π_bm
χ         1.00   0.30   0.45   0.20   0.55  -0.25   0.40
ξ         0.30   1.00   0.15   0.05   0.10  -0.10   0.10
λ₁(level) 0.45   0.15   1.00   0.20   0.30  -0.20   0.20
λ₂(slope) 0.20   0.05   0.20   1.00   0.15  -0.10   0.05
Δ(imbals) 0.55   0.10   0.30   0.15   1.00  -0.30   0.35
π_DC     -0.25  -0.10  -0.20  -0.10  -0.30   1.00  -0.15
π_bm      0.40   0.10   0.20   0.05   0.35  -0.15   1.00
```

---

## 3. LSMC implementation

### 3.1 Algorithm

Backward induction over a `(SoC, SoH)` grid following Boogert & de Jong (2008).

- **Backward pass**: at each `(t, SoC_j, SoH_k)`, find the mode `m` that
  maximises `CF(n,m) + γ · Ê[V_{t+1}(E'_m)]`, fit `β[t,j,k]` by OLS.
- **Forward pass**: track `(E_n, SoH_n)` per path; interpolate continuation
  value from stored `β`; accumulate discounted cashflows.
- **SoC grid**: `N_SOC_NODES_PER_HOUR × duration_h` nodes, linearly spaced
  in `[E_min, E_max(SoH)]`. Default 3–4 nodes/hour.
- **SoH grid**: 3 nodes — `[1.00, 0.90, 0.82]` (medium run).
- **SoC normalisation**: `E_scaled = E / E_nameplate` (always in `[0.10, 0.90]`
  regardless of duration). Bug: was hardcoded `/ 100.0` before May 2026 fix.

### 3.2 Regression basis (26 base features + up to 12 EFA-block features)

```python
BASIS_NAMES = [
    "const",
    "P_da", "P_da_sq", "P_da_cu",          # DA price polynomial
    "P_id_spread",                           # intraday vs DA
    "delta_signal",                          # lagged imbalance signal (t−1)
    "pi_dc", "pi_qr", "pi_bm",              # ancillary and BM clearing
    "E", "E_sq", "E_cu",                    # SoC polynomial
    "E_x_Pda",                               # SoC × DA price interaction
    "E_x_pi_bm",                             # SoC × BM clearing
    "P_da_x_delta", "P_da_x_dc", "dc_x_qr", # cross-state interactions
    "hpfc_fwd_spread",                       # deterministic HPFC carry signal
    "da_fwd_max", "da_fwd_min",             # path-specific next-window DA strip
    "da_fwd_mean", "da_fwd_spread",
    "soc_x_da_spread",                       # SoC × forward spread
    "dc_x_da_spread",                        # DC × forward spread
    "E_sq_x_da_mean",                        # SoC² × forward mean
    "soc_x_da_max",                          # SoC × forward peak — suppresses DC
                                             # when battery charged + big spike ahead
]
# N_BASIS = 26

# da_forward_feature_mode = "efa_blocks" (set by nb13) appends 12 more:
#   da_fwd_efa_mean_b{0..5}    — mean DA price per EFA block over next 48 HH
#   da_fwd_efa_spread_b{0..5}  — max-min spread per EFA block
# Total in efa_blocks mode: 38 features
```

The `da_forward_feature_mode = "efa_blocks"` option gives the LSMC the same
forward-price information as the DA rolling intrinsic benchmark (which observes
the next 24 h of DA prices at each gate). This is the current default in nb13.

### 3.3 Action space

```python
DispatchMode(net_frac, r_dc_frac, r_qr_frac, r_bm_frac)
# net_frac in [-1, -0.5, 0, 0.5, 1]
# r_dc_frac in [0, 0.5]      (nb12 medium default; nb13 sweep uses [0.0, 0.25, 0.5])
# r_qr_frac in [0, 0.25]
# r_bm_frac in [0, 0.25, 0.5]
# Feasibility: |net| + r_dc + r_qr + r_bm ≤ SoH (power headroom)
# allow_ancillary_stacking = 0: DC and QR mutually exclusive
```

Approximately 40–48 feasible modes after power-headroom filtering.

### 3.4 Key modelling choices (current central case)

| Setting | Value | Rationale |
|---|---|---|
| `imbalance_signal_lag_hh` | 1 | Dispatch sees `δ[t−1]`; removes clairvoyance |
| `reserve_sustain_h` | 1.0 | Energy headroom required for 1h of reserve delivery |
| `allow_ancillary_stacking` | 0 | DC and QR mutually exclusive; avoids unrealistic joint reservation |
| `da_forward_feature_mode` | `efa_blocks` | 38-feature basis; matches DA RI information set |
| `continuation_value_cap_gbp` | 25 000 000 | Prevents regression blow-up; replaces old 3 000 000 cap that clipped 4h runs |
| `da_forward_feature_hh` | 48 | Next-48HH (24h) forward DA window |

---

## 4. Co-optimisation constraints (per HH)

```
Power headroom:   |net_mw| + r_dc + r_qr + r_bm  ≤  P_bar × SoH
Energy (down):    E_t − (r_dc + r_qr) × sustain / η_d  ≥  E_min
Energy (up):      E_t + (r_dc + r_qr) × η_c × sustain  ≤  E_max(SoH)
BM energy:        r_bm_mw × sustain_bm_hh / (2 × η_d)  ≤  E_t − E_min
SoC transition:   E_{t+1} = E_t − net × (η_d if d; 1/η_c if c) × dt
```

---

## 5. Valuation benchmarks

### Initial hourly intrinsic
Single-pass LP over the deterministic HPFC for the full horizon.
No ancillary, no imbalance, no stochastic uncertainty.
**Reference (2h): ~£1.86m/year.**

### DA rolling intrinsic (V_RI)
At each EFA gate (every 8 HH = 4h), solve a 48-HH deterministic LP on the
DA price strip for that path; apply the first 8 HH of decisions; re-solve.
DA-only, no ancillary, 100% capacity available.
Non-anticipative: uses only prices observable at gate closure — a real,
executable strategy, not an oracle.
`gate_hh = 8, window_hh = 48` (nb12 cell `method-comparison-rolling`).
**Reference (2h, 250 paths): ~£3.21m/year.**

### WD rolling intrinsic
Identical gate/window cadence to DA RI (`gate_hh = 8, window_hh = 48`).
At each gate, the LP sees intraday prices `P_da + clip(delta_imb, ±cap)` for
the committed 8 HH; the remaining look-ahead uses DA prices only. Cashflow
settles at the same capped intraday price.
`NB12_WD_UPLIFT_CAP_GBP_MWH` default: **10 £/MWh** (reflects typical GB M7
continuous ID-DA spread; the raw settlement basis std ≈35 £/MWh must not be
used here as it overstates WD optionality by 2–3×).
The only structural difference vs DA RI is the intraday price premium;
re-optimisation frequency is identical.
**Reference (2h): pending rerun with corrected gate and cap (2026-05-10 fix).**

### Forward simulation (LSMC, V_LSMC)
Full stochastic backward induction. Co-optimises DA energy, imbalance,
DC, QR, and BM simultaneously.
Current central case: no-stacking, 1h reserve sustain, 1 HH lag, efa_blocks basis.
**Must satisfy V_LSMC ≥ V_RI (LSMC includes ancillary on top of DA energy).**

### Perfect foresight (DA energy only)
Full-horizon LP with known realised DA prices on each path. No ancillary.
Sets the DA-energy upper bound.
**Reference (2h, 250 paths): ~£3.15m/year.**

---

## 6. Current Phase 4 duration-sweep results

_Best complete run (LSMC + DA RI + PF): 250 paths, 2,160 HH backward steps,
efa_blocks basis, no-stacking, sustain 1h, 1 HH lag, dc_levels=[0.0, 0.25, 0.5],
qr_levels=[0.0, 0.25]. Run completed 2026-05-07._

_WD RI values below are **stale**: they were produced with the pre-fix settings
(DA RI gate_hh=48, WD cap=30 £/MWh) and significantly overstate WD revenue.
Updated WD RI values require a fresh nb13 rerun (priority 5b)._

### Method comparison (£m / year annualised, 100 MW)

| Duration | Initial RI | DA RI | WD RI ⚠ | LSMC | PF (DA) |
|---:|---:|---:|---:|---:|---:|
| 1h | 0.97 | 1.69 | ~~3.68~~ | **2.61** | 1.66 |
| 2h | 1.86 | 3.21 | ~~5.59~~ | **3.57** | 3.15 |
| 3h | 2.60 | 4.48 | — | **3.93** | — |
| 4h | 3.17 | 5.47 | — | **4.07** | — |

⚠ WD RI values are from the pre-fix run (gate_hh asymmetry + cap=30). After the
2026-05-10 fix (gate_hh=8 for both DA/WD; cap=10), WD RI is expected to be
modestly above DA RI (reflecting only the ±10 £/MWh intraday premium).

Duration ratios (LSMC): **2h/1h = 1.37×** (Modo benchmark: 1.52×).

LSMC vs DA RI delta: 1h +£0.93m ✓, 2h +£0.36m ✓, 3h −£0.54m ✗, 4h −£1.40m ✗.
The 3h/4h gap indicates the policy still over-allocates to DC at the expense of
DA energy arbitrage. The `soc_x_da_max` basis feature (action 38) was added to
address this and has been validated in a debug run (125 paths), but a full
250-path rerun is still pending (priority 5a).

### LSMC attribution (£m/year, same run)

| Component | 1h | 2h | 3h | 4h |
|---|---:|---:|---:|---:|
| HPFC anchor | −0.10 | −0.11 | −0.10 | −0.10 |
| DA surprise | +0.15 | +0.26 | +0.23 | +0.18 |
| Imbalance proxy (BM/ID substitute) | +0.98 | +1.52 | +1.81 | +1.90 |
| DC ancillary | +1.23 | +1.42 | +1.52 | +1.51 |
| QR ancillary | +0.53 | +0.65 | +0.66 | +0.64 |
| Costs (deg + VOM) | −0.18 | −0.17 | −0.19 | −0.06 |
| **Net LSMC** | **2.61** | **3.57** | **3.93** | **4.07** |

_Attribution for the action-29 efa_blocks run; components may not sum exactly
due to rounding in stored JSON._

The imbalance proxy now contributes ~£1–2m/year (vs £7–8m/year pre-fix).
DC ancillary contributes ~£1.2–1.5m and is the largest single source.

### Per-MW-year summary

| Duration | £k/MW/yr | Modo benchmark |
|---:|---:|---:|
| 1h | 26.1 | 47.7 |
| 2h | 35.7 | 72.5 |
| 3h | 39.3 | — |
| 4h | 40.7 | — |

LSMC is below Modo because **BM (~30%) and CM (~7%) are absent**. The correct
apples-to-apples comparison is Modo's DA+ID+BM bucket (~65% × £72k = ~£47k/MW
for 2h) vs LSMC DA+imbalance+DC+QR net of costs.

---

## 7. Why LSMC DA component ≠ V_RI

| | V_RI | LSMC DA cashflow component |
|---|---|---|
| Capacity | 100% for DA | Shared with DC/QR/BM headroom |
| Dispatch driver | DA spread alone | Imbalance + ancillary signals dominate |
| Price information | Realised next-24h DA (non-anticipative) | 6 EFA-block means/spreads of next-48HH |
| Uncertainty | None (deterministic LP per path) | Full stochastic path |

When the policy co-optimises with ancillary, capacity committed to reserve is
unavailable for net dispatch. This is rational: the policy may choose
`+£1.5m DC + £1.0m imbalance − small DA drag` over `+£3.2m pure DA`
only if the former is genuinely higher. The 3h/4h shortfall suggests this
trade is value-destroying at longer durations.

---

## 8. Market calibration anchors

```python
# src/config.py → CALIBRATION_ANCHORS
Modo GB BESS index (May 2025 – May 2026):
  1h: £47.7k/MW/yr  ←→  LSMC 1h: £26.1k (model misses BM + CM)
  2h: £72.5k/MW/yr  ←→  LSMC 2h: £35.7k
  2h/1h ratio: 1.52× (Modo)  vs  1.37× (LSMC current)

Revenue stack fractions (config anchors):
  DA + ID:         35%
  BM:              30%   ← absent from LSMC, largest missing stream
  DC / DM / DR:    18%
  QR / BR:         10%
  Capacity Market:  7%   ← absent; deterministic overlay only

KYOS 2026 assessment: EUR 96.4k/MW/yr (DA+ID+passive imbalance only;
  excludes ancillary and CM — not comparable to Modo all-in).
```

---

## 9. Open gaps and next steps

| Priority | Item | Status |
|---|---|---|
| 5a | Full medium rerun with `soc_x_da_max` active (250 paths, 2,160 steps) | **Pending** |
| 5b | Rerun nb13 WD RI with corrected gate (gate_hh=8) and cap (10 £/MWh) | **Pending — code fixed 2026-05-10** |
| 6 | nb16 structural counterfactuals rerun with new basis | Pending after 5a |
| 7 | Phase 3: explicit BM revenue layer | Design ready; not yet implemented in cashflow |
| — | CM contracted overlay | Not modelled |
| — | 5,000-path production bundle | Pending after central-case settings are settled |

---

## 10. Infrastructure notes

### Simulation bundle
`data/processed/sim_bundle.pkl` — 1,000-path corrected bundle (imbalance
params from volume-weighted MID, p_pos = 0.358). Built by
`scripts/generate_sim_bundle.py --paths 1000 --steps 17520 --seed 42`.

Debug bundle (`sim_bundle_debug.pkl`) is auto-generated by nb13 for fast
iterative runs (125 paths × 1,180 steps by default).

### WD rolling intrinsic benchmark — fix history

**Pre-fix (before 2026-05-10):**
- DA RI used `gate_hh=48` (daily optimizer; applied all 48 HH decisions at once, never rolled).
- WD RI used `gate_hh=8` with `delta_imb` capped at ±30 £/MWh.
- Structural asymmetry: WD re-optimised every 4h while DA committed for 24h → WD had
  a re-optimisation frequency advantage unrelated to intraday prices.
- Cap of 30 £/MWh still admitted the full settlement-imbalance volatility (std ≈35),
  giving the LP unrealistic 4h perfect-foresight option value on a volatile process.
- Result: WD RI was 2.5–3.5× DA RI across durations (e.g. 2h WD £8.1m vs DA £3.1m).

**Post-fix (2026-05-10):**
- Both DA RI and WD RI use `gate_hh=8, window_hh=48` — identical re-optimisation cadence.
- WD cap reduced from 30 → **10 £/MWh** to reflect typical GB M7 continuous ID-DA spread.
- Override: `NB12_WD_UPLIFT_CAP_GBP_MWH` env var.
- Expected post-fix WD/DA ratio: ~1.1–1.3× (intraday premium only, no frequency advantage).

### Windows zmq kernel fix
Two-layer fix for Python 3.12 Windows `ProactorEventLoop` deadlock:

1. `data/processed/_nb_launch_win.py` — sets `WindowsSelectorEventLoopPolicy`
   in the **nbconvert process** before importing jupyter.
2. `~/.ipython/profile_default/startup/00_win_selector_loop.py` — sets the
   same policy in every **kernel subprocess** at startup. This is the
   layer that actually prevented the deadlock (action 40, 2026-05-10).

The two-layer fix reduces but does not fully eliminate deadlocks — they still
occur intermittently under parallel load. nb13 now adds a third,
**application-level** stall guard: if a child produces no `phase4_status_*.jsonl`
events within `STALL_TIMEOUT_S` (default 4 min), it is killed via
`taskkill /T /F` and relaunched serially with `NB12_FWD_WORKERS=1` (which
eliminates the zmq fan-out that triggers the race). The retry is automatic and
transparently labelled `[stall+retry]` in the run summary.

### nb13 sweep controls

| Env var | Default | Purpose |
|---|---|---|
| `PHASE4_DEBUG` | `1` | Use 125-path debug bundle |
| `PHASE4_DEBUG_N_PATHS` | `125` | Paths per child |
| `PHASE4_DEBUG_BWD_STEPS` | `1080` | Backward HH steps |
| `PHASE4_DA_FORWARD_FEATURE_MODE` | `efa_blocks` (cell 2) | Basis mode |
| `PHASE4_ALLOW_ANCILLARY_STACKING` | `0` | DC+QR mutual exclusion |
| `PHASE4_RESERVE_SUSTAIN_H` | `1.0` | Energy sustain hours |
| `PHASE4_IMBALANCE_SIGNAL_LAG_HH` | `1` | Dispatch signal lag |
| `PHASE4_MAX_PARALLEL` | `2` | Concurrent child processes |
| `PHASE4_CHILD_MAX_MINUTES` | `45` | Hard wall-time cap per child |
| `PHASE4_STALL_MINUTES` | `4` | Kill + serial retry if no status events within N min |

### nb12 rolling intrinsic controls

| Env var | Default | Purpose |
|---|---|---|
| `NB12_WD_UPLIFT_CAP_GBP_MWH` | `10` | Intraday price cap for WD RI (£/MWh) |
| `NB12_RI_PATHS` | (run-mode default) | Number of RI evaluation paths |
| `NB12_DURATION_H` | `2.0` | Asset duration for single-duration run |
| `NB12_BUNDLE_PATH` | (auto) | Override simulation bundle path |
