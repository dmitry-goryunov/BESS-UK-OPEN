# BESS Stochastic Valuation — Claude Code Context

This project contains a stochastic MTM valuation framework for a GB fast-cycle
(1–2h) Battery Energy Storage System, implemented in Python/Jupyter.

The full methodology, vendor landscape, risk taxonomy, and revenue stack analysis
are in the `docs/` folder. The stochastic valuation plan is in
`docs/stochastic_plan.md`. The pricing items reference is in
`docs/pricing_items.md`.

---

## Project structure

```
bess_project/
├── CLAUDE.md                   ← you are here (auto-loaded by Claude Code)
├── docs/
│   ├── stochastic_plan.md      ← 10-phase MTM valuation plan
│   ├── pricing_items.md        ← full list of priceable items
│   ├── revenue_stack.md        ← GB revenue stack, current numbers
│   └── vendor_landscape.md     ← KYOS vs Aurora, Modo, optimisers
├── src/
│   ├── processes/
│   │   ├── schwartz_smith.py   ← two-factor Kalman calibration
│   │   ├── hpfc.py             ← hourly price forward curve (PCA shape)
│   │   ├── imbalance.py        ← imbalance basis OU + jump process
│   │   └── ancillary.py        ← DC/DM/DR/QR AR(1) + saturation curve
│   ├── asset/
│   │   ├── battery.py          ← asset envelope, SoC/SoH state variables
│   │   └── degradation.py      ← calendar + cycle fade, rainflow
│   ├── optimisation/
│   │   ├── lsmc.py             ← LSMC backward induction core
│   │   ├── rolling_intrinsic.py← deterministic LP benchmark
│   │   └── dual_bound.py       ← Andersen-Broadie upper bound
│   ├── valuation/
│   │   ├── mtm.py              ← MTM aggregation + contract overlay
│   │   ├── greeks.py           ← bump-and-revalue Greek engine
│   │   └── var_cvar.py         ← VaR / CVaR / scenario stress
│   └── attribution/
│       └── pnl_explain.py      ← daily P&L decomposition
├── notebooks/
│   ├── 01_calibration.ipynb
│   ├── 02_simulation.ipynb
│   ├── 03_lsmc_valuation.ipynb
│   ├── 04_greeks_var.ipynb
│   └── 05_backtest.ipynb
├── data/
│   ├── raw/                    ← NESO EAC, Elexon BMRS, EPEX DA/ID
│   └── processed/
└── tests/
```

---

## Asset envelope (reference case)

```python
ASSET = {
    "power_mw":        50.0,      # MW nameplate
    "energy_mwh":     100.0,      # MWh nameplate (2h duration)
    "eta_charge":       0.938,    # √0.88 AC-AC round-trip
    "eta_discharge":    0.938,
    "soc_min":          0.10,     # 10% floor
    "soc_max":          0.90,     # 90% ceiling
    "c_rate_max":       1.0,      # 1C for DC qualification
    "aux_load_mw":      0.35,     # HVAC + BMS
    "availability":     0.96,     # annual uptime
    "efc_per_year":     520,      # from OEM warranty
    "soh_augment_trigger": 0.82,  # augment when usable < 82%
    "capex_gbp_kwh":  220.0,      # UK 2h LFP, 2025 vintage
    "fom_gbp_kw_yr":    8.0,
    "vom_gbp_mwh":      1.2,
    "life_years":      15,
    "augment_years":  [4, 8, 12],
    "augment_gbp_kwh": 60.0,
}
```

---

## Stochastic process spec (quick reference)

### Baseload price — Schwartz-Smith two-factor

```
ln P_t  =  χ_t  +  ξ_t  +  f(t)          f(t) = seasonal/hourly shape

dχ_t  =  −κ · χ_t · dt  +  σ_χ · dW_χ   (short-term, mean-reverting)
dξ_t  =  μ_ξ · dt        +  σ_ξ · dW_ξ   (long-term, drifting equilibrium)
corr(dW_χ, dW_ξ) = ρ

Calibrate: Kalman filter on log-forwards (EEX GB baseload 1m–3y)
```

### Hourly shape — PCA decomposition

```
ln P_{h,t}  =  ln P_t  +  Σ_{k=1}^{3} λ_k(t) · φ_k(h)
dλ_k  =  −α_k · λ_k · dt  +  σ_λk · dW_λk
Calibrate: eigendecomposition of daily 24-dim shape matrix
```

### Imbalance basis

```
P_IMB,t  =  P_DA,t  +  Δ_t
dΔ_t  =  −θ_Δ · Δ_t · dt  +  σ_Δ · dW_Δ  +  J_t    (OU + asymmetric jumps)
J_t ~ compound Poisson: intensity λ_J, size ~ asymmetric double-exponential
Calibrate: MLE on Elexon DA–SP settlement price pairs
```

### Ancillary clearing by product

```
π_{k,t}  =  (fleet-level) · max(0, p_res · (1 − Q_t / Q_req)^γ)
Within-path variation: AR(1) per EFA block
π_{k,b,t+1}  =  φ_k · π_{k,b,t}  +  ε_{k,b,t}
γ calibrated to observed DCL saturation 2021–2024 (γ ≈ 2.1 historically)
```

### Joint correlation matrix

```
             χ      ξ      λ₁     λ₂     Δ      π_DC
χ          1.00   0.30   0.45   0.20   0.55   -0.25
ξ          0.30   1.00   0.15   0.05   0.10   -0.10
λ₁ (level) 0.45   0.15   1.00   0.20   0.30   -0.20
λ₂ (slope) 0.20   0.05   0.20   1.00   0.15   -0.10
Δ (imbals) 0.55   0.10   0.30   0.15   1.00   -0.30
π_DC       -0.25  -0.10  -0.20  -0.10  -0.30   1.00

Note: negative DA–ancillary correlation captures the fact that
high-price days (scarcity) are days when batteries switch from
ancillary mode to wholesale; must be captured jointly.
```

---

## LSMC basis functions

```python
# At each SoC/SoH grid node, regress continuation value on:
def basis(P_da, P_id, delta_imb, pi_dc, pi_qr, E, t, EFA_block):
    return [
        1,
        P_da, P_da**2, P_da**3,
        P_id - P_da,              # intraday premium
        delta_imb,                # imbalance basis
        pi_dc, pi_qr,             # ancillary clearing
        E, E**2,                  # SoC (endogenous)
        E * P_da,                 # SoC × price interaction
        np.sin(2*np.pi*t/24),     # hour-of-day
        np.cos(2*np.pi*t/24),
        float(EFA_block),         # EFA block ID (0–5)
    ]
# Polynomial degree 2, plus hour dummies
# Nadarajah et al. 2017 EJOR 256: use regress-later (LSML) for tighter dual bounds
```

---

## Co-optimisation constraints

```python
# At each half-hour t, decision vector u = (c, d, r_DC, r_DM, r_DR, r_QR, r_BR)
# All in MW

# Power headroom
abs(d - c) + r_DC + r_DM + r_DR + r_QR + r_BR  <=  P_bar

# Energy headroom for discharge services (need stored energy to deliver)
E_t - (r_DC + r_DM + r_DR + r_QR) * dt / eta_d  >=  E_min

# Energy headroom for charge services (need space to absorb)
E_t + (r_DC_dn + r_DR_dn + r_BR_dn) * eta_c * dt  <=  E_max(SoH_t)

# Service sustain requirements (simplified)
r_DC_t >= 0  for at least 2 consecutive EFA blocks  (15-min delivery)
r_QR_t >= 0  for at least 1 half-hour period       (1-min delivery)
r_DR_t >= 0  for at least 4 EFA blocks             (60-min sustain)

# Opportunity cost condition — bid ancillary only if:
# pi_k  >=  E[max intra-block DA/ID spread | E_t]
# This is the key LSMC continuation value comparison
```

---

## Degradation model

```python
# Calendar fade (Arrhenius approximation)
def calendar_fade(dt_years, avg_soc, temp_celsius=20):
    A_cal = 4.14e-10   # LFP pre-exponential (illustrative)
    Ea    = 2.47e4     # activation energy J/mol
    R     = 8.314
    rate  = A_cal * np.exp(-Ea / (R * (temp_celsius + 273.15)))
    soc_factor = 1 + 0.5 * (avg_soc - 0.5)  # penalises high SoC
    return rate * soc_factor * dt_years

# Cycle fade (Wöhler / power law in DoD)
def cycle_fade(efc, dod, beta=2.3):   # beta=2.3 typical LFP
    return (dod ** beta) * efc / N_f_reference

# Shadow price of degradation (endogenous)
# lambda_deg = (replacement_capex_gbp_mwh) * (dN_cycles / d_throughput_mwh)
# Enters dispatch as a per-MWh throughput cost:
# c_deg_t = lambda_deg * (c_t + d_t) * dt
```

---

## MTM aggregation

```python
# Full MTM (t=0)
MTM = (
    alpha   * E_Q[ sum_t discount(t) * cashflow_merchant(S_t, policy(state_t)) ]
  + (1-alpha) * PV_contracted_legs     # toll fee, floor, CM contracts
  + MTM_floor_optionality             # put on annual revenue, LSMC-on-LSMC
  - PV_optimiser_fee
  - PV_opex
  - PV_augmentation
  - PV_degradation_shadow_cost
)
# alpha = fraction of portfolio that is merchant (vs contracted)
```

---

## Greek definitions (bump-and-revalue)

```python
GREEKS = {
    "delta_baseload":   "∂MTM/∂F_baseload   — shift all baseload forwards +£1/MWh",
    "delta_peak_twist": "∂MTM/∂(F_peak−F_off) — shift peak forwards +£1/MWh",
    "delta_pc1_shape":  "∂MTM/∂λ₁           — shift PCA level factor +1σ",
    "delta_pc2_shape":  "∂MTM/∂λ₂           — shift PCA slope factor +1σ",
    "vega_da":          "∂MTM/∂σ_DA          — shift DA vol +10%",
    "vega_id":          "∂MTM/∂σ_ID          — shift ID vol +10%",
    "delta_imb_drift":  "∂MTM/∂E[Δ]          — shift imbalance mean +£5/MWh",
    "delta_imb_vol":    "∂MTM/∂σ_Δ           — shift imbalance vol +10%",
    "delta_dc":         "∂MTM/∂E[π_DC]       — shift DC clearing +£1/MW/h",
    "delta_qr":         "∂MTM/∂E[π_QR]       — shift QR clearing +£1/MW/h",
    "delta_saturation": "∂MTM/∂γ             — shift saturation exponent +0.5",
    "delta_skip_rate":  "∂MTM/∂skip          — shift BM skip rate +10pp",
    "delta_soh":        "∂MTM/∂SoH_rate      — shift degradation rate +20%",
    "delta_rte":        "∂MTM/∂η             — shift RTE −2pp",
    "delta_avail":      "∂MTM/∂avail         — shift availability −2pp",
    "rho":              "∂MTM/∂r             — shift discount rate +50bps",
}
```

---

## Daily P&L attribution

```
ΔMTM  =  Θ (theta, time decay)
       +  Σ_k Greek_k × ΔFactor_k    (delta-explain, bump-and-revalue)
       +  [Realised CF − E[CF]]       (execution surprise vs model)
       +  ΔSoH_actual − ΔSoH_model   (degradation surprise)
       +  Calibration effect          (recalibration to new history)
       +  Residual                    (< 5% of |ΔMTM| target)
```

---

## Data sources (priority order)

| Priority | Source | What to pull |
|---|---|---|
| 1 | NESO Data Portal API | EAC results, skip rates, balancing costs, ancillary volumes |
| 1 | Elexon BMRS API | DA prices, SP, NIV, BOA data, BMU data |
| 1 | EPEX SPOT | DA auction (N2EX/EPEX), ID continuous (M7) |
| 2 | Modo Energy API | ME BESS GB Index, actual battery revenues, asset benchmarks |
| 2 | ICE/EEX | GB power forwards (baseload, peak) for SS calibration |
| 3 | Cornwall Insight | BESS Revenue Index, saturation forecasts |
| 3 | Aurora/Baringa | Revenue forecasts for reconciliation |

---

## Key literature

- Boogert & de Jong (2008) — LSMC for gas storage (*J. Derivatives* 15(3))
- Schwartz & Smith (2000) — two-factor commodity model (*Mgt Sci* 46)
- Lucia & Schwartz (2002) — seasonality in electricity
- Cartea & Figueroa (2005) — MRJD for power (*Appl. Math. Finance* 12)
- Nadarajah, Margot & Secomandi (2017) — LSMC dual bounds (*EJOR* 256)
- Löhndorf, Wozabal & Minner (2013) — SDDP+ADP for hydro storage (*OR* 61)
- Finnah, Gönsch & Ziel (2022) — GB imbalance modelling (*EJOR* 301)
- Shi, Xu & Baldick (2019) — convex cycle-based degradation cost (*IEEE T-SG*)
- Brown, Smith & Sun (2010) — information relaxation dual bounds (*OR*)

---

## Commands Claude Code should know

```bash
# Install dependencies
pip install numpy scipy pandas matplotlib scikit-learn cvxpy filterpy joblib

# Run calibration notebook
jupyter nbconvert --to notebook --execute notebooks/01_calibration.ipynb

# Run full LSMC valuation (single path test)
python -m src.optimisation.lsmc --paths 100 --steps 17520 --seed 42

# Run backtest
python -m src.attribution.pnl_explain --start 2024-01-01 --end 2025-12-31

# Run Greek bump engine
python -m src.valuation.greeks --factor baseload --bump 1.0

# Run CVaR
python -m src.valuation.var_cvar --alpha 0.95 --horizon 10
```
