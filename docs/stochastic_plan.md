# Stochastic MTM Valuation Plan — GB Fast-Cycle BESS

The right mental model: **batteries are gas storage where some underlyings are
not tradable**. LSMC machinery from gas transfers directly. The strict
arbitrage-free framing breaks because there are no liquid forward instruments
for hourly shape, ancillary clearing, or imbalance basis — so MTM is
*risk-adjusted expected value under physical measure*, calibrated to whatever
forwards are tradable, with explicit risk premia on un-hedgeable legs.

---

## Phase 0 — Define the MTM contract

Pin down what is being valued before any code runs.

- **Asset envelope**: P̄, Ē, η_c, η_d, SoC bounds, C-rate, ramp, availability,
  calendar+cycle degradation curve, augmentation schedule, BMU registration,
  EAC eligibility.
- **Contract overlay**: merchant / toll / floor+share / hybrid. Each leg has
  a separate MTM treatment.
  - Toll fee → riskless annuity discounted at counterparty curve
  - Floor → put on annual revenue, priced by LSMC-on-LSMC
  - Share above floor → scaled merchant exposure
  - CM contract → deterministic for awarded vintages
- **Tenor**: residual life from t₀ split into merchant horizon and
  augmentation/repower envelope.
- **Discount basis**: risk-free for contracted cashflows, project WACC for
  merchant residual, explicit credit spread on contracted legs.

**Deliverable**: one-page asset-and-contract sheet that the rest of the model
reads as a config file.

---

## Phase 1 — Build input curves

Three tiers:

**1.1 Tradable (no-arbitrage anchors)**
- Baseload + peak forwards out to ~3y (EEX/ICE GB)
- CM clearing for awarded vintages (riskless)
- Toll fees and floor strikes (riskless given counterparty)
- Gas, carbon, FX for fundamentals overlay

**1.2 Benchmark (calibration only, not hedging)**
- Modo ME BESS GB Index (FCA-regulated)
- Aurora / Baringa / LCP / Cornwall battery revenue forecasts
- Cornwall BESS Revenue Index

**1.3 Reconstructed**
- Hourly price forward curve (HPFC): baseload+peak forwards + PCA hourly shape
  → must reproduce monthly settles
- Ancillary clearing curves (DC/DM/DR/QR/BR) by EFA block, forward
- Imbalance basis curve (SP minus DA)
- Skip-rate trajectory

**⚠ Failure mode**: using a single forecaster's hourly shape and inheriting
their saturation assumption implicitly. Blend at least 2–3.

---

## Phase 2 — Specify and calibrate stochastic processes

### Baseload — Schwartz-Smith two-factor

```
ln P_t  =  χ_t + ξ_t + f(t)

dχ_t  =  −κ χ_t dt + σ_χ dW_χ       short-term mean-reverting
dξ_t  =  μ_ξ dt   + σ_ξ dW_ξ       long-term drifting equilibrium
corr(dW_χ, dW_ξ) = ρ
```

Calibrate via Kalman filter on log-forwards, jointly with futures history.

### Hourly shape — PCA

```
ln P_{h,t}  =  ln P_t + Σ_{k=1}^{3} λ_k(t) · φ_k(h)
dλ_k  =  −α_k λ_k dt + σ_λk dW_λk
```

Eigendecompose the daily 24-dim shape matrix. Keep 3 PCs (level, slope,
curvature) — explains 85–95% of variance.

### Spikes and negative prices

Use Cartea-Figueroa (2005) MRJD or a 2-state regime switch. GB had 53 hours
of negative prices in April 2024 alone — arithmetic (not log) formulation
required for negative-price territory.

### Imbalance basis

```
P_IMB,t  =  P_DA,t + Δ_t
dΔ_t  =  −θ_Δ Δ_t dt + σ_Δ dW_Δ + J_t
J_t ~ compound Poisson: λ_J intensity, asymmetric double-exponential size
```

Calibrate MLE on Elexon DA–SP pairs. Separate jump intensities for short
(negative NIV, system buying) vs long (positive NIV, system selling).

### Ancillary clearing

```
π_{k,t}  =  AR(1)_{EFA block} × saturation factor
saturation:  max(0, p_res × (1 − Q_BESS,t / Q_req,t)^γ)
γ ≈ 2.1  (calibrated to DCL saturation 2021–2024)
```

Q_BESS,t is exogenous fleet build (scenario input, not stochastic within-path).

### Joint correlation matrix

```
            χ      ξ      λ₁     λ₂     Δ      π_DC
χ         1.00   0.30   0.45   0.20   0.55  −0.25
ξ         0.30   1.00   0.15   0.05   0.10  −0.10
λ₁(level) 0.45   0.15   1.00   0.20   0.30  −0.20
λ₂(slope) 0.20   0.05   0.20   1.00   0.15  −0.10
Δ(imbals) 0.55   0.10   0.30   0.15   1.00  −0.30
π_DC     −0.25  −0.10  −0.20  −0.10  −0.30   1.00
```

Negative DA–ancillary correlation: high-price (scarcity) days are days when
batteries switch from ancillary to wholesale. Must be captured jointly.

**⚠ Failure mode**: calibrating each process independently — biases revenue
upward because the opportunity cost linkage is broken.

---

## Phase 3 — Build the asset model

State variables: (E_t, SoH_t, Q_t)

```
SoC dynamics:
  E_{t+1} = E_t + η_c·c_t·Δt − d_t·Δt/η_d
  0 ≤ E_t ≤ Ē(SoH_t) = SoH_t · Ē_nameplate

Calendar fade:   ΔSoH_cal = A·exp(−Ea/RT) · soc_factor · Δt
Cycle fade:      ΔSoH_cyc = f_rainflow(SoC path)

Augment trigger: if SoH_t < SoH_aug_trigger:
                     Ē_t += ΔE_aug
                     log capex hit
```

Discretise (E, SoH) on a grid: 21 SoC levels × 5 SoH levels is usually
sufficient. Throughput Q implicit in SoH.

**⚠ Failure mode**: exogenous degradation schedule biases dispatch intensity
upward by 10–25% vs warranty-compliant endogenous treatment.

---

## Phase 4 — Solve dispatch: the LSMC core

### 4.1 Backward induction

For t = T−1 down to t₀, at each grid point (E, SoH, S_t):

```
V_t(E, SoH, S_t) = max_u { h(S_t, u, SoH)
                         + e^{−rΔt} · Ê[ V_{t+1}(E', SoH', S_{t+1}) | S_t ] }

u = (c_t, d_t, r_DC, r_DM, r_DR, r_QR, r_BR)  all in MW

Constraints:
  power:      |d_t − c_t| + Σ r_k  ≤  P̄
  energy up:  E_t − Σ_up r_k·Δt/η_d  ≥  E_min
  energy dn:  E_t + Σ_dn r_k·η_c·Δt  ≤  E_max(SoH_t)
  DC sustain: 15-min minimum delivery block
  QR sustain: 1-min minimum delivery block
  DR sustain: 60-min minimum delivery block
```

Basis functions for Ê (regress continuation value onto):

```python
ψ(S_t) = [1,
           P_da, P_da**2, P_da**3,
           P_id - P_da,             # intraday premium
           delta_imb,               # imbalance basis
           pi_dc, pi_qr,            # ancillary clearing
           E, E**2, E * P_da,       # SoC terms
           sin(2π·h/24), cos(2π·h/24),  # hour-of-day
           EFA_block]               # 0–5
```

### 4.2 Forward simulation for valuation

Generate N = 5,000–20,000 joint paths. Apply optimal policy from 4.1.
Accumulate discounted cashflows.

### 4.3 Andersen-Broadie dual upper bound

Information-relaxation dual to bound the optimality gap.
- Gap < 2%: policy is good enough
- Gap > 5%: increase basis degree, refine SoC grid, add features

### 4.4 Rolling intrinsic benchmark

Re-solve the deterministic LP at each curve refresh. Gives V_RI.
**Must have V_LSMC ≥ V_RI — if not, the stochastic optimiser is broken.**

**Deliverable**: V_t₀, policy function π*(state), N-path PV distribution.

---

## Phase 5 — Aggregate to MTM

```
MTM(t₀)  =  E_Q[ Σ_{t=t₀}^{T} discount(t) · cashflow(S_t, π*(state_t)) ]
           (LSMC primal lower bound, bias-corrected toward Broadie midpoint)

MTM_total  =  α · MTM_merchant
           + (1−α) · MTM_contracted
           +  MTM_floor_optionality    ← put on annual revenue
           −  PV(optimiser_fee)
           −  PV(opex + augmentation)
           +  MTM_capacity_market
           −  PV(degradation_shadow_cost)
```

Floor pricing: nested Monte Carlo or LSMC-on-LSMC. Strike is contractual.

**⚠ Failure mode**: double-counting CM in both merchant residual and
contracted leg.

---

## Phase 6 — Greeks (bump-and-revalue)

Shift one factor, re-run forward simulation with fixed policy.

| Greek | Description | Hedgeable? |
|---|---|---|
| Δ_baseload | ∂MTM/∂F_baseload | Partially |
| Δ_peak twist | ∂MTM/∂(F_pk − F_off) | Partially |
| Δ_PC1/PC2/PC3 shape | ∂MTM/∂λ_k | No |
| Vega_DA | ∂MTM/∂σ_DA | Partially |
| Vega_ID | ∂MTM/∂σ_ID | No |
| Δ_imbalance drift | ∂MTM/∂E[Δ] | No |
| Δ_imbalance vol | ∂MTM/∂σ_Δ | No |
| Δ_DC / Δ_QR | ∂MTM/∂E[π_k] | No |
| Δ_saturation γ | ∂MTM/∂γ (model risk) | No |
| Δ_skip_rate | ∂MTM/∂skip | No |
| Δ_degradation rate | ∂MTM/∂fade_rate | No |
| Δ_RTE | ∂MTM/∂η | No |
| Δ_availability | ∂MTM/∂avail | Partially (insurance) |
| Rho | ∂MTM/∂r | Yes (IR swap) |
| CVA | counterparty default on toll/floor | Partially (CDS) |

---

## Phase 7 — VaR / CVaR / Stress

**7.1 Parametric VaR**: Greek vector × covariance matrix. Fast, misses tails.

**7.2 Historical simulation**: replay 250–500 days of factor moves.

**7.3 Full Monte Carlo VaR**: re-simulate over VaR horizon, fully revalue.
Slowest but most accurate.

**7.4 CVaR (Rockafellar-Uryasev)**:
```
CVaR_α(L) = max_η { η + (1/(1−α)) · E[(L − η)⁺] }
where L = −ΔMTM
```

**7.5 Structural stress scenarios**:
- Ancillary saturation accelerates by 12 months
- DC/QR clears at 50% of base for 3 years
- Skip rate fails to fall below 80%
- CM de-rating methodology cuts 2h factor by 30%
- BM access friction returns (OBP rollback)
- Wholesale spreads compress to 60% of base
- Degradation 20% faster than OEM warranty
- Counterparty default on toll/floor
- Any 3 of the above simultaneously (the hard stress)

**Deliverable**: P&L distribution at 1d, 1m, 1y, project life.
CVaR_95 and CVaR_99.

---

## Phase 8 — Daily P&L attribution

```
ΔMTM  =  Θ                           time decay (theta)
       +  Σ_k Greek_k × ΔFactor_k   delta-explain (bump-and-revalue)
       +  Realised − E[CF]           execution surprise
       +  ΔSoH_actual − ΔSoH_model  degradation surprise
       +  Calibration effect         recalibration to new history
       +  Residual                   target: < 5% of |ΔMTM|
```

Persistent residual > 5% signals model misspecification — usually imbalance
basis or ancillary saturation parameter.

---

## Phase 9 — Validation and backtest

**9.1 Rolling-window backtest**
- 24 months of GB data
- For each date d: predicted distribution from d to d+T vs realised revenue
- Coverage test: realised in P10–P90 ~80% of the time

**9.2 Optimiser parity check**
- Compare model-implied capture rates vs top-quartile public optimiser
  performance (Modo leaderboard)
- If model implies >90% intraday capture → perfect foresight leak

**9.3 Lender reference reconciliation**
- Run with Aurora/Baringa central-case curves
- Output within ±15% of their published BESS revenue forecast
- Reconcile assumption-by-assumption if outside

---

## Phase 10 — Operational cadence

| Frequency | Activity |
|---|---|
| Intraday | Curve refresh (DA close, ID gate), dispatch policy update |
| Daily T+1 | Full MTM revalue, P&L explain, Greek update, VaR refresh |
| Weekly | Calibration drift check, residual diagnostic |
| Monthly | Full process recalibration (Kalman re-fit, ancillary AR re-estimation) |
| Quarterly | Saturation γ refit, degradation curve vs SCADA actuals |
| Semi-annual | Lender-reference reconciliation, scenario stress refresh |
| Annual | Full model validation, basis function review, dual-bound regression |

---

## Three things that determine whether this works

1. **Joint calibration of imbalance and ancillary with wholesale** — modelled
   in isolation, both will look benign and bias revenue upward.

2. **Endogenous degradation as a state variable** — λ_deg shadow price changes
   optimal dispatch intensity by 15–25% vs exogenous treatment. The full curve
   of (capture rate, throughput, augmentation timing) shifts.

3. **Acknowledging incomplete markets** — no liquid hourly-shape, ancillary, or
   imbalance forwards. MTM is risk-adjusted expected value, not arbitrage-free
   price. Risk premia on un-hedgeable legs (typically 5–15% haircut) must be
   set explicitly and reviewed quarterly. Pretending otherwise is how P&L drift
   accumulates into ugly year-end revaluations.
