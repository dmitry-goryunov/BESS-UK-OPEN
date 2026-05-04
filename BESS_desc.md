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

## Intrinsic vs extrinsic split

```
V_LSMC  =  V_RI  +  (V_LSMC − V_RI)
         = intrinsic  +  extrinsic

Intrinsic  ≈ £34k/MW/yr   — deterministic DA arbitrage value
Extrinsic  ≈ £38k/MW/yr   — option value from uncertainty:
                             imbalance spikes, ancillary, flexible re-dispatch
```
