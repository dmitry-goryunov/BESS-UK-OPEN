# BESS Pricing Items — Complete Reference

## Revenue items (the stack)

### Wholesale arbitrage
- Day-ahead (N2EX + EPEX): top-N vs bottom-N hour spread × capture rate × cycles
- Intraday auctions (IDA1 at 17:00 D-1, IDA2 at 08:00 D)
- EPEX continuous intraday M7: 5-min gate closure, most liquid intraday in Europe
- Imbalance / cash-out (single System Price post-SPS reform)
- Interconnector basis vs FR/BE: £5–15/MWh typical
- Capture rate (the KPI): 50–70% on DA for 1–2h GB BESS; top-quartile BMUs >75%

### Balancing Mechanism
- BOA offers (discharge) and bids (charge): priced per £/MWh
- Skip-rate haircut: realised dispatch / theoretical merit-order dispatch
- OBP-driven volume uplift (milestones: Jan 2026, Apr 2026, Mar 2027)
- Typical range: £15–30k/MW/yr median; top-quartile £30–50k/MW/yr

### Frequency response (EAC daily pay-as-clear auctions)
- Dynamic Containment Low (DCL): ~1.1 GW requirement; £1–5/MW/h now
- Dynamic Containment High (DCH): ~1.2 GW requirement; can clear negative
- Dynamic Moderation Low (DML): pre-fault, < 1s, ≤ 20 min
- Dynamic Moderation High (DMH)
- Dynamic Regulation Low (DRL): continuous AGC, 2s, ≥ 60 min
- Dynamic Regulation High (DRH)
- Availability capacity payment (£/MW/h) + utilisation payment (£/MWh)

### Reserve
- Quick Reserve Positive: £8.75/MW/h early 2025; 1-min response
- Quick Reserve Negative: £5.12/MW/h early 2025
- Balancing Reserve: longer notification; £15.5m NESO 2024/25 spend

### Capacity Market
- T-4: 4 years ahead; 15-year new-build contracts
- T-1: 1 year ahead; shorter contracts
- Revenue = clearing price × de-rating factor × nameplate
- De-rating: 1h ~9–12%; 2h ~18–22%; 4h ~45–60%; 6h ~85%+
- Latest T-4 clears: 2028/29 = £60/kW/yr; 2029/30 = £27.10/kW/yr
- Latest T-1 clear: 2026/27 = £5/kW/yr (Mar 2026)

### Other potential revenue
- Black Start / Restoration Services (contracted)
- Stability services: inertia (Zenobē West Burton C precedent)
- Negative generation TNUoS: southern GB (site-specific, can be £2–8/kW/yr)
- Grid-forming uplift: emerging NESO procurement

---

## Cost items

### Variable costs
- Charging cost (DA/ID prices × charge volume)
- Degradation shadow cost λ_deg per MWh throughput
  λ_deg ≈ replacement_capex × (dN_cycles/d_throughput_mwh) / lifetime_mwh
- VOM: £0.5–2/MWh throughput
- Optimiser/RTM fee: 10–20% profit-share OR fixed toll fee
- Imbalance cost when short/long on dispatch
- TNUoS generation charge: can be positive (Scotland) or negative (south England)

### Fixed costs
- FOM: £6–10/kW-yr
- LTSA / OEM warranty payments
- Insurance: property, business interruption, battery-specific fire cover, cyber
- Land lease + business rates
- Connection charges (DUoS where applicable)
- Metering + settlement agent fees
- DC provider / communication fees

### Capex
- Battery modules (cells + racks)
- PCS / inverters
- Transformers
- BOP: HVAC, fire suppression, SCADA, containers, civils
- Grid connection + DNO / NGET works
- Development costs: planning, legal, finance
- Interest during construction
- Anchor: £170–280/kWh all-in for UK 2h LFP, 2025 vintage

### Reinvestment
- Augmentation: years 3/6/9/12 typically; £40–80/kWh added per wave
- Trigger: SOH < 80–85% of COD nameplate or throughput warranty cap
- Mid-life refurbishment: PCS replacement around year 10
- Repower vs run-down decision at year 12–15
- Decommissioning reserve

---

## Technical inputs that drive cashflow

- Power rating P̄ (MW)
- Energy rating Ē (MWh) → duration = Ē/P̄
- Round-trip efficiency: 82–88% AC-AC; split η_c = η_d = √RTE
- SoC bounds: [10%, 90%] typical; [5%, 95%] with warranty risk
- C-rate limits: max 1C for DC, 0.5C for BM typically
- Ramp rates (milliseconds; rarely binding)
- Auxiliary load: HVAC + BMS (~0.5–1% of nameplate)
- Availability: 94–97% typical annual
- Calendar degradation: Arrhenius in T and SoC̄; ~1–2%/yr at 25°C, 50% SoC
- Cycle degradation: Wöhler N_f(DoD) ∝ DoD^{−β}; β ≈ 2.3 for LFP
- Warranty EFC: 365–548/yr for LFP; ≥80% SOH at 10yr or throughput cap
- Response time and telemetry quality → EAC service eligibility

---

## Contract overlay items

- Toll fee: fixed £/MW/yr (often CPI-linked); optimiser keeps all revenue
- Floor strike: guaranteed £/MW/yr; price as put on annual merchant revenue
- Upside share: % split above floor
- Merchant profit-share: 10–20%
- Counterparty credit: rating, parent guarantee, collateral thresholds, margining
- Term, break clauses, force majeure carve-outs
- Dispatch rights allocation (owner vs optimiser)
- Market anchor examples:
  - Shell-BW ESS Bramley: 7-year fixed-price toll (100 MW / 331 MWh)
  - Drax-Fidra West Burton C: 10-year CPI-linked toll (250 MW)
  - Gresham House: 789 MW under floors worth ≥ £35m/yr

---

## Financing items

- Senior debt capacity: sized on P90 merchant or floor price
- Debt tenor: 15–18 years (mini-perm common)
- WACC: 7–10% real (merchant); 5–7% (contracted)
- Gearing: 50–65%
- DSCR covenants: ≥1.3x base, ≥1.1x downside
- Reserve accounts: DSRA, MMRA, augmentation reserve
- Refinancing: typically year 5–7
- Tax: capital allowances, NI/SEM differences, loss-use profile
- Interest rate hedging cost

---

## Risk haircuts to price separately

| Haircut | How to model |
|---|---|
| Capture rate by service | Per-service, never blended |
| Ancillary saturation | Supply curve, not stationary AR(1) |
| Skip-rate trajectory | Scenario tree, OBP reform timeline |
| CM de-rating revisions | Low/base/high scenario |
| Scarcity-weighted outage | Higher cost per MW than average hour |
| Counterparty default | CVA from CDS or hazard rate |
| Construction delay + COD slip | Scenario: 3/6/12 month delay |
| Forecaster choice | GRID NAV impact: 15% on switch to conservative |
| REMA/zonal reform | Post-2027 structural scenario |

---

## The single most important modelling principle

**Do not sum revenue legs gross and independently.**

MW and MWh are the binding constraints. The correct formulation:

```
max_u  E[ Σ_t p_t·(d_t−c_t) + Σ_k π_k·r_k ]
s.t.   |d_t − c_t| + Σ_k r_k  ≤  P̄             (power headroom)
       E_t − Σ_{up} r_k·Δt/η_d  ≥  E_min         (discharge energy)
       E_t + Σ_{dn} r_k·η_c·Δt  ≤  E_max(SoH_t)  (charge energy)
```

Bid into ancillary only if π_k ≥ E[max intra-block wholesale spread | SoC path].
Miscalibrating this conditional expectation by ±30% translates one-for-one into
total revenue variance — the most sensitive single parameter in the model.
