# Key Data & Assumptions Extracted from Uploaded Files
*Extracted April 2026 — compared against CLAUDE.md reference case*

---

## 1. GB Market Context (new data not in CLAUDE.md)

### Fleet size & saturation pressure
| Metric | Value | Source |
|---|---|---|
| GB operational BESS end-2025 | 6.8 GW / 11 GWh | Valuing_BESS_Master |
| Average duration | 1.62 h | Valuing_BESS_Master |
| Connection queue to 2030 | ~61 GW | Cornwall Insight via BESS UK 10 years |
| Connection queue to 2035 | ~129 GW | Cornwall Insight via BESS UK 10 years |
| Clean Power 2030 implied need | 23–27 GW | Valuing_BESS_Master |

**Implication for CLAUDE.md**: the saturation saturation curve (γ calibration) and fleet-growth assumption in the ancillary module need to reflect ~6 GW → 10+ GW over the model horizon, not a static fleet. The 61 GW queue is far above system need — cannibalisation must be endogenised.

---

## 2. Revenue Benchmarks (concrete numbers to anchor calibration)

### Modo ME BESS GB Index (FCA-regulated, excludes CM)
| Period | £/MW/yr |
|---|---|
| 2025 annual average | ~£72k |
| 2025 YoY change | +41% |
| Dec 2025 | £47k |
| Jan 2026 | £52k |
| Feb 2026 | £41k |

### Cornwall Insight BESS Revenue Index (30-day rolling)
| Period | £/MW/yr |
|---|---|
| Jan 2024 | £21k |
| Jan 2025 | £92k (~50% attributable to QR launch) |

### All-in revenue (merchant + CM) for well-optimised 2h BMU, 2025
- Central: **~£82–87k/MW/yr**
- Modo 2028 forecast (merchant only, two-cycle 2h): **~£87k/MW/yr**

### KYOS Battery Index — GB (ID + passive imbalance trading, in k€/MW/yr)
| Report date | 12-month avg | Year-ahead assessment |
|---|---|---|
| Aug 2025 (for 2026) | 184 €/MW/day × 365 ≈ 67k€/MW/yr | 276k€/MW = 276 k€/MW |
| Nov 2025 (for 2026) | 183 €/MW/day | 96.4k€/MW (2026 assessment) |
| Feb 2026 (for 2027) | 63k€/MW/yr (2025 actual) | 83k€/MW (2027 assessment) |

*Note: KYOS uses 0.5C battery, 90% RTE, 730 cycles/yr, no degradation, standalone, 30% PIT cap.*

#### ⚠ KYOS vs Modo reconciliation (do not compare directly)

KYOS's revenue assessment covers **DA + ID + passive imbalance trading only** — no ancillary (DC/DM/DR/QR), no Capacity Market. In GB, "passive imbalance" means the battery takes positions settled at the System Price (cash-out), which functionally overlaps with Balancing Mechanism revenue in Modo's taxonomy. The correct Modo comparison bucket is therefore DA + ID + BM — approximately **65% of the Modo index** (35% + 30% = £47k at 2025 Modo levels).

KYOS's 2027 forward assessment (~83k€/MW/yr ≈ **£70k/MW/yr** at current €/£) therefore implies the DA+ID+BM sub-stack in 2027 is roughly 50% above the 2025 realised Modo equivalent (~£47k). The gap is partly explained by: (a) KYOS is forward-looking (2027 vs 2025 actuals), (b) the standard KYOS battery assumes no degradation and 730 EFC/yr (vs 520 EFC/yr in CLAUDE.md), and (c) KYOS's passive imbalance model may capture BM upside more aggressively than realised execution. **Before using KYOS figures as calibration anchors for the ID/IB process, reconcile assumption-by-assumption against the realised 2024–2025 Modo sub-stack.** The 50–70% gap is too large to absorb in exchange-rate or vintage adjustments alone.

---

## 3. Revenue Stack Breakdown

### Illustrative GB BESS central case (2025/26, from Valuing_BESS_Master)
| Stream | Share | £/MW/yr (indicative) |
|---|---|---|
| Wholesale DA + ID | 35% | ~£25–30k |
| Balancing Mechanism + imbalance | 30% | ~£21–25k |
| Frequency response (DC/DM/DR) | 18% | ~£13–15k |
| Quick Reserve + Balancing Reserve | 10% | ~£7–9k |
| Capacity Market | 7% | ~£5–8k |

*Note: QR belongs in the Reserve (10%) bucket alongside BR, not in the frequency response (18%) bucket. DC/DM/DR are the three EAC dynamic response products in the 18% tier.*

### Modo long-run 2030 forecast
- Wholesale: 60%, BM: 20%, Frequency response: 15%, CM: 10%

---

## 4. Frequency Response Products (EAC)

### Dynamic Containment (DC)
- Trigger: ±0.2 Hz, response 500–1000 ms, 15 min sustain
- Requirement: DCL ~1.1 GW, DCH ~1.2 GW
- Historical flat cap: **£17/MW/h** (2020–21)
- **Current: £1–5/MW/h**; cleared negative on 7 Nov 2023 (DCH −£7.56/MW/h)
- NESO 2024/25 spend: DC **£52.3m** (DCL + DCH)

### Dynamic Moderation (DM)
- Pre-fault, <1s full response, ≤20 min sustain
- Clears **£3–20/MW/h**
- NESO 2024/25 spend: **£9.6m**

### Dynamic Regulation (DR)
- 2s response, ≥60 min sustain, symmetric
- Clears **£3–20/MW/h**; 20% energy reservation per SP for symmetric contracts
- NESO 2024/25 spend: **£9.0m**

### Quick Reserve (QR) — live 3 December 2024
- 1-minute full response, 30-min granularity, co-optimised with DC/DM/DR
- Initially capped at 500 MW
- **Early 2025 clearing: ~£8.75/MW/h positive, ~£5.12/MW/h negative**
- Pure QR uplift: up to **£94k/MW/yr** in Gridcog simulations
- ~25% of a 1h battery's stack at launch
- Cornwall expects saturation within **12–24 months** given 6 GW+ 2025 pipeline

### Balancing Reserve (BR) — launched March 2024
- NESO 2024/25 spend: **£15.5m**
- Replaced Fast Reserve (Fast Reserve 2024/25: **£103m** before December 2024 replacement)

**Note for CLAUDE.md**: the saturation exponent γ≈2.1 is calibrated to DC 2021–2024 data. The ~7× revenue collapse in DC (from ~£150k/MW/yr equivalent to ~£14–23k by late 2024) is the empirical anchor. QR is currently following the same trajectory.

---

## 5. Balancing Mechanism (BM)

| Metric | Value |
|---|---|
| Typical 2025 revenue (1–2h BMU, well-optimised) | £15–30k/MW/yr |
| Top quartile | £30–50k/MW/yr |
| Bottom quartile | <£5k/MW/yr |
| BM overtook frequency response | January 2024 |
| OBP launch | 12 December 2023 |
| Dispatch volumes increase (pre-OBP → Q1 2025) | +425% |
| Instruction count increase | +1,347% |
| Skip rates Dec 2023 | ~90% |
| Skip rates Aug 2024 | ~76% |

### OBP roadmap (important policy dates)
| Date | Milestone | Status (as of Apr 2026) |
|---|---|---|
| Dec 2023 | OBP goes live | ✅ Delivered |
| March 2024 | 15-min rule → 30-min rule; Balancing Reserve launches | ✅ Delivered |
| Sep 2025 | ABSVD rule changes (+22% short-term uplift in freq response revenue); Ofgem determination: NESO must set skip-rate numerical targets | ✅ Delivered |
| Jan 2026 | DC/DM/DR activation moved into OBP | ✅ Delivered (confirm vs NESO bulletin — verify no slip) |
| Apr 2026 | Optimisation within constraints | ⏳ Current — confirm delivery with NESO |
| Mar 2027 | Full EBS replacement | 🔲 Pending |

*Action: verify Jan 2026 and Apr 2026 milestone delivery against NESO/Elexon operational notices before using BM capture rates calibrated to pre-OBP data. If Apr 2026 optimisation milestone delivered as planned, skip-rate trajectory assumption (target <70% by end-2026) should be confirmed or revised.*

---

## 6. Capacity Market

| Auction | Result | BESS de-rating (2h) | Implied CM revenue (2h) |
|---|---|---|---|
| T-4 2028/29 (Feb 2025) | £60/kW/yr, 1.8 GW BESS | ~18–22% | **~£12k/MW/yr** |
| T-1 2026/27 (Mar 2026) | £5/kW/yr (lowest since 2020/21) | — | ~£1k/MW/yr |
| T-4 2029/30 (Mar 2026) | £27.10/kW/yr | 4h+5h dominated | ~£5.4k/MW/yr for 2h |

### De-rating factors by duration
| Duration | De-rating |
|---|---|
| 1h | ~9–12% |
| 2h | ~18–22% |
| 4h | ~45–60% |
| 6h | ~85%+ |

*NESO moved from 4.5h stress to 6h and 8h+ stress events, collapsing short-duration de-rating.*

---

## 7. Wholesale Market Structure

- Two DA auctions: EPEX 09:20 GMT D-1, N2EX 09:50 GMT D-1
- Two intraday auctions: IDA1 17:00 D-1, IDA2 08:00 D
- Continuous intraday: EPEX M7, 5-minute gate closure
- Since Brexit: decoupled from SDAC; interconnector basis £5–15/MWh vs FR/BE
- GB had **53 hours of negative prices in April 2024**
- DA spreads 2024: £50–70/MWh; late 2025: £57–65/MWh intraday

### KYOS DA/ID/Imbalance daily spread data (to Dec 2025, in €/MWh)
| Market | GB DA (36m avg) | GB DA (12m avg) | GB ID (36m avg) | GB ID (12m avg) | GB Imbalance (36m avg) |
|---|---|---|---|---|---|
| From Feb 2026 report | 76.7 | 79.8 | 93.1 | 104.0 | 153.3 |
| From Nov 2025 report | 84.1 | 87.4 | 101.3 | 97.1 | 166.0 |

### Capture rates (Modo methodology)
- Fast-cycle 1–2h GB BESS: **50–70% capture on DA**, lower on ID
- Non-BMUs trade only when EFA-block spreads exceed £50/MWh
- BMUs: 1.5–2 cycles/day; non-BMUs: ~1 cycle/day at 70% capture

---

## 8. Asset Parameters — Comparison with CLAUDE.md

| Parameter | CLAUDE.md | Docs range | Status |
|---|---|---|---|
| Power MW | 50 MW | Notebooks use 100 MW | CLAUDE.md uses 50 MW; notebooks illustrate 100 MW |
| Energy MWh | 100 MWh | Configurable from selected duration | Consistent with selected duration |
| RTE (AC-AC) | √0.88 = 88% | 82–88%; KYOS benchmark: 90% | At top of range; KYOS 90% is common benchmark |
| η charge/discharge | 0.938 | 94.9% one-way (KYOS) | CLAUDE.md consistent |
| SoC min/max | 10%/90% | 10%/90% or 5%/95% | ✅ Consistent |
| C-rate max | 1.0 | 1C for DC; ≥0.5C for BM | ✅ Consistent |
| EFC/yr | 520 | 365–548 LFP warranty; KYOS uses 730 | CLAUDE.md is conservative vs KYOS benchmark; within warranty |
| SOH augment trigger | 82% | ≥80% SOH at 10yr typical | ✅ Consistent |
| Capex £/kWh | £220 | £170–280 UK 2h LFP all-in | ✅ Middle of range |
| FOM £/kW/yr | £8 | £6–10 | ✅ Consistent |
| VOM £/MWh | £1.2 | £0.5–2.0 | ✅ Consistent |
| Life years | 15 | 15–20 | ✅ Conservative end |
| Augment years | [4, 8, 12] | 3/6/9/12 (some sources) | Slight discrepancy |
| Augment £/kWh | £60 | £40–80 | ✅ Mid-range |
| aux load MW | 0.35 | Losses in PCS/transformer/HVAC | ✅ Consistent |
| Availability | 0.96 | Standard | ✅ |

---

## 9. Degradation Model

### From docs (consistent with CLAUDE.md approach)
- Calendar: Arrhenius, 1–2%/yr at 25°C, 50% SoC
- Cycle: Wöhler N_f(DoD) ∝ DoD^{−β}
  - NMC: β ≈ 1.5–2.0
  - LFP: β ≈ 2–2.5 **(CLAUDE.md uses β=2.3 — ✅ consistent with LFP)**
- Typical LFP warranty: 365–548 EFC/yr, ≥80% SOH at 10 years
- Rainflow counting: Kwon & Zhu (2021) Markovian approximation
- Endogenous shadow cost: λ_deg ≈ (replacement capex) × (dN_cycles / d(dispatched MWh)) / (economic battery lifetime in MWh)

### Notebook degradation cost (illustrative)
- Notebook 1: £6/MWh discharged
- Notebook 2: £5/MWh discharged

---

## 10. Stochastic Price Process Parameters (notebook illustrations)

From `UK_BESS_Stochastic_Model_and_Contract_Layer.ipynb`:
```python
price = {
    "base_price_gbp_mwh":    75.0,
    "seasonal_amp":           18.0,
    "daily_peak_amp":         38.0,
    "ou_kappa":               0.08,    # mean reversion speed
    "ou_sigma":               12.0,    # diffusion
    "shape_sigma":            16.0,    # hourly shape noise
    "imbalance_sigma":        35.0,    # heavy-tailed (t₄)
    "imbalance_jump_prob":     0.015,
    "imbalance_jump_mean":   150.0,
    "imbalance_jump_sigma":   90.0,
    "negative_price_prob":     0.01,
    "negative_price_mean":   -30.0,
}
```

*These are synthetic scaffold values — replace with Kalman-calibrated Schwartz-Smith parameters from EEX GB baseload forwards.*

### Ancillary saturation (notebook)
```python
ancillary = {
    "available_bess_mw_start":  6_000,   # current fleet
    "available_bess_mw_end":   11_000,   # end-of-project fleet
    "service_requirement_mw":   2_500,
    "saturation_gamma":           1.8,   # CLAUDE.md uses 2.1
    "min_saturation_multiplier":  0.15,
}
```
*CLAUDE.md γ=2.1 vs notebook γ=1.8 — the DC empirical collapse supports the higher value.*

---

## 11. Contract Structures (real-world evidence)

| Deal | Structure | Terms |
|---|---|---|
| Shell / BW ESS Bramley | Fixed-price toll | 7-year, 100 MW / 331 MWh |
| Drax / Fidra West Burton C | CPI-linked toll | 10-year, 250 MW / ~1 GWh; construction + maintenance risk stays with Fidra |
| Octopus Kraken / Gresham House | Fixed toll | 2-year, 568 MW / 920 MWh |
| Statkraft / Statera Thurrock | Floor (Quality Factor Index Swap) | 300 MW / 600 MWh; uses Modo index |
| Gresham House (789 MW) | Long-term floors | ≥£35m/yr (~£44k/MW/yr inferred floor) |

### Notebook contract parameters (illustrative, not bankable)
```python
contract = {
    "optimiser_fee_pct":              0.12,     # 12% profit share
    "fixed_toll_gbp_per_mw_year":    78_000,
    "floor_gbp_per_mw_year":         62_000,
    "floor_share_owner":              0.55,      # 55% of upside above floor
    "cap_gbp_per_mw_year":          105_000,
    "swap_fixed_gbp_per_mw_year":    75_000,
    "index_beta_to_asset":            0.92,
}
```

---

## 12. Financial / Valuation Parameters

| Parameter | Notebook values | Docs guidance |
|---|---|---|
| Discount rate | 8% | 7–10% merchant; 5–7% contracted/tolled |
| Project life | 15 years | 15–20 years |
| Gearing | — | 50–65% |
| WACC range | — | 7–10% merchant, 5–7% contracted |
| Debt sizing | — | P90 merchant or floor prices |
| Annual revenue decay | 1.5–2% | Implicit in saturation curves |
| Augmentation (year 8) | £7m for 100 MW = £70/kWh | CLAUDE.md £60/kWh; docs say £40–80/kWh |
| CAPEX (notebook) | £190/kW + £130/kWh = £225/kWh all-in | Consistent with CLAUDE.md £220/kWh |

### LCOS reference (Ember Oct 2025)
- Global 4h+ utility: **US$65/MWh** (down from $155–320/MWh in 2023)
- BNEF pack price 2025: **US$70/kWh** (lowest ever for Li-ion)
- BNEF global turnkey 2025: **US$117/kWh** (−31% YoY); forecast US$101/kWh by 2035

---

## 13. Key Deltas vs CLAUDE.md — Action Items

### Items to add/update in CLAUDE.md or docs:

1. **Revenue benchmarks** — add Modo index values and Cornwall BESS Revenue Index as calibration anchors for the MTM model output:
   - 2025 Modo annual avg: £72k/MW/yr
   - Well-optimised 2h all-in 2025: £82–87k/MW/yr

2. **OBP roadmap** — add critical dates (Jan 2026 DC/DM/DR into OBP, Apr 2026 constraint optimisation, Mar 2027 full EBS replacement); these affect BM capture rates in the model.

3. **CM de-rating** — add duration de-rating table; 2h battery at ~20% de-rating changes CM revenue calculation vs nameplate. T-4 2028/29 £60/kW/yr → £12k/MW/yr for 2h.

4. **Skip rate trajectory** — add 90% (Dec 2023) → 76% (Aug 2024) as BM capture-rate calibration input; Ofgem determination requires NESO numerical targets.

5. **QR pricing** — add £8.75/MW/h positive, £5.12/MW/h negative (early 2025) as QR calibration anchors; saturation expected within 12–24 months.

6. **Fleet size** — add 6.8 GW / 11 GWh (end 2025) as saturation baseline for ancillary supply curve calibration.

7. **Saturation γ** — CLAUDE.md uses 2.1, notebooks use 1.8. Both are plausible; the DC empirical evidence supports ≥2 for a fast saturation regime. Keep 2.1 but note sensitivity.

8. **Negative price handling** — docs cite 53 negative-price hours in April 2024 alone; ensure price process allows negative values (CLAUDE.md uses log-normal SS which can't go negative — needs arithmetic OU or truncation layer).

9. **Contract overlay** — add real-world floor level (~£44k/MW/yr inferred from Gresham House) and toll pricing (~£78k/MW/yr in notebook illustration) as calibration targets.

10. **Augmentation years** — docs say years 3/6/9/12 (some sources); CLAUDE.md uses [4,8,12]. Consider aligning.

11. **Capex update** — CLAUDE.md £220/kWh is still in range; global pack prices at US$70/kWh signal further downward pressure. Worth noting the global → UK premium (£170–280/kWh all-in).

---

## 14. KYOS KySim / KyBattery Architecture (for vendor_landscape.md)

- **KySim**: mean-reverting multi-factor price simulation, arbitrage-free to forward prices; same engine as gas curves
- **KyBattery**: LSMC-based dispatch optimiser; same kernel as KyStore (gas storage)
- Market coverage: DA, ID, passive imbalance (capped at 30%), FCR, aFRR
- Standard benchmark asset: 0.5C, 90% RTE, 730 cycles/yr, no degradation, standalone
- Passive imbalance strategy: multi-linear regression forecasts; long/short position when forecasted IB price falls/rises above thresholds; 30% capacity cap due to liquidity / cannibalization

**Aurora Chronos**: fundamental-market-led dispatch engine; does not disclose LSMC or stochastic asset-level formulation publicly; 200+ subscribers; claims ~70% of GB installed battery capacity served.

---

## 15. Literature Additions (from docs — not yet in CLAUDE.md)

The uploaded docs cite additional relevant papers not in CLAUDE.md:
- Kwon & Zhu (2021, arXiv:2108.02374) — RL + rainflow-consistent degradation
- Shi et al. (2019), He et al. (2016) — convex cycle-based degradation cost
- Jiang & Powell (2015, INFORMS JoC 27) — ADP for hour-ahead battery bidding (monotonicity speedup)
- Shengren et al. (2023) — MIP-DQN for physical feasibility in RL
- Oeltz & Pfingsten — Rolling intrinsic for battery in DA/ID markets
- Schmidt et al. (2019, Joule 3(1)) — LCOS canonical reference
- IRENA (2020) — Electricity Storage Valuation Framework
- US DOE (2022) — Energy Storage Valuation: Review of Use Cases and Modeling Tools

---

*End of extraction. Cross-reference with `docs/revenue_stack.md`, `docs/vendor_landscape.md`, and `docs/stochastic_plan.md` for integration.*
