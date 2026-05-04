# BESS UK Valuation Framework

Stochastic mark-to-market valuation for fast-cycle battery energy storage in Great Britain (1–4 hours).

This repository contains:
1. **Streamlit dashboard** — Comprehensive BESS valuation outputs (all phases)
2. **Phase 4 Streamlit app** — Interactive duration sweep analysis (method comparison & attribution)
3. **Processed output files** — All valuation results, charts, and diagnostics
4. **Documentation** — Methodology, assumptions, and usage guides

The research notebooks and modelling source code remain local; they are not needed to view the dashboards.

## 📊 Dashboards

### 1. Main BESS Dashboard (`streamlit_app.py`)

Comprehensive outputs from all phases (1–7):

| Tab | Contents |
|---|---|
| **Overview** | Headline LSMC, MTM, risk, backtest, perfect-foresight |
| **Calibration** | Schwartz-Smith, PCA, imbalance, ancillary parameters |
| **Simulation** | Sample paths, cross-correlation, diagnostic charts |
| **LSMC** | Duration-specific valuation results and charts |
| **MTM Risk** | Greeks (delta, vega, rho), VaR, CVaR, stress tests, SoH |
| **Backtest** | P&L attribution, dual bounds, execution analysis |
| **Perfect Foresight** | Upper-bound benchmarks, dispatch dispatch analysis |
| **Files** | Inventory and raw JSON viewer |

### 2. Phase 4 Duration Sweep App (`streamlit_phase4_sweep.py`) ⭐

Interactive **method comparison & revenue attribution** across 1h–4h durations:

| View | Purpose |
|---|---|
| **Overview** | Key metrics and findings |
| **Method Comparison** | Line/bar charts of 5 valuation methods by duration |
| **Attribution Analysis** | Revenue breakdown by component (HPFC, DA, ancillary, costs) |
| **Detailed Tables** | Full data tables (downloadable CSV) |
| **Run Diagnostics** | Execution logs, file inventory, regeneration instructions |

**See:** [STREAMLIT_PHASE4_APP.md](docs/STREAMLIT_PHASE4_APP.md) for full documentation.

## 🚀 Run Dashboards Locally

### Main Dashboard
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Phase 4 Duration Sweep
```bash
streamlit run streamlit_phase4_sweep.py
```

Both will open at `http://localhost:8501` by default.

## 📦 Repository Contents

### Streamlit Apps
- `streamlit_app.py` — Main BESS valuation dashboard
- `streamlit_phase4_sweep.py` — Phase 4 duration sweep analysis ⭐ **New!**

### Documentation
- `README.md` — This file
- `docs/STREAMLIT_PHASE4_APP.md` — Phase 4 app guide
- `docs/stochastic_plan.md` — Full methodology
- `docs/pricing_items.md` — Priceable revenue components
- `docs/revenue_stack.md` — GB market assumptions
- `CLAUDE.md` — Project context (for AI assistants)

### Data
- `data/processed/` — All output files, charts, JSON summaries
- `requirements.txt` — Python dependencies

## 📝 Notes

These outputs are research artifacts from stochastic valuation of a 50 MW / 100 MWh (2h nominal) fast-cycle BESS asset.

**Phases 1–3:** Data pipeline, calibration, simulation  
**Phase 4:** Duration sweep – method comparison for 1h, 2h, 3h, 4h duration batteries  
**Phases 5–7:** MTM risk (Greeks, VaR, CVaR), backtest attribution, perfect-foresight benchmark

### Phase 4 Highlights
- Compares 5 valuation methods: intrinsic, rolling, LSMC, perfect foresight
- Duration sweep: 1h, 2h, 3h, 4h batteries
- Revenue attribution: HPFC anchor, DA surprise, intraday, DC/QR ancillary, costs
- **App:** `streamlit_phase4_sweep.py` — Interactive dashboard with filtering, drill-down, CSV export

### Important Caveats
- Perfect-foresight is an **upper-bound benchmark**, not a tradable strategy
- Optimiser fee: assumed 12% of gross positive merchant revenue
- SoH augmentation: modelled at years 4, 8, 12 of asset life
- All outputs research/illustrative; not for investment decisions
