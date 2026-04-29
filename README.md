# BESS UK Valuation Dashboard

Streamlit dashboard for the important outputs and graphs from the BESS UK
stochastic valuation notebooks `01`-`07`.

The published repository is intentionally small: it contains the Streamlit app,
processed output files, and this README. The research notebooks and modelling
source code remain local working files and are not needed to view the dashboard.

## Dashboard Contents

| Tab | Source output |
|---|---|
| Overview | Headline LSMC, MTM, risk, backtest, and perfect-foresight outputs |
| Calibration | `ss_params.json`, `pca_params.json`, `imbalance_params.json`, `ancillary_params.json` |
| Simulation | `sim_summary.json` and simulation diagnostic charts |
| LSMC | `lsmc_valuation_summary.json`, `lsmc_valuation.png` |
| MTM Risk | `mtm_summary.json`, MTM, Greeks, VaR/CVaR, stress and SOH charts |
| Backtest | `phase6_summary.json`, dual-bound and P&L attribution charts |
| Perfect Foresight | `perfect_foresight_summary.json`, dispatch parquet, high-value week chart |
| Files | Inventory and raw JSON viewer for published outputs |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Published Files

```text
streamlit_app.py
requirements.txt
README.md
data/processed/
```

## Notes

These outputs are research artifacts. Phase 4-6 are saved from the current
fast-development partial mode rather than a final production solve. The
perfect-foresight benchmark is an upper benchmark and is not a tradable
strategy. The Phase 5 optimiser fee is shown as an assumed 12% negotiated
share of gross positive merchant revenue.
