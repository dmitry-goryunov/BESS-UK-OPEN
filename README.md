# BESS-UK

Research code and notebooks for stochastic valuation of a Great Britain battery energy storage system.

The project includes:

- Data fetchers for Elexon, NESO ancillary services, and forward-curve inputs.
- Calibration notebooks for price, imbalance, and ancillary processes.
- Joint path simulation and LSMC valuation components.
- Backtest, P&L attribution, MTM, Greeks, and stress scenario utilities.

Main notebook:

- `notebooks/bess_valuation_full.ipynb`

## Streamlit App

Run the cached-output dashboard locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app reads the parquet and JSON files in `data/raw` and `data/processed`.

## Generate Phase 3 Bundle

Phase 4 can run without rerunning Phases 1-3 if `data/processed/sim_bundle.pkl`
exists. Generate it with:

```bash
python scripts/generate_sim_bundle.py --paths 1000 --steps 17520
```

For a quick development bundle:

```bash
python scripts/generate_sim_bundle.py --paths 200 --steps 240
```
