# Streamlit Phase 4 Duration Sweep Application

A **web-based interactive dashboard** for analyzing BESS valuation across multiple battery durations using Streamlit.

## 📊 Overview

The `streamlit_phase4_sweep.py` application visualizes the results from **notebook 13** (`13_phase4_duration_sweep.ipynb`), which sequentially runs Phase 4 method comparison for battery durations of **1h, 2h, 3h, and 4h**.

### Key Features

✅ **Method Comparison** — Compare 5 valuation approaches:
- Initial hourly intrinsic
- DA rolling intrinsic  
- WD rolling intrinsic
- Forward simulation (LSMC)
- Perfect foresight (DA energy)

✅ **Revenue Attribution** — Break down LSMC value by revenue source:
- HPFC anchor (baseload forward curve)
- DA surprise (day-ahead deviation)
- Imbalance proxy (BM/ID substitute) (imbalance/BM proxy value)
- DC ancillary (dynamic containment)
- QR ancillary (quick reserve)
- Degradation & O&M costs

✅ **Duration Sweep** — Analyze value across 1h–4h battery durations

✅ **Interactive Charts** — Plotly-based hover, zoom, filter capabilities

✅ **Diagnostics** — View run logs, execution status, and file inventory

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install streamlit pandas plotly

# (or use the project requirements)
pip install -r requirements.txt
```

### Run the App

```bash
# From the project root directory
streamlit run streamlit_phase4_sweep.py
```

The app will open at `http://localhost:8501` by default.

### Generate Data (Required)

Before running the Streamlit app, you must execute **notebook 13** to generate the aggregated results:

```bash
cd notebooks
jupyter notebook 13_phase4_duration_sweep.ipynb
```

Or via command line:

```bash
jupyter nbconvert --to notebook --execute notebooks/13_phase4_duration_sweep.ipynb
```

This produces:
- `data/processed/phase4_all_durations_comparison.{csv,json,png}`
- `data/processed/phase4_all_durations_attribution.{csv,json,png}`
- `data/processed/phase4_sweep_run_log.json`

## 📖 Navigation

The app has **5 main views**:

### 1️⃣ **Overview**
- Summary metrics (durations, peak value, methods, components)
- Key findings
- Revenue component definitions

### 2️⃣ **Method Comparison**
- Aggregated comparison chart (bar + line)
- Interactive Plotly chart with filtering
- Pivot table: method × duration
- Per-duration rankings

### 3️⃣ **Attribution Analysis**
- Aggregated attribution chart (grouped bar + stacked)
- Interactive component analysis
- Mean value (£m) and percentage share tables

### 4️⃣ **Detailed Tables**
- Full comparison dataset (downloadable CSV)
- Full attribution dataset (downloadable CSV)

### 5️⃣ **Run Diagnostics**
- Run summary log (status, elapsed time, errors)
- Output file inventory
- Per-duration file checklist
- Instructions to regenerate results

## 📁 Data Files

### Required Input Files
Located in `data/processed/`:

```
phase4_all_durations_comparison.csv     ← Main comparison table
phase4_all_durations_comparison.json    ← JSON format
phase4_all_durations_comparison.png     ← Visualization
phase4_all_durations_attribution.csv    ← Attribution table
phase4_all_durations_attribution.json   ← JSON format
phase4_all_durations_attribution.png    ← Visualization
phase4_sweep_run_log.json               ← Execution log
```

### Optional Per-Duration Files
```
phase4_method_comparison_{d}h.{csv,json,png}  ← Per-duration method comparison
lsmc_attribution_{d}h.{json,png}              ← Per-duration attribution
```

## 🔧 Configuration & Customization

### Sidebar Information
The sidebar displays:
- Navigation menu
- File location info
- Data source paths

### Performance Notes
- **Caching:** Uses Streamlit `@st.cache_data` for fast reloads
- **Memory:** Typically <100 MB for full dataset
- **Load time:** <1s after initial data cache

### Extending the App

To add new analyses:

1. Add a new `elif page == "..."` block in the main navigation
2. Load additional data via `optional_csv()` or `optional_json()`
3. Create Plotly charts or tables
4. Commit and push to GitHub

Example:

```python
elif page == "Custom Analysis":
    st.header("🔬 Custom Analysis")
    
    new_data = optional_csv("custom_output.csv")
    if not new_data.empty:
        st.dataframe(new_data)
    else:
        st.info("Data not available")
```

## 📊 Data Schema

### Comparison Table Columns
| Column | Type | Example |
|--------|------|---------|
| `duration_h` | float | 1.0, 2.0, 3.0, 4.0 |
| `method` | str | "Forward simulation (LSMC)" |
| `value_gbp_annualized_m` | float | 45.23 |
| `p5_ann_m` | float | 42.1 |
| `p95_ann_m` | float | 48.3 |

### Attribution Table Columns
| Column | Type | Example |
|--------|------|---------|
| `duration_h` | float | 1.0 |
| `component` | str | "DC ancillary" |
| `mean_m` | float | 8.5 |
| `std_m` | float | 1.2 |
| `pct_of_gross` | float | 22.5 |

## 🐛 Troubleshooting

### "No comparison data loaded"
→ Run notebook 13 first:
```bash
jupyter nbconvert --to notebook --execute notebooks/13_phase4_duration_sweep.ipynb
```

### "Chart image not yet generated"
→ Re-run section 4–5 in notebook 13 to regenerate PNG outputs.

### Missing per-duration files
→ Check `data/processed/phase4_sweep_run_log.json` for execution errors.
→ Re-run notebook 12 with `NB12_DURATION_H` environment variable set.

### Streamlit cache issues
→ Clear cache: `streamlit cache clear`
→ Restart server: Stop and re-run `streamlit run streamlit_phase4_sweep.py`

## 🔗 Related Files

- **Notebook:** [13_phase4_duration_sweep.ipynb](../notebooks/13_phase4_duration_sweep.ipynb)
- **Main dashboard:** [streamlit_app.py](../streamlit_app.py) — Full BESS outputs
- **Documentation:** [docs/stochastic_plan.md](../docs/stochastic_plan.md) — Methodology
- **Phase 4 results:** [notebooks/12_phase4_method_comparison.ipynb](../notebooks/12_phase4_method_comparison.ipynb)

## 📦 Deployment

### Local Development
```bash
streamlit run streamlit_phase4_sweep.py --logger.level=info
```

### Streamlit Cloud
Push to GitHub and deploy via [Streamlit Cloud](https://streamlit.io/cloud):

1. Connect GitHub repository: `https://github.com/dmitry-goryunov/BESS-UK`
2. Select branch: `main`
3. Set main file: `streamlit_phase4_sweep.py`
4. Deploy

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_phase4_sweep.py"]
```

Build and run:
```bash
docker build -t bess-streamlit .
docker run -p 8501:8501 bess-streamlit
```

## 📝 License & Attribution

**BESS UK Valuation Framework**  
Stochastic MTM valuation for fast-cycle battery storage (1–4 hours)  
Repository: https://github.com/dmitry-goryunov/BESS-UK

---

**Questions?** Refer to [CLAUDE.md](../CLAUDE.md) for project overview and methodology.
