from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


st.set_page_config(
    page_title="BESS UK Valuation",
    page_icon="B",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path, mtime_ns: int) -> dict:
    _ = mtime_ns
    with path.open() as f:
        return json.load(f)


def format_gbp(value, decimals: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{sign}GBP {abs_value / 1_000_000:,.{decimals}f}M"
    if abs_value >= 1_000:
        return f"{sign}GBP {abs_value / 1_000:,.0f}k"
    return f"{sign}GBP {abs_value:,.0f}"


def file_status() -> pd.DataFrame:
    rows = []
    for path in sorted([*RAW_DIR.glob("*"), *PROCESSED_DIR.glob("*")]):
        if path.is_file():
            rows.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                }
            )
    return pd.DataFrame(rows)


def read_optional_json(name: str) -> dict | None:
    path = PROCESSED_DIR / name
    return load_json(path, path.stat().st_mtime_ns) if path.exists() else None


def show_optional_image(name: str, caption: str | None = None) -> None:
    path = PROCESSED_DIR / name
    if path.exists():
        st.image(str(path), caption=caption or name, use_container_width=True)
    else:
        st.info(f"Not generated yet: data/processed/{name}")


def metric_card(label: str, value, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def kv_table(data: dict, title: str) -> None:
    st.markdown(f"**{title}**")
    if data:
        st.dataframe(
            pd.DataFrame(data.items(), columns=["parameter", "value"]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No data available.")


def flatten_distribution(summary: dict) -> pd.DataFrame:
    rows = []
    for group, values in summary.items():
        if isinstance(values, dict) and {"p5", "p50", "p95"}.issubset(values):
            rows.append({"series": group, **values})
    return pd.DataFrame(rows)


st.title("BESS UK Stochastic Valuation")
st.caption("Phase-by-phase dashboard using cached notebook outputs from the GB BESS valuation workflow.")

with st.sidebar:
    st.header("Project")
    st.markdown("[GitHub repo](https://github.com/dmitry-goryunov/BESS-UK)")
    st.markdown("Run locally with `streamlit run streamlit_app.py`.")
    st.divider()
    st.write("Cached files")
    st.dataframe(file_status(), hide_index=True, use_container_width=True)


da_path = RAW_DIR / "elexon_da_prices.parquet"
sp_path = RAW_DIR / "elexon_sp_prices.parquet"
anc_path = RAW_DIR / "neso_eac_clearing.parquet"
fwd_path = RAW_DIR / "ice_eex_forwards.parquet"

missing = [p.name for p in [da_path, sp_path, anc_path, fwd_path] if not p.exists()]
if missing:
    st.error(f"Missing cached data files: {', '.join(missing)}")
    st.stop()

df_da = load_parquet(da_path)
df_sp = load_parquet(sp_path)
df_anc = load_parquet(anc_path)
df_fwd = load_parquet(fwd_path)

df_da["settlement_date"] = pd.to_datetime(df_da["settlement_date"])
df_sp["settlement_date"] = pd.to_datetime(df_sp["settlement_date"])
if "date" in df_anc.columns:
    df_anc["date"] = pd.to_datetime(df_anc["date"])
if "delivery_start" in df_fwd.columns:
    df_fwd["delivery_start"] = pd.to_datetime(df_fwd["delivery_start"])

tabs = st.tabs(
    [
        "Phase 1 Data",
        "Phase 2 Calibration",
        "Phase 3 Simulation",
        "Phase 4 LSMC",
        "Phase 5 MTM Risk",
        "Phase 6 Backtest",
    ]
)

with tabs[0]:
    st.header("Phase 1: Data Pipeline")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("DA rows", f"{len(df_da):,}")
    with c2:
        metric_card("System price rows", f"{len(df_sp):,}")
    with c3:
        metric_card("Ancillary rows", f"{len(df_anc):,}")
    with c4:
        metric_card("Forward rows", f"{len(df_fwd):,}")

    da_daily = df_da.groupby("settlement_date")["price_gbp_mwh"].mean().sort_index()
    sp_daily = (
        df_sp.groupby("settlement_date")["system_price"].mean().sort_index()
        if "system_price" in df_sp.columns
        else pd.Series(dtype=float)
    )
    price_panel = pd.DataFrame(
        {
            "day_ahead_gbp_mwh": da_daily,
            "system_price_gbp_mwh": sp_daily,
        }
    )
    st.subheader("Daily Average Prices")
    st.line_chart(price_panel, height=340)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Day-Ahead Price Distribution")
        hist = df_da["price_gbp_mwh"].clip(-100, 300).value_counts(bins=40).sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist, height=280)
    with c2:
        st.subheader("Forward Curve")
        if {"delivery_start", "price_gbp_mwh"}.issubset(df_fwd.columns):
            fwd_plot = df_fwd.sort_values("delivery_start").set_index("delivery_start")[
                "price_gbp_mwh"
            ]
            st.line_chart(fwd_plot, height=280)
        else:
            st.dataframe(df_fwd, use_container_width=True)

    st.subheader("Raw Data Sample")
    sample_choice = st.radio(
        "Dataset",
        ["Day-ahead", "System price", "Ancillary", "Forwards"],
        index=0,
        horizontal=True,
    )
    sample_map = {
        "Day-ahead": df_da,
        "System price": df_sp,
        "Ancillary": df_anc,
        "Forwards": df_fwd,
    }
    st.dataframe(sample_map[sample_choice].head(200), use_container_width=True)

with tabs[1]:
    st.header("Phase 2: Model Calibration")
    ss = read_optional_json("ss_params.json")
    pca = read_optional_json("pca_params.json")
    imb = read_optional_json("imbalance_params.json")
    anc = read_optional_json("ancillary_params.json")

    c1, c2, c3 = st.columns(3)
    with c1:
        kv_table(ss or {}, "Schwartz-Smith")
    with c2:
        kv_table(imb or {}, "Imbalance OU + jumps")
    with c3:
        st.markdown("**PCA Variance Explained**")
        if pca and "explained_variance_ratio" in pca:
            evr = pd.DataFrame(
                {
                    "factor": [f"PC{i+1}" for i in range(len(pca["explained_variance_ratio"]))],
                    "variance_pct": [100 * x for x in pca["explained_variance_ratio"]],
                }
            )
            st.bar_chart(evr.set_index("factor"), height=250)
        else:
            st.info("No PCA parameters found.")

    st.subheader("Ancillary Product Parameters")
    if anc and anc.get("products"):
        products = pd.DataFrame(anc["products"]).T.reset_index(drop=True)
        st.dataframe(products, use_container_width=True)
    else:
        st.info("No ancillary parameter file found.")

with tabs[2]:
    st.header("Phase 3: Joint Path Simulation")
    summary = read_optional_json("sim_summary.json")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Paths", f"{summary.get('n_paths', 0):,}")
        with c2:
            metric_card("Steps", f"{summary.get('n_steps', 0):,}")
        with c3:
            metric_card("Seed", summary.get("seed", "-"))
        with c4:
            validation = summary.get("validation", {})
            passed = sum(bool(v) for v in validation.values())
            metric_card("Validation", f"{passed}/{len(validation)}")

        st.subheader("Terminal Distributions")
        st.dataframe(flatten_distribution(summary), hide_index=True, use_container_width=True)

        st.subheader("Validation Checks")
        st.dataframe(
            pd.DataFrame(validation.items(), columns=["check", "passed"]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No simulation summary found. Run Phase 3 cells to generate sim_summary.json.")

    st.subheader("Simulation Figures")
    for fig_name in [
        "sim_sample_paths.png",
        "sim_validation.png",
        "sim_corr_matrix.png",
        "sim_cross_corr.png",
        "sim_3yr_chain.png",
    ]:
        show_optional_image(fig_name)

with tabs[3]:
    st.header("Phase 4: LSMC Valuation")
    lsmc_summary = read_optional_json("lsmc_valuation_summary.json")
    if lsmc_summary:
        mtm = lsmc_summary.get("mtm_gbp", {})
        per_mw = lsmc_summary.get("mtm_gbp_per_mw", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("MTM mean", format_gbp(mtm.get("mean")))
        with c2:
            metric_card("MTM P50", format_gbp(mtm.get("p50")))
        with c3:
            metric_card("Per MW mean", format_gbp(per_mw.get("mean"), decimals=1))
        with c4:
            ratio = lsmc_summary.get("lsmc_ri_ratio")
            metric_card("LSMC / RI", f"{ratio:.2f}x" if isinstance(ratio, (int, float)) else "-")
        st.dataframe(pd.json_normalize(lsmc_summary).T, use_container_width=True)
    else:
        st.info("No LSMC summary yet. Run Phase 4 in the notebook to create lsmc_valuation_summary.json.")

    show_optional_image("lsmc_valuation.png", "Phase 4 LSMC valuation diagnostics")

with tabs[4]:
    st.header("Phase 5: MTM, Greeks, VaR and Stress")
    mtm_summary = read_optional_json("mtm_summary.json")
    if mtm_summary:
        mtm = mtm_summary.get("mtm", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            mtm_mean_val = mtm.get("mtm_mean", mtm.get("mean", mtm.get("total_mean", "-")))
            metric_card("MTM mean", format_gbp(mtm_mean_val))
        with c2:
            risk_95 = mtm_summary.get("risk_95", {})
            var_val = risk_95.get("var_gbp", "-")
            metric_card("VaR 95", format_gbp(var_val))
        with c3:
            cvar_val = risk_95.get("cvar_gbp", "-")
            metric_card("CVaR 95", format_gbp(cvar_val))

        st.subheader("Summary JSON")
        st.json(mtm_summary, expanded=False)
    else:
        st.info("No Phase 5 summary found. Run the MTM / Greeks / VaR cells to create mtm_summary.json.")

    c1, c2 = st.columns(2)
    with c1:
        show_optional_image("mtm_distribution.png", "MTM distribution")
    with c2:
        show_optional_image("scenario_stress.png", "Stress scenario impact")

with tabs[5]:
    st.header("Phase 6: Backtest and P&L Attribution")
    phase6 = read_optional_json("phase6_summary.json")
    if phase6:
        st.json(phase6, expanded=False)
    else:
        st.info("No Phase 6 summary found. Run Phase 6 cells to create phase6_summary.json.")

    st.subheader("Optional Backtest Figures")
    for fig_name in [
        "backtest_pnl.png",
        "pnl_attribution.png",
        "dual_bound.png",
    ]:
        show_optional_image(fig_name)
