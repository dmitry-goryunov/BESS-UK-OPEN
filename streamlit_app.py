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
def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


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


def metric_card(label: str, value, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


st.title("BESS UK Stochastic Valuation")
st.caption("Cached data and model outputs from the GB battery storage valuation workflow.")

with st.sidebar:
    st.header("Project")
    st.markdown("[GitHub repo](https://github.com/dmitry-goryunov/BESS-UK)")
    st.divider()
    st.write("Data source files")
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

tab_data, tab_calibration, tab_simulation = st.tabs(
    ["Market Data", "Calibration", "Simulation"]
)

with tab_data:
    st.subheader("Market Data Coverage")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("DA rows", f"{len(df_da):,}")
    with c2:
        metric_card("System price rows", f"{len(df_sp):,}")
    with c3:
        metric_card("Ancillary rows", f"{len(df_anc):,}")
    with c4:
        metric_card("Forward rows", f"{len(df_fwd):,}")

    da_daily = (
        df_da.groupby("settlement_date", as_index=True)["price_gbp_mwh"]
        .mean()
        .sort_index()
    )
    sp_daily = (
        df_sp.groupby("settlement_date", as_index=True)["system_price"]
        .mean()
        .sort_index()
        if "system_price" in df_sp.columns
        else pd.Series(dtype=float)
    )

    st.subheader("Daily Average Prices")
    price_panel = pd.DataFrame(
        {
            "day_ahead_gbp_mwh": da_daily,
            "system_price_gbp_mwh": sp_daily,
        }
    )
    st.line_chart(price_panel, height=360)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Day-Ahead Distribution")
        hist = (
            df_da["price_gbp_mwh"]
            .clip(-100, 300)
            .value_counts(bins=40)
            .sort_index()
        )
        hist.index = hist.index.astype(str)
        st.bar_chart(hist, height=300)
    with c2:
        st.subheader("Forward Curve")
        if {"delivery_start", "price_gbp_mwh"}.issubset(df_fwd.columns):
            fwd_plot = df_fwd.sort_values("delivery_start").set_index("delivery_start")[
                "price_gbp_mwh"
            ]
            st.line_chart(fwd_plot, height=300)
        else:
            st.dataframe(df_fwd, use_container_width=True)

    st.subheader("Raw Samples")
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

with tab_calibration:
    st.subheader("Calibrated Parameters")
    ss = load_json(PROCESSED_DIR / "ss_params.json")
    pca = load_json(PROCESSED_DIR / "pca_params.json")
    imb = load_json(PROCESSED_DIR / "imbalance_params.json")
    anc = load_json(PROCESSED_DIR / "ancillary_params.json")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Schwartz-Smith**")
        st.dataframe(
            pd.DataFrame(ss.items(), columns=["parameter", "value"]),
            hide_index=True,
            use_container_width=True,
        )
    with c2:
        st.markdown("**Imbalance OU + jumps**")
        st.dataframe(
            pd.DataFrame(imb.items(), columns=["parameter", "value"]),
            hide_index=True,
            use_container_width=True,
        )
    with c3:
        st.markdown("**PCA variance explained**")
        evr = pd.DataFrame(
            {
                "factor": [f"PC{i+1}" for i in range(len(pca["explained_variance_ratio"]))],
                "variance_pct": [
                    100 * x for x in pca["explained_variance_ratio"]
                ],
            }
        )
        st.bar_chart(evr.set_index("factor"), height=250)

    st.subheader("Ancillary Product Parameters")
    products = pd.DataFrame(anc.get("products", {})).T.reset_index(drop=True)
    st.dataframe(products, use_container_width=True)

with tab_simulation:
    st.subheader("Simulation Summary")
    summary_path = PROCESSED_DIR / "sim_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
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

        st.markdown("**Terminal distributions**")
        distributions = []
        for group, values in summary.items():
            if isinstance(values, dict) and {"p5", "p50", "p95"}.issubset(values):
                distributions.append({"series": group, **values})
        st.dataframe(pd.DataFrame(distributions), hide_index=True, use_container_width=True)

        st.markdown("**Validation checks**")
        st.dataframe(
            pd.DataFrame(validation.items(), columns=["check", "passed"]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("No simulation summary found.")

    st.subheader("Generated Figures")
    figures = [
        "sim_sample_paths.png",
        "sim_validation.png",
        "sim_corr_matrix.png",
        "sim_cross_corr.png",
        "sim_3yr_chain.png",
    ]
    for fig_name in figures:
        fig_path = PROCESSED_DIR / fig_name
        if fig_path.exists():
            st.image(str(fig_path), caption=fig_name, use_container_width=True)
