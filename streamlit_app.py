from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"
POWER_MW = 100
BUILT_VALUATION_DURATIONS = {2}

st.set_page_config(
    page_title="BESS UK Valuation Outputs",
    page_icon="B",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_json(path: Path, mtime_ns: int) -> dict[str, Any]:
    _ = mtime_ns
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_image(path: Path, mtime_ns: int) -> bytes:
    _ = mtime_ns
    return path.read_bytes()


@st.cache_data(show_spinner=False)
def load_parquet(path: Path, mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    return pd.read_parquet(path)


def optional_json(name: str) -> dict[str, Any]:
    path = OUT / name
    if not path.exists():
        return {}
    return load_json(path, path.stat().st_mtime_ns)


def optional_df(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        return pd.DataFrame()
    return load_parquet(path, path.stat().st_mtime_ns)


def duration_label(duration_h: int) -> str:
    return f"{duration_h:g}h"


def duration_file(stem: str, suffix: str, duration_h: int) -> str:
    return f"{stem}_{duration_label(duration_h)}{suffix}"


def formatted_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f} mil"
    if abs(value) > 100:
        return f"{value:,.0f}"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}"
    return f"{value:,.0f}"


def money(value: Any, decimals: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"GBP {formatted_number(value)}"


def pct(value: Any, decimals: int = 2, fraction: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    value = value * 100 if fraction else value
    return f"{value:,.{decimals}f}%"


def metric(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def format_table_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return formatted_number(value)
    return value


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(format_table_value).astype(str)


def show_table(df: pd.DataFrame, hide_index: bool = True) -> None:
    st.dataframe(format_table(df), hide_index=hide_index, width="stretch")


def show_image(name: str, caption: str) -> None:
    path = OUT / name
    if path.exists():
        st.image(load_image(path, path.stat().st_mtime_ns), caption=caption, width="stretch")
    else:
        st.info(f"Missing output: {name}")


def show_duration_image(stem: str, caption: str, duration_h: int) -> None:
    if not duration_is_published(duration_h):
        st.info(f"Not built for {duration_label(duration_h)} yet.")
        return
    name = duration_file(stem, ".png", duration_h)
    path = OUT / name
    if path.exists():
        st.image(load_image(path, path.stat().st_mtime_ns), caption=f"{caption} ({name})", width="stretch")
    else:
        st.info(f"Not built for {duration_label(duration_h)} yet: {name}")


def duration_output_notice(duration_h: int, source: str | None) -> None:
    if not source:
        st.info(
            f"No {duration_label(duration_h)} valuation output has been built yet. "
            f"Run the notebooks for {duration_label(duration_h)} to generate duration-labelled files."
        )


def duration_is_published(duration_h: int) -> bool:
    if duration_h not in BUILT_VALUATION_DURATIONS:
        return False
    label = duration_label(duration_h)
    required = [
        OUT / f"lsmc_valuation_summary_{label}.json",
        OUT / f"mtm_summary_{label}.json",
        OUT / f"phase6_summary_{label}.json",
        OUT / f"perfect_foresight_summary_{label}.json",
    ]
    return any(path.exists() for path in required)


def dict_table(data: dict[str, Any], title: str) -> None:
    st.subheader(title)
    if not data:
        st.info("No cached output found.")
        return
    rows = [{"metric": key, "value": value} for key, value in data.items() if not isinstance(value, (dict, list))]
    show_table(pd.DataFrame(rows))


def output_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(OUT.glob("*")):
        if path.is_file():
            rows.append(
                {
                    "file": path.name,
                    "type": path.suffix.lower().lstrip("."),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                }
            )
    return pd.DataFrame(rows)


def simulation_distribution(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, value in summary.items():
        if isinstance(value, dict) and {"p5", "p50", "p95"}.issubset(value):
            rows.append({"series": key, **value})
    return pd.DataFrame(rows)


def lsmc_metrics(summary: dict[str, Any]) -> None:
    mtm = summary.get("mtm_gbp_annualized") or summary.get("mtm_gbp", {})
    per_mw = summary.get("mtm_gbp_per_mw_year") or summary.get("mtm_gbp_per_mw", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Annual MTM mean", money(mtm.get("mean")), "Annualized valuation output")
    with c2:
        metric("Annual MTM P50", money(mtm.get("p50")))
    with c3:
        metric("Mean GBP/MW/yr", money(per_mw.get("mean")))
    with c4:
        metric("LSMC / RI", f"{summary.get('v_lsmc_over_v_ri', '-')}x")

    rows = [
        {"metric": "P5 annual GBP", "value": mtm.get("p5")},
        {"metric": "P50 annual GBP", "value": mtm.get("p50")},
        {"metric": "P95 annual GBP", "value": mtm.get("p95")},
        {"metric": "P5 GBP/MW/year", "value": per_mw.get("p5")},
        {"metric": "P50 GBP/MW/year", "value": per_mw.get("p50")},
        {"metric": "P95 GBP/MW/year", "value": per_mw.get("p95")},
        {"metric": "Backward pass seconds", "value": summary.get("bwd_time_s")},
        {"metric": "Forward pass seconds", "value": summary.get("fwd_time_s")},
    ]
    show_table(pd.DataFrame(rows))


def component_rows(values: dict[str, Any]) -> pd.DataFrame:
    labels = {
        "merchant": "merchant",
        "toll": "toll",
        "floor_contracted": "floor contracted",
        "cm": "capacity market",
        "floor_optionality": "floor optionality",
        "optimiser_fee": "optimiser fee, 12% negotiated",
        "opex_fixed": "fixed O&M",
        "augmentation": "augmentation",
        "total_mean": "total mean",
    }
    return pd.DataFrame(
        [
            {"component": labels[key], "GBP/MW/year": values[key]}
            for key in labels
            if key in values
        ]
    )


st.title("BESS UK Valuation Outputs")
st.caption("Streamlit dashboard for the important outputs and graphs from notebooks 01-07.")

if not OUT.exists():
    st.error("Missing data/processed output folder.")
    st.stop()

with st.sidebar:
    st.header("Asset")
    selected_duration_h = st.radio("Duration", [1, 2, 4], index=1, horizontal=True)
    st.metric("Power", f"{POWER_MW} MW")
    st.metric("Energy", f"{POWER_MW * selected_duration_h} MWh")
    st.caption("Uses duration-labelled valuation outputs when available.")
    st.divider()
    st.header("Published Files")
    show_table(output_inventory())
    st.divider()
    st.markdown("Run locally with `streamlit run streamlit_app.py`.")

sim_summary = optional_json("sim_summary.json")
selected_label = duration_label(selected_duration_h)
duration_published = duration_is_published(selected_duration_h)
lsmc_source = f"lsmc_valuation_summary_{selected_label}.json"
mtm_source = f"mtm_summary_{selected_label}.json"
phase6_source = f"phase6_summary_{selected_label}.json"
pf_source = f"perfect_foresight_summary_{selected_label}.json"
lsmc_view = optional_json(lsmc_source) if duration_published else {}
mtm_view = optional_json(mtm_source) if duration_published else {}
phase6_summary = optional_json(phase6_source) if duration_published else {}
pf_summary = optional_json(pf_source) if duration_published else {}
duration_basis = "duration-labelled notebook output" if duration_published else "not built"

tabs = st.tabs(
    [
        "Overview",
        "Calibration",
        "Simulation",
        "LSMC",
        "MTM Risk",
        "Backtest",
        "Perfect Foresight",
        "Files",
    ]
)

with tabs[0]:
    st.header("Executive Outputs")
    if duration_basis == "not built":
        duration_output_notice(selected_duration_h, None)
    lsmc = lsmc_view.get("mtm_gbp_annualized") or lsmc_view.get("mtm_gbp", {})
    mtm = mtm_view.get("mtm", {})
    risk = mtm_view.get("risk_95", {})
    pf_results = pf_summary.get("results", {})
    da_pf = pf_results.get("DA", {})
    phase6 = phase6_summary.get("dual_bound", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("LSMC annual mean", money(lsmc.get("mean")))
    with c2:
        metric("Lifecycle MTM mean", money(risk.get("mean_gbp_per_year", risk.get("mean_gbp"))))
    with c3:
        metric("Total GBP/MW/yr", money(mtm.get("total_mean")))
    with c4:
        metric("DA perfect foresight", money(da_pf.get("value_gbp_per_mw_year")), "GBP/MW/year")

    c1, c2, c3 = st.columns(3)
    with c1:
        show_duration_image("lsmc_valuation", "Phase 4 LSMC valuation diagnostics", selected_duration_h)
    with c2:
        show_duration_image("mtm_distribution", "Phase 5 MTM distribution", selected_duration_h)
    with c3:
        show_duration_image("perfect_foresight_da_high_value_week", "Phase 7 high-value DA dispatch week", selected_duration_h)

    st.subheader("Current Caveats")
    caveats = [
        f"Selected asset: {POWER_MW} MW / {POWER_MW * selected_duration_h} MWh ({selected_duration_h}h duration).",
        f"Valuation basis: {duration_basis}.",
        "Phase 4-6 outputs are fast-development partial-mode outputs, not final production economics.",
        "The dual bound is a clairvoyant information-relaxation benchmark, not a martingale-penalty proof.",
        "The perfect-foresight benchmark is an upper benchmark and is not a tradable strategy.",
    ]
    for item in caveats:
        st.write(f"- {item}")

    if phase6:
        st.subheader("Backtest Headline")
        st.write(f"Dual gap: {pct(phase6.get('gap_pct'))}")

with tabs[1]:
    st.header("Phase 2: Calibration Outputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        dict_table(optional_json("ss_params.json"), "Schwartz-Smith")
    with c2:
        dict_table(optional_json("imbalance_params.json"), "Imbalance")
    with c3:
        pca = optional_json("pca_params.json")
        st.subheader("PCA Variance")
        if pca.get("explained_variance_ratio"):
            pca_df = pd.DataFrame(
                {
                    "factor": [f"PC{i + 1}" for i in range(len(pca["explained_variance_ratio"]))],
                    "variance_pct": [100 * value for value in pca["explained_variance_ratio"]],
                }
            )
            st.bar_chart(pca_df.set_index("factor"), height=260)
        else:
            st.info("No PCA output found.")

    anc = optional_json("ancillary_params.json")
    st.subheader("Ancillary Parameters")
    if anc.get("products"):
        show_table(pd.DataFrame(anc["products"]).T.reset_index())
    else:
        st.json(anc, expanded=False)

with tabs[2]:
    st.header("Phase 3: Joint Simulation")
    if sim_summary:
        c1, c2, c3, c4 = st.columns(4)
        validation = sim_summary.get("validation", {})
        with c1:
            metric("Paths", f"{sim_summary.get('n_paths', 0):,}")
        with c2:
            metric("Half-hours", f"{sim_summary.get('n_steps', 0):,}")
        with c3:
            metric("Seed", sim_summary.get("seed", "-"))
        with c4:
            metric("Validation", f"{sum(bool(v) for v in validation.values())}/{len(validation)}")

        dist = simulation_distribution(sim_summary)
        if not dist.empty:
            st.subheader("Terminal Distribution")
            show_table(dist)

        if validation:
            st.subheader("Validation Checks")
            show_table(pd.DataFrame(validation.items(), columns=["check", "passed"]))
    else:
        st.info("No simulation summary found.")

    for row in [
        [("sim_sample_paths.png", "Sample paths"), ("sim_validation.png", "Validation checks")],
        [("sim_cross_corr.png", "Cross correlations"), ("sim_3yr_chain.png", "Three-year chain")],
    ]:
        cols = st.columns(len(row))
        for col, (name, caption) in zip(cols, row):
            with col:
                show_image(name, caption)

with tabs[3]:
    st.header("Phase 4: LSMC Valuation")
    st.caption(f"Showing {selected_duration_h}h output. Source: {lsmc_source}.")
    if lsmc_view:
        lsmc_metrics(lsmc_view)
        with st.expander("Raw LSMC summary"):
            st.json(lsmc_view, expanded=False)
    else:
        duration_output_notice(selected_duration_h, None)
    show_duration_image("lsmc_valuation", "Phase 4 LSMC valuation diagnostics", selected_duration_h)

with tabs[4]:
    st.header("Phase 5: MTM, Greeks, VaR and Stress")
    st.caption(f"Showing {selected_duration_h}h output. Source: {mtm_source}.")
    if mtm_view:
        mtm = mtm_view.get("mtm", {})
        risk95 = mtm_view.get("risk_95", {})
        risk99 = mtm_view.get("risk_99", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric("Annual mean", money(risk95.get("mean_gbp_per_year", risk95.get("mean_gbp"))))
        with c2:
            metric("VaR 95 (P5-P50)", money(risk95.get("var_gbp_per_year", risk95.get("var_gbp"))))
        with c3:
            metric("CVaR 95 (tail-P50)", money(risk95.get("cvar_gbp_per_year", risk95.get("cvar_gbp"))))
        with c4:
            metric("MTM GBP/MW/yr", money(mtm.get("total_mean")))

        st.subheader("Components")
        show_table(component_rows(mtm))

        st.subheader("Risk Metrics")
        show_table(
            pd.DataFrame(
                [
                    {"confidence": "95%", **risk95},
                    {"confidence": "99%", **risk99},
                ]
            )
        )

        greeks = mtm_view.get("greeks", {})
        if greeks:
            st.subheader("Greek Ladder")
            show_table(pd.DataFrame(greeks).T.reset_index())

        scenarios = mtm_view.get("scenarios", {})
        if scenarios:
            st.subheader("Scenario Stress")
            show_table(pd.DataFrame(scenarios).T.reset_index())
    else:
        duration_output_notice(selected_duration_h, None)

    for row in [
        [("mtm_components", "MTM components"), ("mtm_distribution", "MTM distribution")],
        [("greek_ladder", "Greek ladder"), ("var_cvar", "VaR / CVaR")],
        [("scenario_stress", "Scenario stress")],
    ]:
        cols = st.columns(len(row))
        for col, (stem, caption) in zip(cols, row):
            with col:
                show_duration_image(stem, caption, selected_duration_h)

    show_duration_image("soh_trajectory", "SOH trajectory", selected_duration_h)

with tabs[5]:
    st.header("Phase 6: Dual Bound and Backtest")
    st.caption(f"Showing {selected_duration_h}h output. Source: {phase6_source}.")
    if phase6_summary:
        dual = phase6_summary.get("dual_bound", {})
        backtest = phase6_summary.get("backtest", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric("Dual gap", pct(dual.get("gap_pct")))
        with c2:
            metric("V LSMC", money(dual.get("v_lsmc_gbp", dual.get("v_lsmc"))))
        with c3:
            metric("Residual / total", pct(backtest.get("residual_pct_total")))
        with c4:
            metric("Residual target", pct(backtest.get("target_residual_pct"), fraction=True))

        st.subheader("Backtest")
        show_table(pd.DataFrame(backtest.items(), columns=["metric", "value"]))
        with st.expander("Raw Phase 6 summary"):
            st.json(phase6_summary, expanded=False)
    else:
        duration_output_notice(selected_duration_h, None)

    c1, c2, c3 = st.columns(3)
    with c1:
        show_duration_image("dual_bound", "Dual bound", selected_duration_h)
    with c2:
        show_duration_image("backtest_pnl", "Backtest P&L", selected_duration_h)
    with c3:
        show_duration_image("pnl_attribution", "P&L attribution", selected_duration_h)

with tabs[6]:
    st.header("Phase 7: Historical Perfect-Foresight Benchmark")
    st.caption(f"Showing {selected_duration_h}h output. Source: {pf_source}.")
    if pf_summary:
        results = pf_summary.get("results", {})
        rows = []
        for market, values in results.items():
            rows.append({"market": market, **values})
        table = pd.DataFrame(rows)
        if not table.empty:
            c1, c2, c3, c4 = st.columns(4)
            da = results.get("DA", {})
            sp = results.get("SP", {})
            with c1:
                metric("DA value", money(da.get("value_gbp")))
            with c2:
                metric("DA GBP/MW/yr", money(da.get("value_gbp_per_mw_year")))
            with c3:
                metric("SP value", money(sp.get("value_gbp")))
            with c4:
                metric("SP GBP/MW/yr", money(sp.get("value_gbp_per_mw_year")))

            show_table(table)

        dispatch = optional_df(duration_file("perfect_foresight_da_dispatch", ".parquet", selected_duration_h))
        if not dispatch.empty:
            st.subheader("DA Dispatch Sample")
            show_table(dispatch.head(300), hide_index=False)
    else:
        duration_output_notice(selected_duration_h, None)

    show_duration_image("perfect_foresight_da_high_value_week", "Highest-value day-ahead dispatch week", selected_duration_h)

with tabs[7]:
    st.header("Output Files")
    show_table(output_inventory())
    st.subheader("Raw JSON Outputs")
    selected = st.selectbox(
        "Summary file",
        [
            "sim_summary.json",
            f"lsmc_valuation_summary_{selected_label}.json",
            f"mtm_summary_{selected_label}.json",
            f"phase6_summary_{selected_label}.json",
            f"perfect_foresight_summary_{selected_label}.json",
            "ss_params.json",
            "pca_params.json",
            "imbalance_params.json",
            "ancillary_params.json",
        ],
    )
    st.json(optional_json(selected), expanded=False)
