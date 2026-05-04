"""
Streamlit app for BESS Phase 4 Duration Sweep Analysis

Displays:
- Method comparison across durations (1h, 2h, 3h, 4h)
- LSMC value attribution by revenue component
- Detailed results tables
- Run diagnostics and logs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Configuration ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
EXECUTED = PROCESSED / "executed"

st.set_page_config(
    page_title="BESS Phase 4: Duration Sweep",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔋 BESS Phase 4: Duration Sweep Analysis")
st.markdown("""
**Phase 4 Method Comparison & Attribution Across Battery Durations (1h–4h)**

This analysis sequentially runs valuation for multiple battery durations and compares:
- Method performance (initial intrinsic, DA rolling, LSMC, perfect foresight)
- Revenue attribution by component (HPFC, DA surprise, ancillary, degradation)
""")

# ── Cache data loaders ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_json(path: Path, mtime_ns: int) -> dict[str, Any]:
    _ = mtime_ns
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(path: Path, mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_image(path: Path, mtime_ns: int) -> bytes:
    _ = mtime_ns
    return path.read_bytes()


def optional_json(name: str) -> dict[str, Any]:
    path = PROCESSED / name
    if not path.exists():
        return {}
    return load_json(path, path.stat().st_mtime_ns)


def optional_csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        return pd.DataFrame()
    return load_csv(path, path.stat().st_mtime_ns)


def optional_image(name: str) -> bytes | None:
    path = PROCESSED / name
    if not path.exists():
        return None
    return load_image(path, path.stat().st_mtime_ns)


# ── Load data ───────────────────────────────────────────────────────────────
df_comparison = optional_csv("phase4_all_durations_comparison.csv")
df_attribution = optional_csv("phase4_all_durations_attribution.csv")
img_comparison = optional_image("phase4_all_durations_comparison.png")
img_attribution = optional_image("phase4_all_durations_attribution.png")
run_log = optional_json("phase4_sweep_run_log.json")

# ── Method & component definitions ──────────────────────────────────────────
METHODS = [
    'Initial hourly intrinsic',
    'DA rolling intrinsic',
    'WD rolling intrinsic',
    'Forward simulation (LSMC)',
    'Perfect foresight (DA energy)',
]

COMPONENTS = [
    'HPFC anchor',
    'DA surprise',
    'WD/intraday',
    'DC ancillary',
    'QR ancillary',
    'Costs (deg+VOM)',
]

# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Select view:",
    options=[
        "Overview",
        "Method Comparison",
        "Attribution Analysis",
        "Detailed Tables",
        "Run Diagnostics",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "**Data Location:** `data/processed/`\n\n"
    "**Output Files:**\n"
    "- `phase4_all_durations_comparison.{csv,json,png}`\n"
    "- `phase4_all_durations_attribution.{csv,json,png}`\n"
    "- Per-duration: `phase4_method_comparison_{d}h.*`\n"
    "- Attribution: `lsmc_attribution_{d}h.json`"
)

# ── Page: Overview ───────────────────────────────────────────────────────────
if page == "Overview":
    st.header("📋 Overview")

    if df_comparison.empty:
        st.warning(
            "⚠️ No comparison data loaded. "
            "Run notebook 13 (`13_phase4_duration_sweep.ipynb`) to generate results."
        )
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        durations = sorted(df_comparison["duration_h"].unique())
        lsmc_data = df_comparison[df_comparison["method"] == "Forward simulation (LSMC)"]

        with col1:
            st.metric("Durations analysed", len(durations))

        with col2:
            if not lsmc_data.empty:
                max_val = lsmc_data["value_gbp_annualized_m"].max()
                max_dur = lsmc_data[
                    lsmc_data["value_gbp_annualized_m"] == max_val
                ]["duration_h"].iloc[0]
                st.metric("Peak LSMC value", f"£{max_val:.1f}m", f"at {max_dur}h")
            else:
                st.metric("Peak LSMC value", "—")

        with col3:
            if not df_comparison.empty:
                n_methods = df_comparison["method"].nunique()
                st.metric("Methods compared", n_methods)
            else:
                st.metric("Methods compared", "—")

        with col4:
            if not df_attribution.empty:
                n_components = df_attribution["component"].nunique()
                st.metric("Revenue components", n_components)
            else:
                st.metric("Revenue components", "—")

        st.divider()
        st.subheader("Key Findings")

        if not lsmc_data.empty:
            dur_sorted = lsmc_data.sort_values("value_gbp_annualized_m", ascending=False)
            st.success(
                f"✓ **Optimal duration:** {dur_sorted.iloc[0]['duration_h']}h"
                f" (£{dur_sorted.iloc[0]['value_gbp_annualized_m']:.1f}m/year LSMC)"
            )

        st.markdown(
            "**Revenue components** (from LSMC attribution):\n"
            "- **HPFC anchor:** Baseload forward curve valuation\n"
            "- **DA surprise:** Deviation from day-ahead prices\n"
            "- **WD/intraday:** Intraday and within-day trading opportunities\n"
            "- **DC ancillary:** Dynamic containment (rapid frequency response)\n"
            "- **QR ancillary:** Quick reserve (minute-rated reserve)\n"
            "- **Costs:** Degradation shadow cost + variable O&M"
        )

# ── Page: Method Comparison ───────────────────────────────────────────────────
elif page == "Method Comparison":
    st.header("📈 Method Comparison Across Durations")

    if df_comparison.empty:
        st.warning("No comparison data available.")
    else:
        st.subheader("Aggregated Comparison Chart")
        if img_comparison:
            st.image(img_comparison, use_container_width=True)
        else:
            st.info("Chart image not yet generated. Run notebook 13 to create it.")

        st.divider()
        st.subheader("Interactive Method Comparison")

        col1, col2 = st.columns([2, 1])

        with col2:
            selected_methods = st.multiselect(
                "Filter methods:",
                options=METHODS,
                default=METHODS,
            )
            selected_durations = st.multiselect(
                "Filter durations (h):",
                options=sorted(df_comparison["duration_h"].unique()),
                default=sorted(df_comparison["duration_h"].unique()),
            )

        with col1:
            filtered = df_comparison[
                (df_comparison["method"].isin(selected_methods))
                & (df_comparison["duration_h"].isin(selected_durations))
            ]

            fig = go.Figure()

            for method in selected_methods:
                sub = filtered[filtered["method"] == method].sort_values("duration_h")
                if not sub.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sub["duration_h"],
                            y=sub["value_gbp_annualized_m"],
                            mode="lines+markers",
                            name=method,
                            hovertemplate="%{x}h: £%{y:.2f}m<extra></extra>",
                        )
                    )

            fig.update_layout(
                title="LSMC Value by Method and Duration",
                xaxis_title="Battery Duration (hours)",
                yaxis_title="Annual Value (£ million)",
                hovermode="x unified",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Pivot Table: Method vs. Duration")

        pivot = df_comparison.pivot_table(
            index="method",
            columns="duration_h",
            values="value_gbp_annualized_m",
            aggfunc="first",
        )
        pivot = pivot.loc[
            [m for m in METHODS if m in pivot.index]
        ]  # Reorder by METHODS
        pivot.columns = [f"{c:g}h" for c in pivot.columns]

        st.dataframe(pivot.round(2), use_container_width=True)

        st.divider()
        st.subheader("Individual Duration Results")

        selected_dur = st.selectbox(
            "Select duration to detail:",
            options=sorted(df_comparison["duration_h"].unique()),
        )

        dur_detail = df_comparison[df_comparison["duration_h"] == selected_dur].sort_values(
            "value_gbp_annualized_m", ascending=False
        )

        if not dur_detail.empty:
            st.markdown(f"#### {selected_dur}h Duration Ranking")
            dur_detail_display = dur_detail[
                ["method", "value_gbp_annualized_m", "p5_ann_m", "p95_ann_m"]
            ].copy()
            dur_detail_display.columns = [
                "Method",
                "Value (£m)",
                "5th percentile",
                "95th percentile",
            ]
            st.dataframe(dur_detail_display, use_container_width=True, hide_index=True)

# ── Page: Attribution Analysis ──────────────────────────────────────────────
elif page == "Attribution Analysis":
    st.header("🧩 Revenue Attribution Analysis")

    if df_attribution.empty:
        st.warning("No attribution data available.")
    else:
        st.subheader("Aggregated Attribution Chart")
        if img_attribution:
            st.image(img_attribution, use_container_width=True)
        else:
            st.info("Chart image not yet generated. Run notebook 13 to create it.")

        st.divider()
        st.subheader("Interactive Attribution Analysis")

        col1, col2 = st.columns([2, 1])

        with col2:
            selected_components = st.multiselect(
                "Filter components:",
                options=COMPONENTS,
                default=[c for c in COMPONENTS if c != "Costs (deg+VOM)"],
            )
            chart_type = st.radio(
                "Chart type:",
                options=["Mean value by component", "Percentage of gross revenue"],
            )

        with col1:
            filtered_attr = df_attribution[
                df_attribution["component"].isin(selected_components)
            ].sort_values("duration_h")

            fig = go.Figure()

            for component in selected_components:
                sub = filtered_attr[filtered_attr["component"] == component]
                if not sub.empty:
                    y_col = "pct_of_gross" if chart_type.startswith("Percentage") else "mean_m"
                    y_label = "%" if chart_type.startswith("Percentage") else "£m"
                    fig.add_trace(
                        go.Scatter(
                            x=sub["duration_h"],
                            y=sub[y_col],
                            mode="lines+markers",
                            name=component,
                            hovertemplate=f"%{{x}}h: %{{y:.1f}}{y_label}<extra></extra>",
                        )
                    )

            y_title = "% of gross revenue" if chart_type.startswith("Percentage") else "Annual value (£m)"
            fig.update_layout(
                title=f"Attribution — {chart_type.lower()}",
                xaxis_title="Battery Duration (hours)",
                yaxis_title=y_title,
                hovermode="x unified",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Attribution Summary Tables")

        tab1, tab2 = st.tabs(["Mean value (£m)", "Share of gross (%)"])

        with tab1:
            pivot_mean = df_attribution[
                df_attribution["component"].isin(COMPONENTS)
            ].pivot_table(
                index="component",
                columns="duration_h",
                values="mean_m",
                aggfunc="first",
            )
            pivot_mean = pivot_mean.loc[[c for c in COMPONENTS if c in pivot_mean.index]]
            pivot_mean.columns = [f"{c:g}h" for c in pivot_mean.columns]
            st.dataframe(pivot_mean.round(2), use_container_width=True)

        with tab2:
            pivot_pct = df_attribution[
                df_attribution["component"].isin(
                    [c for c in COMPONENTS if c != "Costs (deg+VOM)"]
                )
            ].pivot_table(
                index="component",
                columns="duration_h",
                values="pct_of_gross",
                aggfunc="first",
            )
            pivot_pct = pivot_pct.loc[
                [
                    c
                    for c in COMPONENTS
                    if c != "Costs (deg+VOM)" and c in pivot_pct.index
                ]
            ]
            pivot_pct.columns = [f"{c:g}h" for c in pivot_pct.columns]
            st.dataframe(pivot_pct.round(1), use_container_width=True)

# ── Page: Detailed Tables ─────────────────────────────────────────────────────
elif page == "Detailed Tables":
    st.header("📊 Detailed Data Tables")

    st.subheader("Method Comparison (Full Dataset)")
    if not df_comparison.empty:
        st.dataframe(
            df_comparison.sort_values(["duration_h", "method"]),
            use_container_width=True,
            height=400,
        )
        csv_comparison = df_comparison.to_csv(index=False)
        st.download_button(
            "Download comparison as CSV",
            csv_comparison,
            "phase4_comparison.csv",
            "text/csv",
        )
    else:
        st.info("No comparison data available.")

    st.divider()

    st.subheader("Attribution (Full Dataset)")
    if not df_attribution.empty:
        st.dataframe(
            df_attribution.sort_values(["duration_h", "component"]),
            use_container_width=True,
            height=400,
        )
        csv_attribution = df_attribution.to_csv(index=False)
        st.download_button(
            "Download attribution as CSV",
            csv_attribution,
            "phase4_attribution.csv",
            "text/csv",
        )
    else:
        st.info("No attribution data available.")

# ── Page: Run Diagnostics ─────────────────────────────────────────────────────
elif page == "Run Diagnostics":
    st.header("🔧 Run Diagnostics & Logs")

    st.subheader("Run Summary Log")
    if run_log:
        df_log = pd.DataFrame(run_log).T
        st.dataframe(df_log, use_container_width=True)
        st.success(f"✓ Total runs: {len(run_log)}")
    else:
        st.info("No run log available. Check `data/processed/phase4_sweep_run_log.json`.")

    st.divider()

    st.subheader("Generated Output Files")
    output_files = [
        "phase4_all_durations_comparison.csv",
        "phase4_all_durations_comparison.json",
        "phase4_all_durations_comparison.png",
        "phase4_all_durations_attribution.csv",
        "phase4_all_durations_attribution.json",
        "phase4_all_durations_attribution.png",
        "phase4_sweep_run_log.json",
    ]

    cols = st.columns(2)
    for i, file in enumerate(output_files):
        path = PROCESSED / file
        exists = path.exists()
        status = "✓" if exists else "✗"
        size = f"({path.stat().st_size / 1024:.1f} KB)" if exists else ""
        cols[i % 2].text(f"{status} {file} {size}")

    st.divider()

    st.subheader("Per-Duration Outputs")
    durations = [1.0, 2.0, 3.0, 4.0]

    for dur in durations:
        col1, col2, col3 = st.columns([1, 1, 1])
        files = [
            f"phase4_method_comparison_{dur:g}h.csv",
            f"phase4_method_comparison_{dur:g}h.json",
            f"lsmc_attribution_{dur:g}h.json",
        ]
        with col1:
            st.text(f"**{dur:g}h**")
        for i, file in enumerate(files):
            path = PROCESSED / file
            exists = path.exists()
            status = "✓" if exists else "✗"
            if i == 0:
                col1.text(f"  {status} {file}")
            elif i == 1:
                col2.text(f"  {status} {file}")
            else:
                col3.text(f"  {status} {file}")

    st.divider()

    st.subheader("Executed Notebooks")
    executed_nbs = list(EXECUTED.glob("12_phase4_*h.ipynb"))
    if executed_nbs:
        for nb in sorted(executed_nbs):
            size_kb = nb.stat().st_size / 1024
            st.text(f"✓ {nb.name} ({size_kb:.1f} KB)")
    else:
        st.info("No executed notebooks found.")

    st.divider()

    st.subheader("How to regenerate results")
    st.markdown("""
    Run notebook 13 to regenerate Phase 4 results:
    
    ```bash
    cd notebooks
    jupyter notebook 13_phase4_duration_sweep.ipynb
    ```
    
    Or execute via command line:
    ```bash
    jupyter nbconvert --to notebook --execute notebooks/13_phase4_duration_sweep.ipynb
    ```
    
    **Configuration:**
    - Sweep durations: 1h, 2h, 3h, 4h (edit `SWEEP_DURATIONS_H` in notebook)
    - Timeout per run: 3 hours (edit `TIMEOUT_S` in notebook)
    """)

# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**BESS Phase 4 Duration Sweep** — "
    "Method comparison & attribution analysis for fast-cycle battery storage (1–4 hours). "
    "[Repo](https://github.com/dmitry-goryunov/BESS-UK)"
)
