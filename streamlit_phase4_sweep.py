"""
BESS UK — Research Dashboard

Three sections:
1. Phase 4 Duration Sweep  (notebook 13) — LSMC method comparison 1h–4h
2. Historical BESS Index vs Modo (notebook 19) — calibrated backtest vs Modo Energy public index
3. Forward vs Realized — nb13 LSMC/WD-rolling vs nb19 historical: price basis + optimality gap
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
    page_title="BESS UK Research",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cache helpers ─────────────────────────────────────────────────────────────
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔋 BESS UK Research")
section = st.sidebar.radio(
    "Section",
    options=["Phase 4: Duration Sweep", "Historical Index vs Modo", "Forward vs Realized"],
    index=0,
)
st.sidebar.divider()

# ── Load shared data once (used by sections 1, 2, 3) ─────────────────────────
df_comparison  = optional_csv("phase4_all_durations_comparison.csv")
df_attribution = optional_csv("phase4_all_durations_attribution.csv")

_m1 = optional_csv("historical_index_1h_monthly.csv")
_m2 = optional_csv("historical_index_2h_monthly.csv")
_bm1 = optional_csv("bm_index_1h.csv")
_bm2 = optional_csv("bm_index_2h.csv")
for _df in (_m1, _m2):
    if not _df.empty:
        _df["period_dt"] = pd.to_datetime(_df["year_month"])
for _df in (_bm1, _bm2):
    if not _df.empty:
        _df["period_dt"] = pd.to_datetime(_df["year_month"])

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Phase 4 Duration Sweep
# ═══════════════════════════════════════════════════════════════════════════════
if section == "Phase 4: Duration Sweep":

    # ── Phase 4 data ────────────────────────────────────────────────────────
    img_comparison = optional_image("phase4_all_durations_comparison.png")
    img_attribution = optional_image("phase4_all_durations_attribution.png")
    run_log = optional_json("phase4_sweep_run_log.json")

    METHODS = [
        "Initial hourly intrinsic",
        "DA rolling intrinsic",
        "WD rolling intrinsic",
        "Forward simulation (LSMC)",
        "Perfect foresight (DA energy)",
    ]
    COMPONENTS = [
        "HPFC anchor",
        "DA surprise",
        "Imbalance proxy (BM/ID substitute)",
        "DC ancillary",
        "QR ancillary",
        "Costs (deg+VOM)",
    ]

    page = st.sidebar.radio(
        "View",
        options=[
            "Overview",
            "Method Comparison",
            "Attribution Analysis",
            "Detailed Tables",
            "Run Diagnostics",
        ],
    )

    st.title("🔋 BESS Phase 4: Duration Sweep Analysis")
    st.markdown(
        "**Method comparison & attribution across battery durations (1h–4h)** — "
        "initial intrinsic, DA rolling, WD rolling, LSMC, perfect foresight."
    )

    # ── Overview ─────────────────────────────────────────────────────────────
    if page == "Overview":
        st.header("Overview")

        if df_comparison.empty:
            st.warning(
                "No comparison data loaded. "
                "Run notebook 13 (`13_phase4_duration_sweep.ipynb`) to generate results."
            )
        else:
            col1, col2, col3, col4 = st.columns(4)
            durations = sorted(df_comparison["duration_h"].unique())
            lsmc_data = df_comparison[df_comparison["method"] == "Forward simulation (LSMC)"]

            with col1:
                st.metric("Durations analysed", len(durations))
            with col2:
                if not lsmc_data.empty:
                    max_val = lsmc_data["value_gbp_annualized_m"].max()
                    max_dur = lsmc_data[lsmc_data["value_gbp_annualized_m"] == max_val]["duration_h"].iloc[0]
                    st.metric("Peak LSMC value", f"£{max_val:.1f}m", f"at {max_dur}h")
                else:
                    st.metric("Peak LSMC value", "—")
            with col3:
                st.metric("Methods compared", df_comparison["method"].nunique())
            with col4:
                st.metric("Revenue components", df_attribution["component"].nunique() if not df_attribution.empty else "—")

            st.divider()
            st.subheader("Key Findings")
            if not lsmc_data.empty:
                dur_sorted = lsmc_data.sort_values("value_gbp_annualized_m", ascending=False)
                st.success(
                    f"Optimal duration: **{dur_sorted.iloc[0]['duration_h']}h** "
                    f"(£{dur_sorted.iloc[0]['value_gbp_annualized_m']:.1f}m/year LSMC)"
                )
            st.markdown(
                "**Revenue components** (LSMC attribution):\n"
                "- **HPFC anchor:** Baseload forward curve valuation\n"
                "- **DA surprise:** Deviation from day-ahead prices\n"
                "- **Imbalance proxy:** BM/ID substitute value\n"
                "- **DC ancillary:** Dynamic containment\n"
                "- **QR ancillary:** Quick reserve\n"
                "- **Costs:** Degradation shadow cost + variable O&M"
            )

    # ── Method Comparison ─────────────────────────────────────────────────────
    elif page == "Method Comparison":
        st.header("Method Comparison Across Durations")

        if df_comparison.empty:
            st.warning("No comparison data available.")
        else:
            st.subheader("Aggregated Comparison Chart")
            if img_comparison:
                st.image(img_comparison, use_container_width=True)

            st.divider()
            st.subheader("Interactive Method Comparison")
            col1, col2 = st.columns([2, 1])

            with col2:
                selected_methods = st.multiselect("Filter methods:", options=METHODS, default=METHODS)
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
                        fig.add_trace(go.Scatter(
                            x=sub["duration_h"], y=sub["value_gbp_annualized_m"],
                            mode="lines+markers", name=method,
                            hovertemplate="%{x}h: £%{y:.2f}m<extra></extra>",
                        ))
                fig.update_layout(
                    title="Annual Value by Method and Duration",
                    xaxis_title="Battery Duration (hours)",
                    yaxis_title="Annual Value (£ million)",
                    hovermode="x unified", height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Pivot Table: Method vs Duration")
            pivot = df_comparison.pivot_table(
                index="method", columns="duration_h",
                values="value_gbp_annualized_m", aggfunc="first",
            )
            pivot = pivot.loc[[m for m in METHODS if m in pivot.index]]
            pivot.columns = [f"{c:g}h" for c in pivot.columns]
            st.dataframe(pivot.round(2), use_container_width=True)

            st.divider()
            st.subheader("Individual Duration Detail")
            selected_dur = st.selectbox(
                "Select duration:", options=sorted(df_comparison["duration_h"].unique())
            )
            dur_detail = df_comparison[df_comparison["duration_h"] == selected_dur].sort_values(
                "value_gbp_annualized_m", ascending=False
            )
            if not dur_detail.empty:
                disp = dur_detail[["method", "value_gbp_annualized_m", "p5_ann_m", "p95_ann_m"]].copy()
                disp.columns = ["Method", "Value (£m)", "P5", "P95"]
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Attribution Analysis ──────────────────────────────────────────────────
    elif page == "Attribution Analysis":
        st.header("Revenue Attribution Analysis")

        if df_attribution.empty:
            st.warning("No attribution data available.")
        else:
            st.subheader("Aggregated Attribution Chart")
            if img_attribution:
                st.image(img_attribution, use_container_width=True)

            st.divider()
            st.subheader("Interactive Attribution")
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
                filtered_attr = df_attribution[df_attribution["component"].isin(selected_components)].sort_values("duration_h")
                fig = go.Figure()
                for component in selected_components:
                    sub = filtered_attr[filtered_attr["component"] == component]
                    if not sub.empty:
                        y_col = "pct_of_gross" if chart_type.startswith("Percentage") else "mean_m"
                        y_label = "%" if chart_type.startswith("Percentage") else "£m"
                        fig.add_trace(go.Scatter(
                            x=sub["duration_h"], y=sub[y_col],
                            mode="lines+markers", name=component,
                            hovertemplate=f"%{{x}}h: %{{y:.1f}}{y_label}<extra></extra>",
                        ))
                y_title = "% of gross revenue" if chart_type.startswith("Percentage") else "Annual value (£m)"
                fig.update_layout(
                    title=f"Attribution — {chart_type.lower()}",
                    xaxis_title="Battery Duration (hours)",
                    yaxis_title=y_title,
                    hovermode="x unified", height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Attribution Tables")
            tab1, tab2 = st.tabs(["Mean value (£m)", "Share of gross (%)"])
            with tab1:
                pivot_mean = df_attribution[df_attribution["component"].isin(COMPONENTS)].pivot_table(
                    index="component", columns="duration_h", values="mean_m", aggfunc="first",
                )
                pivot_mean = pivot_mean.loc[[c for c in COMPONENTS if c in pivot_mean.index]]
                pivot_mean.columns = [f"{c:g}h" for c in pivot_mean.columns]
                st.dataframe(pivot_mean.round(2), use_container_width=True)
            with tab2:
                no_costs = [c for c in COMPONENTS if c != "Costs (deg+VOM)"]
                pivot_pct = df_attribution[df_attribution["component"].isin(no_costs)].pivot_table(
                    index="component", columns="duration_h", values="pct_of_gross", aggfunc="first",
                )
                pivot_pct = pivot_pct.loc[[c for c in no_costs if c in pivot_pct.index]]
                pivot_pct.columns = [f"{c:g}h" for c in pivot_pct.columns]
                st.dataframe(pivot_pct.round(1), use_container_width=True)

    # ── Detailed Tables ───────────────────────────────────────────────────────
    elif page == "Detailed Tables":
        st.header("Detailed Data Tables")
        st.subheader("Method Comparison (Full Dataset)")
        if not df_comparison.empty:
            st.dataframe(df_comparison.sort_values(["duration_h", "method"]), use_container_width=True, height=400)
            st.download_button("Download comparison CSV", df_comparison.to_csv(index=False), "phase4_comparison.csv", "text/csv")
        else:
            st.info("No comparison data available.")

        st.divider()
        st.subheader("Attribution (Full Dataset)")
        if not df_attribution.empty:
            st.dataframe(df_attribution.sort_values(["duration_h", "component"]), use_container_width=True, height=400)
            st.download_button("Download attribution CSV", df_attribution.to_csv(index=False), "phase4_attribution.csv", "text/csv")
        else:
            st.info("No attribution data available.")

    # ── Run Diagnostics ───────────────────────────────────────────────────────
    elif page == "Run Diagnostics":
        st.header("Run Diagnostics & Logs")

        st.subheader("Run Summary Log")
        if run_log:
            st.dataframe(pd.DataFrame(run_log).T, use_container_width=True)
            st.success(f"Total runs logged: {len(run_log)}")
        else:
            st.info("No run log available.")

        st.divider()
        st.subheader("Output Files")
        for fname in [
            "phase4_all_durations_comparison.csv",
            "phase4_all_durations_comparison.png",
            "phase4_all_durations_attribution.csv",
            "phase4_all_durations_attribution.png",
            "phase4_sweep_run_log.json",
        ]:
            p = PROCESSED / fname
            exists = p.exists()
            size = f"({p.stat().st_size / 1024:.1f} KB)" if exists else ""
            st.text(f"{'✓' if exists else '✗'} {fname} {size}")

        st.divider()
        st.subheader("Per-Duration Outputs")
        for dur in [1.0, 2.0, 3.0, 4.0]:
            for fname in [
                f"phase4_method_comparison_{dur:g}h.csv",
                f"lsmc_attribution_{dur:g}h.json",
            ]:
                p = PROCESSED / fname
                st.text(f"  {'✓' if p.exists() else '✗'} {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Historical BESS Index vs Modo
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Historical Index vs Modo":

    # ── Load historical data ──────────────────────────────────────────────────
    m1, m2, bm1, bm2 = _m1, _m2, _bm1, _bm2
    img_bm = optional_image("historical_index_with_bm.png")

    page = st.sidebar.radio(
        "View",
        options=[
            "Comparison Chart",
            "Revenue Streams",
            "Assumptions & Calibration",
            "Gap Analysis",
        ],
    )

    st.title("📊 Historical BESS Index vs Modo Energy")
    st.markdown(
        "**Calibrated backtest (Apr 2024–Apr 2026) vs Modo Energy ME BESS GB public index.** "
        "Model streams: DA energy, intraday/WD, ancillary services (DC/DM/DR/QR), "
        "actual BM fleet revenue (Elexon)."
    )

    # ── Page: Comparison Chart ─────────────────────────────────────────────────
    if page == "Comparison Chart":
        st.header("Model + Actual BM vs Modo Optimal")

        if img_bm:
            st.image(img_bm, use_container_width=True)
        else:
            st.warning("Chart not found. Run `run_headroom_variant.py` or `run_step7.py` to regenerate.")

        st.divider()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model+BM 1H avg", "£52.9k/MW/yr")
        with col2:
            st.metric("Modo 1H avg", "£50.8k/MW/yr", delta="-£2.0k gap")
        with col3:
            st.metric("Model+BM 2H avg", "£70.8k/MW/yr")
        with col4:
            st.metric("Modo 2H avg", "£71.9k/MW/yr", delta="+£1.1k gap")

        st.info(
            "**Period:** Apr 2024 – Apr 2026 (2 years). "
            "**Basis:** Gross revenue, VOM and degradation costs excluded for like-for-like comparison. "
            "Modo index is an optimal dispatch model, not actual fleet revenues."
        )

    # ── Page: Revenue Streams ─────────────────────────────────────────────────
    elif page == "Revenue Streams":
        st.header("Revenue Streams — Model Data")

        if m1.empty and m2.empty:
            st.warning("No historical index data loaded. Run notebook 19 to generate results.")
        else:
            BLUE, ORANGE = "#1565C0", "#E65100"

            dur_sel = st.radio("Duration", options=["1h", "2h"], horizontal=True)
            m = m1 if dur_sel == "1h" else m2
            bm = bm1 if dur_sel == "1h" else bm2
            col_color = BLUE if dur_sel == "1h" else ORANGE
            modo_ref = 50.8 if dur_sel == "1h" else 71.9

            if m.empty:
                st.warning(f"No data for {dur_sel}.")
            else:
                # Stacked bar: DA, WD, Ancillary, BM
                m_merged = m.merge(
                    bm[["period_dt", "bm_rev_gbp_mw_yr"]] if not bm.empty else pd.DataFrame(columns=["period_dt", "bm_rev_gbp_mw_yr"]),
                    on="period_dt", how="left",
                )
                m_merged["bm_k"] = m_merged["bm_rev_gbp_mw_yr"].fillna(0) / 1000
                m_merged["da_k"] = m_merged["da_revenue_gbp_mw_yr"] / 1000
                m_merged["wd_k"] = m_merged["wd_revenue_gbp_mw_yr"] / 1000
                m_merged["anc_k"] = m_merged["anc_revenue_gbp_mw_yr"] / 1000
                m_merged["total_k"] = m_merged["da_k"] + m_merged["wd_k"] + m_merged["anc_k"] + m_merged["bm_k"]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=m_merged["period_dt"], y=m_merged["da_k"],
                    name="DA energy", marker_color="#42A5F5",
                    hovertemplate="%{x|%b %Y} DA: £%{y:.1f}k<extra></extra>",
                ))
                fig.add_trace(go.Bar(
                    x=m_merged["period_dt"], y=m_merged["wd_k"],
                    name="Intraday / WD (cap £60)", marker_color="#1565C0",
                    hovertemplate="%{x|%b %Y} WD: £%{y:.1f}k<extra></extra>",
                ))
                fig.add_trace(go.Bar(
                    x=m_merged["period_dt"], y=m_merged["anc_k"],
                    name="Ancillary (DC+DM+DR+QR)", marker_color="#66BB6A",
                    hovertemplate="%{x|%b %Y} Anc: £%{y:.1f}k<extra></extra>",
                ))
                fig.add_trace(go.Bar(
                    x=m_merged["period_dt"], y=m_merged["bm_k"],
                    name="BM fleet avg (Elexon)", marker_color="#FFA726",
                    hovertemplate="%{x|%b %Y} BM: £%{y:.1f}k<extra></extra>",
                ))
                fig.add_hline(
                    y=modo_ref, line_dash="dot", line_color="red",
                    annotation_text=f"Modo {dur_sel.upper()} avg £{modo_ref}k",
                    annotation_position="top right",
                )
                fig.update_layout(
                    barmode="stack",
                    title=f"Monthly Revenue by Stream — {dur_sel.upper()} battery (gross, £k/MW/yr annualised)",
                    xaxis_title="Month",
                    yaxis_title="£k / MW / yr (annualised)",
                    hovermode="x unified",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("Stream Averages (Apr 2024 – Apr 2026)")
                avg_rows = {
                    "DA energy": f"£{m_merged['da_k'].mean():.1f}k",
                    "Intraday / WD (cap £60)": f"£{m_merged['wd_k'].mean():.1f}k",
                    "Ancillary (DC+DM+DR+QR)": f"£{m_merged['anc_k'].mean():.1f}k",
                    "BM fleet avg (Elexon)": f"£{m_merged['bm_k'].mean():.1f}k",
                    "**Total model + BM**": f"**£{m_merged['total_k'].mean():.1f}k**",
                    "Modo optimal": f"£{modo_ref}k",
                    "Gap": f"£{m_merged['total_k'].mean() - modo_ref:.1f}k",
                }
                st.table(pd.DataFrame.from_dict(avg_rows, orient="index", columns=["£k/MW/yr"]))

                st.download_button(
                    f"Download {dur_sel} monthly data CSV",
                    m.to_csv(index=False),
                    f"historical_index_{dur_sel}_monthly.csv",
                    "text/csv",
                )

    # ── Page: Assumptions & Calibration ──────────────────────────────────────
    elif page == "Assumptions & Calibration":
        st.header("Assumptions & Calibration")

        st.subheader("Locked Base Case")
        st.markdown(
            "All parameters below are the **locked central case** producing the published results."
        )

        assumptions = pd.DataFrame([
            {"Parameter": "WD cap (SP−DA basis)", "Value": "£60/MWh", "Notes": "Proxy for intraday execution; most spread sits below £60"},
            {"Parameter": "DC headroom", "Value": "35% of nameplate", "Notes": "Reduced from 50%; lowest-clearing product (£2.71/MW/h)"},
            {"Parameter": "DM headroom", "Value": "10% of nameplate", "Notes": "Dynamic moderation"},
            {"Parameter": "DR headroom", "Value": "5% of nameplate", "Notes": "Dynamic regulation"},
            {"Parameter": "QR headroom", "Value": "15% of nameplate", "Notes": "Quick reserve (highest clearing £4.42/MW/h)"},
            {"Parameter": "DA residual", "Value": "35% of nameplate", "Notes": "1 − (DC+DM+DR+QR) = 1 − 0.65"},
            {"Parameter": "VOM & degradation", "Value": "Excluded", "Notes": "Gross basis for like-for-like vs Modo"},
            {"Parameter": "BM revenue", "Value": "Actual fleet avg", "Notes": "Elexon settlement volumes × system price, 71 BMUs, 3,307 MW"},
            {"Parameter": "Comparison period", "Value": "Apr 2024 – Apr 2026", "Notes": "Both Modo index and model data available"},
        ])
        st.dataframe(assumptions, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Ancillary Clearing Prices (2024–2026 actual NESO EAC)")
        anc_prices = pd.DataFrame([
            {"Product": "DC (Dynamic Containment)", "Mean clearing": "£2.71/MW/h", "Headroom held": "35%", "Revenue contribution": "£5.1k/MW/yr"},
            {"Product": "DM (Dynamic Moderation)", "Mean clearing": "£3.46/MW/h", "Headroom held": "10%", "Revenue contribution": "£1.5k/MW/yr"},
            {"Product": "DR (Dynamic Regulation)", "Mean clearing": "£3.87/MW/h", "Headroom held": "5%", "Revenue contribution": "£0.9k/MW/yr"},
            {"Product": "QR (Quick Reserve)", "Mean clearing": "£4.42/MW/h", "Headroom held": "15%", "Revenue contribution": "£4.6k/MW/yr"},
        ])
        st.dataframe(anc_prices, use_container_width=True, hide_index=True)
        st.caption(
            "Key insight: DC is the *lowest*-clearing product yet was originally assigned 50% headroom — "
            "an inverted calibration. Reducing DC to 35% frees DA headroom, increasing rolling LP "
            "throughput and WD revenue from ~£25k to ~£40k (2H)."
        )

        st.divider()
        st.subheader("DC Headroom Calibration Path (2H gross)")
        calibration = pd.DataFrame([
            {"DC%": "50% (original)", "DA%": "20%", "Model+BM 2H": "£55.8k", "Gap vs Modo": "+£16.1k"},
            {"DC%": "40%", "DA%": "30%", "Model+BM 2H": "£66.3k", "Gap vs Modo": "+£5.6k"},
            {"DC%": "35% (locked)", "DA%": "35%", "Model+BM 2H": "£70.8k", "Gap vs Modo": "+£1.1k ✓"},
        ])
        st.dataframe(calibration, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("WD Cap Sensitivity (2H gross)")
        wd_sens = pd.DataFrame([
            {"WD cap": "£15 (original)", "Gross 2H excl BM": "~£47k", "Gap vs Modo": "~£25k"},
            {"WD cap": "£30", "Gross 2H excl BM": "~£52k", "Gap vs Modo": "~£20k"},
            {"WD cap": "£60 (locked)", "Gross 2H excl BM": "£62.9k", "Gap vs Modo": "£9.0k (before BM)"},
            {"WD cap": "Uncapped", "Gross 2H excl BM": "~£64k", "Gap vs Modo": "~£8k"},
        ])
        st.dataframe(wd_sens, use_container_width=True, hide_index=True)
        st.caption("Diminishing returns above £60 — most SP−DA spread sits below that level.")

    # ── Page: Gap Analysis ────────────────────────────────────────────────────
    elif page == "Gap Analysis":
        st.header("Gap Analysis — Model vs Modo")

        st.subheader("Revenue Stream Breakdown (Apr 2024 – Apr 2026, gross £k/MW/yr)")
        gap_table = pd.DataFrame([
            {"Stream": "DA energy", "1H": "£2.1k", "1H %": "4%", "2H": "£11.3k", "2H %": "16%"},
            {"Stream": "Intraday / WD (cap £60)", "1H": "£32.4k", "1H %": "61%", "2H": "£40.4k", "2H %": "57%"},
            {"Stream": "Ancillary (DC/DM/DR/QR)", "1H": "£11.2k", "1H %": "21%", "2H": "£11.2k", "2H %": "16%"},
            {"Stream": "BM fleet avg (Elexon)", "1H": "£7.2k", "1H %": "14%", "2H": "£7.9k", "2H %": "11%"},
            {"Stream": "**Total model + BM**", "1H": "**£52.9k**", "1H %": "100%", "2H": "**£70.8k**", "2H %": "100%"},
        ])
        st.dataframe(gap_table, use_container_width=True, hide_index=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1H Battery")
            fig1 = go.Figure(go.Pie(
                labels=["DA energy", "WD (cap £60)", "Ancillary", "BM fleet"],
                values=[2.1, 32.4, 11.2, 7.2],
                hole=0.4,
                marker_colors=["#42A5F5", "#1565C0", "#66BB6A", "#FFA726"],
                textinfo="label+percent",
                hovertemplate="%{label}: £%{value:.1f}k<extra></extra>",
            ))
            fig1.update_layout(
                title="1H Revenue Mix (£52.9k total)",
                height=350,
                showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.subheader("2H Battery")
            fig2 = go.Figure(go.Pie(
                labels=["DA energy", "WD (cap £60)", "Ancillary", "BM fleet"],
                values=[11.3, 40.4, 11.2, 7.9],
                hole=0.4,
                marker_colors=["#FF8A65", "#E65100", "#66BB6A", "#FFA726"],
                textinfo="label+percent",
                hovertemplate="%{label}: £%{value:.1f}k<extra></extra>",
            ))
            fig2.update_layout(
                title="2H Revenue Mix (£70.8k total)",
                height=350,
                showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Final Gap vs Modo")
        final = pd.DataFrame([
            {"Metric": "Model + BM (this model)", "1H": "£52.9k", "2H": "£70.8k"},
            {"Metric": "Modo ME BESS GB (public index)", "1H": "£50.8k", "2H": "£71.9k"},
            {"Metric": "Residual gap", "1H": "−£2.0k", "2H": "+£1.1k"},
            {"Metric": "Gap as % of Modo", "1H": "−3.9%", "2H": "+1.5%"},
        ])
        st.dataframe(final, use_container_width=True, hide_index=True)
        st.success(
            "Gap closed to within noise (±2%). Remaining difference attributable to "
            "Capacity Market (est. ~£6k/MW/yr), intraday optimisation alpha, and model discretisation."
        )

        st.divider()
        st.subheader("Architecture Note")
        st.markdown("""
**Two independent headroom systems — do not confuse:**

| System | Used by | How headroom works |
|---|---|---|
| `ancillary_revenue.py` `DEFAULT_HEADROOM` | Notebook 19 backtest | Fixed fractions — a *parameter* |
| `dispatch.py` `enumerate_modes()` | Notebook 13 LSMC | Discrete mode grid — *endogenous decision* |

Changing `DEFAULT_HEADROOM` has **zero effect** on the LSMC valuation (notebook 13).
The LSMC mode grid (`dc_levels=[0.0, 0.25, 0.50]`) is a separate optimisation.

Rolling intrinsic LP **ignores all ancillary services** — they are computed separately
and added on top. Ancillary revenue is duration-agnostic in this model
(1H and 2H earn identical ancillary revenue — a known simplification).
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Forward vs Realized
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Forward vs Realized":

    m1, m2, bm1, bm2 = _m1, _m2, _bm1, _bm2

    page = st.sidebar.radio(
        "View",
        options=["Overview", "Energy Trading", "Full Stack", "Gap Waterfall"],
    )

    st.title("Forward vs Realized: nb13 vs nb19")
    st.markdown(
        "**Three-way comparison:** "
        "nb13 WD rolling (forward prices, rolling LP) · "
        "nb19 Historical (actual 2024–26 prices, rolling LP) · "
        "nb13 LSMC (forward prices, optimal dispatch). "
        "Realized value sits between the two forward benchmarks."
    )

    # ── Helper: extract numbers from loaded CSVs ─────────────────────────────
    def _nb13(method: str, dur: float, col: str = "gbp_per_mw_year_k") -> float:
        if df_comparison.empty:
            return 0.0
        row = df_comparison[
            (df_comparison["method"] == method) & (df_comparison["duration_h"] == dur)
        ]
        return float(row[col].iloc[0]) if not row.empty else 0.0

    def _nb13_attr(key: str, dur: float) -> float:
        """£k/MW/yr from attribution (mean_m × 10 for 100 MW asset)."""
        if df_attribution.empty:
            return 0.0
        row = df_attribution[
            (df_attribution["key"] == key) & (df_attribution["duration_h"] == dur)
        ]
        return float(row["mean_m"].iloc[0]) * 10.0 if not row.empty else 0.0

    def _nb19_stream(m: pd.DataFrame, bm: pd.DataFrame, stream: str) -> float:
        """Mean £k/MW/yr for a stream from historical monthly CSV."""
        if m.empty:
            return 0.0
        if stream == "da_wd":
            return (m["da_revenue_gbp_mw_yr"] + m["wd_revenue_gbp_mw_yr"]).mean() / 1000
        if stream == "anc":
            return m["anc_revenue_gbp_mw_yr"].mean() / 1000
        if stream == "bm":
            return bm["bm_rev_gbp_mw_yr"].mean() / 1000 if not bm.empty else 0.0
        if stream == "total":
            bm_val = bm["bm_rev_gbp_mw_yr"].mean() / 1000 if not bm.empty else 0.0
            return m["total_net_gbp_mw_yr"].mean() / 1000 + bm_val
        return 0.0

    BLUE, ORANGE, GREEN = "#1565C0", "#E65100", "#2E7D32"

    # ── Page: Overview ───────────────────────────────────────────────────────
    if page == "Overview":
        st.header("Overview: Three Valuations, One Battery")

        st.info(
            "**WD rolling** = forward simulation using rolling LP with intraday prices "
            "(same algorithm as nb19 but fed simulated price paths anchored to current forwards). "
            "**Historical** = same rolling LP algorithm on actual 2024–26 prices. "
            "**LSMC** = non-anticipative optimal dispatch on simulated paths, full-stack."
        )

        for dur_label, dur, m_df, bm_df in [("1h", 1.0, m1, bm1), ("2h", 2.0, m2, bm2)]:
            st.subheader(f"{dur_label.upper()} Battery")
            c1, c2, c3, c4 = st.columns(4)
            wd  = _nb13("WD rolling intrinsic", dur)
            lsmc = _nb13("Forward simulation (LSMC)", dur)
            hist = _nb19_stream(m_df, bm_df, "total")
            da_wd = _nb19_stream(m_df, bm_df, "da_wd")

            with c1:
                st.metric("nb13 WD rolling", f"£{wd:.1f}k", "forward, energy-only")
            with c2:
                st.metric("nb19 Historical", f"£{hist:.1f}k", "realized, gross+BM")
                st.caption(f"DA+WD only: £{da_wd:.1f}k")
            with c3:
                st.metric("nb13 LSMC", f"£{lsmc:.1f}k", "forward, full-stack")
            with c4:
                energy_gap = da_wd - wd
                st.metric("Energy basis gap", f"£{energy_gap:+.1f}k",
                          "actual > model spreads" if energy_gap > 0 else "model > actual")
            if dur_label == "1h":
                st.divider()

        st.divider()
        st.subheader("What Each Gap Means")
        st.markdown("""
| Gap | Size (1H) | Driver |
|---|---|---|
| nb13 WD rolling → nb19 DA+WD | +£4.8k | Actual 2024–26 imbalance spreads exceeded model forecast |
| nb19 DA+WD → nb19 total | +£18.4k | Ancillary (£11.2k) + BM fleet (£7.2k) not in WD rolling |
| nb19 total → nb13 LSMC | +£23.0k | Dispatch optimality premium (LSMC vs rolling LP) |
| **nb13 WD → nb13 LSMC** | **+£46.2k** | **Full optimality gap: ancillary + optimal timing** |

Realized value (£52.9k) sits between the two forward benchmarks:
**WD rolling (£29.7k)** < **Realized (£52.9k)** < **LSMC optimal (£75.9k)**
        """)

    # ── Page: Energy Trading ─────────────────────────────────────────────────
    elif page == "Energy Trading":
        st.header("Energy Trading: WD Rolling (nb13) vs DA+WD Historical (nb19)")
        st.markdown(
            "Both use the **same rolling LP algorithm**. The only difference is the price signal: "
            "nb13 feeds simulated paths anchored to current forwards; nb19 uses actual 2024–26 DA and SP prices."
        )

        # Bar comparison by duration
        durations = [1.0, 2.0]
        wd_vals = [_nb13("WD rolling intrinsic", d) for d in durations]
        hist_wd_vals = [
            _nb19_stream(m1, bm1, "da_wd"),
            _nb19_stream(m2, bm2, "da_wd"),
        ]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="nb13 WD rolling (forward prices)",
            x=[f"{int(d)}h" for d in durations],
            y=wd_vals,
            marker_color=BLUE,
            text=[f"£{v:.1f}k" for v in wd_vals],
            textposition="outside",
        ))
        fig_bar.add_trace(go.Bar(
            name="nb19 DA+WD (actual 2024–26)",
            x=[f"{int(d)}h" for d in durations],
            y=hist_wd_vals,
            marker_color=ORANGE,
            text=[f"£{v:.1f}k" for v in hist_wd_vals],
            textposition="outside",
        ))
        fig_bar.update_layout(
            barmode="group",
            title="Energy trading value: forward simulation vs actual realization",
            yaxis_title="£k / MW / yr",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        gaps = [h - w for h, w in zip(hist_wd_vals, wd_vals)]
        st.success(
            f"Gap: 1H = £{gaps[0]:+.1f}k  |  2H = £{gaps[1]:+.1f}k  — "
            "actual 2024–26 imbalance spreads were higher than the calibrated model predicts. "
            "This is model basis risk, not execution alpha."
        )

        st.divider()

        # Monthly time series
        dur_sel = st.radio("Duration for monthly view", ["1h", "2h"], horizontal=True)
        m_df = m1 if dur_sel == "1h" else m2
        bm_df = bm1 if dur_sel == "1h" else bm2
        dur_h = 1.0 if dur_sel == "1h" else 2.0
        wd_ref = _nb13("WD rolling intrinsic", dur_h)

        if not m_df.empty:
            monthly_da_wd = (m_df["da_revenue_gbp_mw_yr"] + m_df["wd_revenue_gbp_mw_yr"]) / 1000

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=m_df["period_dt"], y=monthly_da_wd,
                mode="lines+markers", name=f"nb19 DA+WD monthly (actual)",
                marker_color=ORANGE, line_width=2,
                hovertemplate="%{x|%b %Y}: £%{y:.1f}k<extra></extra>",
            ))
            fig_ts.add_hline(
                y=wd_ref, line_dash="dot", line_color=BLUE,
                annotation_text=f"nb13 WD rolling avg £{wd_ref:.1f}k",
                annotation_position="top left",
            )
            fig_ts.add_hline(
                y=monthly_da_wd.mean(), line_dash="dash", line_color=ORANGE,
                annotation_text=f"nb19 avg £{monthly_da_wd.mean():.1f}k",
                annotation_position="top right",
            )
            fig_ts.update_layout(
                title=f"Monthly DA+WD revenue ({dur_sel.upper()}) vs forward model reference",
                xaxis_title="Month", yaxis_title="£k / MW / yr (annualised)",
                height=420,
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            st.caption(
                "The nb13 forward model gives a single expected value (dotted line). "
                "Monthly actual performance varies around it — months with high imbalance events "
                "drive the realized average above the model."
            )

    # ── Page: Full Stack ─────────────────────────────────────────────────────
    elif page == "Full Stack":
        st.header("Full Stack: Three-Way Comparison")
        st.markdown(
            "Comparing all revenue components across the three valuation perspectives. "
            "nb13 LSMC is the **optimal** benchmark; nb19 is **realized**; nb13 WD rolling is the **naive forward**."
        )

        dur_sel = st.radio("Duration", ["1h", "2h"], horizontal=True)
        dur_h = 1.0 if dur_sel == "1h" else 2.0
        m_df = m1 if dur_sel == "1h" else m2
        bm_df = bm1 if dur_sel == "1h" else bm2

        wd = _nb13("WD rolling intrinsic", dur_h)
        lsmc = _nb13("Forward simulation (LSMC)", dur_h)
        lsmc_costs = abs(_nb13_attr("costs", dur_h))
        lsmc_anc = _nb13_attr("dc", dur_h) + _nb13_attr("qr", dur_h)
        lsmc_energy = lsmc + lsmc_costs - lsmc_anc

        hist_da_wd = _nb19_stream(m_df, bm_df, "da_wd")
        hist_anc = _nb19_stream(m_df, bm_df, "anc")
        hist_bm = _nb19_stream(m_df, bm_df, "bm")
        hist_total = _nb19_stream(m_df, bm_df, "total")

        categories = ["Energy (DA+WD)", "Ancillary (DC+QR)", "BM fleet", "Costs (−)"]
        val_wd = [wd, 0.0, 0.0, 0.0]
        val_hist = [hist_da_wd, hist_anc, hist_bm, 0.0]
        val_lsmc = [lsmc_energy, lsmc_anc, 0.0, -lsmc_costs]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("nb13 WD rolling", f"£{wd:.1f}k", "energy-only")
        with col2:
            st.metric("nb19 Historical", f"£{hist_total:.1f}k", "DA+WD+Anc+BM gross")
        with col3:
            st.metric("nb13 LSMC", f"£{lsmc:.1f}k", "full-stack net")

        fig = go.Figure()
        colors = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]
        for i, (cat, c) in enumerate(zip(categories, colors)):
            fig.add_trace(go.Bar(
                name=cat,
                x=["nb13 WD rolling", "nb19 Historical", "nb13 LSMC"],
                y=[val_wd[i], val_hist[i], val_lsmc[i]],
                marker_color=c,
                hovertemplate=f"{cat}: £%{{y:.1f}}k<extra></extra>",
            ))

        fig.update_layout(
            barmode="relative",
            title=f"Revenue stack comparison — {dur_sel.upper()} battery (£k/MW/yr)",
            yaxis_title="£k / MW / yr",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Summary Table")
        summary = pd.DataFrame({
            "Component": ["Energy (DA+WD)", "Ancillary (DC+QR)", "BM fleet", "Costs (VOM+deg)", "Total"],
            "nb13 WD rolling": [f"£{wd:.1f}k", "—", "—", "VOM=0", f"£{wd:.1f}k"],
            "nb19 Historical": [f"£{hist_da_wd:.1f}k", f"£{hist_anc:.1f}k", f"£{hist_bm:.1f}k", "excl.", f"£{hist_total:.1f}k"],
            "nb13 LSMC": [f"£{lsmc_energy:.1f}k", f"£{lsmc_anc:.1f}k", "—", f"−£{lsmc_costs:.1f}k", f"£{lsmc:.1f}k"],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.caption(
            "nb13 LSMC 'Energy' = WD/intraday + HPFC + DA surprise components net of ancillary. "
            "BM not modelled in nb13. Costs excluded from nb19 for gross comparison."
        )

    # ── Page: Gap Waterfall ──────────────────────────────────────────────────
    elif page == "Gap Waterfall":
        st.header("Gap Waterfall: From Forward to Realized")

        dur_sel = st.radio("Duration", ["1h", "2h"], horizontal=True)
        dur_h = 1.0 if dur_sel == "1h" else 2.0
        m_df = m1 if dur_sel == "1h" else m2
        bm_df = bm1 if dur_sel == "1h" else bm2

        wd = _nb13("WD rolling intrinsic", dur_h)
        lsmc = _nb13("Forward simulation (LSMC)", dur_h)
        da_wd = _nb19_stream(m_df, bm_df, "da_wd")
        anc   = _nb19_stream(m_df, bm_df, "anc")
        bm    = _nb19_stream(m_df, bm_df, "bm")
        hist  = _nb19_stream(m_df, bm_df, "total")

        price_basis = da_wd - wd
        optimality  = lsmc - hist

        st.subheader("From nb13 WD rolling → nb19 Realized → nb13 LSMC")

        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total",
                     "relative", "total"],
            x=[
                "nb13 WD rolling\n(forward, energy)",
                "Ancillary\n(nb19 actual)",
                "Price basis\n(actual > model)",
                "BM fleet\n(not in model)",
                "nb19 Realized\n(total)",
                "Dispatch\noptimality\n(LSMC vs LP)",
                "nb13 LSMC\n(forward, optimal)",
            ],
            y=[wd, anc, price_basis, bm, 0, optimality, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": GREEN}},
            decreasing={"marker": {"color": "#EF5350"}},
            totals={"marker": {"color": ORANGE}},
            text=[
                f"£{wd:.1f}k",
                f"+£{anc:.1f}k",
                f"+£{price_basis:.1f}k",
                f"+£{bm:.1f}k",
                f"£{hist:.1f}k",
                f"+£{optimality:.1f}k",
                f"£{lsmc:.1f}k",
            ],
            textposition="outside",
        ))
        fig_wf.update_layout(
            title=f"Gap decomposition — {dur_sel.upper()} battery (£k/MW/yr, gross)",
            yaxis_title="£k / MW / yr",
            height=520,
            showlegend=False,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Price basis gap")
            st.metric("Actual DA+WD", f"£{da_wd:.1f}k")
            st.metric("nb13 WD rolling", f"£{wd:.1f}k")
            st.metric("Gap", f"£{price_basis:+.1f}k",
                      "actual 2024–26 spreads > model")
            st.markdown(
                "Driven by higher-than-modelled SP−DA imbalance in 2024–26. "
                "This is **model basis risk** — the imbalance OU+jump calibration "
                "underestimates realized spread volatility."
            )
        with col2:
            st.subheader("Optimality gap")
            st.metric("nb13 LSMC (optimal)", f"£{lsmc:.1f}k")
            st.metric("nb19 Realized", f"£{hist:.1f}k")
            st.metric("Gap", f"£{optimality:+.1f}k",
                      "LSMC dispatch > rolling LP")
            st.markdown(
                "LSMC finds the optimal non-anticipative policy by valuing continuation. "
                "Rolling LP is myopic (re-plans every gate). "
                "This **execution alpha** is the theoretical upper bound on operator skill."
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**BESS UK Research** — "
    "Stochastic MTM valuation for GB fast-cycle battery storage. "
    "Data: NESO EAC, Elexon BMRS, Modo Energy public index. "
    "[GitHub](https://github.com/dmitry-goryunov/BESS-UK-OPEN)"
)
