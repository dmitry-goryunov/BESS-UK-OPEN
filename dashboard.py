"""
BESS Valuation Dashboard
Run: streamlit run dashboard.py   (from bess_project/)
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BESS Valuation",
    page_icon="⚡",
    layout="wide",
)

DASH_FILE = Path("data/processed/dashboard.json")

# ── load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load(path: Path) -> dict:
    return json.loads(path.read_text())


if not DASH_FILE.exists():
    st.error(f"dashboard.json not found at {DASH_FILE}. Run notebook 11 first.")
    st.stop()

data = load(DASH_FILE)
available = sorted(data.keys(), key=lambda x: float(x.replace("h", "")))

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("BESS Dashboard")
    duration = st.selectbox("Asset duration", available, index=available.index("2h") if "2h" in available else 0)
    d = data[duration]

    st.divider()
    st.caption(f"Curve date: **{d['as_of_date']}**")
    st.caption(f"Generated : **{d['generated_at']}**")
    st.caption(f"FX: 1 GBP = **{d['fx_gbp_eur']} EUR**")
    st.caption(f"Decay: **{d['decay_pct_yr']:.1f}%/yr** market saturation")
    st.caption(f"LSMC paths: **{d['total_value']['n_paths']:,}** over **{d['total_value']['horizon_years']:.2f}yr**")

    st.divider()
    st.caption(
        "**Total value** = LSMC stochastic MTM (DA + ID + ancillary + imbalance), annualised.\n\n"
        "**Intrinsic** = DA-only perfect-foresight LP dispatch on hourly forward curve.\n\n"
        "\\* = flat-extrapolated beyond forward curve."
    )

# ── header ─────────────────────────────────────────────────────────────────────
a = d["asset"]
st.title(f"BESS Valuation — {duration} ({a['power_mw']:.0f} MW / {a['energy_mwh']:.0f} MWh)")

# ── KPI metrics ────────────────────────────────────────────────────────────────
tv   = d["total_value"]
intr = d["intrinsic"]
y1   = intr[0]
avg_cycles = sum(r["cycles"] for r in intr) // len(intr)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Total value (mean)",
    f"{tv['mean']:.0f} k€/MW/yr",
    help="LSMC MTM mean across paths, annualised",
)
col2.metric(
    "Total value P5 / P95",
    f"{tv['p5']:.0f} / {tv['p95']:.0f}",
    help="5th and 95th percentile across LSMC paths",
)
col3.metric(
    "Intrinsic Y1 gross",
    f"{y1['gross']:.0f} k€/MW/yr",
    help="DA-only perfect-foresight LP, Year 1",
)
col4.metric(
    "Intrinsic / Total value",
    f"{y1['gross'] / tv['mean'] * 100:.0f}%",
    help="Intrinsic Y1 gross as % of LSMC total value mean",
)
col5.metric(
    "Avg cycles / yr",
    f"{avg_cycles}",
    help=f"Average equivalent full cycles per year (Y1–Y{len(intr)})",
)

st.divider()

# ── main chart: intrinsic bars + LSMC band ────────────────────────────────────
years  = [r["year"] for r in intr]
gross  = [r["gross"] for r in intr]
marker_colors = [
    "steelblue" if r["in_fwd"] else "slategray"
    for r in intr
]

fig = go.Figure()

# P5–P95 band (filled area across all years)
fig.add_trace(go.Scatter(
    x=years + years[::-1],
    y=[tv["p95"]] * len(years) + [tv["p5"]] * len(years),
    fill="toself",
    fillcolor="rgba(255,140,0,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    name="Total value P5–P95",
    showlegend=True,
))

# LSMC mean line
fig.add_trace(go.Scatter(
    x=years,
    y=[tv["mean"]] * len(years),
    mode="lines",
    line=dict(color="darkorange", width=2, dash="dash"),
    name=f"Total value mean ({tv['mean']:.0f} k€/MW/yr)",
))

# P50 line
fig.add_trace(go.Scatter(
    x=years,
    y=[tv["p50"]] * len(years),
    mode="lines",
    line=dict(color="darkorange", width=1, dash="dot"),
    name=f"Total value P50 ({tv['p50']:.0f} k€/MW/yr)",
))

# Intrinsic gross bars
fig.add_trace(go.Bar(
    x=years,
    y=gross,
    name="Intrinsic gross",
    marker_color=marker_colors,
    text=[f"{v:.0f}" for v in gross],
    textposition="outside",
    hovertemplate="%{x}: %{y:.1f} k€/MW/yr<extra>Intrinsic gross</extra>",
))

fig.update_layout(
    title=dict(
        text="Intrinsic gross vs LSMC total value  (k€/MW/yr, grey = extrapolated)",
        font=dict(size=15),
    ),
    yaxis_title="k€/MW/yr",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="white",
    height=420,
    margin=dict(t=80, b=40),
)
fig.update_yaxes(gridcolor="rgba(0,0,0,0.07)", zeroline=True, zerolinecolor="black")

st.plotly_chart(fig, use_container_width=True)

# ── year-by-year table ─────────────────────────────────────────────────────────
st.subheader("Year-by-year intrinsic breakdown  (k€/MW/yr)")

rows = []
for r in intr:
    rows.append({
        "Year":    r["year"] + (" *" if not r["in_fwd"] else ""),
        "Gross":   r["gross"],
        "VOM":     r["vom"],
        "Opt fee": r["optim"],
        "FOM":     r["fom"],
        "Cycles":  r["cycles"],
    })

df_table = pd.DataFrame(rows)

# Totals row
df_table.loc[len(df_table)] = {
    "Year":    "Total",
    "Gross":   round(df_table["Gross"].sum(), 1),
    "VOM":     round(df_table["VOM"].sum(), 1),
    "Opt fee": round(df_table["Opt fee"].sum(), 1),
    "FOM":     round(df_table["FOM"].sum(), 1),
    "Cycles":  int(df_table["Cycles"].sum()),
}

st.dataframe(
    df_table.style
    .format({"Gross": "{:.1f}", "VOM": "{:.1f}", "Opt fee": "{:.1f}", "FOM": "{:.1f}", "Cycles": "{:.0f}"})
    .set_properties(**{"text-align": "right"})
    .set_properties(subset=["Year"], **{"text-align": "left"})
    .apply(lambda s: ["font-weight: bold"] * len(s) if s.name == len(df_table) - 1 else [""] * len(s), axis=1),
    use_container_width=True,
    hide_index=True,
)

st.caption("\\* flat-extrapolated beyond forward curve  |  Gross = DA spread capture before any costs  |  FX: 1 GBP = " + str(d["fx_gbp_eur"]) + " EUR")

# ── asset info ─────────────────────────────────────────────────────────────────
with st.expander("Asset & run details"):
    col1, col2 = st.columns(2)
    with col1:
        st.json({
            "power_mw":    a["power_mw"],
            "energy_mwh":  a["energy_mwh"],
            "duration_h":  a["duration_h"],
            "capex_gbp_m": a["capex_gbp_m"],
        })
    with col2:
        st.json({
            "lsmc_n_paths":       tv["n_paths"],
            "lsmc_horizon_years": tv["horizon_years"],
            "decay_pct_yr":       d["decay_pct_yr"],
            "intrinsic_horizon":  f"{d['horizon_years']}yr",
            "fx_gbp_eur":         d["fx_gbp_eur"],
        })
