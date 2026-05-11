"""Re-run index with DC=40%, DA=30%, regenerate comparison chart."""
import copy, base64, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from src.backtest.historical_index import load_data, build_index, monthly_index
from src.backtest.bm_revenue import build_bm_index
from src.config import ASSET, configure_asset_duration

PROCESSED  = Path("data/processed")
PLOT_START = pd.Timestamp("2024-01-01")
HEADROOM   = {"DC": 0.35, "DM": 0.10, "DR": 0.05, "QR": 0.15, "BR": 0.0}  # DA=35%

da, sp, anc = load_data(start="2024-04-01", end="2026-04-25")

monthly = {}
for dur in [1.0, 2.0]:
    tag = f"{int(dur)}h"
    asset = copy.deepcopy(ASSET)
    configure_asset_duration(asset, dur)
    df = build_index(da, sp, anc, duration_h=dur, asset_cfg=asset,
                     headroom=HEADROOM, include_costs=False, verbose=False)
    m = monthly_index(df, asset["power_mw"])
    m["period_dt"] = m["year_month"].dt.to_timestamp()
    m["net_k"] = m["total_net_gbp_mw_yr"] / 1000
    monthly[tag] = m
    P = asset["power_mw"]; yrs = len(df)/365.25
    print(f"{tag.upper()}  DA=£{df['da_revenue'].sum()/P/yrs/1000:.1f}k  "
          f"WD=£{df['wd_revenue'].sum()/P/yrs/1000:.1f}k  "
          f"Anc=£{df['anc_revenue'].sum()/P/yrs/1000:.1f}k  "
          f"Gross=£{df['total_net'].sum()/P/yrs/1000:.1f}k/MW/yr")

bm_idx = build_bm_index()
for tag, bm in bm_idx.items():
    bm["period_dt"] = bm["year_month"].dt.to_timestamp()
    bm["bm_k"] = bm["bm_rev_gbp_mw_yr"] / 1000

def stack(m, bm):
    s = m[["period_dt","net_k"]].merge(bm[["period_dt","bm_k"]], on="period_dt", how="left")
    s["stack_k"] = s["net_k"] + s["bm_k"].fillna(0)
    return s

s1 = stack(monthly["1h"], bm_idx["1h"])
s2 = stack(monthly["2h"], bm_idx["2h"])

def _load_modo(fname, col):
    df = pd.read_csv(fname, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date", col: "modo_k"})
    df["modo_k"] = pd.to_numeric(df["modo_k"], errors="coerce") / 1000
    return df.dropna(subset=["modo_k"]).sort_values("date").reset_index(drop=True)

modo1h = _load_modo("MODO 1H.csv", "ME BESS GB (1H)")
modo2h = _load_modo("MODO 2H.csv", "ME BESS GB (2H)")
modo1h_plot = modo1h[modo1h["date"] >= PLOT_START]
modo2h_plot = modo2h[modo2h["date"] >= PLOT_START]
x_end = max(modo1h_plot["date"].max(), modo2h_plot["date"].max()) + pd.Timedelta(days=30)
model_start = monthly["1h"]["period_dt"].min()

BLUE, ORANGE = "#1565C0", "#E65100"
fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 2]})
fig.suptitle("GB BESS Gross Revenue — DC=35% DA=35% vs Modo Optimal (Jan 2024–)",
             fontsize=13, fontweight="bold")
ax1, ax2 = axes

ax1.plot(modo2h_plot["date"], modo2h_plot["modo_k"], color=ORANGE, linewidth=2.4,
         label=f"Modo 2H  (avg £{modo2h_plot['modo_k'].mean():.0f}k/MW/yr)")
ax1.plot(modo1h_plot["date"], modo1h_plot["modo_k"], color=BLUE, linewidth=2.4,
         label=f"Modo 1H  (avg £{modo1h_plot['modo_k'].mean():.0f}k/MW/yr)")
ax1.fill_between(s2["period_dt"], s2["net_k"], s2["stack_k"], step="mid", alpha=0.50, color="#FFA726", label="Actual BM 2H")
ax1.step(s2["period_dt"], s2["net_k"],    where="mid", color=ORANGE, linewidth=1.6, linestyle="--", alpha=0.65,
         label=f"Model 2H excl BM  (£{monthly['2h']['net_k'].mean():.0f}k)")
ax1.step(s2["period_dt"], s2["stack_k"], where="mid", color=ORANGE, linewidth=2.2,
         label=f"Model 2H+BM  (£{s2['stack_k'].mean():.0f}k)")
ax1.fill_between(s1["period_dt"], s1["net_k"], s1["stack_k"], step="mid", alpha=0.40, color="#42A5F5", label="Actual BM 1H")
ax1.step(s1["period_dt"], s1["net_k"],    where="mid", color=BLUE, linewidth=1.6, linestyle="--", alpha=0.65,
         label=f"Model 1H excl BM  (£{monthly['1h']['net_k'].mean():.0f}k)")
ax1.step(s1["period_dt"], s1["stack_k"], where="mid", color=BLUE, linewidth=2.2,
         label=f"Model 1H+BM  (£{s1['stack_k'].mean():.0f}k)")
ax1.axvline(model_start, color="grey", linewidth=0.8, linestyle=":", zorder=0)
ax1.set_ylabel("£k / MW / yr (annualised)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:.0f}k"))
ax1.legend(fontsize=8, loc="upper right", framealpha=0.95, ncol=2)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.set_title("Gross: DA+WD(cap=60)+Ancillary(DC=35%,DA=35%) + BM  vs  Modo optimal", fontsize=10)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1,13,2)))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax1.set_xlim(PLOT_START - pd.Timedelta(days=15), x_end)

ax2.step(bm_idx["2h"]["period_dt"], bm_idx["2h"]["bm_k"], where="mid", color=ORANGE, linewidth=2.0,
         label=f"BM 2H  (avg £{bm_idx['2h']['bm_k'].mean():.1f}k/MW/yr)")
ax2.step(bm_idx["1h"]["period_dt"], bm_idx["1h"]["bm_k"], where="mid", color=BLUE, linewidth=2.0,
         label=f"BM 1H  (avg £{bm_idx['1h']['bm_k'].mean():.1f}k/MW/yr)")
ax2.set_ylabel("BM £k/MW/yr")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:.0f}k"))
ax2.legend(fontsize=9)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.set_title("BM component: Elexon settlement volumes × system price (fleet-normalised)", fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1,13,2)))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax2.set_xlim(PLOT_START - pd.Timedelta(days=15), x_end)

plt.tight_layout()
out = PROCESSED / "historical_index_with_bm.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {out}")

ov = model_start
for label, s, modo in [("2H", s2, modo2h_plot), ("1H", s1, modo1h_plot)]:
    gap = modo[modo.date>=ov]["modo_k"].mean() - s[s.period_dt>=ov]["stack_k"].mean()
    print(f"  {label}: Model+BM £{s[s.period_dt>=ov]['stack_k'].mean():.1f}k  Modo £{modo[modo.date>=ov]['modo_k'].mean():.1f}k  Gap £{gap:.1f}k")
