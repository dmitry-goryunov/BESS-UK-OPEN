"""Standalone Step 7: model + BM vs Modo comparison chart."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
PLOT_START   = pd.Timestamp("2024-01-01")

# ── Load model monthly ────────────────────────────────────────────────────────
m1 = pd.read_csv(PROCESSED / "historical_index_1h_monthly.csv")
m2 = pd.read_csv(PROCESSED / "historical_index_2h_monthly.csv")
for df in (m1, m2):
    df["period_dt"] = pd.to_datetime(df["year_month"])

m1["net_k"] = m1["total_net_gbp_mw_yr"] / 1000
m2["net_k"] = m2["total_net_gbp_mw_yr"] / 1000
model_start = m1["period_dt"].min()

# ── Load BM index ─────────────────────────────────────────────────────────────
bm1 = pd.read_csv(PROCESSED / "bm_index_1h.csv")
bm2 = pd.read_csv(PROCESSED / "bm_index_2h.csv")
for df in (bm1, bm2):
    df["period_dt"] = pd.to_datetime(df["year_month"])
    df["bm_k"] = df["bm_rev_gbp_mw_yr"] / 1000

# ── Load Modo CSVs ────────────────────────────────────────────────────────────
def _load_modo(fname, col):
    df = pd.read_csv(PROJECT_ROOT / fname, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date", col: "modo_k"})
    df["modo_k"] = pd.to_numeric(df["modo_k"], errors="coerce") / 1000
    return df.dropna(subset=["modo_k"]).sort_values("date").reset_index(drop=True)

modo1h = _load_modo("MODO 1H.csv", "ME BESS GB (1H)")
modo2h = _load_modo("MODO 2H.csv", "ME BESS GB (2H)")
modo1h_plot = modo1h[modo1h["date"] >= PLOT_START]
modo2h_plot = modo2h[modo2h["date"] >= PLOT_START]
x_end = max(modo1h_plot["date"].max(), modo2h_plot["date"].max()) + pd.Timedelta(days=30)

# ── Stacked model + BM ────────────────────────────────────────────────────────
def stack(m, bm):
    s = m[["period_dt", "net_k"]].merge(bm[["period_dt", "bm_k"]], on="period_dt", how="left")
    s["stack_k"] = s["net_k"] + s["bm_k"].fillna(0)
    return s

s1 = stack(m1, bm1)
s2 = stack(m2, bm2)

BLUE   = "#1565C0"
ORANGE = "#E65100"

fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 2]})
fig.suptitle("GB BESS Gross Revenue — Model + Actual BM vs Modo Optimal (Jan 2024–)",
             fontsize=13, fontweight="bold")
ax1, ax2 = axes

# ── Panel 1: all-in comparison ────────────────────────────────────────────────
ax1.plot(modo2h_plot["date"], modo2h_plot["modo_k"], color=ORANGE, linewidth=2.4,
         label=f"Modo 2H optimal  (avg £{modo2h_plot['modo_k'].mean():.0f}k/MW/yr)")
ax1.plot(modo1h_plot["date"], modo1h_plot["modo_k"], color=BLUE, linewidth=2.4,
         label=f"Modo 1H optimal  (avg £{modo1h_plot['modo_k'].mean():.0f}k/MW/yr)")

ax1.fill_between(s2["period_dt"], s2["net_k"], s2["stack_k"],
                 step="mid", alpha=0.50, color="#FFA726", label="Actual BM 2H (fleet avg)")
ax1.step(s2["period_dt"], s2["net_k"], where="mid", color=ORANGE,
         linewidth=1.6, linestyle="--", alpha=0.65,
         label=f"Model 2H excl. BM  (avg £{m2['net_k'].mean():.0f}k)")
ax1.step(s2["period_dt"], s2["stack_k"], where="mid", color=ORANGE, linewidth=2.2,
         label=f"Model 2H + BM  (avg £{s2['stack_k'].mean():.0f}k)")

ax1.fill_between(s1["period_dt"], s1["net_k"], s1["stack_k"],
                 step="mid", alpha=0.40, color="#42A5F5", label="Actual BM 1H (fleet avg)")
ax1.step(s1["period_dt"], s1["net_k"], where="mid", color=BLUE,
         linewidth=1.6, linestyle="--", alpha=0.65,
         label=f"Model 1H excl. BM  (avg £{m1['net_k'].mean():.0f}k)")
ax1.step(s1["period_dt"], s1["stack_k"], where="mid", color=BLUE, linewidth=2.2,
         label=f"Model 1H + BM  (avg £{s1['stack_k'].mean():.0f}k)")

ax1.axvline(model_start, color="grey", linewidth=0.8, linestyle=":", zorder=0)
ax1.set_ylabel("£k / MW / yr (annualised)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:.0f}k"))
ax1.legend(fontsize=8, loc="upper right", framealpha=0.95, ncol=2)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.set_title("Gross revenue: Model (DA+WD+Ancillary, excl. VOM/deg/fees) + Actual fleet BM  vs  Modo optimal", fontsize=10)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, 2)))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax1.set_xlim(PLOT_START - pd.Timedelta(days=15), x_end)

# ── Panel 2: BM component alone ───────────────────────────────────────────────
ax2.step(bm2["period_dt"], bm2["bm_k"], where="mid", color=ORANGE, linewidth=2.0,
         label=f"BM 2H  (avg £{bm2['bm_k'].mean():.1f}k/MW/yr)")
ax2.step(bm1["period_dt"], bm1["bm_k"], where="mid", color=BLUE,   linewidth=2.0,
         label=f"BM 1H  (avg £{bm1['bm_k'].mean():.1f}k/MW/yr)")
ax2.set_ylabel("BM revenue £k/MW/yr")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:.0f}k"))
ax2.legend(fontsize=9)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.set_title("BM component: actual Elexon settlement volumes × system price  (fleet normalised by nameplate MW)", fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, 2)))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax2.set_xlim(PLOT_START - pd.Timedelta(days=15), x_end)

plt.tight_layout()
out = PROCESSED / "historical_index_with_bm.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {out}")

# ── Gap table ─────────────────────────────────────────────────────────────────
ov_start = model_start
m1o  = m1[m1.period_dt >= ov_start]
m2o  = m2[m2.period_dt >= ov_start]
s1o  = s1[s1.period_dt >= ov_start]
s2o  = s2[s2.period_dt >= ov_start]
bm1o = bm1[bm1.period_dt >= ov_start]
bm2o = bm2[bm2.period_dt >= ov_start]
md1o = modo1h_plot[modo1h_plot.date >= ov_start]
md2o = modo2h_plot[modo2h_plot.date >= ov_start]

print(f"\n{'Metric':<42}  {'1H':>8}  {'2H':>8}")
print("-" * 62)
print(f"{'Modo optimal (all-in)':<42}  £{md1o['modo_k'].mean():>5.1f}k  £{md2o['modo_k'].mean():>5.1f}k")
print(f"{'Model DA+WD+Ancillary':<42}  £{m1o['net_k'].mean():>5.1f}k  £{m2o['net_k'].mean():>5.1f}k")
print(f"{'  + Actual fleet BM (Elexon)':<42}  £{bm1o['bm_k'].mean():>5.1f}k  £{bm2o['bm_k'].mean():>5.1f}k")
print(f"{'Model + BM subtotal':<42}  £{s1o['stack_k'].mean():>5.1f}k  £{s2o['stack_k'].mean():>5.1f}k")
gap1 = md1o["modo_k"].mean() - s1o["stack_k"].mean()
gap2 = md2o["modo_k"].mean() - s2o["stack_k"].mean()
print(f"{'Residual gap (CM + ID alpha + model)':<42}  £{gap1:>5.1f}k  £{gap2:>5.1f}k")
