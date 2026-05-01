"""
Run a historical perfect-foresight BESS arbitrage benchmark for the selected duration.

Inputs:
    data/raw/elexon_da_prices.parquet
    data/raw/elexon_sp_prices.parquet

Outputs:
    data/processed/perfect_foresight_summary.json
    data/processed/perfect_foresight_da_dispatch.parquet
    data/processed/perfect_foresight_sp_dispatch.parquet
    data/processed/perfect_foresight_da_high_value_week.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ASSET  # noqa: E402
from src.optimisation.perfect_foresight import solve_perfect_foresight  # noqa: E402
from src.utils import find_project_root  # noqa: E402


PROJECT_ROOT = find_project_root(PROJECT_ROOT)


RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def prepare_price_frame(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    df = df.copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
    df["timestamp_index"] = np.arange(len(df))
    return df[
        ["settlement_date", "settlement_period", "timestamp_index", price_col]
    ].rename(columns={price_col: "price_gbp_mwh"})


def run_one(label: str, df: pd.DataFrame) -> dict:
    print(f"\nSolving {label} perfect-foresight LP on {len(df):,} half-hours...")
    result = solve_perfect_foresight(df["price_gbp_mwh"].to_numpy(), ASSET, dt_h=0.5)
    horizon_years = len(df) * 0.5 / 8760.0

    dispatch = df.copy()
    dispatch["charge_mw"] = result.charge_mw
    dispatch["discharge_mw"] = result.discharge_mw
    dispatch["net_export_mw"] = result.discharge_mw - result.charge_mw
    dispatch["soc_mwh"] = result.soc_mwh[:-1]
    dispatch["cashflow_gbp"] = result.cashflow_gbp
    dispatch.to_parquet(PROCESSED / f"perfect_foresight_{label.lower()}_dispatch.parquet", index=False)

    summary = {
        "rows": int(len(df)),
        "start_date": str(df["settlement_date"].min().date()),
        "end_date": str(df["settlement_date"].max().date()),
        "price_min_gbp_mwh": float(df["price_gbp_mwh"].min()),
        "price_mean_gbp_mwh": float(df["price_gbp_mwh"].mean()),
        "price_max_gbp_mwh": float(df["price_gbp_mwh"].max()),
        "value_gbp": float(result.objective_gbp),
        "value_gbp_per_mw": float(result.objective_gbp / ASSET["power_mw"]),
        "value_gbp_per_mw_year": float(result.objective_gbp / ASSET["power_mw"] / horizon_years),
        "horizon_years": float(horizon_years),
        "equivalent_cycles": float(result.cycles_equiv),
        "mean_daily_gbp": float(result.objective_gbp / df["settlement_date"].nunique()),
        "terminal_soc_mwh": float(result.soc_mwh[-1]),
    }
    print(json.dumps(summary, indent=2))
    return summary


def save_high_value_week_plot() -> None:
    dispatch = pd.read_parquet(PROCESSED / "perfect_foresight_da_dispatch.parquet")
    settlement_date = pd.to_datetime(dispatch["settlement_date"])
    weekly = dispatch.groupby(settlement_date.dt.to_period("W"))["cashflow_gbp"].sum()
    week = weekly.idxmax()
    plot = dispatch.loc[settlement_date.dt.to_period("W") == week].reset_index(drop=True)

    x = np.arange(len(plot)) / 48.0
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(x, plot["price_gbp_mwh"], color="black", lw=1.0, label="DA price")
    ax1.set_ylabel("GBP/MWh")

    ax2 = ax1.twinx()
    ax2.step(x, plot["net_export_mw"], where="post", color="tab:blue", alpha=0.75, label="net export MW")
    ax2.plot(x, plot["soc_mwh"], color="tab:green", lw=1.2, label="SoC MWh")
    ax2.set_ylabel("MW / MWh")

    ax1.set_xlabel("Days in selected week")
    ax1.set_title(f"Perfect Foresight DA Dispatch - highest value week {week}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(PROCESSED / "perfect_foresight_da_high_value_week.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    da = prepare_price_frame(pd.read_parquet(RAW / "elexon_da_prices.parquet"), "price_gbp_mwh")
    sp = prepare_price_frame(pd.read_parquet(RAW / "elexon_sp_prices.parquet"), "system_price")

    print(f"DA data: {len(da):,} rows, {da.settlement_date.min().date()} to {da.settlement_date.max().date()}")
    print(f"SP data: {len(sp):,} rows, {sp.settlement_date.min().date()} to {sp.settlement_date.max().date()}")
    print(f"Settlement periods: DA max={da.settlement_period.max()}, SP max={sp.settlement_period.max()}")
    print(
        "Asset: "
        f"{ASSET['power_mw']:.0f} MW / {ASSET['energy_mwh']:.0f} MWh, "
        f"RTE={ASSET['rte']:.1%}, SoC={ASSET['soc_min_frac']:.0%}-{ASSET['soc_max_frac']:.0%}"
    )

    results = {"DA": run_one("DA", da), "SP": run_one("SP", sp)}
    summary = {
        "asset": {
            k: float(ASSET[k])
            for k in [
                "power_mw",
                "energy_mwh",
                "rte",
                "eta_charge",
                "eta_discharge",
                "soc_min_frac",
                "soc_max_frac",
                "soc_init_frac",
                "vom_gbp_mwh",
            ]
            if k in ASSET
        },
        "assumptions": {
            "dt_h": 0.5,
            "perfect_foresight": True,
            "terminal_soc_equals_initial": True,
        },
        "results": results,
    }
    with open(PROCESSED / "perfect_foresight_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_high_value_week_plot()
    print("\nSaved outputs in:", PROCESSED)


if __name__ == "__main__":
    main()
