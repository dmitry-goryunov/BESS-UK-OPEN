#!/usr/bin/env python
"""Notebook 20 / Option B1: historical perfect-foresight LP on actual prices.

This is the first historical-LSMC bridge: value a 100 MW battery on actual
2024-2026 GB DA and system-price paths using the existing deterministic LP.
It is intentionally a perfect-foresight upper benchmark, not a realized trading
strategy.  Results are written in GBPk/MW/year for easy comparison to Modo.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as cfg
from src.config import ASSET, configure_asset_duration
from src.optimisation.perfect_foresight import solve_perfect_foresight

RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2024-04-01")
END = pd.Timestamp("2026-04-25")
DT_H = 0.5
WD_CAP = 60.0
DURATIONS_H = [1.0, 2.0]
COMPARISON_DURATIONS_H = [1.0, 2.0]

cfg.VALID_ASSET_DURATIONS_H = (1.0, 2.0, 3.0, 4.0)


def load_da() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "elexon_da_prices.parquet")
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df = df[df["settlement_date"].between(START, END)].copy()
    if "volume_mwh" in df.columns:
        df["weighted_price"] = df["price_gbp_mwh"] * df["volume_mwh"].clip(lower=0.0)
        grouped = df.groupby(["settlement_date", "settlement_period"], as_index=False).agg(
            volume_mwh=("volume_mwh", "sum"),
            weighted_price=("weighted_price", "sum"),
            price_mean=("price_gbp_mwh", "mean"),
        )
        grouped["price_gbp_mwh"] = np.where(
            grouped["volume_mwh"] > 0,
            grouped["weighted_price"] / grouped["volume_mwh"],
            grouped["price_mean"],
        )
        return grouped[["settlement_date", "settlement_period", "price_gbp_mwh"]]
    return df[["settlement_date", "settlement_period", "price_gbp_mwh"]]


def load_sp() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "elexon_sp_prices.parquet")
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df = df[df["settlement_date"].between(START, END)].copy()
    df = df.groupby(["settlement_date", "settlement_period"], as_index=False)["system_price"].mean()
    return df.rename(columns={"system_price": "sp_gbp_mwh"})


def aligned_prices() -> pd.DataFrame:
    da = load_da()
    sp = load_sp()
    frame = da.merge(sp, on=["settlement_date", "settlement_period"], how="inner")
    frame = frame.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
    frame["delta_sp_da"] = frame["sp_gbp_mwh"] - frame["price_gbp_mwh"]
    frame["wd60_gbp_mwh"] = frame["price_gbp_mwh"] + frame["delta_sp_da"].clip(-WD_CAP, WD_CAP)
    return frame


def run_case(price: np.ndarray, duration_h: float, stream: str, asset: dict) -> dict:
    result = solve_perfect_foresight(
        price,
        asset,
        dt_h=DT_H,
        terminal_soc_mwh=asset["soc_init_mwh"],
        vom_gbp_mwh=asset["vom_gbp_mwh"],
    )
    horizon_years = len(price) * DT_H / 8760.0
    return {
        "duration_h": duration_h,
        "stream": stream,
        "rows": int(len(price)),
        "horizon_years": horizon_years,
        "value_gbp": result.objective_gbp,
        "gbp_per_mw_year_k": result.objective_gbp / asset["power_mw"] / horizon_years / 1000.0,
        "cycles_equiv": result.cycles_equiv,
        "terminal_soc_mwh": float(result.soc_mwh[-1]),
        "price_mean_gbp_mwh": float(np.mean(price)),
        "price_min_gbp_mwh": float(np.min(price)),
        "price_max_gbp_mwh": float(np.max(price)),
    }


def load_nb13_reference(durations_h: list[float]) -> pd.DataFrame:
    """Load the per-duration nr13 method-comparison rows for the same durations."""
    rows: list[pd.DataFrame] = []
    for duration_h in durations_h:
        label = f"{duration_h:g}h"
        path = PROCESSED / f"phase4_method_comparison_{label}.csv"
        if not path.exists():
            print(f"WARNING: nr13 reference missing: {path}")
            continue
        frame = pd.read_csv(path)
        frame["duration_h"] = duration_h
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_nb13_all_methods(durations_h: list[float] | None = None) -> pd.DataFrame:
    """Load all six nr13 valuation methods, including the separate MODO row file."""
    if durations_h is None:
        durations_h = [1.0, 2.0, 3.0, 4.0]

    rows: list[pd.DataFrame] = []
    for duration_h in durations_h:
        label = f"{duration_h:g}h"
        path = PROCESSED / f"phase4_method_comparison_{label}.csv"
        if not path.exists():
            print(f"WARNING: nr13 reference missing: {path}")
            continue
        frame = pd.read_csv(path)
        frame["duration_h"] = duration_h
        rows.append(frame)

    modo_path = PROCESSED / "phase4_modo_wd_rows.json"
    if modo_path.exists():
        modo = pd.read_json(modo_path)
        modo = modo[modo["duration_h"].isin(durations_h)].copy()
        rows.append(modo)
    else:
        print(f"WARNING: MODO reference missing: {modo_path}")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_nb13_comparison(df_rows: pd.DataFrame, power_mw: float) -> pd.DataFrame:
    b1 = df_rows.copy()
    b1["b1_value_gbp_annualized_m"] = b1["gbp_per_mw_year_k"] * power_mw / 1000.0
    b1 = b1.rename(columns={"gbp_per_mw_year_k": "b1_gbp_per_mw_year_k"})

    nb13 = load_nb13_reference(sorted(b1["duration_h"].unique()))
    if nb13.empty:
        return pd.DataFrame()

    reference_methods = [
        "Perfect foresight (DA energy)",
        "DA rolling intrinsic",
        "WD rolling intrinsic",
        "Forward simulation (LSMC)",
    ]
    nb13 = nb13[nb13["method"].isin(reference_methods)].copy()
    nb13 = nb13.rename(
        columns={
            "method": "nb13_method",
            "value_gbp_annualized_m": "nb13_value_gbp_annualized_m",
            "gbp_per_mw_year_k": "nb13_gbp_per_mw_year_k",
        }
    )

    direct = b1[b1["stream"].eq("DA")].merge(
        nb13[nb13["nb13_method"].eq("Perfect foresight (DA energy)")],
        on="duration_h",
        how="inner",
    )
    direct["comparison_type"] = "direct_DA_perfect_foresight"

    context_pairs = [
        ("SP_perfect_foresight", "WD rolling intrinsic"),
        ("WD60_perfect_foresight", "WD rolling intrinsic"),
        ("DA", "Forward simulation (LSMC)"),
    ]
    context_rows: list[pd.DataFrame] = []
    for stream, method in context_pairs:
        left = b1[b1["stream"].eq(stream)]
        right = nb13[nb13["nb13_method"].eq(method)]
        merged = left.merge(right, on="duration_h", how="inner")
        merged["comparison_type"] = "context_not_strictly_like_for_like"
        context_rows.append(merged)

    comparison = pd.concat([direct, *context_rows], ignore_index=True)
    comparison["diff_gbp_annualized_m"] = (
        comparison["b1_value_gbp_annualized_m"] - comparison["nb13_value_gbp_annualized_m"]
    )
    comparison["diff_pct_vs_nb13"] = (
        comparison["diff_gbp_annualized_m"] / comparison["nb13_value_gbp_annualized_m"] * 100.0
    )

    cols = [
        "comparison_type",
        "duration_h",
        "stream",
        "nb13_method",
        "b1_gbp_per_mw_year_k",
        "b1_value_gbp_annualized_m",
        "nb13_gbp_per_mw_year_k",
        "nb13_value_gbp_annualized_m",
        "diff_gbp_annualized_m",
        "diff_pct_vs_nb13",
    ]
    return comparison[cols]


def build_stacked_valuation_table(df_rows: pd.DataFrame, power_mw: float) -> pd.DataFrame:
    """Stack the three nr20 streams and six nr13 methods for 1h and 2h only."""
    b1 = df_rows[df_rows["duration_h"].isin(COMPARISON_DURATIONS_H)].copy()
    b1["source"] = "nr20"
    b1["valuation"] = b1["stream"]
    b1["unit"] = "GBPm/year for 100 MW"
    b1["value_gbp_annualized_m"] = b1["gbp_per_mw_year_k"] * power_mw / 1000.0
    b1["market_basis"] = b1["stream"].map(
        {
            "DA": "Actual DA prices",
            "SP_perfect_foresight": "Actual system prices",
            "WD60_perfect_foresight": "Actual DA plus SP-DA spread capped at GBP60/MWh",
        }
    )
    b1["foresight"] = "Full historical perfect foresight"
    b1["policy_type"] = "Full-horizon deterministic LP"
    b1["scope"] = "Energy-only price path, VOM/degradation as configured"
    b1["strict_like_for_like_to"] = b1["stream"].map(
        {
            "DA": "nr13 Perfect foresight (DA energy)",
            "SP_perfect_foresight": "None; upper-bound context for imbalance value",
            "WD60_perfect_foresight": "None; context for WD/MODO volatility capture",
        }
    )
    b1["key_difference"] = b1["stream"].map(
        {
            "DA": "Actual historical DA path rather than HPFC-anchored simulated DA paths.",
            "SP_perfect_foresight": "Uses realized system prices with perfect foresight; not a tradable nr13 policy.",
            "WD60_perfect_foresight": "Uses realized imbalance spread capped at GBP60/MWh with perfect foresight.",
        }
    )

    nb20_cols = [
        "source",
        "valuation",
        "duration_h",
        "unit",
        "value_gbp_annualized_m",
        "gbp_per_mw_year_k",
        "market_basis",
        "foresight",
        "policy_type",
        "scope",
        "strict_like_for_like_to",
        "key_difference",
    ]

    nb13 = load_nb13_all_methods(COMPARISON_DURATIONS_H)
    if nb13.empty:
        return b1[nb20_cols].copy()

    nb13 = nb13.copy()
    nb13["source"] = "nr13"
    nb13["valuation"] = nb13["method"]
    nb13["unit"] = "GBPm/year for 100 MW"
    nb13["market_basis"] = nb13["method"].map(
        {
            "Initial hourly intrinsic": "HPFC hourly curve",
            "DA rolling intrinsic": "HPFC-anchored simulated DA paths",
            "WD rolling intrinsic": "HPFC-anchored simulated WD paths",
            "MODO style forward look": "HPFC-anchored simulated WD paths, GBP60/MWh WD cap",
            "Forward simulation (LSMC)": "HPFC-anchored simulated full-stack paths",
            "Perfect foresight (DA energy)": "HPFC-anchored simulated DA paths",
        }
    )
    nb13["foresight"] = nb13["method"].map(
        {
            "Initial hourly intrinsic": "Deterministic daily HPFC look-ahead",
            "DA rolling intrinsic": "Rolling finite-window look-ahead",
            "WD rolling intrinsic": "Rolling finite-window look-ahead",
            "MODO style forward look": "Rolling finite-window look-ahead",
            "Forward simulation (LSMC)": "Non-anticipative learned policy",
            "Perfect foresight (DA energy)": "Full simulated-path perfect foresight",
        }
    )
    nb13["policy_type"] = nb13["method"].map(
        {
            "Initial hourly intrinsic": "Daily intrinsic LP",
            "DA rolling intrinsic": "Rolling intrinsic LP",
            "WD rolling intrinsic": "Rolling intrinsic LP",
            "MODO style forward look": "Rolling intrinsic LP",
            "Forward simulation (LSMC)": "Forward policy simulation",
            "Perfect foresight (DA energy)": "Full-horizon deterministic LP",
        }
    )
    nb13["scope"] = nb13["method"].map(
        {
            "Initial hourly intrinsic": "Energy-only",
            "DA rolling intrinsic": "DA energy-only",
            "WD rolling intrinsic": "WD energy-only",
            "MODO style forward look": "WD energy-only with MODO-style cap",
            "Forward simulation (LSMC)": "Full stack: DA plus ancillary/BM modes",
            "Perfect foresight (DA energy)": "DA energy-only upper bound",
        }
    )
    nb13["strict_like_for_like_to"] = nb13["method"].map(
        {
            "Perfect foresight (DA energy)": "nr20 DA",
        }
    ).fillna("None; methodological context")
    nb13["key_difference"] = nb13["method"].map(
        {
            "Initial hourly intrinsic": "Deterministic forward curve, not historical actual prices.",
            "DA rolling intrinsic": "Rolling policy on simulated paths, not full historical perfect foresight.",
            "WD rolling intrinsic": "Rolling policy on simulated WD paths, not realized SP/WD perfect foresight.",
            "MODO style forward look": "Uses MODO-style WD cap on simulated paths; closest context to nr20 WD60 but not like-for-like.",
            "Forward simulation (LSMC)": "Non-anticipative full-stack policy, not energy-only perfect foresight.",
            "Perfect foresight (DA energy)": "Same LP style as nr20 DA, but on HPFC-anchored simulated DA paths.",
        }
    )

    stacked = pd.concat([b1[nb20_cols], nb13[nb20_cols]], ignore_index=True)
    order_valuation = {
        "DA": 0,
        "SP_perfect_foresight": 1,
        "WD60_perfect_foresight": 2,
        "Initial hourly intrinsic": 3,
        "DA rolling intrinsic": 4,
        "WD rolling intrinsic": 5,
        "MODO style forward look": 6,
        "Forward simulation (LSMC)": 7,
        "Perfect foresight (DA energy)": 8,
    }
    stacked["_valuation_order"] = stacked["valuation"].map(order_valuation).fillna(99)
    stacked = stacked.sort_values(["duration_h", "_valuation_order", "source"])
    return stacked.drop(columns=["_valuation_order"]).reset_index(drop=True)


def main() -> None:
    prices = aligned_prices()
    price_map = {
        "DA": prices["price_gbp_mwh"].to_numpy(dtype=float),
        "SP_perfect_foresight": prices["sp_gbp_mwh"].to_numpy(dtype=float),
        "WD60_perfect_foresight": prices["wd60_gbp_mwh"].to_numpy(dtype=float),
    }

    rows: list[dict] = []
    for duration_h in DURATIONS_H:
        asset = copy.deepcopy(ASSET)
        configure_asset_duration(asset, duration_h)
        for stream, arr in price_map.items():
            print(f"Solving {stream} {duration_h:g}h on {len(arr):,} HH...")
            rows.append(run_case(arr, duration_h, stream, asset))

    summary = {
        "as_of": "2026-05-17",
        "method": "Option B1 historical perfect-foresight LP",
        "start_date": str(prices["settlement_date"].min().date()),
        "end_date": str(prices["settlement_date"].max().date()),
        "wd_cap_gbp_mwh": WD_CAP,
        "dt_h": DT_H,
        "rows": rows,
    }

    out_json = PROCESSED / "historical_lsmc_b1_summary.json"
    out_csv = PROCESSED / "historical_lsmc_b1_summary.csv"
    out_prices = PROCESSED / "historical_lsmc_b1_prices.parquet"
    out_png = PROCESSED / "historical_lsmc_b1_summary.png"
    out_cmp_csv = PROCESSED / "historical_lsmc_b1_vs_nb13.csv"
    out_cmp_png = PROCESSED / "historical_lsmc_b1_vs_nb13.png"
    out_stack_csv = PROCESSED / "historical_lsmc_b1_nr13_stacked_table.csv"
    out_stack_wide_csv = PROCESSED / "historical_lsmc_b1_nr13_stacked_wide.csv"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df_rows = pd.DataFrame(rows)
    power_mw = float(ASSET["power_mw"])
    df_rows["value_gbp_annualized_m"] = df_rows["gbp_per_mw_year_k"] * power_mw / 1000.0
    df_rows.to_csv(out_csv, index=False)
    prices.to_parquet(out_prices, index=False)

    print("\nHistorical B1 summary (GBPk/MW/year):")
    pivot = df_rows.pivot(index="stream", columns="duration_h", values="gbp_per_mw_year_k")
    print(pivot.round(1).to_string())

    ax = pivot.T.plot(kind="bar", figsize=(8, 4.5))
    ax.set_title("Historical B1 perfect-foresight LP on actual prices")
    ax.set_xlabel("Battery duration (hours)")
    ax.set_ylabel("GBPk/MW/year")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()

    print(f"\nConverted B1 summary for {power_mw:g} MW asset (GBPm/year):")
    converted = df_rows.pivot(index="stream", columns="duration_h", values="value_gbp_annualized_m")
    print(converted.round(2).to_string())

    comparison = build_nb13_comparison(df_rows, power_mw)
    if not comparison.empty:
        comparison.to_csv(out_cmp_csv, index=False)
        print("\nB1 vs nr13 comparison (GBPm/year):")
        display_cols = [
            "comparison_type",
            "duration_h",
            "stream",
            "nb13_method",
            "b1_value_gbp_annualized_m",
            "nb13_value_gbp_annualized_m",
            "diff_gbp_annualized_m",
            "diff_pct_vs_nb13",
        ]
        print(comparison[display_cols].round(2).to_string(index=False))

        direct = comparison[comparison["comparison_type"].eq("direct_DA_perfect_foresight")]
        if not direct.empty:
            plot = direct.set_index("duration_h")[
                ["b1_value_gbp_annualized_m", "nb13_value_gbp_annualized_m"]
            ]
            plot = plot.rename(
                columns={
                    "b1_value_gbp_annualized_m": "nr20 actual DA PF",
                    "nb13_value_gbp_annualized_m": "nr13 simulated DA PF",
                }
            )
            ax = plot.plot(kind="bar", figsize=(7.5, 4.2))
            ax.set_title("nr20 vs nr13: DA perfect-foresight benchmark")
            ax.set_xlabel("Battery duration (hours)")
            ax.set_ylabel("GBPm/year for 100 MW asset")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(title="", fontsize=8)
            plt.tight_layout()
            plt.savefig(out_cmp_png, dpi=140, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out_cmp_png}")

    stacked = build_stacked_valuation_table(df_rows, power_mw)
    stacked.to_csv(out_stack_csv, index=False)
    stacked_wide = (
        stacked.pivot_table(
            index=[
                "source",
                "valuation",
                "unit",
                "market_basis",
                "foresight",
                "policy_type",
                "scope",
                "strict_like_for_like_to",
                "key_difference",
            ],
            columns="duration_h",
            values="value_gbp_annualized_m",
            aggfunc="first",
        )
        .rename(columns=lambda c: f"value_{c:g}h_gbp_m_per_year")
        .reset_index()
    )
    stacked_wide.to_csv(out_stack_wide_csv, index=False)

    print("\nStacked nr20/nr13 valuation table (GBPm/year for 100 MW):")
    show_cols = ["source", "valuation", "duration_h", "value_gbp_annualized_m", "strict_like_for_like_to"]
    print(stacked[show_cols].round(2).to_string(index=False))

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_prices}")
    print(f"Saved: {out_png}")
    if comparison.empty:
        print("Skipped nr13 comparison outputs because no nr13 reference rows were found.")
    else:
        print(f"Saved: {out_cmp_csv}")
    print(f"Saved: {out_stack_csv}")
    print(f"Saved: {out_stack_wide_csv}")


if __name__ == "__main__":
    main()
