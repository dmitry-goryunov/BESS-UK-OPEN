"""
historical_index.py — Realized GB BESS revenue index from historical market data.

Builds a daily revenue time series for a standardized 100 MW GB battery
(1h or 2h duration) using actual DA prices, SP prices, and NESO ancillary
clearing data. Comparable to the Modo Energy GB BESS 1H / 2H public indices.

Revenue streams
---------------
  DA energy arbitrage  : LP on actual DA prices, rolling EFA-gate re-optimisation
  Within-day (WD) uplift: clip(SP - DA, -wd_cap, +wd_cap) applied at each gate
  DC ancillary         : actual clearing price × headroom held × 4h per block
  DM ancillary         : same
  DR ancillary         : same (sustain 1h → primary duration scaling mechanism)
  QR ancillary         : same (sustain 0.5h, available from Dec 2024)
  BR ancillary         : same (optional, typically low price)

Dispatch methodology
--------------------
  DA schedule: LP maximises revenue given the full next-day DA strip, solved
               at each EFA gate (every 8 HH = 4h) with 48 HH look-ahead.
               Uses solve_daily_lp from rolling_intrinsic.py.

  WD update  : at each gate, the LP also sees the capped SP-DA basis for the
               committed gate window. Cashflow settled at DA + clipped basis.

  Ancillary  : computed independently from the DA LP using actual NESO clearing
               prices. Headroom reduces effective power available for DA.

  Co-optimisation: ancillary power fraction is fixed; DA dispatch uses
               P_bar * (1 - total_headroom_fraction). Energy headroom constraint
               is enforced via the SoC trajectory from the LP.

Usage
-----
  from src.backtest.historical_index import build_index, load_data
  da, sp, anc = load_data()
  df = build_index(da, sp, anc, asset_cfg=ASSET_1H)
  df.to_csv('data/processed/historical_index_1h.csv')
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.ancillary_revenue import (
    DEFAULT_HEADROOM,
    SUSTAIN_H,
    compute_daily_ancillary,
    hh_to_efa,
)


def _ancillary_energy_floor(
    headroom: dict[str, float],
    P_bar: float,
    eta_d: float,
) -> float:
    """
    Additional energy floor (MWh) the battery must hold above E_min so that
    all upward ancillary services can be simultaneously sustained.

    Each product requires headroom_mw * sustain_h / eta_d of stored energy.
    Services are assumed to activate simultaneously (conservative).
    """
    return sum(
        headroom.get(p, 0.0) * P_bar * SUSTAIN_H.get(p, 0.0) / eta_d
        for p in headroom
        if headroom.get(p, 0.0) > 0.0
    )
from src.config import ASSET, PATHS, configure_asset_duration
from src.optimisation.rolling_intrinsic import solve_daily_lp

log = logging.getLogger(__name__)

WD_CAP_DEFAULT   = 60.0    # £/MWh cap on SP-DA uplift
GATE_HH          = 8       # re-solve every 8 HH (= 4h = 1 EFA block)
WINDOW_HH        = 48      # look-ahead window (1 full day)
DT_H             = 0.5     # half-hourly
VOM_GBP_MWH      = 1.2     # variable O&M (£/MWh throughput)
DEG_GBP_MWH      = 6.0     # degradation shadow cost (£/MWh throughput)

def _total_headroom_frac(headroom: dict[str, float]) -> float:
    total = sum(headroom.values())
    if total > 1.0:
        raise ValueError(
            f"Ancillary headroom fractions sum to {total:.2f} > 1.0. "
            "Reduce fractions so the remainder (≥ 0) is available for DA trading."
        )
    return total


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data(
    raw_dir: str | Path | None = None,
    start: str = "2024-04-01",
    end:   str = "2026-04-25",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load DA prices, SP prices, and NESO ancillary clearing data.

    Returns
    -------
    da_prices  : DataFrame with settlement_date, settlement_period, price_gbp_mwh
    sp_prices  : DataFrame with settlement_date, settlement_period, system_price
    anc        : DataFrame with date, efa_block, product, direction,
                              clearing_price_gbp_mw_h, volume_mw
    """
    if raw_dir is None:
        raw_dir = PATHS["data_raw"]
    raw_dir = Path(raw_dir)

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)

    da = pd.read_parquet(raw_dir / "elexon_da_prices.parquet")
    da["settlement_date"] = pd.to_datetime(da["settlement_date"])
    da = da[da["settlement_date"].between(start_dt, end_dt)].copy()

    sp = pd.read_parquet(raw_dir / "elexon_sp_prices.parquet")
    sp["settlement_date"] = pd.to_datetime(sp["settlement_date"])
    sp = sp[sp["settlement_date"].between(start_dt, end_dt)].copy()

    anc = pd.read_parquet(raw_dir / "neso_eac_clearing.parquet")
    anc["date"] = pd.to_datetime(anc["date"])
    anc = anc[anc["date"].between(start_dt, end_dt)].copy()

    log.info(
        "Loaded: DA %d rows (%s→%s), SP %d rows, Ancillary %d rows",
        len(da), da["settlement_date"].min().date(), da["settlement_date"].max().date(),
        len(sp), len(anc),
    )
    return da, sp, anc


def _build_hh_arrays(
    da: pd.DataFrame,
    sp: pd.DataFrame,
    dates: list[pd.Timestamp],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build aligned (T_hh,) numpy arrays of DA and SP prices for the date range.

    Settlement periods run 1-48 per day. Missing SPs are forward-filled.

    Returns
    -------
    da_arr : (T_hh,) £/MWh DA prices
    sp_arr : (T_hh,) £/MWh system prices (for WD basis)
    """
    da_pivot = (
        da.set_index(["settlement_date", "settlement_period"])["price_gbp_mwh"]
        .unstack("settlement_period")
        .reindex(index=dates, fill_value=np.nan)
    )
    sp_pivot = (
        sp.set_index(["settlement_date", "settlement_period"])["system_price"]
        .unstack("settlement_period")
        .reindex(index=dates, fill_value=np.nan)
    )

    # Forward-fill missing SPs within each day, then fill with 0
    da_pivot = da_pivot.ffill(axis=1).fillna(0.0)
    sp_pivot = sp_pivot.ffill(axis=1).fillna(0.0)

    da_arr = da_pivot.values.ravel().astype(np.float64)
    sp_arr = sp_pivot.values.ravel().astype(np.float64)
    return da_arr, sp_arr


# ---------------------------------------------------------------------------
# Main index builder
# ---------------------------------------------------------------------------

def build_index(
    da_prices:    pd.DataFrame,
    sp_prices:    pd.DataFrame,
    anc_clearing: pd.DataFrame,
    duration_h:   float = 2.0,
    asset_cfg:    dict  | None = None,
    wd_cap:       float = WD_CAP_DEFAULT,
    headroom:     dict[str, float] | None = None,
    include_costs: bool = True,
    verbose:      bool = True,
) -> pd.DataFrame:
    """
    Build historical realized BESS revenue index.

    Parameters
    ----------
    da_prices    : output of load_data()[0]
    sp_prices    : output of load_data()[1]
    anc_clearing : output of load_data()[2]
    duration_h   : battery duration (1.0 or 2.0 hours)
    asset_cfg    : override asset config (default: src.config.ASSET scaled to duration_h)
    wd_cap       : WD uplift cap £/MWh (clip on SP - DA)
    headroom     : ancillary headroom fractions per product (default: DEFAULT_HEADROOM)
    include_costs: deduct VOM + degradation costs from revenue
    verbose      : print progress

    Returns
    -------
    DataFrame with daily resolution, columns:
        date, da_revenue, wd_revenue,
        dc_revenue, dm_revenue, dr_revenue, qr_revenue, br_revenue,
        anc_revenue, total_revenue_gross, costs, total_revenue_net,
        throughput_mwh,
        gbp_per_mw_day, gbp_per_mw_year_ann
    """
    import copy

    if headroom is None:
        headroom = DEFAULT_HEADROOM

    if asset_cfg is None:
        asset_cfg = copy.deepcopy(ASSET)
        configure_asset_duration(asset_cfg, duration_h)

    P_bar   = float(asset_cfg["power_mw"])
    E_nm    = float(asset_cfg["energy_mwh"])
    eta_c   = float(asset_cfg["eta_charge"])
    eta_d   = float(asset_cfg["eta_discharge"])
    E_min   = float(asset_cfg["soc_min_frac"]) * E_nm
    E_max   = float(asset_cfg["soc_max_frac"]) * E_nm
    E_init  = float(asset_cfg["soc_init_frac"]) * E_nm

    # Power available for DA after reserving ancillary headroom
    total_hf = _total_headroom_frac(headroom)
    P_da = P_bar * (1.0 - total_hf)

    # Energy floor: LP must leave enough SoC for all ancillary sustain requirements
    anc_energy_floor = _ancillary_energy_floor(headroom, P_bar, eta_d)
    E_min_da = min(E_min + anc_energy_floor, E_max - 0.05 * E_nm)

    log.info(
        "Duration=%.0fh | P_bar=%.0f MW | ancillary=%.0f%% | P_da=%.0f MW | "
        "E_min_base=%.1f MWh | anc_floor=%.1f MWh | E_min_da=%.1f MWh",
        duration_h, P_bar, total_hf * 100, P_da,
        E_min, anc_energy_floor, E_min_da,
    )

    # Build sorted date list
    dates = sorted(set(
        da_prices["settlement_date"].dt.normalize().unique().tolist() +
        sp_prices["settlement_date"].dt.normalize().unique().tolist()
    ))
    dates = [d for d in dates if not pd.isna(d)]

    da_arr, sp_arr = _build_hh_arrays(da_prices, sp_prices, dates)
    T = len(da_arr)
    delta_wd = np.clip(sp_arr - da_arr, -wd_cap, wd_cap)

    # ── Rolling LP dispatch ───────────────────────────────────────────────────
    cf_da   = np.zeros(T, dtype=np.float64)
    cf_wd   = np.zeros(T, dtype=np.float64)
    soc_arr = np.zeros(T + 1, dtype=np.float64)
    tp_arr  = np.zeros(T, dtype=np.float64)   # throughput (d + c) per HH

    soc_arr[0] = E_init
    E_n = E_init
    t = 0

    while t < T:
        t_end  = min(t + WINDOW_HH, T)
        window = da_arr[t:t_end].copy()

        # For the committed gate window, use the WD-adjusted price for LP
        near = min(GATE_HH, len(window))
        window[:near] = window[:near] + delta_wd[t:t + near]

        d_opt, c_opt, _ = solve_daily_lp(
            window, E_n, E_min_da, E_max, P_da, eta_c, eta_d, DT_H,
        )

        apply_len = min(GATE_HH, T - t)
        for s in range(apply_len):
            d_s = float(d_opt[s]) if s < len(d_opt) else 0.0
            c_s = float(c_opt[s]) if s < len(c_opt) else 0.0
            step = t + s

            da_price = da_arr[step]
            wd_basis = delta_wd[step]

            cf_da[step]  = da_price * (d_s - c_s) * DT_H
            cf_wd[step]  = wd_basis  * (d_s - c_s) * DT_H
            tp_arr[step] = (d_s + c_s) * DT_H

            dE = (-d_s / eta_d + c_s * eta_c) * DT_H
            E_n = float(np.clip(E_n + dE, E_min, E_max))
            soc_arr[step + 1] = E_n

        t += apply_len

        if verbose and (t // GATE_HH) % 200 == 0:
            pct = t / T * 100
            print(f"  Dispatch {pct:.0f}% ({t}/{T} HH) ...", end="\r", flush=True)

    if verbose:
        print(f"  Dispatch 100% ({T}/{T} HH) done.    ")

    # ── Aggregate to daily ────────────────────────────────────────────────────
    n_days = len(dates)
    hh_per_day = T // n_days

    records = []
    prods = [p for p in headroom if p in anc_clearing["product"].unique()]

    for i, date in enumerate(dates):
        sl = slice(i * hh_per_day, (i + 1) * hh_per_day)
        da_rev  = float(cf_da[sl].sum())
        wd_rev  = float(cf_wd[sl].sum())
        tp_day  = float(tp_arr[sl].sum())

        # SoC at start of each EFA block (mapped from HH index).
        # efa_block=0 (BST EFA 1) uses the SoC at midnight (SP 1 of the day).
        sps_day = range(i * hh_per_day, (i + 1) * hh_per_day)
        soc_by_efa: dict[int, float] = {0: float(soc_arr[i * hh_per_day])}
        for hh_idx in sps_day:
            sp_num = (hh_idx % hh_per_day) + 1   # 1-based SP within day
            efa = hh_to_efa(sp_num)
            if efa not in soc_by_efa:
                soc_by_efa[efa] = float(soc_arr[hh_idx])

        anc_day = compute_daily_ancillary(
            date, anc_clearing, headroom, P_bar, soc_by_efa, E_min, eta_d, products=prods,
        )

        anc_total = sum(anc_day.values())
        gross = da_rev + wd_rev + anc_total
        costs = (VOM_GBP_MWH + DEG_GBP_MWH) * tp_day if include_costs else 0.0
        net   = gross - costs

        rec = {
            "date":              date,
            "da_revenue":        round(da_rev, 2),
            "wd_revenue":        round(wd_rev, 2),
            "anc_revenue":       round(anc_total, 2),
            "total_gross":       round(gross, 2),
            "costs":             round(costs, 2),
            "total_net":         round(net, 2),
            "throughput_mwh":    round(tp_day, 2),
            "gbp_per_mw_day":    round(net / P_bar, 4),
        }
        for prod, rev in anc_day.items():
            rec[f"{prod.lower()}_revenue"] = round(rev, 2)

        records.append(rec)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Annualised daily rate (assuming full year)
    df["gbp_per_mw_year_ann"] = df["gbp_per_mw_day"] * 365.25

    log.info(
        "Index built: %d days | DA=£%.0fk/MW/yr | WD=£%.0fk/MW/yr | Anc=£%.0fk/MW/yr | Net=£%.0fk/MW/yr",
        len(df),
        df["da_revenue"].sum() / P_bar / (len(df) / 365.25) / 1000,
        df["wd_revenue"].sum() / P_bar / (len(df) / 365.25) / 1000,
        df["anc_revenue"].sum() / P_bar / (len(df) / 365.25) / 1000,
        df["total_net"].sum() / P_bar / (len(df) / 365.25) / 1000,
    )

    return df


# ---------------------------------------------------------------------------
# Monthly / annual aggregation
# ---------------------------------------------------------------------------

def monthly_index(df: pd.DataFrame, P_bar: float) -> pd.DataFrame:
    """
    Aggregate daily revenue DataFrame to monthly £/MW/year (annualised).
    """
    d = df.copy()
    d["year_month"] = d["date"].dt.to_period("M")

    agg = d.groupby("year_month").agg(
        days          = ("date", "count"),
        da_revenue    = ("da_revenue",  "sum"),
        wd_revenue    = ("wd_revenue",  "sum"),
        anc_revenue   = ("anc_revenue", "sum"),
        total_net     = ("total_net",   "sum"),
        throughput    = ("throughput_mwh", "sum"),
    ).reset_index()

    # Annualise: multiply monthly sum by (12 / days_in_month * days_in_year / 12)
    # Equivalently: monthly_total / days * 365.25
    for col in ["da_revenue", "wd_revenue", "anc_revenue", "total_net"]:
        agg[f"{col}_gbp_mw_yr"] = agg[col] / P_bar / agg["days"] * 365.25

    return agg


def annual_index(df: pd.DataFrame, P_bar: float) -> pd.DataFrame:
    """
    Aggregate daily revenue DataFrame to annual £/MW/year.
    """
    d = df.copy()
    d["year"] = d["date"].dt.year

    agg = d.groupby("year").agg(
        days          = ("date", "count"),
        da_revenue    = ("da_revenue",  "sum"),
        wd_revenue    = ("wd_revenue",  "sum"),
        anc_revenue   = ("anc_revenue", "sum"),
        total_net     = ("total_net",   "sum"),
        throughput    = ("throughput_mwh", "sum"),
    ).reset_index()

    for col in ["da_revenue", "wd_revenue", "anc_revenue", "total_net"]:
        agg[f"{col}_gbp_mw_yr"] = agg[col] / P_bar / agg["days"] * 365.25

    return agg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse, json, copy

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build historical BESS revenue index")
    parser.add_argument("--start",      default="2024-04-01")
    parser.add_argument("--end",        default="2026-04-25")
    parser.add_argument("--duration",   type=float, default=2.0, choices=[1.0, 2.0])
    parser.add_argument("--wd-cap",     type=float, default=WD_CAP_DEFAULT)
    parser.add_argument("--out-dir",    default="data/processed")
    parser.add_argument("--no-costs",   action="store_true")
    args = parser.parse_args()

    dur  = args.duration
    da, sp, anc = load_data(start=args.start, end=args.end)

    asset = copy.deepcopy(ASSET)
    configure_asset_duration(asset, dur)

    df = build_index(
        da, sp, anc,
        duration_h   = dur,
        asset_cfg    = asset,
        wd_cap       = args.wd_cap,
        include_costs= not args.no_costs,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{int(dur)}h"
    df.to_csv(out / f"historical_index_{tag}.csv", index=False)
    log.info("Saved daily index to %s/historical_index_%s.csv", args.out_dir, tag)

    mo = monthly_index(df, float(ASSET["power_mw"]))
    mo.to_csv(out / f"historical_index_{tag}_monthly.csv", index=False)
    log.info("Saved monthly index to %s/historical_index_%s_monthly.csv", args.out_dir, tag)


if __name__ == "__main__":
    main()
