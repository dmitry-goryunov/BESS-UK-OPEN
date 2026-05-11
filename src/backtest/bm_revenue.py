"""
bm_revenue.py — GB BESS fleet BM (Balancing Mechanism) revenue index.

Revenue calculation
-------------------
For each BESS BMU, each settlement period:
  offer_revenue = offer_volume_MWh * system_sell_price_GBP_MWh
  bid_revenue   = bid_volume_MWh   * system_buy_price_GBP_MWh

offer_volume_MWh and bid_volume_MWh come from the settlement acceptance
volumes endpoint (totalVolumeAccepted per SP per BMU), which captures the
full sustained dispatch including hold periods — not just ramp transitions.

System prices are used as the price proxy (SSP for offers, SBP for bids).
Actual accepted bid/offer prices differ but correlate with system prices;
this approximation gives the correct order of magnitude and captures the
monthly revenue trend accurately.

Fleet normalization
-------------------
Daily fleet BM revenue is divided by total active fleet MW (non-zero capacity
BMUs in the register) to give £/MW/day, then annualised ×365.25.

Duration scaling
----------------
BM revenue per MW is approximately duration-neutral (same MW headroom for
1H and 2H). A small premium (~15%) is applied for 2H to reflect that longer-
duration batteries can sustain BM activations without early curtailment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

BM_DURATION_PREMIUM_2H = 1.10   # 2H earns ~10% more BM/MW than 1H

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_bm_volumes(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW / "elexon_bm_volumes.parquet"
    df = pd.read_parquet(path)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df


def load_sp_prices(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW / "elexon_sp_prices.parquet"
    df = pd.read_parquet(path)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df[["settlement_date", "settlement_period",
               "system_sell_price", "system_buy_price"]].copy()


def load_bess_bmu_list(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW / "bess_bmu_list.parquet"
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Revenue computation
# ---------------------------------------------------------------------------

def compute_daily_bm_revenue(
    bm_volumes: pd.DataFrame,
    sp_prices: pd.DataFrame,
    bess_bmu_list: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily BM revenue (GBP total fleet) from settlement volumes + system prices.

    Returns DataFrame: settlement_date, bm_revenue_gbp, fleet_mw, bm_rev_per_mw
    """
    df = bm_volumes.copy()
    # Bid volumes arrive as negative MWh (import/charge direction).
    # Take absolute value so both sides contribute positively to revenue.
    df["volume_mwh"] = df["volume_mwh"].abs()
    df = df[df["volume_mwh"] > 0]

    if df.empty:
        log.warning("No BM volume rows > 0")
        return pd.DataFrame()

    sp = sp_prices.rename(columns={
        "system_sell_price": "ssp",
        "system_buy_price":  "sbp",
    })
    df = df.merge(sp, on=["settlement_date", "settlement_period"], how="left")

    # Fill missing prices
    df["ssp"] = df["ssp"].fillna(df["ssp"].median())
    df["sbp"] = df["sbp"].fillna(df["sbp"].median())

    # Revenue: offer × SSP, bid × SBP (prices clipped to 0 — negative price
    # periods mean the grid pays the BMU to charge, but we conservatively zero)
    is_offer = df["side"] == "offer"
    df["price"] = df["ssp"].where(is_offer, df["sbp"]).clip(lower=0)
    df["revenue_gbp"] = df["volume_mwh"] * df["price"]

    daily = (
        df.groupby("settlement_date")["revenue_gbp"]
        .sum()
        .reset_index()
        .rename(columns={"revenue_gbp": "bm_revenue_gbp"})
    )

    fleet_mw = bess_bmu_list.loc[
        bess_bmu_list["generationCapacity"] > 0, "generationCapacity"
    ].sum()
    if fleet_mw <= 0:
        log.warning("Fleet MW = 0 — using 1500 MW fallback")
        fleet_mw = 1500.0

    daily["fleet_mw"]      = fleet_mw
    daily["bm_rev_per_mw"] = daily["bm_revenue_gbp"] / fleet_mw

    log.info(
        "Daily BM revenue: %d days, mean GBP%.0f/MW/day, fleet=%.0f MW",
        len(daily), daily["bm_rev_per_mw"].mean(), fleet_mw,
    )
    return daily


# ---------------------------------------------------------------------------
# Monthly index
# ---------------------------------------------------------------------------

def monthly_bm_index(
    daily_rev: pd.DataFrame,
    duration_h: float = 2.0,
) -> pd.DataFrame:
    """
    Aggregate to monthly GBP/MW/yr (annualised).

    duration_h: 1.0 or 2.0 — applies duration scaling factor.
    """
    df = daily_rev.copy()
    df["year_month"] = df["settlement_date"].dt.to_period("M")

    scale = BM_DURATION_PREMIUM_2H if duration_h >= 2.0 else 1.0

    monthly = (
        df.groupby("year_month")
        .agg(days=("settlement_date", "nunique"),
             rev_sum=("bm_rev_per_mw", "sum"))
        .reset_index()
    )
    monthly["bm_rev_gbp_mw_yr"] = (
        monthly["rev_sum"] / monthly["days"] * 365.25 * scale
    )
    monthly["duration_h"] = duration_h
    return monthly[["year_month", "days", "bm_rev_gbp_mw_yr", "duration_h"]]


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_bm_index(
    start: str = "2024-01-01",
    end:   str = "2026-04-25",
    volumes_path:  Path | None = None,
    sp_path:       Path | None = None,
    bmu_list_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Build BM revenue index for 1H and 2H. Returns {'1h': df, '2h': df}."""
    vols     = load_bm_volumes(volumes_path)
    sp       = load_sp_prices(sp_path)
    bmu_list = load_bess_bmu_list(bmu_list_path)

    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
    vols = vols[(vols["settlement_date"] >= start_dt) &
                (vols["settlement_date"] <= end_dt)]

    daily = compute_daily_bm_revenue(vols, sp, bmu_list)

    return {
        "1h": monthly_bm_index(daily, duration_h=1.0),
        "2h": monthly_bm_index(daily, duration_h=2.0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compute GB BESS BM revenue index")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end",   default="2026-04-25")
    args = parser.parse_args()

    idx = build_bm_index(args.start, args.end)
    for tag, df in idx.items():
        print(f"\n=== BM index {tag.upper()} ===")
        df2 = df.copy()
        df2["bm_rev_k"] = (df2["bm_rev_gbp_mw_yr"] / 1000).round(1)
        print(df2[["year_month", "days", "bm_rev_k"]].to_string(index=False))
        print(f"Period avg: GBP{df2['bm_rev_gbp_mw_yr'].mean()/1000:.1f}k/MW/yr")
        out = PROCESSED / f"bm_index_{tag}.csv"
        df.to_csv(out, index=False)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
