"""
fetch_forwards.py — GB power forward curve data (ICE/EEX).

ICE and EEX require exchange membership / vendor credentials.
This module provides:
  1. A loader for ICE CSV exports (manual download from ICE WebICE terminal)
  2. A loader for EEX CSV exports (manual download from EEX transparency platform)
  3. A synthetic fallback using Schwartz-Smith parameters from config — for
     development / backtesting when live forwards are unavailable.

Output file:
  data/raw/ice_eex_forwards.parquet
    columns: as_of_date, contract, delivery_start, delivery_end,
             price_gbp_mwh, type (baseload/peak)

Usage (manual export path):
  python -m src.data.fetch_forwards --source ice  --file path/to/ice_export.csv
  python -m src.data.fetch_forwards --source eex  --file path/to/eex_export.csv
  python -m src.data.fetch_forwards --source synthetic  # uses config SS params
"""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# ICE export loader
# ---------------------------------------------------------------------------

def load_ice_export(file_path: Path) -> pd.DataFrame:
    """
    Parse a manual ICE WebICE CSV export for GB power forwards.
    ICE exports typically have columns like:
      Contract, Settlement Date, Settlement Price, ...
    Adjust column mapping to match your actual export format.
    """
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    log.info("ICE export columns: %s", list(df.columns))

    # Common ICE column variants — adapt if your export differs
    col_map = {
        "Contract":           "contract",
        "Settlement Date":    "as_of_date",
        "Settlement Price":   "price_gbp_mwh",
        "Delivery Start":     "delivery_start",
        "Delivery End":       "delivery_end",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df["as_of_date"]     = pd.to_datetime(df["as_of_date"], errors="coerce")
    df["delivery_start"] = pd.to_datetime(df.get("delivery_start", pd.NaT), errors="coerce")
    df["delivery_end"]   = pd.to_datetime(df.get("delivery_end", pd.NaT), errors="coerce")
    df["price_gbp_mwh"]  = pd.to_numeric(df.get("price_gbp_mwh", np.nan), errors="coerce")
    df["type"]           = "baseload"   # override if export includes peak contracts
    df["source"]         = "ICE"

    keep = ["as_of_date", "contract", "delivery_start", "delivery_end",
            "price_gbp_mwh", "type", "source"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["price_gbp_mwh"])


# ---------------------------------------------------------------------------
# EEX export loader
# ---------------------------------------------------------------------------

def load_eex_export(file_path: Path) -> pd.DataFrame:
    """
    Parse a manual EEX transparency platform CSV for GB (UK) power forwards.
    EEX exports: Date, Product, Price, Volume, ...
    """
    df = pd.read_csv(file_path, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]

    log.info("EEX export columns: %s", list(df.columns))

    col_map = {
        "Date":             "as_of_date",
        "Product":          "contract",
        "Price":            "price_gbp_mwh",
        "Delivery Period":  "delivery_period",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df["as_of_date"]    = pd.to_datetime(df["as_of_date"], errors="coerce")
    df["price_gbp_mwh"] = pd.to_numeric(df.get("price_gbp_mwh", np.nan), errors="coerce")
    df["type"]          = df["contract"].str.lower().apply(
        lambda x: "peak" if "peak" in str(x) else "baseload"
    )
    df["source"] = "EEX"

    keep = ["as_of_date", "contract", "price_gbp_mwh", "type", "source"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["price_gbp_mwh"])


# ---------------------------------------------------------------------------
# Synthetic forward curve (development fallback)
# ---------------------------------------------------------------------------

def build_synthetic_forwards(
    as_of: date = None,
    tenors_years: list = None,
    anchor_gbp_mwh: float = 76.7,
    seasonal_amp: float = 5.0,
    noise_std: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic GB baseload forward curve for development use.
    Uses Schwartz-Smith long-run anchor from config with added seasonal shape.

    NOT for production valuation — replace with real ICE/EEX data.
    """
    if as_of is None:
        as_of = date.today()
    if tenors_years is None:
        tenors_years = [1/12, 2/12, 3/12, 6/12, 1, 1.5, 2, 2.5, 3]

    rng = np.random.default_rng(seed)
    rows = []
    for t in tenors_years:
        delivery_start = date(as_of.year, as_of.month, 1) + timedelta(days=int(t * 365))
        delivery_end   = delivery_start + timedelta(days=int(30.5))
        # Seasonal uplift: winter premium, summer discount
        month = delivery_start.month
        seasonal = seasonal_amp * np.cos(2 * np.pi * (month - 1) / 12)
        noise    = rng.normal(0, noise_std)
        price    = anchor_gbp_mwh + seasonal + noise

        rows.append({
            "as_of_date":     pd.Timestamp(as_of),
            "contract":       f"SYN_BL_{delivery_start.strftime('%Y%m')}",
            "delivery_start": pd.Timestamp(delivery_start),
            "delivery_end":   pd.Timestamp(delivery_end),
            "price_gbp_mwh":  round(price, 2),
            "type":           "baseload",
            "source":         "synthetic",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load GB power forward curves")
    parser.add_argument("--source", choices=["ice", "eex", "synthetic"],
                        default="synthetic")
    parser.add_argument("--file",   help="Path to ICE/EEX CSV export", default=None)
    parser.add_argument("--out",    default="data/raw")
    args = parser.parse_args()

    out_path = Path(args.out) / "ice_eex_forwards.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "ice":
        if not args.file:
            raise ValueError("--file required for source=ice")
        df = load_ice_export(Path(args.file))
    elif args.source == "eex":
        if not args.file:
            raise ValueError("--file required for source=eex")
        df = load_eex_export(Path(args.file))
    else:
        log.warning("Using SYNTHETIC forward curve — not for production use")
        df = build_synthetic_forwards()

    df.to_parquet(out_path, index=False)
    log.info("Saved %d forward curve rows to %s", len(df), out_path)
    log.info("Contracts: %s", sorted(df["contract"].unique()))


if __name__ == "__main__":
    main()
