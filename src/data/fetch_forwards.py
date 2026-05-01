"""
fetch_forwards.py — GB power forward curve data.

Priority 1 (default):
  "forward UK.xlsx" — proprietary GB forward curve workbook with two sheets:
    Base  → UBLIMc1 … UBLIMc36  (UK baseload, monthly tenors 1–36 m forward)
    Peak  → UPLMc1  … UPLMc36   (UK peak,     monthly tenors 1–36 m forward)
  Default location: data/raw/forward_uk.xlsx
  Loader: load_forward_uk_export()

Priority 2 (fallback — requires exchange credentials):
  ICE WebICE terminal CSV export  → load_ice_export()
  EEX transparency platform CSV   → load_eex_export()

Priority 3 (development only):
  Synthetic Schwartz-Smith curve  → build_synthetic_forwards()

Output file:
  data/raw/forwards.parquet
    columns: as_of_date, contract, delivery_start, delivery_end,
             price_gbp_mwh, type (baseload/peak), source

Usage:
  # Tier 1 — forward UK workbook (default):
  python -m src.data.fetch_forwards --source forward_uk
  python -m src.data.fetch_forwards --source forward_uk --file path/to/forward_uk.xlsx

  # Tier 2 — ICE / EEX manual exports:
  python -m src.data.fetch_forwards --source ice --file path/to/ice_export.csv
  python -m src.data.fetch_forwards --source eex --file path/to/eex_export.csv

  # Tier 3 — synthetic fallback:
  python -m src.data.fetch_forwards --source synthetic
"""

import argparse
import logging
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# forward UK.xlsx loader  (Priority 1)
# ---------------------------------------------------------------------------

_DEFAULT_FORWARD_UK_PATH = Path("data/raw/forward_uk.xlsx")

# Maps sheet name → (column prefix, type label)
_FORWARD_UK_SHEETS = {
    "Base": ("UBLIM", "baseload"),
    "Peak": ("UPLM",  "peak"),
}


def load_forward_uk_export(file_path: Path = None) -> pd.DataFrame:
    """
    Load the proprietary "forward UK.xlsx" workbook.

    Layout (both sheets identical in structure):
      Row 0    — blank / filler
      Row 1    — column headers: NaT, UBLIMc1, UBLIMc2, … UBLIMc36
                 (or UPLMc1 … UPLMc36 on the Peak sheet)
      Row 2    — blank / filler
      Row 3+   — data: col 0 = as_of_date, col 1-36 = price (£/MWh)

    Contract naming: cN means the Nth calendar month forward from as_of_date.
    delivery_start = first day of that calendar month
    delivery_end   = last day of that calendar month

    Returns the standard schema:
      as_of_date, contract, delivery_start, delivery_end,
      price_gbp_mwh, type, source
    """
    if file_path is None:
        file_path = _DEFAULT_FORWARD_UK_PATH
    file_path = Path(file_path)

    log.info("Loading forward UK workbook: %s", file_path)
    frames = []

    for sheet, (prefix, curve_type) in _FORWARD_UK_SHEETS.items():
        raw = pd.read_excel(file_path, sheet_name=sheet, header=None)

        # Row 1 holds the column headers; rows 3+ hold the data
        headers = raw.iloc[1].tolist()           # [NaT, 'UBLIMc1', ...]
        data    = raw.iloc[3:].copy()
        data.columns = headers

        # Rename the date column (whatever the first element is)
        date_col = headers[0]
        data = data.rename(columns={date_col: "as_of_date"})
        data["as_of_date"] = pd.to_datetime(data["as_of_date"], errors="coerce")
        data = data.dropna(subset=["as_of_date"])

        # Identify contract columns by prefix
        contract_cols = [c for c in data.columns if isinstance(c, str)
                         and c.startswith(prefix) and re.search(r"c(\d+)$", c)]

        # Melt wide → long
        melted = data.melt(
            id_vars=["as_of_date"],
            value_vars=contract_cols,
            var_name="contract",
            value_name="price_gbp_mwh",
        )
        melted["price_gbp_mwh"] = pd.to_numeric(melted["price_gbp_mwh"], errors="coerce")
        melted = melted.dropna(subset=["price_gbp_mwh"])

        # Derive delivery window: cN → Nth month forward from as_of_date
        def _delivery_bounds(row):
            m = re.search(r"c(\d+)$", row["contract"])
            n = int(m.group(1)) if m else 1
            start = (row["as_of_date"] + relativedelta(months=n)).replace(day=1)
            end   = (start + relativedelta(months=1)) - pd.Timedelta(days=1)
            return pd.Series({"delivery_start": start, "delivery_end": end})

        bounds = melted.apply(_delivery_bounds, axis=1)
        melted = pd.concat([melted, bounds], axis=1)

        melted["type"]   = curve_type
        melted["source"] = "forward_uk"
        frames.append(melted)

    df = pd.concat(frames, ignore_index=True)
    df = df[["as_of_date", "contract", "delivery_start", "delivery_end",
             "price_gbp_mwh", "type", "source"]]
    df = df.sort_values(["as_of_date", "type", "contract"]).reset_index(drop=True)

    log.info(
        "forward_uk: %d rows, %d unique dates, baseload/peak split: %s",
        len(df),
        df["as_of_date"].nunique(),
        df.groupby("type").size().to_dict(),
    )
    return df


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
    parser.add_argument(
        "--source",
        choices=["forward_uk", "ice", "eex", "synthetic"],
        default="forward_uk",
        help="Data source (default: forward_uk)",
    )
    parser.add_argument(
        "--file",
        help="Path to input file (xlsx for forward_uk, csv for ice/eex). "
             "Defaults to data/raw/forward_uk.xlsx for forward_uk source.",
        default=None,
    )
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    out_path = Path(args.out) / "forwards.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "forward_uk":
        file_arg = Path(args.file) if args.file else None
        df = load_forward_uk_export(file_arg)
    elif args.source == "ice":
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
    log.info("Date range: %s → %s", df['as_of_date'].min().date(), df['as_of_date'].max().date())
    log.info("Contracts per date (sample): %d", df.groupby('as_of_date')['contract'].count().iloc[0])


if __name__ == "__main__":
    main()
