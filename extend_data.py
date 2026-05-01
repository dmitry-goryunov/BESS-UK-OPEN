"""
extend_data.py — Extend raw data files to cover a full 2-year window.

Strategy (Option B):
  1. Inspect existing parquet files to find the last date covered.
  2. Fetch only the gap (last_date + 1 day → END).
  3. Merge new rows with existing data; de-duplicate; save in-place.
  4. Re-fetch NESO ancillary for the full 2-year window (NESO CKAN
     returns the full dataset in one call, so no merge step needed).

Run from the project root:
    python extend_data.py
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT    = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FULL_START = date(2024, 4, 1)
FULL_END   = date(2026, 4, 25)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_date(path: Path, date_col: str) -> date | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=[date_col])
    if df.empty:
        return None
    return pd.to_datetime(df[date_col]).max().date()


def _extend_elexon(
    name: str,
    date_col: str,
    fetch_fn,          # callable(start, end, out_path) -> DataFrame
    end: date,
) -> None:
    out_path = RAW_DIR / f"{name}.parquet"
    tmp_path = RAW_DIR / f"{name}_gap.parquet"

    last = _last_date(out_path, date_col)
    if last is None:
        log.info("%s: no cache found — fetching full range %s to %s", name, FULL_START, end)
        fetch_fn(FULL_START, end, out_path)
        return

    gap_start = last + timedelta(days=1)
    if gap_start > end:
        log.info("%s: already covers up to %s — nothing to fetch", name, last)
        return

    log.info("%s: existing data ends %s — fetching gap %s to %s", name, last, gap_start, end)
    gap_df = fetch_fn(gap_start, end, tmp_path)

    if gap_df.empty:
        log.warning("%s: gap fetch returned no rows", name)
        tmp_path.unlink(missing_ok=True)
        return

    existing = pd.read_parquet(out_path)
    merged   = (
        pd.concat([existing, gap_df], ignore_index=True)
        .drop_duplicates(subset=[date_col, "settlement_period"])
        .sort_values([date_col, "settlement_period"])
        .reset_index(drop=True)
    )
    merged.to_parquet(out_path, index=False)
    tmp_path.unlink(missing_ok=True)

    new_last = pd.to_datetime(merged[date_col]).max().date()
    log.info("%s: merged → %d rows, now covers %s to %s", name, len(merged), FULL_START, new_last)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.data.fetch_elexon import fetch_mid_range, fetch_system_prices_range
    from src.data.fetch_neso   import fetch_all_ancillary

    end = FULL_END

    # ── 1. DA prices ──────────────────────────────────────────────────────────
    log.info("=== Step 1: Elexon DA prices ===")
    _extend_elexon(
        name      = "elexon_da_prices",
        date_col  = "settlement_date",
        fetch_fn  = fetch_mid_range,
        end       = end,
    )

    # ── 2. System (imbalance) prices ──────────────────────────────────────────
    log.info("=== Step 2: Elexon system prices ===")
    _extend_elexon(
        name      = "elexon_sp_prices",
        date_col  = "settlement_date",
        fetch_fn  = fetch_system_prices_range,
        end       = end,
    )

    # ── 3. NESO ancillary — full re-fetch (CKAN returns all records at once) ──
    log.info("=== Step 3: NESO EAC ancillary (full re-fetch) ===")
    fetch_all_ancillary(
        start    = FULL_START.isoformat(),
        end      = end.isoformat(),
        out_path = RAW_DIR / "neso_eac_clearing.parquet",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=== Summary ===")
    for fname, dcol in [
        ("elexon_da_prices.parquet",    "settlement_date"),
        ("elexon_sp_prices.parquet",    "settlement_date"),
        ("neso_eac_clearing.parquet",   "date"),
    ]:
        p = RAW_DIR / fname
        if p.exists():
            df = pd.read_parquet(p, columns=[dcol])
            mn = pd.to_datetime(df[dcol]).min().date()
            mx = pd.to_datetime(df[dcol]).max().date()
            log.info("  %-35s  %d rows   %s → %s", fname, len(df), mn, mx)
        else:
            log.warning("  %-35s  NOT FOUND", fname)


if __name__ == "__main__":
    main()
