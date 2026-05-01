"""
pipeline.py — Orchestrates the full Phase 1 data pull.

Runs all three fetchers in sequence, validates outputs, and logs a summary.

Usage:
  python -m src.data.pipeline --start 2024-04-01 --end 2026-04-25
  python -m src.data.pipeline --start 2024-04-01 --end 2026-04-25 --forwards synthetic
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def validate_parquet(path: Path, min_rows: int = 1) -> bool:
    if not path.exists():
        log.error("MISSING: %s", path)
        return False
    df = pd.read_parquet(path)
    if len(df) < min_rows:
        log.error("TOO FEW ROWS (%d): %s", len(df), path)
        return False
    log.info("OK: %s — %d rows, cols: %s", path.name, len(df), list(df.columns))
    return True


def run(start: str, end: str, out_dir: Path, forwards_source: str = "synthetic"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Elexon DA prices ---
    log.info("=" * 60)
    log.info("STEP 1/3: Elexon day-ahead + system prices")
    log.info("=" * 60)
    from src.data.fetch_elexon import fetch_mid_range, fetch_system_prices_range
    from datetime import date as _date
    s = _date.fromisoformat(start)
    e = _date.fromisoformat(end)

    fetch_mid_range(s, e, out_path=out_dir / "elexon_da_prices.parquet")
    fetch_system_prices_range(s, e, out_path=out_dir / "elexon_sp_prices.parquet")

    # --- 2. NESO EAC ancillary ---
    log.info("=" * 60)
    log.info("STEP 2/3: NESO EAC ancillary clearing")
    log.info("=" * 60)
    from src.data.fetch_neso import fetch_all_ancillary
    fetch_all_ancillary(start, end, out_path=out_dir / "neso_eac_clearing.parquet")

    # --- 3. Forward curve ---
    log.info("=" * 60)
    log.info("STEP 3/3: GB power forward curve (%s)", forwards_source)
    log.info("=" * 60)
    from src.data.fetch_forwards import build_synthetic_forwards
    if forwards_source == "synthetic":
        df_fwd = build_synthetic_forwards(as_of=_date.fromisoformat(start))
        fwd_path = out_dir / "ice_eex_forwards.parquet"
        df_fwd.to_parquet(fwd_path, index=False)
        log.info("Saved synthetic forwards (%d rows) to %s", len(df_fwd), fwd_path)
    else:
        log.warning("Non-synthetic forwards require manual export — skipping auto-fetch")

    # --- Validation summary ---
    log.info("=" * 60)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 60)
    files = {
        "elexon_da_prices.parquet":  100,
        "elexon_sp_prices.parquet":  100,
        "neso_eac_clearing.parquet":   0,   # may be empty if NESO IDs changed
        "ice_eex_forwards.parquet":    1,
    }
    results = {name: validate_parquet(out_dir / name, min_rows) for name, min_rows in files.items()}
    passed  = sum(results.values())
    total   = len(results)

    log.info("-" * 60)
    log.info("Passed: %d/%d", passed, total)
    if passed < total:
        log.warning("Some files missing or thin — check NESO resource IDs and Elexon API status")
    return passed == total


def main():
    parser = argparse.ArgumentParser(description="BESS Phase 1 data pipeline")
    parser.add_argument("--start",    default="2024-04-01")
    parser.add_argument("--end",      default="2026-04-25")
    parser.add_argument("--out",      default="data/raw")
    parser.add_argument("--forwards", choices=["synthetic", "ice", "eex"],
                        default="synthetic",
                        help="Forward curve source (ice/eex require manual export)")
    args = parser.parse_args()

    ok = run(args.start, args.end, Path(args.out), forwards_source=args.forwards)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
