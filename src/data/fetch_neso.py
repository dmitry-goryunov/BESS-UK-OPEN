"""
fetch_neso.py — Pull EAC ancillary clearing data from the NESO Data Portal.

Covers ERA 2 only (Nov 2023 – present, EAC Response-Reserve + QR + BR framework):
    Package: 291e3c28-75f2-4a8f-b5f5-008bebaac368  (46 resources)
    Schema: auctionProduct | deliveryStart | deliveryEnd | clearedVolume | clearingPrice
    Price unit: £/MW/h  (already hourly)
    Products: DCH DCL DRH DRL DMH DML NQR PQR NBR PBR NSR PSR
    Key result-summary resource IDs (discovered at runtime via package_show):
      FY2023  be5c6b0d-a335-4859-93f2-389585b4e9a1
      FY2024  (discovered at runtime)
      FY2025  (discovered at runtime)
      Current 596f29ac-0387-4ba4-a6d3-95c243140707

  BR standalone
    Package: 94c65383-a108-468f-aad4-1d96380be93f
    Results Summary: 1b3f2ee1-74a0-4939-a5a3-f01f19e663e4

Output file:
  data/raw/neso_eac_clearing.parquet
    columns: date, efa_block, product, direction, clearing_price_gbp_mw_h, volume_mw

Usage:
  python -m src.data.fetch_neso --start 2023-11-01 --end 2026-05-11
"""

import argparse
import logging
import socket
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CKAN_BASE = "https://api.neso.energy/api/3/action"
TIMEOUT   = 60
RETRY     = 3

# ---------------------------------------------------------------------------
# Era 2: post-Nov 2023 — EAC Response-Reserve package (discover archives
# at runtime so new monthly files are picked up automatically)
# ---------------------------------------------------------------------------
ERA2_PACKAGE_ID   = "291e3c28-75f2-4a8f-b5f5-008bebaac368"
ERA2_CURRENT_ID   = "596f29ac-0387-4ba4-a6d3-95c243140707"   # running current
ERA2_FY2023_ID    = "be5c6b0d-a335-4859-93f2-389585b4e9a1"   # FY2023 archive

# ---------------------------------------------------------------------------
# BR standalone package
# ---------------------------------------------------------------------------
BR_RESULTS_ID = "1b3f2ee1-74a0-4939-a5a3-f01f19e663e4"

# EFA block boundaries (start hour, UTC-equivalent used by NESO)
# EFA 1: 23:00-03:00, EFA 2: 03:00-07:00, ... EFA 6: 19:00-23:00
_EFA_START_HOURS = {23: 1, 3: 2, 7: 3, 11: 4, 15: 5, 19: 6}


def _efa_block_from_hour(hour: int) -> int:
    return _EFA_START_HOURS.get(hour, 0)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None) -> requests.Response:
    for attempt in range(1, RETRY + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == RETRY:
                raise
            log.warning("Attempt %d failed: %s — retrying in %.0fs", attempt, 2.0 * attempt, e)
            time.sleep(2.0 * attempt)


def _is_connectivity_error(exc: Exception) -> bool:
    current = exc
    while current is not None:
        if isinstance(current, (requests.ConnectionError, requests.Timeout, socket.gaierror)):
            return True
        current = current.__cause__ or current.__context__
    return False


def _read_cached_or_raise(out_path: Path, exc: Exception) -> pd.DataFrame:
    if out_path.exists():
        log.warning("NESO unreachable; using cached %s", out_path)
        return pd.read_parquet(out_path)
    raise RuntimeError(
        f"NESO unreachable and no cache at {out_path}"
    ) from exc


# ---------------------------------------------------------------------------
# CKAN fetch
# ---------------------------------------------------------------------------

def _fetch_ckan(resource_id: str, limit: int = 200_000) -> pd.DataFrame:
    url    = f"{CKAN_BASE}/datastore_search"
    params = {"resource_id": resource_id, "limit": limit}
    r      = _get(url, params)
    data   = r.json()
    if not data.get("success"):
        log.warning("CKAN %s returned success=False", resource_id)
        return pd.DataFrame()
    records = data["result"]["records"]
    total   = data["result"].get("total", len(records))
    log.info("  %s: %d / %d records", resource_id[:8], len(records), total)
    if len(records) < total:
        log.warning("  Truncated — increase limit or paginate. Got %d of %d", len(records), total)
    return pd.DataFrame(records)


def _discover_era2_result_summary_ids() -> list[str]:
    """
    Call package_show on the EAC Response-Reserve package and return
    resource IDs whose name contains 'Results Summary' (case-insensitive),
    excluding the current-running resource (already included explicitly).
    """
    url  = f"{CKAN_BASE}/package_show"
    r    = _get(url, {"id": ERA2_PACKAGE_ID})
    data = r.json()
    if not data.get("success"):
        log.warning("package_show failed for EAC package")
        return []
    resources = data["result"].get("resources", [])
    ids = []
    for res in resources:
        name = res.get("name", "")
        rid  = res.get("id", "")
        if "results summary" in name.lower() and rid != ERA2_CURRENT_ID:
            ids.append(rid)
            log.info("  Discovered archive: %s  (%s)", name, rid[:8])
    return ids


# ---------------------------------------------------------------------------
# Era 2 normalisation
# ---------------------------------------------------------------------------

_ERA2_PRODUCT_MAP = {
    "DCH": ("DC", "High"), "DCL": ("DC", "Low"),
    "DRH": ("DR", "High"), "DRL": ("DR", "Low"),
    "DMH": ("DM", "High"), "DML": ("DM", "Low"),
    "NQR": ("QR", "Neg"),  "PQR": ("QR", "Pos"),
    "NBR": ("BR", "Neg"),  "PBR": ("BR", "Pos"),
    "NSR": ("SR", "Neg"),  "PSR": ("SR", "Pos"),
}


def _normalise_era2(df: pd.DataFrame) -> pd.DataFrame:
    """
    New schema (post-Nov 2023):
      auctionID | auctionProduct | serviceType | deliveryStart | deliveryEnd |
      clearedVolume | clearingPrice | linkedServiceWindowID

    Price is already in £/MW/h.
    EFA block and date are derived from deliveryStart.
    """
    if df.empty:
        return df

    rows = []
    for _, row in df.iterrows():
        auc_prod  = str(row.get("auctionProduct", "")).strip().upper()
        product, direction = _ERA2_PRODUCT_MAP.get(auc_prod, (auc_prod, "unknown"))

        start = pd.to_datetime(row.get("deliveryStart", None), errors="coerce")
        if pd.isna(start):
            continue

        efa   = _efa_block_from_hour(start.hour)
        date  = start.normalize()
        if start.hour == 23:   # EFA 1 belongs to the next calendar day
            date = date + pd.Timedelta(days=1)

        vol   = float(pd.to_numeric(row.get("clearedVolume",  0), errors="coerce") or 0)
        price = float(pd.to_numeric(row.get("clearingPrice",  0), errors="coerce") or 0)

        rows.append({
            "date":                  date,
            "efa_block":             efa,
            "product":               product,
            "direction":             direction,
            "clearing_price_gbp_mw_h": price,
            "volume_mw":             vol,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------

def fetch_all_ancillary(start: str, end: str, out_path: Path) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    frames   = []

    # ── Discover and fetch all EAC Results Summary archives ──────────────────
    log.info("Discovering archive resources from EAC package...")
    try:
        archive_ids = _discover_era2_result_summary_ids()
    except Exception as e:
        if _is_connectivity_error(e):
            return _read_cached_or_raise(out_path, e)
        log.warning("Archive discovery failed: %s — using known IDs only", e)
        archive_ids = [ERA2_FY2023_ID]

    resource_ids = archive_ids + [ERA2_CURRENT_ID]
    for rid in resource_ids:
        log.info("  Fetching resource %s", rid[:8])
        try:
            df2 = _fetch_ckan(rid)
            df2 = _normalise_era2(df2)
            if not df2.empty:
                df2 = df2[df2["date"].between(start_dt, end_dt)]
                log.info("  %s: %d rows after date filter", rid[:8], len(df2))
                if not df2.empty:
                    frames.append(df2)
        except Exception as e:
            if _is_connectivity_error(e):
                return _read_cached_or_raise(out_path, e)
            log.warning("  Resource %s failed: %s", rid[:8], e)

    # ── BR standalone ─────────────────────────────────────────────────────────
    log.info("Fetching BR Results Summary %s", BR_RESULTS_ID[:8])
    try:
        dfbr = _fetch_ckan(BR_RESULTS_ID)
        dfbr = _normalise_era2(dfbr)
        if not dfbr.empty:
            dfbr = dfbr[dfbr["date"].between(start_dt, end_dt)]
            if not dfbr.empty:
                frames.append(dfbr)
                log.info("  BR: %d rows", len(dfbr))
    except Exception as e:
        log.warning("  BR fetch failed: %s", e)

    # ── Combine and save ─────────────────────────────────────────────────────
    if not frames:
        log.error("No ancillary data retrieved — check NESO resource IDs and connectivity")
        empty = pd.DataFrame(columns=[
            "date", "efa_block", "product", "direction",
            "clearing_price_gbp_mw_h", "volume_mw",
        ])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(out_path, index=False)
        return empty

    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date", "efa_block", "product", "direction"])
        .sort_values(["date", "product", "efa_block", "direction"])
        .reset_index(drop=True)
    )
    out["date"]      = pd.to_datetime(out["date"])
    out["efa_block"] = out["efa_block"].astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info("Saved %d ancillary rows to %s", len(out), out_path)

    _print_coverage(out)
    return out


def _print_coverage(df: pd.DataFrame) -> None:
    log.info("Coverage by product:")
    for prod, g in df.groupby("product"):
        log.info("  %-4s  %s → %s  (%d rows)",
                 prod, g["date"].min().date(), g["date"].max().date(), len(g))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch NESO EAC ancillary clearing data")
    parser.add_argument("--start", default="2023-11-01",
                        help="Start date (default: 2023-11-01, start of Era 2 EAC framework)")
    parser.add_argument("--end",   default="2026-05-11")
    parser.add_argument("--out",   default="data/raw")
    args = parser.parse_args()

    out_path = Path(args.out) / "neso_eac_clearing.parquet"
    fetch_all_ancillary(args.start, args.end, out_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
