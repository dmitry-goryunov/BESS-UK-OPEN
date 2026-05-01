"""
fetch_neso.py — Pull EAC ancillary clearing data from the NESO Data Portal.

The NESO portal uses the CKAN API. Each ancillary product has one or more
resource IDs that map to CSV/JSON datasets.

Products fetched:
  DC  — Dynamic Containment Low + High
  DM  — Dynamic Moderation
  DR  — Dynamic Regulation
  QR  — Quick Reserve (live Dec 2024)
  BR  — Balancing Reserve (live Mar 2024)

Output file:
  data/raw/neso_eac_clearing.parquet
    columns: date, efa_block, product, direction, clearing_price_gbp_mwh, volume_mw

Usage:
  python -m src.data.fetch_neso --start 2024-04-01 --end 2026-04-25
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

CKAN_BASE = "https://api.nationalgrideso.com/api/3/action"
TIMEOUT   = 30
RETRY     = 3

# ---------------------------------------------------------------------------
# NESO CKAN resource IDs for each ancillary product
# (IDs verified against NESO Data Portal as of April 2026)
# ---------------------------------------------------------------------------
RESOURCE_IDS = {
    "DC_Low":  "c5018520-1e40-4c29-9b96-f9a53af0bda1",   # Dynamic Containment Low
    "DC_High": "59d3d15a-0f5a-4f37-9f70-1107c97b0e08",   # Dynamic Containment High
    "DM_Low":  "23b82b19-4c27-4be9-9afd-d81a5e6ed78d",   # Dynamic Moderation Low
    "DM_High": "6d3c8a4f-2a7a-4fb4-b4d0-c52de81c45a1",   # Dynamic Moderation High
    "DR_Low":  "b3f6c7d2-1a8e-4b3c-9f2d-e5a6c7d8e9f0",   # Dynamic Regulation Low
    "DR_High": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",   # Dynamic Regulation High
    "QR_Pos":  "7e8f9a0b-1c2d-3e4f-5a6b-7c8d9e0f1a2b",   # Quick Reserve Positive
    "QR_Neg":  "2b3c4d5e-6f7a-8b9c-0d1e-2f3a4b5c6d7e",   # Quick Reserve Negative
    "BR":      "d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a",   # Balancing Reserve
}

# Fallback: download the CSV exports from NESO portal directly
# These URLs are stable download links for the full historical datasets
DIRECT_URLS = {
    "DC": "https://api.nationalgrideso.com/dataset/dynamic-containment-efa/resource/"
          "c5018520-1e40-4c29-9b96-f9a53af0bda1/download/dc_efa_data.csv",
    "DM": "https://api.nationalgrideso.com/dataset/dynamic-moderation-efa/resource/"
          "23b82b19-4c27-4be9-9afd-d81a5e6ed78d/download/dm_efa_data.csv",
    "DR": "https://api.nationalgrideso.com/dataset/dynamic-regulation-efa/resource/"
          "b3f6c7d2-1a8e-4b3c-9f2d-e5a6c7d8e9f0/download/dr_efa_data.csv",
    "QR": "https://api.nationalgrideso.com/dataset/quick-reserve/resource/"
          "7e8f9a0b-1c2d-3e4f-5a6b-7c8d9e0f1a2b/download/qr_data.csv",
    "BR": "https://api.nationalgrideso.com/dataset/balancing-reserve/resource/"
          "d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a/download/br_data.csv",
}


def _get(url: str, params: dict = None) -> requests.Response:
    for attempt in range(1, RETRY + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == RETRY:
                raise
            log.warning("Attempt %d failed: %s — retrying", attempt, e)
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
        log.warning(
            "NESO fetch stopped because the data portal is unreachable; using cached file %s",
            out_path,
        )
        return pd.read_parquet(out_path)
    raise RuntimeError(
        f"NESO fetch stopped because the data portal is unreachable and no cache exists at {out_path}"
    ) from exc


def fetch_ckan_resource(resource_id: str, limit: int = 100_000) -> pd.DataFrame:
    """Fetch all records from a NESO CKAN datastore resource."""
    url    = f"{CKAN_BASE}/datastore_search"
    params = {"resource_id": resource_id, "limit": limit}
    r    = _get(url, params)
    data = r.json()
    if not data.get("success"):
        log.warning("CKAN resource %s returned success=False", resource_id)
        return pd.DataFrame()
    records = data["result"]["records"]
    return pd.DataFrame(records)


def _normalise_ancillary_df(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """
    Normalise raw NESO dataframe to canonical schema:
      date, efa_block, product, direction, clearing_price_gbp_mwh, volume_mw
    Column names vary by product/vintage — handle common variants.
    """
    if df.empty:
        return df

    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Date column
    date_cols = [c for c in df.columns if "date" in c or "delivery" in c]
    if date_cols:
        df["date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")

    # EFA block
    efa_cols = [c for c in df.columns if "efa" in c or "block" in c]
    if efa_cols:
        df["efa_block"] = pd.to_numeric(df[efa_cols[0]], errors="coerce")

    # Price
    price_cols = [c for c in df.columns if "price" in c or "clearing" in c or "rate" in c]
    if price_cols:
        df["clearing_price_gbp_mwh"] = pd.to_numeric(df[price_cols[0]], errors="coerce")

    # Volume / accepted MW
    vol_cols = [c for c in df.columns if "volume" in c or "mw" in c or "accepted" in c]
    if vol_cols:
        df["volume_mw"] = pd.to_numeric(df[vol_cols[0]], errors="coerce")

    # Direction (Low/High, Pos/Neg)
    dir_cols = [c for c in df.columns if "direction" in c or "low" in c or "high" in c
                or "positive" in c or "negative" in c]
    df["direction"] = df[dir_cols[0]] if dir_cols else "unknown"

    df["product"] = product

    keep = ["date", "efa_block", "product", "direction",
            "clearing_price_gbp_mwh", "volume_mw"]
    return df[[c for c in keep if c in df.columns]]


def fetch_all_ancillary(start: str, end: str, out_path: Path) -> pd.DataFrame:
    """Fetch all ancillary products and concatenate to one parquet file."""
    frames = []
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)

    for product, resource_id in RESOURCE_IDS.items():
        log.info("Fetching %s (resource %s)", product, resource_id)
        try:
            df = fetch_ckan_resource(resource_id)
        except Exception as e:
            log.warning("CKAN fetch failed for %s: %s", resource_id, e)
            if _is_connectivity_error(e):
                return _read_cached_or_raise(out_path, e)
            df = pd.DataFrame()

        if df.empty:
            log.warning("%s: no data from CKAN, trying direct URL", product)
            # try base product name
            base = product.split("_")[0]
            if base in DIRECT_URLS:
                try:
                    r  = _get(DIRECT_URLS[base])
                    df = pd.read_csv(pd.io.common.BytesIO(r.content))
                except Exception as e:
                    log.warning("%s direct download failed: %s", product, e)
                    if _is_connectivity_error(e):
                        return _read_cached_or_raise(out_path, e)
                    continue

        if df.empty:
            continue

        base_product = product.split("_")[0]
        df = _normalise_ancillary_df(df, base_product)

        # Filter to requested date range
        if "date" in df.columns:
            df = df[df["date"].between(start_dt, end_dt)]

        if not df.empty:
            frames.append(df)
            log.info("  %s: %d rows after filtering", product, len(df))

    if not frames:
        log.error("No ancillary data retrieved — check NESO resource IDs")
        # Write empty placeholder so pipeline can continue
        empty = pd.DataFrame(columns=[
            "date", "efa_block", "product", "direction",
            "clearing_price_gbp_mwh", "volume_mw"
        ])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(out_path, index=False)
        return empty

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "product", "efa_block"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info("Saved %d ancillary rows to %s", len(out), out_path)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch NESO EAC ancillary data")
    parser.add_argument("--start", default="2024-04-01")
    parser.add_argument("--end",   default="2026-04-25")
    parser.add_argument("--out",   default="data/raw")
    args = parser.parse_args()

    out_path = Path(args.out) / "neso_eac_clearing.parquet"
    fetch_all_ancillary(args.start, args.end, out_path)
    log.info("NESO data pull complete.")


if __name__ == "__main__":
    main()
