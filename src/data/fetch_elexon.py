"""
fetch_elexon.py — Pull day-ahead and settlement price data from Elexon BMRS.

Endpoints used (all public, no auth required):
  - /datasets/MID       : Market Index Data — N2EX + EPEX day-ahead prices
  - /balancing/settlement/system-prices/{date} : System sell/buy price per SP
  - /datasets/SYSDEM    : System demand (NIV proxy)

Output files (parquet):
  data/raw/elexon_da_prices.parquet   — half-hourly DA prices (GBP/MWh)
  data/raw/elexon_sp_prices.parquet   — half-hourly System Price (GBP/MWh)

Usage:
  python -m src.data.fetch_elexon --start 2024-04-01 --end 2026-04-25
"""

import argparse
import os
import socket
import time
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE_URL = os.getenv("ELEXON_BASE_URL", "https://data.elexon.co.uk/bmrs/api/v1")
HEADERS  = {"Accept": "application/json"}
TIMEOUT  = 30    # seconds per request
RETRY    = 3
BACKOFF  = 2.0   # seconds between retries


def _get(url: str, params: dict = None) -> dict:
    """GET with retry/backoff."""
    for attempt in range(1, RETRY + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == RETRY:
                raise
            log.warning("Attempt %d/%d failed: %s — retrying in %.0fs", attempt, RETRY, e, BACKOFF)
            time.sleep(BACKOFF * attempt)


def _is_connectivity_error(exc: Exception) -> bool:
    """True when retrying every settlement date would likely fail the same way."""
    current = exc
    while current is not None:
        if isinstance(current, (requests.ConnectionError, requests.Timeout, socket.gaierror)):
            return True
        current = current.__cause__ or current.__context__
    return False


def _read_cached_or_raise(out_path: Path, label: str, exc: Exception) -> pd.DataFrame:
    if out_path.exists():
        log.warning(
            "%s fetch stopped because Elexon is unreachable; using cached file %s",
            label,
            out_path,
        )
        return pd.read_parquet(out_path)
    raise RuntimeError(
        f"{label} fetch stopped because Elexon is unreachable and no cache exists at {out_path}"
    ) from exc


# ---------------------------------------------------------------------------
# Day-ahead prices (Market Index Data)
# ---------------------------------------------------------------------------

def fetch_mid_day(settlement_date: date) -> pd.DataFrame:
    """
    Fetch Market Index Data for one settlement date.
    Returns DataFrame with columns: settlement_date, settlement_period, price_gbp_mwh, volume_mwh.
    MID combines N2EX and EPEX into a volume-weighted index.
    """
    url    = f"{BASE_URL}/balancing/pricing/market-index"
    params = {
        # Query a wide enough UTC start-time window to cover BST and clock-change
        # settlement days, then filter on settlementDate below.
        "from":   (settlement_date - timedelta(days=1)).isoformat(),
        "to":     (settlement_date + timedelta(days=1)).isoformat(),
        "format": "json",
    }
    data = _get(url, params)
    records = data.get("data", [])
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        row_date = pd.to_datetime(r.get("settlementDate") or r.get("SettlementDate"))
        if row_date.date() != settlement_date:
            continue
        rows.append({
            "settlement_date":   row_date,
            "settlement_period": int(r.get("settlementPeriod") or r.get("SettlementPeriod", 0)),
            "price_gbp_mwh":     float(r.get("price") or r.get("Price") or 0),
            "volume_mwh":        float(r.get("volume") or r.get("Volume") or 0),
            "data_provider":     r.get("dataProvider") or r.get("DataProvider", ""),
        })
    return pd.DataFrame(rows)


def _aggregate_mid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse provider-level MID rows to one price per settlement period.

    Elexon returns provider rows separately. In the current GB sample APXMIDP
    carries nearly all volume while N2EXMIDP often appears as zero volume and
    zero price. Keeping both rows makes downstream SP-DA calibration treat the
    zero-volume N2EX placeholders as real day-ahead prices.
    """
    if df.empty:
        return df

    keys = ["settlement_date", "settlement_period"]
    work = df.copy()
    work["price_x_volume"] = work["price_gbp_mwh"] * work["volume_mwh"]

    positive_volume = work[work["volume_mwh"] > 0].copy()
    if positive_volume.empty:
        out = (
            work.groupby(keys, as_index=False)
            .agg(
                price_gbp_mwh=("price_gbp_mwh", "mean"),
                volume_mwh=("volume_mwh", "sum"),
            )
        )
    else:
        out = (
            positive_volume.groupby(keys, as_index=False)
            .agg(
                price_x_volume=("price_x_volume", "sum"),
                volume_mwh=("volume_mwh", "sum"),
            )
        )
        out["price_gbp_mwh"] = out["price_x_volume"] / out["volume_mwh"]
        out = out.drop(columns=["price_x_volume"])

    out["data_provider"] = "MID_VOLUME_WEIGHTED"
    return out[["settlement_date", "settlement_period", "price_gbp_mwh", "volume_mwh", "data_provider"]]


def fetch_mid_range(start: date, end: date, out_path: Path) -> pd.DataFrame:
    """Fetch MID for a date range; save to parquet."""
    frames = []
    current = start
    done = 0
    total = (end - start).days + 1
    t0 = time.monotonic()
    log.info("Fetching MID (day-ahead) from %s to %s (%d days)", start, end, total)

    while current <= end:
        try:
            df = fetch_mid_day(current)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            if _is_connectivity_error(e):
                return _read_cached_or_raise(out_path, "MID", e)
            log.warning("MID fetch failed for %s: %s", current, e)
        done += 1
        if done == 1 or done % 25 == 0 or current >= end:
            rows = sum(len(frame) for frame in frames)
            elapsed = time.monotonic() - t0
            log.info(
                "MID progress: %d/%d days through %s, %d rows collected, %.1fs elapsed",
                done,
                total,
                current,
                rows,
                elapsed,
            )
        current += timedelta(days=1)
        time.sleep(0.1)   # be polite to the API

    if not frames:
        log.error("No MID data retrieved")
        return pd.DataFrame()

    out = _aggregate_mid_prices(pd.concat(frames, ignore_index=True))
    out = out.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(out), out_path)
    return out


# ---------------------------------------------------------------------------
# System Price (cash-out / imbalance settlement price)
# ---------------------------------------------------------------------------

def fetch_system_prices_day(settlement_date: date) -> pd.DataFrame:
    """
    Fetch settlement system prices for one settlement date.
    Returns DataFrame with: settlement_date, settlement_period,
                            system_sell_price, system_buy_price, net_imbalance_volume.
    """
    url    = f"{BASE_URL}/balancing/settlement/system-prices/{settlement_date.isoformat()}"
    params = {"format": "json"}
    data   = _get(url, params)
    records = data.get("data", data) if isinstance(data, dict) else data

    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        rows.append({
            "settlement_date":     pd.to_datetime(
                r.get("settlementDate") or r.get("SettlementDate")
            ),
            "settlement_period":   int(
                r.get("settlementPeriod") or r.get("SettlementPeriod", 0)
            ),
            "system_sell_price":   float(
                r.get("systemSellPrice") or r.get("SystemSellPrice") or 0
            ),
            "system_buy_price":    float(
                r.get("systemBuyPrice") or r.get("SystemBuyPrice") or 0
            ),
            "net_imbalance_volume": float(
                r.get("netImbalanceVolume") or r.get("NetImbalanceVolume") or 0
            ),
        })
    return pd.DataFrame(rows)


def fetch_system_prices_range(start: date, end: date, out_path: Path) -> pd.DataFrame:
    """Fetch system prices for a date range; save to parquet."""
    frames = []
    current = start
    done = 0
    total = (end - start).days + 1
    t0 = time.monotonic()
    log.info("Fetching System Prices from %s to %s (%d days)", start, end, total)

    while current <= end:
        try:
            df = fetch_system_prices_day(current)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            if _is_connectivity_error(e):
                return _read_cached_or_raise(out_path, "System price", e)
            log.warning("System price fetch failed for %s: %s", current, e)
        done += 1
        if done == 1 or done % 25 == 0 or current >= end:
            rows = sum(len(frame) for frame in frames)
            elapsed = time.monotonic() - t0
            log.info(
                "System price progress: %d/%d days through %s, %d rows collected, %.1fs elapsed",
                done,
                total,
                current,
                rows,
                elapsed,
            )
        current += timedelta(days=1)
        time.sleep(0.1)

    if not frames:
        log.error("No system price data retrieved")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    # Derive imbalance basis: SP = single price post-Nov 2015
    # System Price = (SSP + SBP) / 2 when NIV-weighted; use SSP as cash-out proxy
    out["system_price"] = out["system_sell_price"]   # single cash-out price

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(out), out_path)
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Elexon BMRS data")
    parser.add_argument("--start", default="2024-04-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2026-04-25", help="End date YYYY-MM-DD")
    parser.add_argument("--out",   default="data/raw",   help="Output directory")
    args = parser.parse_args()

    start    = date.fromisoformat(args.start)
    end      = date.fromisoformat(args.end)
    out_dir  = Path(args.out)

    fetch_mid_range(
        start, end,
        out_path=out_dir / "elexon_da_prices.parquet"
    )
    fetch_system_prices_range(
        start, end,
        out_path=out_dir / "elexon_sp_prices.parquet"
    )
    log.info("Elexon data pull complete.")


if __name__ == "__main__":
    main()
