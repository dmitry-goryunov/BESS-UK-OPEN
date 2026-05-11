"""
ancillary_revenue.py — EFA-block ancillary income from actual NESO clearing data.

Each EFA block delivers 4 hours of ancillary revenue.
Headroom fractions are fixed model parameters; revenue is reduced when the
battery's stored energy cannot sustain the service for the required duration.

Products in scope (ERA 2, post-Nov 2023):
  DC  — Dynamic Containment (bi-directional)
  DM  — Dynamic Moderation (bi-directional)
  DR  — Dynamic Regulation (bi-directional)
  QR  — Quick Reserve (Pos/Neg)
  BR  — Balancing Reserve (Pos/Neg) — optional, low price typically

Direction conventions in neso_eac_clearing.parquet:
  DC/DM/DR : direction = 'High' (upward/discharge) or 'Low' (downward/charge)
  QR/BR    : direction = 'Pos' (upward) or 'Neg' (downward)

Revenue formula (per EFA block, per product):
  revenue = effective_price × headroom_mw × BLOCK_H

where headroom_mw = min(P_bar × fraction, max_sustainable_mw)
and max_sustainable_mw = (E_curr - E_min) × eta_d / sustain_h    (upward services)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK_H = 4.0   # hours per EFA block

# Default headroom fractions (share of nameplate power held for each product).
# Fractions must sum to ≤ 1.0; the remainder is available for DA energy trading.
# Calibrated to produce ancillary revenue broadly consistent with Modo 2h index:
#   DC: ~50 MW held → ~£12k/MW/yr at 2024-25 clearing prices
#   DM: ~10 MW held → ~£3k/MW/yr
#   DR: ~5 MW held  → ~£2k/MW/yr  (energy-limited for 1h battery, scales with duration)
#   QR: ~15 MW held → ~£4k/MW/yr  (available from Dec 2024 only)
#   DA: remaining 20% → ~£5-8k/MW/yr
DEFAULT_HEADROOM: dict[str, float] = {
    "DC": 0.35,
    "DM": 0.10,
    "DR": 0.05,
    "QR": 0.15,
    "BR": 0.00,
}

# Sustain duration required by each product (hours).
# This determines how much stored energy the battery must have to provide the service.
SUSTAIN_H: dict[str, float] = {
    "DC": 0.25,   # 15-min sustain
    "DM": 0.25,   # 15-min sustain
    "DR": 1.00,   # 60-min sustain ← primary duration scaling driver
    "QR": 0.50,   # 30-min sustain
    "BR": 1.00,   # 60-min sustain
}


def _effective_price(anc_block: pd.DataFrame, product: str) -> float:
    """
    Mean clearing price across directions for a product in one EFA block.

    For bi-directional products (DC/DM/DR), the asset earns both High and Low
    clearing prices on its full headroom. We average them to get a single
    price for the headroom_mw held.

    For unidirectional products (QR/BR), Pos = upward, Neg = downward.
    """
    rows = anc_block[anc_block["product"] == product]
    if rows.empty:
        return 0.0
    return float(rows["clearing_price_gbp_mw_h"].mean())


def _available_upward_mw(
    headroom_frac: float,
    P_bar: float,
    E_curr: float,
    E_min: float,
    sustain_h: float,
    eta_d: float,
) -> float:
    """
    Upward ancillary headroom limited by power and stored energy.

    A battery can only sustain upward (discharge) service if it holds
    enough stored energy: E_curr - E_min >= headroom_mw * sustain_h / eta_d.
    """
    power_limit = headroom_frac * P_bar
    energy_limit = (E_curr - E_min) * eta_d / max(sustain_h, 1e-6)
    return max(0.0, min(power_limit, energy_limit))


def compute_block_ancillary(
    anc_block: pd.DataFrame,
    headroom_fracs: dict[str, float],
    P_bar: float,
    E_curr: float,
    E_min: float,
    eta_d: float,
    products: list[str] | None = None,
) -> dict[str, float]:
    """
    Revenue by product for one EFA block.

    Parameters
    ----------
    anc_block   : rows from neso_eac_clearing for a single (date, efa_block)
    headroom_fracs : product -> fraction of P_bar held for ancillary
    P_bar       : nameplate power (MW)
    E_curr      : SoC at start of this EFA block (MWh)
    E_min       : minimum SoC (MWh)
    eta_d       : discharge efficiency
    products    : products to include (default: all keys in headroom_fracs)

    Returns
    -------
    dict: product -> GBP revenue for this EFA block
    """
    if products is None:
        products = list(headroom_fracs.keys())

    revenue = {}
    for prod in products:
        if prod not in headroom_fracs:
            continue
        price = _effective_price(anc_block, prod)
        if price <= 0.0:
            continue
        h_mw = _available_upward_mw(
            headroom_fracs[prod],
            P_bar,
            E_curr,
            E_min,
            SUSTAIN_H.get(prod, 1.0),
            eta_d,
        )
        revenue[prod] = price * h_mw * BLOCK_H

    return revenue


def compute_daily_ancillary(
    date: pd.Timestamp,
    anc_clearing: pd.DataFrame,
    headroom_fracs: dict[str, float],
    P_bar: float,
    soc_by_efa: dict[int, float],   # efa_block (1-6) -> SoC at block start (MWh)
    E_min: float,
    eta_d: float,
    products: list[str] | None = None,
) -> dict[str, float]:
    """
    Total daily ancillary revenue for a given date, summed over all EFA blocks.

    Parameters
    ----------
    date         : the date to compute (pd.Timestamp or date)
    anc_clearing : full NESO clearing DataFrame (date, efa_block, product, direction, ...)
    soc_by_efa   : EFA block number (1-6) -> SoC (MWh) at start of that block
    """
    if products is None:
        products = list(headroom_fracs.keys())

    date_ts = pd.Timestamp(date).normalize()
    day_anc = anc_clearing[anc_clearing["date"].dt.normalize() == date_ts]

    daily = {p: 0.0 for p in products}

    # efa_block=0 occurs when deliveryStart is at 00:00 local (BST summer),
    # representing EFA 1 — mutually exclusive with efa_block=1 on almost all days.
    for efa in range(0, 7):
        block_anc = day_anc[day_anc["efa_block"] == efa]
        if block_anc.empty:
            continue
        E_curr = soc_by_efa.get(efa, (E_min + P_bar) / 2)  # fallback: midpoint
        block_rev = compute_block_ancillary(
            block_anc, headroom_fracs, P_bar, E_curr, E_min, eta_d, products,
        )
        for prod, rev in block_rev.items():
            daily[prod] = daily.get(prod, 0.0) + rev

    return daily


def hh_to_efa(settlement_period: int) -> int:
    """
    Map Elexon settlement period (1-48) to EFA block (1-6).

    EFA block boundaries (UTC):
      EFA 1: 23:00-03:00  → SP 47-48 (prev day) + SP 1-6
      EFA 2: 03:00-07:00  → SP 7-14
      EFA 3: 07:00-11:00  → SP 15-22
      EFA 4: 11:00-15:00  → SP 23-30
      EFA 5: 15:00-19:00  → SP 31-38
      EFA 6: 19:00-23:00  → SP 39-46
      SP 47-48 → EFA 1 of next day (treated as EFA 1 of current day here)
    """
    sp = int(settlement_period)
    if sp <= 6:
        return 1
    elif sp <= 14:
        return 2
    elif sp <= 22:
        return 3
    elif sp <= 30:
        return 4
    elif sp <= 38:
        return 5
    elif sp <= 46:
        return 6
    else:  # SP 47-48
        return 1
