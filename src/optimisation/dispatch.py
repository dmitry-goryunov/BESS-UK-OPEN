"""
Co-optimisation dispatch module — Phase 4.

Enumerates feasible dispatch modes for a BESS at each half-hour, computes
immediate cashflows and SoC transitions.  The LSMC solver iterates over
these modes to find the Q-maximising action at each state.

Decision vector (all in MW at nameplate power):
    net_mw   — positive = discharge, negative = charge
    r_dc     — DC reserve headroom committed (MW)
    r_qr     — QR reserve headroom committed (MW)

Simplified for Phase 4 (DM / DR / BR folded into DC for headroom arithmetic;
added back as separate products in Phase 5 when the full revenue stack is live).

Co-optimisation constraints (from CLAUDE.md):
    |net_mw| + r_dc + r_qr  <=  P_bar × SoH        (power headroom)
    E - (r_dc + r_qr) × dt / eta_d  >=  E_min       (energy headroom up)
    E + (r_dc + r_qr) × eta_c × dt  <=  E_max(SoH)  (energy headroom dn)

Cashflow per half-hour:
    CF = P_da × net_mw × dt                          (wholesale DA)
       + delta × max(net_mw, 0) × dt                 (imbalance uplift on discharge)
       + pi_dc × r_dc × dt                           (DC clearing)
       + pi_qr × r_qr × dt                           (QR clearing)
       - deg_cost × (|net_mw| × dt)                  (cycle degradation shadow price)
       - vom × |net_mw| × dt                         (variable O&M)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class DispatchMode:
    """
    One feasible dispatch action expressed as fractions of P_bar.

    Attributes
    ----------
    net_frac  : discharge fraction (positive) or charge fraction (negative)
    r_dc_frac : DC reserve as fraction of P_bar
    r_qr_frac : QR reserve as fraction of P_bar
    """
    net_frac:   float    # -1..+1 (charge is negative)
    r_dc_frac:  float    # 0..1
    r_qr_frac:  float    # 0..1
    r_bm_frac:  float    # 0..1

    @property
    def headroom_used(self) -> float:
        return abs(self.net_frac) + self.r_dc_frac + self.r_qr_frac + self.r_bm_frac

    def __repr__(self) -> str:
        return (f"DispatchMode(net={self.net_frac:+.2f}, "
                f"dc={self.r_dc_frac:.2f}, qr={self.r_qr_frac:.2f}, "
                f"bm={self.r_bm_frac:.2f})")


def enumerate_modes(
    net_levels:  List[float] = None,
    dc_levels:   List[float] = None,
    qr_levels:   List[float] = None,
    bm_levels:   List[float] = None,
    headroom_tol: float = 1e-6,
) -> List[DispatchMode]:
    """
    Enumerate all feasible (net_frac, r_dc_frac, r_qr_frac) combinations.

    Parameters
    ----------
    net_levels  : net power fractions to enumerate (default: ±1, ±0.5, 0)
    dc_levels   : DC reserve fractions (default: 0, 0.25, 0.5, 1.0)
    qr_levels   : QR reserve fractions (default: 0, 0.25, 0.5)
    headroom_tol: numerical tolerance on headroom constraint

    Returns a list of DispatchMode objects with total headroom ≤ 1.
    """
    if net_levels is None:
        net_levels  = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    if dc_levels is None:
        dc_levels   = [0.0, 0.25, 0.5]
    if qr_levels is None:
        qr_levels   = [0.0, 0.25]
    if bm_levels is None:
        bm_levels   = [0.0, 0.25, 0.5]

    modes = []
    seen  = set()
    for net in net_levels:
        for dc in dc_levels:
            for qr in qr_levels:
                for bm in bm_levels:
                    # Headroom constraint
                    if abs(net) + dc + qr + bm > 1.0 + headroom_tol:
                        continue
                    key = (round(net, 4), round(dc, 4), round(qr, 4), round(bm, 4))
                    if key in seen:
                        continue
                    seen.add(key)
                    modes.append(
                        DispatchMode(
                            net_frac=net,
                            r_dc_frac=dc,
                            r_qr_frac=qr,
                            r_bm_frac=bm,
                        )
                    )
    return modes


# Default mode set (48 feasible modes after headroom filter)
DEFAULT_MODES: List[DispatchMode] = enumerate_modes()


# ---------------------------------------------------------------------------
# Cashflow computation — vectorised over N_paths
# ---------------------------------------------------------------------------

def cashflow_batch(
    modes:        List[DispatchMode],
    P_da:         np.ndarray,    # (N_paths,)  £/MWh
    delta_imb:    np.ndarray,    # (N_paths,)  £/MWh imbalance basis
    pi_dc:        np.ndarray,    # (N_paths,)  £/MW/h
    pi_qr:        np.ndarray,    # (N_paths,)  £/MW/h
    pi_bm:        np.ndarray,    # (N_paths,)  £/MWh
    P_bar_mw:     float,         # nameplate MW
    dt_h:         float,         # half-hour = 0.5 h
    p_activation: float = 0.12,
    deg_cost:     float = 6.0,   # £/MWh throughput degradation shadow price
    vom:          float = 1.2,   # £/MWh variable O&M
    imbalance_cashflow_mode: str = "discharge_only",
) -> np.ndarray:
    """
    Compute cashflows for all (paths × modes) pairs.

    Returns
    -------
    CF : (N_paths, N_modes)  cashflows in £ for one settlement period
    """
    N = len(P_da)
    M = len(modes)

    # Pre-stack mode parameters as arrays
    net_fracs  = np.array([m.net_frac  for m in modes], dtype=np.float32)   # (M,)
    dc_fracs   = np.array([m.r_dc_frac for m in modes], dtype=np.float32)   # (M,)
    qr_fracs   = np.array([m.r_qr_frac for m in modes], dtype=np.float32)   # (M,)
    bm_fracs   = np.array([m.r_bm_frac for m in modes], dtype=np.float32)   # (M,)

    # Broadcast: (N, 1) × (1, M) → (N, M)
    P   = P_da[:, None].astype(np.float32)        # (N, 1)
    dlt = delta_imb[:, None].astype(np.float32)   # (N, 1)
    pdc = pi_dc[:, None].astype(np.float32)       # (N, 1)
    pqr = pi_qr[:, None].astype(np.float32)       # (N, 1)
    pbm = pi_bm[:, None].astype(np.float32)       # (N, 1)

    net = net_fracs[None, :] * P_bar_mw     # (N, M) — net MW
    dc  = dc_fracs[None, :]  * P_bar_mw     # (N, M) — DC reserve MW
    qr  = qr_fracs[None, :]  * P_bar_mw     # (N, M) — QR reserve MW
    bm  = bm_fracs[None, :]  * P_bar_mw     # (N, M) — BM headroom MW

    d_mw = np.maximum(net,  0.0)   # discharge MW
    c_mw = np.maximum(-net, 0.0)   # charge MW

    # Wholesale DA revenue: discharge positive, charge negative
    wholesale = P * net * dt_h

    if imbalance_cashflow_mode == "discharge_only":
        # Imbalance uplift on discharge (system short -> positive basis)
        imb_uplift = dlt * d_mw * dt_h
    elif imbalance_cashflow_mode == "net":
        # WD-like mode: value the whole net position at DA + delta_imb.
        imb_uplift = dlt * net * dt_h
    else:
        raise ValueError(
            "imbalance_cashflow_mode must be 'discharge_only' or 'net'; "
            f"got {imbalance_cashflow_mode!r}"
        )

    # Ancillary revenues
    anc_rev = (pdc * dc + pqr * qr) * dt_h
    bm_rev  = pbm * bm * p_activation * dt_h

    # Degradation + VoM cost (on absolute throughput)
    throughput = (d_mw + c_mw) * dt_h    # MWh
    costs = (deg_cost + vom) * throughput

    CF = wholesale + imb_uplift + anc_rev + bm_rev - costs   # (N, M)
    return CF.astype(np.float32)


def cashflow_batch_components(
    modes:        List[DispatchMode],
    P_da:         np.ndarray,    # (N_paths,)  £/MWh
    delta_imb:    np.ndarray,    # (N_paths,)  £/MWh
    pi_dc:        np.ndarray,    # (N_paths,)  £/MW/h
    pi_qr:        np.ndarray,    # (N_paths,)  £/MW/h
    pi_bm:        np.ndarray,    # (N_paths,)  £/MWh
    P_bar_mw:     float,
    dt_h:         float,
    p_activation: float = 0.12,
    deg_cost:     float = 6.0,
    vom:          float = 1.2,
    imbalance_cashflow_mode: str = "discharge_only",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Same arithmetic as cashflow_batch but also returns per-source components.

    Returns
    -------
    CF         : (N, M) total cashflow
    components : dict with keys 'da', 'imbalance', 'dc', 'qr', 'costs'; each (N, M)
    """
    net_fracs = np.array([m.net_frac  for m in modes], dtype=np.float32)
    dc_fracs  = np.array([m.r_dc_frac for m in modes], dtype=np.float32)
    qr_fracs  = np.array([m.r_qr_frac for m in modes], dtype=np.float32)
    bm_fracs  = np.array([m.r_bm_frac for m in modes], dtype=np.float32)

    P   = P_da[:, None].astype(np.float32)
    dlt = delta_imb[:, None].astype(np.float32)
    pdc = pi_dc[:, None].astype(np.float32)
    pqr = pi_qr[:, None].astype(np.float32)
    pbm = pi_bm[:, None].astype(np.float32)

    net = net_fracs[None, :] * P_bar_mw
    dc  = dc_fracs[None, :]  * P_bar_mw
    qr  = qr_fracs[None, :]  * P_bar_mw
    bm  = bm_fracs[None, :]  * P_bar_mw
    d_mw = np.maximum(net,  0.0)
    c_mw = np.maximum(-net, 0.0)

    da_comp    = (P * net * dt_h).astype(np.float32)
    if imbalance_cashflow_mode == "discharge_only":
        imb_comp = (dlt * d_mw * dt_h).astype(np.float32)
    elif imbalance_cashflow_mode == "net":
        imb_comp = (dlt * net * dt_h).astype(np.float32)
    else:
        raise ValueError(
            "imbalance_cashflow_mode must be 'discharge_only' or 'net'; "
            f"got {imbalance_cashflow_mode!r}"
        )
    dc_comp    = (pdc * dc * dt_h).astype(np.float32)
    qr_comp    = (pqr * qr * dt_h).astype(np.float32)
    bm_comp    = (pbm * bm * p_activation * dt_h).astype(np.float32)
    costs_comp = ((deg_cost + vom) * (d_mw + c_mw) * dt_h).astype(np.float32)

    CF = da_comp + imb_comp + dc_comp + qr_comp + bm_comp - costs_comp
    return CF.astype(np.float32), {
        'da':        da_comp,
        'imbalance': imb_comp,
        'dc':        dc_comp,
        'qr':        qr_comp,
        'bm':        bm_comp,
        'costs':     costs_comp,
    }


# ---------------------------------------------------------------------------
# SoC transition — vectorised over paths for a single mode
# ---------------------------------------------------------------------------

def next_soc_batch(
    E_curr:     np.ndarray,   # (N_paths,) current SoC in MWh
    net_frac:   float,        # dispatch mode net fraction
    P_bar_mw:   float,
    eta_charge: float,
    eta_discharge: float,
    dt_h:       float,
    E_min:      float,
    E_max:      float,
) -> np.ndarray:
    """
    Compute next SoC for all paths under one dispatch mode.

    Returns
    -------
    E_next : (N_paths,) SoC after one HH period, clipped to [E_min, E_max]
    """
    net_mw = net_frac * P_bar_mw
    if net_mw > 0:      # discharging
        delta_E = -net_mw / eta_discharge * dt_h
    else:               # charging (or idle)
        delta_E = -net_mw * eta_charge * dt_h   # -net_mw is positive when charging
    E_next = np.clip(E_curr + delta_E, E_min, E_max)
    return E_next.astype(np.float32)


def next_soc_grid(
    E_j:         float,        # scalar SoC grid node
    net_fracs:   np.ndarray,   # (N_modes,)
    P_bar_mw:    float,
    eta_charge:  float,
    eta_discharge: float,
    dt_h:        float,
    E_min:       float,
    E_max:       float,
) -> np.ndarray:
    """
    Compute next SoC for all modes given a single starting SoC E_j.

    Returns
    -------
    E_next : (N_modes,) clipped to [E_min, E_max]
    """
    delta_E = np.where(
        net_fracs > 0,
        -net_fracs * P_bar_mw / eta_discharge * dt_h,   # discharge
        -net_fracs * P_bar_mw * eta_charge   * dt_h,    # charge (net_frac ≤ 0)
    )
    return np.clip(E_j + delta_E, E_min, E_max).astype(np.float32)


# ---------------------------------------------------------------------------
# Feasibility mask — for a given (E, SoH) state, which modes are feasible?
# ---------------------------------------------------------------------------

def feasibility_mask(
    modes:         List[DispatchMode],
    E_curr:        float,     # current SoC (MWh)
    SoH:           float,     # state of health (0..1)
    P_bar_mw:      float,
    E_min_frac:    float,     # 0.10
    E_max_frac:    float,     # 0.90
    energy_mwh:    float,     # nameplate MWh
    eta_charge:    float,
    eta_discharge: float,
    dt_h:          float,
    reserve_sustain_h: float = 0.5,   # service must be sustained for ≥ 0.5h
    sustain_bm_hh:   int   = 4,
) -> np.ndarray:
    """
    Return boolean mask (N_modes,): True if mode is feasible at this (E, SoH) state.

    Feasibility checks:
    1. Power headroom: |net| + r_dc + r_qr ≤ SoH (SoH degrades max C-rate)
    2. Energy down: E - (r_dc + r_qr) × sustain / eta_d ≥ E_min
    3. Energy up: E + (r_dc + r_qr) × eta_c × sustain ≤ E_max × SoH
    4. Net discharge: E_curr - net_mw/eta_d × dt ≥ E_min (basic SoC feasibility)
    5. Net charge: E_curr + net_mw × eta_c × dt ≤ E_max × SoH
    """
    E_min_mwh = E_min_frac * energy_mwh
    E_max_mwh = E_max_frac * energy_mwh * SoH

    mask = np.ones(len(modes), dtype=bool)
    for i, m in enumerate(modes):
        net_mw = m.net_frac  * P_bar_mw
        dc_mw  = m.r_dc_frac * P_bar_mw
        qr_mw  = m.r_qr_frac * P_bar_mw
        res_mw = dc_mw + qr_mw

        # Power headroom (SoH reduces effective max power slightly in practice)
        if m.headroom_used > SoH + 1e-6:
            mask[i] = False
            continue

        # Energy needed to sustain reserves
        E_sustain_down = res_mw * reserve_sustain_h / eta_discharge
        E_sustain_up   = res_mw * reserve_sustain_h * eta_charge
        bm_mw          = m.r_bm_frac * P_bar_mw
        E_bm_needed    = bm_mw * (sustain_bm_hh * 0.5) / eta_discharge

        if E_curr - E_sustain_down - E_bm_needed < E_min_mwh - 1e-3:
            mask[i] = False; continue
        if E_curr + E_sustain_up  > E_max_mwh + 1e-3:
            mask[i] = False; continue

        # Net SoC transition after one HH
        if net_mw > 0:
            E_after = E_curr - net_mw / eta_discharge * dt_h
        else:
            E_after = E_curr + (-net_mw) * eta_charge * dt_h

        if E_after < E_min_mwh - 1e-3:
            mask[i] = False; continue
        if E_after > E_max_mwh + 1e-3:
            mask[i] = False

    return mask
