"""
src/asset/battery.py
====================
BatteryAsset — state machine for a grid-scale LFP BESS.

Tracks SoC (MWh) and SoH (fraction) through each dispatch step.
Used by the LSMC forward pass and the backtest engine.

State variables
---------------
    E_mwh   : stored energy (MWh), bounded by [soc_min, soc_max * SoH]
    SoH     : state of health (0..1), degrading over time

Dispatch action
---------------
    net_mw  : net power (MW), positive = discharge, negative = charge
    r_dc_mw : DC ancillary reserved (MW)
    r_qr_mw : QR ancillary reserved (MW)

Physics
-------
    E_next = E + (-net_mw * eta_eff - aux_load) * dt_h
    where eta_eff = eta_c if net < 0 (charging), 1/eta_d if net > 0 (discharging)

References
----------
CLAUDE.md §Asset envelope (reference case)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BatteryState:
    """Snapshot of battery state at a single time step."""
    E_mwh:    float    # stored energy (MWh)
    SoH:      float    # state of health (fraction, 0..1)
    efc:      float    # cumulative equivalent full cycles
    t_years:  float    # elapsed time from commissioning (years)

    def soc_frac(self, energy_mwh: float) -> float:
        """State of charge as fraction of nameplate capacity."""
        return self.E_mwh / energy_mwh

    def usable_mwh(self, energy_mwh: float, soc_min_frac: float, soc_max_frac: float) -> float:
        """Usable energy window (MWh) given current SoH."""
        e_max = energy_mwh * soc_max_frac * self.SoH
        e_min = energy_mwh * soc_min_frac
        return max(0.0, e_max - e_min)


@dataclass
class DispatchAction:
    """Single half-hour dispatch decision."""
    net_mw:   float    # net power (positive = discharge, negative = charge)
    r_dc_mw:  float    # DC reserved (MW)
    r_qr_mw:  float    # QR reserved (MW)


@dataclass
class StepResult:
    """Outcome of one dispatch step."""
    state_next:   BatteryState
    cashflow_gbp: float
    feasible:     bool
    efc_delta:    float    # equivalent full cycles consumed this step
    throughput_mwh: float  # |net_mw| * dt_h


# ---------------------------------------------------------------------------
# BatteryAsset
# ---------------------------------------------------------------------------

class BatteryAsset:
    """
    Grid-scale LFP BESS — physics and feasibility layer.

    Parameters
    ----------
    asset_cfg : ASSET dict from src.config
    """

    def __init__(self, asset_cfg: dict) -> None:
        cfg = asset_cfg
        self.P_mw         = float(cfg["power_mw"])
        self.E_mwh        = float(cfg["energy_mwh"])
        self.eta_c        = float(cfg["eta_charge"])
        self.eta_d        = float(cfg["eta_discharge"])
        self.soc_min_frac = float(cfg["soc_min_frac"])
        self.soc_max_frac = float(cfg["soc_max_frac"])
        self.aux_mw       = float(cfg.get("aux_load_mw", 0.70))
        self.availability = float(cfg.get("availability", 0.96))
        self.vom          = float(cfg.get("vom_gbp_mwh", 1.2))
        self.c_rate_max   = float(cfg.get("c_rate_max", 1.0))

    # ------------------------------------------------------------------
    # State bounds
    # ------------------------------------------------------------------

    def e_min(self) -> float:
        """Minimum stored energy (MWh) — SoC floor."""
        return self.E_mwh * self.soc_min_frac

    def e_max(self, soh: float) -> float:
        """Maximum stored energy (MWh) — SoC ceiling × SoH."""
        return self.E_mwh * self.soc_max_frac * soh

    def max_discharge_mw(self, E: float, soh: float, dt_h: float) -> float:
        """
        Maximum discharge power given SoC and SoH.
        Limited by power rating and available energy.
        """
        energy_headroom = max(0.0, E - self.e_min()) / dt_h * self.eta_d
        return min(self.P_mw, energy_headroom)

    def max_charge_mw(self, E: float, soh: float, dt_h: float) -> float:
        """
        Maximum charge power given SoC and SoH.
        Limited by power rating and available space.
        """
        space_headroom = max(0.0, self.e_max(soh) - E) / dt_h / self.eta_c
        return min(self.P_mw, space_headroom)

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def is_feasible(
        self,
        E:       float,
        soh:     float,
        action:  DispatchAction,
        dt_h:    float = 0.5,
    ) -> bool:
        """
        Check that the dispatch action is physically feasible.

        Constraints:
        1. Power headroom: |net| + r_dc + r_qr ≤ P_bar
        2. Discharge headroom: E - (net + r_dc + r_qr) * dt_h / eta_d ≥ E_min
        3. Charge headroom: E + |net| * eta_c * dt_h ≤ E_max
        4. Non-negativity: r_dc, r_qr ≥ 0
        """
        net = action.net_mw
        rdc = action.r_dc_mw
        rqr = action.r_qr_mw

        # 1. Total power
        if abs(net) + rdc + rqr > self.P_mw + 1e-6:
            return False
        # 2. Non-negativity
        if rdc < -1e-6 or rqr < -1e-6:
            return False
        # 3. SoC bounds after step
        E_next = self._next_e(E, net, dt_h)
        if E_next < self.e_min() - 1e-4:
            return False
        if E_next > self.e_max(soh) + 1e-4:
            return False
        return True

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _next_e(self, E: float, net_mw: float, dt_h: float) -> float:
        """
        Compute next stored energy.

        Discharge (net > 0): E_next = E - net_mw / eta_d * dt_h
        Charge    (net < 0): E_next = E - net_mw * eta_c * dt_h
                             (net_mw is negative so E increases)
        """
        if net_mw >= 0:   # discharging or idle
            return E - net_mw / self.eta_d * dt_h
        else:             # charging
            return E - net_mw * self.eta_c * dt_h   # net_mw < 0, so E increases

    def step(
        self,
        state:   BatteryState,
        action:  DispatchAction,
        prices:  dict,            # keys: P_da, delta_imb, pi_dc, pi_qr
        deg_cfg: dict,
        dt_h:    float = 0.5,
    ) -> StepResult:
        """
        Advance the battery by one half-hour step.

        Parameters
        ----------
        state   : current BatteryState
        action  : DispatchAction
        prices  : dict with P_da, delta_imb, pi_dc, pi_qr (all GBP/MWh or GBP/MW/h)
        deg_cfg : DEGRADATION dict
        dt_h    : step size in hours (0.5 for HH)

        Returns
        -------
        StepResult
        """
        E      = state.E_mwh
        soh    = state.SoH
        net    = action.net_mw
        rdc    = action.r_dc_mw
        rqr    = action.r_qr_mw

        feasible = self.is_feasible(E, soh, action, dt_h)

        # Next SoC
        E_next = np.clip(self._next_e(E, net, dt_h),
                         self.e_min(), self.e_max(soh))

        # Throughput (MWh per step)
        throughput = abs(net) * dt_h

        # EFC delta: throughput / (2 * E_nameplate)
        efc_delta = throughput / (2.0 * self.E_mwh) if self.E_mwh > 0 else 0.0

        # Cashflow (GBP)
        P_da    = prices.get("P_da", 0.0)
        delta   = prices.get("delta_imb", 0.0)
        pi_dc   = prices.get("pi_dc", 0.0)
        pi_qr   = prices.get("pi_qr", 0.0)

        # Wholesale: discharge earns P_da + delta, charge pays P_da + delta
        # Imbalance uplift only on net discharge (asymmetric — BESS takes imbalance on discharge)
        imb_uplift = delta * max(0.0, net) * dt_h if net > 0 else 0.0
        wholesale  = net * P_da * dt_h + imb_uplift

        ancillary  = (rdc * pi_dc + rqr * pi_qr) * dt_h
        vom_cost   = self.vom * throughput
        deg_cost   = deg_cfg.get("lambda_deg_init_gbp_mwh", 6.0) * throughput
        aux_cost   = self.aux_mw * P_da * dt_h   # auxiliary load cost

        cashflow = wholesale + ancillary - vom_cost - deg_cost - aux_cost

        # Next SoH (full degradation handled separately in DegradationModel)
        state_next = BatteryState(
            E_mwh   = E_next,
            SoH     = soh,     # updated by DegradationModel.update_soh()
            efc     = state.efc + efc_delta,
            t_years = state.t_years + dt_h / 8760.0,
        )

        return StepResult(
            state_next    = state_next,
            cashflow_gbp  = cashflow,
            feasible      = feasible,
            efc_delta     = efc_delta,
            throughput_mwh = throughput,
        )

    # ------------------------------------------------------------------
    # Intrinsic value (deterministic DA-only LP)
    # ------------------------------------------------------------------

    def intrinsic_value(
        self,
        da_prices:  np.ndarray,   # (T,) GBP/MWh
        E_init:     float,
        dt_h:       float = 0.5,
        soh:        float = 1.0,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute deterministic DA-only intrinsic value via LP.

        Returns (total_revenue, dispatch_mw, soc_mwh).
        Wraps rolling_intrinsic.solve_daily_lp for a single window.
        """
        from src.optimisation.rolling_intrinsic import solve_daily_lp

        T = len(da_prices)
        rev, d_opt, c_opt = solve_daily_lp(
            prices    = da_prices,
            E_init    = E_init,
            E_min     = self.e_min(),
            E_max     = self.e_max(soh),
            P_bar     = self.P_mw,
            eta_c     = self.eta_c,
            eta_d     = self.eta_d,
            dt_h      = dt_h,
        )

        # Reconstruct SoC path
        net    = d_opt - c_opt
        soc    = np.empty(T + 1)
        soc[0] = E_init
        for t in range(T):
            soc[t + 1] = self._next_e(soc[t], net[t], dt_h)

        return rev, net, soc

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "power_mw":      self.P_mw,
            "energy_mwh":    self.E_mwh,
            "duration_h":    self.E_mwh / self.P_mw,
            "eta_charge":    self.eta_c,
            "eta_discharge": self.eta_d,
            "rte":           self.eta_c * self.eta_d,
            "soc_min_frac":  self.soc_min_frac,
            "soc_max_frac":  self.soc_max_frac,
            "aux_mw":        self.aux_mw,
            "vom_gbp_mwh":   self.vom,
        }


# ---------------------------------------------------------------------------
# Convenience: initial state
# ---------------------------------------------------------------------------

def initial_state(asset_cfg: dict, t_years: float = 0.0) -> BatteryState:
    """Construct initial BatteryState from asset config."""
    E_init = asset_cfg["energy_mwh"] * asset_cfg.get("soc_init_frac", 0.50)
    return BatteryState(E_mwh=E_init, SoH=1.0, efc=0.0, t_years=t_years)
