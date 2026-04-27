"""
src/asset/degradation.py
========================
Degradation model for LFP BESS.

Two degradation mechanisms (CLAUDE.md §Degradation model):

  1. Calendar fade (Arrhenius):
       fade_cal = A_cal * exp(-Ea / (R * T_K)) * soc_stress_factor * dt_years

  2. Cycle fade (Wöhler / power law in DoD):
       fade_cyc = (DoD^beta) * n_cycles / N_f_ref

     where beta=2.3 for LFP (vs 1.5-2.0 for NMC) and N_f_ref=3000 at 100% DoD.

  Total SoH decline: delta_SoH = -(fade_cal + fade_cyc)

Shadow price of degradation:
  lambda_deg = (replacement_capex GBP/MWh) * (delta_SoH / delta_throughput_MWh)
  Used as a per-MWh throughput cost in the dispatch optimiser.

Rainflow cycle counting:
  Half-cycle counting on SoC(t) series to extract (DoD, mean_SoC) pairs
  for accurate cycle fade computation.

References
----------
CLAUDE.md §Degradation model
Shi, Xu & Baldick (2019) — convex cycle-based degradation cost (IEEE T-SG)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DegradationResult:
    """Outcome of one degradation computation period."""
    fade_calendar:  float   # SoH loss from calendar ageing
    fade_cycle:     float   # SoH loss from cycling
    fade_total:     float   # sum of above
    soh_new:        float   # updated SoH
    lambda_deg:     float   # shadow price GBP/MWh


# ---------------------------------------------------------------------------
# Rainflow cycle counting
# ---------------------------------------------------------------------------

def rainflow_count(soc: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    Simplified half-cycle rainflow counting on a SoC (or signal) series.

    Returns list of (amplitude, mean, count) tuples.
    amplitude = DoD (0..1) as fraction of nameplate
    mean      = mean SoC during half-cycle

    Algorithm: ASTM E1049-85 simplified half-cycle (stack-based).
    For full BESS simulations, amplitude = (peak - valley) / E_nameplate.
    """
    # Extract turning points (local minima/maxima)
    if len(soc) < 2:
        return []

    turning = [soc[0]]
    for i in range(1, len(soc) - 1):
        if (soc[i] - soc[i-1]) * (soc[i+1] - soc[i]) < 0:
            turning.append(soc[i])
    turning.append(soc[-1])

    # Stack-based half-cycle extraction
    stack: List[float] = []
    cycles: List[Tuple[float, float, float]] = []

    for val in turning:
        stack.append(val)
        while len(stack) >= 3:
            # Check the last three points
            a, b, c = stack[-3], stack[-2], stack[-1]
            amp_ab = abs(b - a)
            amp_bc = abs(c - b)
            if amp_bc >= amp_ab:
                # Extract half-cycle [a, b]
                dod  = amp_ab   # DoD in MWh (normalise later)
                mean = (a + b) / 2.0
                cycles.append((dod, mean, 0.5))
                stack.pop(-2)   # remove b from stack
                stack.pop(-2)   # remove a from stack
            else:
                break

    # Remaining residue — each contributes 0.5 cycle
    for i in range(len(stack) - 1):
        dod  = abs(stack[i+1] - stack[i])
        mean = (stack[i] + stack[i+1]) / 2.0
        cycles.append((dod, mean, 0.5))

    return cycles


def efc_from_soc(soc: np.ndarray, e_nameplate: float) -> float:
    """
    Compute equivalent full cycles (EFC) from a SoC trajectory.
    EFC = total_throughput / (2 * E_nameplate).
    """
    throughput = np.sum(np.abs(np.diff(soc)))   # MWh round-trip
    return throughput / (2.0 * e_nameplate) if e_nameplate > 0 else 0.0


# ---------------------------------------------------------------------------
# Degradation model
# ---------------------------------------------------------------------------

class DegradationModel:
    """
    LFP battery degradation model.

    Implements Arrhenius calendar fade, Wöhler cycle fade, and rainflow
    cycle counting.  Provides the shadow price of degradation for use
    in the LSMC dispatch objective.

    Parameters
    ----------
    deg_cfg  : DEGRADATION dict from src.config
    asset_cfg : ASSET dict from src.config
    """

    def __init__(self, deg_cfg: dict, asset_cfg: dict) -> None:
        d = deg_cfg
        self.A_cal      = float(d.get("A_cal",     4.14e-10))
        self.Ea         = float(d.get("Ea_cal",    2.47e4))
        self.R          = float(d.get("R_gas",     8.314))
        self.T_ref_C    = float(d.get("T_ref_celsius", 20.0))
        self.soc_stress = float(d.get("soc_stress_coeff", 0.5))
        self.soc_ref    = float(d.get("soc_stress_ref",   0.5))
        self.beta       = float(d.get("beta",      2.3))
        self.N_f_ref    = float(d.get("N_f_ref",   3000.0))
        self.lambda_init = float(d.get("lambda_deg_init_gbp_mwh", 6.0))

        self.E_name     = float(asset_cfg["energy_mwh"])
        self.capex_kwh  = float(asset_cfg.get("capex_gbp_kwh", 220.0))
        self.augment_kwh = float(asset_cfg.get("augment_gbp_kwh", 60.0))

    # ------------------------------------------------------------------
    # Calendar fade
    # ------------------------------------------------------------------

    def calendar_fade_rate(
        self,
        avg_soc:      float,
        temp_celsius:  float = 20.0,
    ) -> float:
        """
        Annual calendar capacity loss rate (fraction/year).

        Arrhenius temperature dependence × SoC stress factor.
        High SoC storage accelerates LFP degradation.
        """
        T_K     = temp_celsius + 273.15
        rate    = self.A_cal * np.exp(-self.Ea / (self.R * T_K))
        sf      = 1.0 + self.soc_stress * (avg_soc - self.soc_ref)
        return rate * sf   # fraction / year

    def calendar_fade(
        self,
        dt_years:     float,
        avg_soc:      float,
        temp_celsius:  float = 20.0,
    ) -> float:
        """Capacity loss from calendar ageing over dt_years."""
        return self.calendar_fade_rate(avg_soc, temp_celsius) * dt_years

    # ------------------------------------------------------------------
    # Cycle fade
    # ------------------------------------------------------------------

    def cycle_fade_per_efc(self, dod_frac: float) -> float:
        """
        Capacity loss per EFC at given DoD (Wöhler power law).

        cycle_fade = (DoD^beta) / N_f_ref
        """
        return (dod_frac ** self.beta) / self.N_f_ref

    def cycle_fade_from_soc(
        self,
        soc_series:  np.ndarray,   # MWh
        soh_current: float = 1.0,
    ) -> float:
        """
        Compute total cycle fade from a SoC trajectory using rainflow.

        Each half-cycle contributes: (DoD_frac^beta) * count / N_f_ref.
        DoD_frac = DoD_MWh / (E_nameplate * SoH).
        """
        usable = self.E_name * soh_current
        if usable <= 0:
            return 0.0

        cycles = rainflow_count(soc_series)
        fade   = 0.0
        for dod_mwh, _, count in cycles:
            dod_frac = min(1.0, dod_mwh / usable)
            fade    += (dod_frac ** self.beta) * count / self.N_f_ref

        return fade

    def cycle_fade_from_efc(
        self,
        efc:         float,
        avg_dod_frac: float = 0.80,
    ) -> float:
        """Approximate cycle fade from EFC and average DoD."""
        return (avg_dod_frac ** self.beta) * efc / self.N_f_ref

    # ------------------------------------------------------------------
    # Shadow price
    # ------------------------------------------------------------------

    def shadow_price(
        self,
        soh:         float,
        dod_frac:    float = 0.80,
        fin_cfg:     Optional[dict] = None,
    ) -> float:
        """
        Shadow price of degradation (GBP/MWh throughput).

        lambda_deg = replacement_capex_gbp_mwh * d(SoH loss) / d(throughput_MWh)

        Replacement capex: augment_gbp_kwh if SoH > trigger, else capex_gbp_kwh.
        Uses the cycle fade gradient: d_fade/d_EFC = (DoD^beta) / N_f_ref.
        """
        # Replacement cost per MWh of usable capacity
        repl_capex_kwh = self.augment_kwh   # GBP/kWh for augmentation
        repl_gbp_mwh   = repl_capex_kwh * 1000.0 * soh   # GBP per MWh usable

        # d(SoH_loss) per MWh throughput
        # EFC = throughput / (2 * E_nameplate)
        # fade = (DoD^beta) * EFC / N_f_ref
        # d_fade/d_throughput = (DoD^beta) / (N_f_ref * 2 * E_nameplate)
        d_fade_per_mwh = (dod_frac ** self.beta) / (self.N_f_ref * 2.0 * self.E_name)

        # Shadow price = replacement_capex * d(usable_lost) / d(throughput)
        # d(usable_lost) = d_fade * E_nameplate
        d_usable_per_mwh = d_fade_per_mwh * self.E_name   # fraction → MWh

        lambda_deg = repl_capex_kwh * 1000.0 * d_usable_per_mwh   # GBP/MWh

        # Clip to reasonable range
        return float(np.clip(lambda_deg, 1.0, 50.0))

    # ------------------------------------------------------------------
    # SoH update
    # ------------------------------------------------------------------

    def update_soh(
        self,
        soh:          float,
        dt_years:     float,
        avg_soc_frac: float,
        soc_series:   Optional[np.ndarray] = None,
        efc:          float = 0.0,
        avg_dod:      float = 0.80,
        temp_celsius:  float = 20.0,
    ) -> DegradationResult:
        """
        Compute degradation over a period and return updated SoH.

        Parameters
        ----------
        soh          : current SoH (fraction)
        dt_years     : elapsed time (years)
        avg_soc_frac : average SoC fraction during period (for calendar fade stress)
        soc_series   : SoC trajectory in MWh (for rainflow); if None, use efc
        efc          : equivalent full cycles if soc_series not provided
        avg_dod      : average DoD fraction (used if soc_series is None)
        temp_celsius : average temperature

        Returns
        -------
        DegradationResult
        """
        # Calendar
        fade_cal = self.calendar_fade(dt_years, avg_soc_frac, temp_celsius)

        # Cycle
        if soc_series is not None and len(soc_series) > 2:
            fade_cyc = self.cycle_fade_from_soc(soc_series, soh)
        else:
            fade_cyc = self.cycle_fade_from_efc(efc, avg_dod)

        fade_total = fade_cal + fade_cyc
        soh_new    = max(0.0, soh - fade_total)

        lam = self.shadow_price(soh)

        return DegradationResult(
            fade_calendar = fade_cal,
            fade_cycle    = fade_cyc,
            fade_total    = fade_total,
            soh_new       = soh_new,
            lambda_deg    = lam,
        )

    # ------------------------------------------------------------------
    # Augmentation check
    # ------------------------------------------------------------------

    def needs_augmentation(
        self,
        soh:         float,
        trigger:     Optional[float] = None,
        asset_cfg:   Optional[dict]  = None,
    ) -> bool:
        """Return True if SoH has fallen below the augmentation trigger."""
        if trigger is None and asset_cfg is not None:
            trigger = asset_cfg.get("soh_augment_trigger", 0.82)
        elif trigger is None:
            trigger = 0.82
        return soh < trigger

    def augment(self, soh: float) -> float:
        """Reset SoH to 1.0 after augmentation (capacity replacement)."""
        return 1.0

    # ------------------------------------------------------------------
    # Life-time simulation
    # ------------------------------------------------------------------

    def simulate_soh_trajectory(
        self,
        efc_per_year: float,
        life_years:   int,
        avg_soc_frac: float = 0.50,
        avg_dod:      float = 0.80,
        temp_celsius:  float = 20.0,
        augment_years: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate annual SoH trajectory over the asset life.

        Returns (years, soh_array) for plotting.
        Augmentations reset SoH to 1.0 at specified years.
        """
        soh_arr = np.zeros(life_years + 1)
        soh_arr[0] = 1.0
        soh = 1.0

        for yr in range(1, life_years + 1):
            res = self.update_soh(
                soh=soh, dt_years=1.0, avg_soc_frac=avg_soc_frac,
                efc=efc_per_year, avg_dod=avg_dod, temp_celsius=temp_celsius,
            )
            soh = res.soh_new

            # Augmentation at specified years
            if augment_years and yr in augment_years:
                soh = 1.0

            soh_arr[yr] = soh

        years = np.arange(life_years + 1, dtype=float)
        return years, soh_arr
