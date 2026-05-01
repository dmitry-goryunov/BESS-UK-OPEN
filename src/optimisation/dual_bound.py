"""
src/optimisation/dual_bound.py
================================
Information-relaxation benchmark for BESS LSMC.

The original implementation in this module attempted to report an
Andersen-Broadie dual bound, but the penalty was heuristic and the final result
was clamped to the LSMC estimate. That could manufacture a misleading 0% gap.

This module now computes an honest clairvoyant upper benchmark on the same
discrete dispatch modes and physical constraints used by the LSMC solver. For
each simulated path it solves a backward dynamic program with full knowledge of
that path's future prices. This is an information-relaxation benchmark, not a
rigorous martingale-penalty Andersen-Broadie proof, but it is useful because:

    gap = (V_clairvoyant - V_LSMC) / |V_LSMC|

is no longer forced to zero. If the benchmark falls below the LSMC forward
value, the result is flagged as a model/implementation diagnostic rather than
silently corrected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.processes.simulate import PathBundle
from src.optimisation.lsmc import Policy
from src.optimisation.dispatch import cashflow_batch, feasibility_mask, next_soc_grid


@dataclass
class DualBoundResult:
    """Results of the information-relaxation benchmark."""

    v_lsmc: float       # LSMC lower estimate on the same path subset
    v_dual: float       # clairvoyant upper benchmark, kept for API compatibility
    v_dual_std: float   # std of clairvoyant PV across paths
    v_ri: float         # rolling-intrinsic lower bound, if available

    gap_abs: float
    gap_pct: float
    n_paths: int
    dual_ok: bool
    threshold: float

    def summary(self) -> str:
        status = "PASS" if self.dual_ok else "REFINE"
        lines = [
            f"\n{'=' * 55}",
            "  Clairvoyant Upper-Benchmark Verification",
            f"{'=' * 55}",
            f"  V_LSMC (same paths):       GBP {self.v_lsmc:>12,.0f}",
            f"  V_upper (clairvoyant):     GBP {self.v_dual:>12,.0f}  +/- {self.v_dual_std:,.0f}",
            f"  Gap:                       GBP {self.gap_abs:>12,.0f}  ({self.gap_pct:.2%})",
            f"  Target gap:                < {self.threshold:.0%}",
            f"  Status:                    {status}",
            f"{'=' * 55}",
        ]
        if self.gap_abs < -1e-6:
            lines.extend([
                "  Diagnostic: upper benchmark is below LSMC on this subset.",
                "  Check policy/value-function consistency before interpreting the gap.",
                f"{'=' * 55}",
            ])
        return "\n".join(lines)


def _path_prices(bundle: PathBundle, n: int, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract clipped one-path market arrays for the dispatch cashflow model."""
    p_da = np.exp(np.clip(bundle.ln_P_base[n, :T], -100.0, np.log(500.0))).astype(np.float32)
    delta = np.clip(bundle.delta_imb[n, :T], -500.0, 500.0).astype(np.float32)
    pi_dc = np.clip(bundle.pi["DC_Low"][n, :T], 0.0, 100.0).astype(np.float32)
    pi_qr_src = bundle.pi.get("QR_Pos", bundle.pi["DC_Low"])
    pi_qr = np.clip(pi_qr_src[n, :T], 0.0, 100.0).astype(np.float32)
    return p_da, delta, pi_dc, pi_qr


def _clairvoyant_path_value(
    bundle: PathBundle,
    path_idx: int,
    policy: Policy,
    asset_cfg: dict,
    deg_cfg: dict,
    fin_cfg: dict,
    T: int,
) -> float:
    """Solve the full-information DP for one path on the policy SoC grid."""
    P_bar = float(asset_cfg["power_mw"])
    E_name = float(asset_cfg["energy_mwh"])
    eta_c = float(asset_cfg["eta_charge"])
    eta_d = float(asset_cfg["eta_discharge"])
    E_min = E_name * float(asset_cfg["soc_min_frac"])
    E_max = E_name * float(asset_cfg["soc_max_frac"])
    E_init = E_name * float(asset_cfg.get("soc_init_frac", 0.5))
    dt_h = float(policy.dt_h)
    deg_cost = float(deg_cfg.get("lambda_deg_init_gbp_mwh", 6.0))
    vom = float(asset_cfg.get("vom_gbp_mwh", 1.2))
    disc = float(np.exp(-float(fin_cfg["wacc_merchant"]) * dt_h / 8760.0))

    modes = policy.modes
    net_fracs = np.array([m.net_frac for m in modes], dtype=np.float32)
    soc_grid = np.asarray(policy.soc_grid, dtype=np.float32)
    p_da, delta, pi_dc, pi_qr = _path_prices(bundle, path_idx, T)

    V_next = np.zeros(len(soc_grid), dtype=np.float64)

    for t in range(T - 1, -1, -1):
        cf_all = cashflow_batch(
            modes,
            P_da=np.array([p_da[t]], dtype=np.float32),
            delta_imb=np.array([delta[t]], dtype=np.float32),
            pi_dc=np.array([pi_dc[t]], dtype=np.float32),
            pi_qr=np.array([pi_qr[t]], dtype=np.float32),
            P_bar_mw=P_bar,
            dt_h=dt_h,
            deg_cost=deg_cost,
            vom=vom,
        )[0].astype(np.float64)

        V_curr = np.empty_like(V_next)
        for j, E in enumerate(soc_grid):
            feasible = feasibility_mask(
                modes,
                float(E),
                1.0,
                P_bar,
                float(asset_cfg["soc_min_frac"]),
                float(asset_cfg["soc_max_frac"]),
                E_name,
                eta_c,
                eta_d,
                dt_h,
            )
            next_E = next_soc_grid(float(E), net_fracs, P_bar, eta_c, eta_d, dt_h, E_min, E_max)
            continuation = np.interp(next_E, soc_grid, V_next)
            q = cf_all + disc * continuation
            q[~feasible] = -1e18
            V_curr[j] = float(np.max(q))

        V_next = V_curr

    return float(np.interp(E_init, soc_grid, V_next))


def compute_dual_bound(
    bundle: PathBundle,
    policy: Policy,
    val_result,
    asset_cfg: dict,
    lsmc_cfg: dict,
    deg_cfg: dict,
    fin_cfg: dict,
    n_dual_paths: int = 200,
    threshold: float = 0.05,
    verbose: bool = True,
) -> DualBoundResult:
    """
    Compute a clairvoyant information-relaxation benchmark.

    The function keeps the historical API name `compute_dual_bound` because the
    notebooks already call it, but the reported `v_dual` value is a pathwise
    clairvoyant upper benchmark rather than a martingale-penalty dual bound.
    """
    del lsmc_cfg  # reserved for future true martingale dual implementation

    n_val = len(np.asarray(val_result.pv_paths))
    N_paths = min(int(n_dual_paths), int(bundle.n_paths), n_val)
    T = min(int(policy.n_steps), int(bundle.n_steps), int(val_result.cashflow_paths.shape[1]))

    if N_paths <= 0 or T <= 0:
        raise ValueError("compute_dual_bound requires at least one path and one time step")

    upper_pv = np.empty(N_paths, dtype=np.float64)
    for n in range(N_paths):
        upper_pv[n] = _clairvoyant_path_value(bundle, n, policy, asset_cfg, deg_cfg, fin_cfg, T)

    lsmc_subset = np.asarray(val_result.pv_paths[:N_paths], dtype=np.float64)
    v_lsmc = float(np.mean(lsmc_subset))
    v_upper = float(np.mean(upper_pv))
    v_upper_std = float(np.std(upper_pv, ddof=1)) if N_paths > 1 else 0.0

    gap_abs = v_upper - v_lsmc
    gap_pct = gap_abs / abs(v_lsmc) if v_lsmc != 0 else float("nan")
    dual_ok = bool(np.isfinite(gap_pct) and 0.0 <= gap_pct < threshold)

    result = DualBoundResult(
        v_lsmc=v_lsmc,
        v_dual=v_upper,
        v_dual_std=v_upper_std,
        v_ri=0.0,
        gap_abs=gap_abs,
        gap_pct=gap_pct,
        n_paths=N_paths,
        dual_ok=dual_ok,
        threshold=threshold,
    )

    if verbose:
        print(result.summary())

    return result
