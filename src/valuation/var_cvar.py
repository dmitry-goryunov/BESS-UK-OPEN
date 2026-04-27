"""VaR/CVaR and simple scenario helpers for Phase 5 notebooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass
class RiskMetrics:
    alpha: float
    mean_gbp: float
    std_gbp: float
    var_gbp: float
    cvar_gbp: float
    p5_gbp: float
    p50_gbp: float
    p95_gbp: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenarioResult:
    name: str
    stress_mtm_mean: float
    delta_gbp: float
    delta_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def _paths_from_mtm(mtm) -> np.ndarray:
    paths = np.asarray(getattr(mtm, "mtm_paths", []), dtype=float)
    if paths.size == 0:
        paths = np.asarray([float(getattr(mtm, "mtm_mean", 0.0))], dtype=float)
    return paths


def compute_risk_metrics(mtm, alpha: float = 0.95) -> RiskMetrics:
    paths = _paths_from_mtm(mtm)
    losses = -paths
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return RiskMetrics(
        alpha=float(alpha),
        mean_gbp=float(paths.mean()),
        std_gbp=float(paths.std()),
        var_gbp=var,
        cvar_gbp=cvar,
        p5_gbp=float(np.percentile(paths, 5)),
        p50_gbp=float(np.percentile(paths, 50)),
        p95_gbp=float(np.percentile(paths, 95)),
    )


def risk_metrics_multi_alpha(mtm, alphas: Iterable[float] = (0.90, 0.95, 0.99)) -> Dict[float, RiskMetrics]:
    return {float(a): compute_risk_metrics(mtm, float(a)) for a in alphas}


def print_risk_summary(rm: RiskMetrics) -> None:
    print(
        f"alpha={rm.alpha:.0%}  VaR=GBP {rm.var_gbp:,.0f}  "
        f"CVaR=GBP {rm.cvar_gbp:,.0f}  mean=GBP {rm.mean_gbp:,.0f}"
    )


def run_scenarios(mtm, alpha: float = 0.95) -> Dict[str, ScenarioResult]:
    base = float(getattr(mtm, "mtm_mean", 0.0))
    shocks = {
        "High price": 0.15,
        "Low price": -0.15,
        "High volatility": 0.08,
        "Low ancillary": -0.07,
        "High discount": -0.05,
    }
    return {
        name: ScenarioResult(
            name=name,
            stress_mtm_mean=base * (1.0 + shock),
            delta_gbp=base * shock,
            delta_pct=shock * 100.0,
        )
        for name, shock in shocks.items()
    }


def print_scenario_table(scenarios: Dict[str, ScenarioResult]) -> None:
    print(f"{'Scenario':<20} {'MTM GBP':>16} {'Delta %':>10}")
    for s in scenarios.values():
        print(f"{s.name:<20} {s.stress_mtm_mean:>16,.0f} {s.delta_pct:>9.1f}%")


def scenarios_to_dict(scenarios: Dict[str, ScenarioResult]) -> Dict[str, dict]:
    return {name: result.to_dict() for name, result in scenarios.items()}

