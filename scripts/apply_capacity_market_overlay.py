#!/usr/bin/env python
"""Apply a deterministic Capacity Market overlay to Phase 4 LSMC outputs.

The LSMC dispatch model does not optimise CM availability. CM is therefore a
post-dispatch annual revenue overlay, expressed in GBPk/MW/year to match the
Phase 4 comparison CSV convention.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASE_METHOD = "Forward simulation (LSMC)"
CM_METHOD = "Forward simulation (LSMC + CM overlay)"

# Central placeholder pending a CM register extract.  For a 100 MW asset this is
# GBP0.6m/year; the value is added outside dispatch and outside LSMC training.
CM_GBP_PER_MW_YEAR_K = 6.0


def _read_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    comparison_json = PROCESSED / "phase4_all_durations_comparison.json"
    if not comparison_json.exists():
        raise FileNotFoundError(f"Missing {comparison_json}")

    rows = _read_rows(comparison_json)
    base_rows = [r for r in rows if r.get("method") == BASE_METHOD]
    if not base_rows:
        raise ValueError(f"No {BASE_METHOD!r} rows found in {comparison_json}")

    cleaned = [r for r in rows if r.get("method") != CM_METHOD]
    overlay_rows: list[dict] = []

    for row in base_rows:
        power_mw = float(row.get("asset_power_mw", 100.0))
        cm_value_m = CM_GBP_PER_MW_YEAR_K * power_mw / 1000.0
        base_value_m = float(row["value_gbp_annualized_m"])
        base_k = float(row.get("gbp_per_mw_year_k", row.get("gbp_per_mw_year")))

        overlay = dict(row)
        overlay.update({
            "method": CM_METHOD,
            "base_method": BASE_METHOD,
            "cm_gbp_per_mw_year_k": CM_GBP_PER_MW_YEAR_K,
            "cm_value_gbp_annualized_m": cm_value_m,
            "base_value_gbp_annualized_m": base_value_m,
            "base_gbp_per_mw_year_k": base_k,
            "value_gbp_annualized_m": base_value_m + cm_value_m,
        })
        if "gbp_per_mw_year" in overlay:
            overlay["gbp_per_mw_year"] = base_k + CM_GBP_PER_MW_YEAR_K
        overlay["gbp_per_mw_year_k"] = base_k + CM_GBP_PER_MW_YEAR_K
        overlay_rows.append(overlay)

    combined = cleaned + overlay_rows
    combined.sort(key=lambda r: (float(r.get("duration_h", 0.0)), str(r.get("method", ""))))

    overlay_audit = [
        {
            "duration_h": r["duration_h"],
            "base_method": BASE_METHOD,
            "overlay_method": CM_METHOD,
            "base_gbp_per_mw_year_k": r["base_gbp_per_mw_year_k"],
            "cm_gbp_per_mw_year_k": r["cm_gbp_per_mw_year_k"],
            "total_gbp_per_mw_year_k": r["gbp_per_mw_year_k"],
            "base_value_gbp_annualized_m": r["base_value_gbp_annualized_m"],
            "cm_value_gbp_annualized_m": r["cm_value_gbp_annualized_m"],
            "total_value_gbp_annualized_m": r["value_gbp_annualized_m"],
            "note": "Deterministic CM overlay; replace 6.0k/MW/yr with CM register-derived value when available.",
        }
        for r in overlay_rows
    ]

    _write_json(PROCESSED / "phase4_all_durations_comparison.json", combined)
    _write_csv(PROCESSED / "phase4_all_durations_comparison.csv", combined)
    _write_json(PROCESSED / "capacity_market_overlay.json", overlay_audit)
    _write_csv(PROCESSED / "capacity_market_overlay.csv", overlay_audit)

    print(f"Applied CM overlay: GBP{CM_GBP_PER_MW_YEAR_K:.1f}k/MW/yr")
    for row in overlay_audit:
        print(
            f"  {row['duration_h']:g}h: "
            f"{row['base_gbp_per_mw_year_k']:.1f} + {row['cm_gbp_per_mw_year_k']:.1f} "
            f"= {row['total_gbp_per_mw_year_k']:.1f}k/MW/yr"
        )


if __name__ == "__main__":
    main()
