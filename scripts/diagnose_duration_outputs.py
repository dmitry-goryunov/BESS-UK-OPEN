from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
DURATIONS = (1, 2, 4)


def read_json(name: str) -> dict:
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required output: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def duration_row(duration_h: int) -> dict:
    label = f"{duration_h}h"
    lsmc = read_json(f"lsmc_valuation_summary_{label}.json")
    mtm = read_json(f"mtm_summary_{label}.json")["mtm"]
    perfect_foresight = read_json(f"perfect_foresight_summary_{label}.json")
    pf_results = perfect_foresight.get("results", {})
    pf_da = pf_results.get("DA", {})
    pf_sp = pf_results.get("SP", {})
    return {
        "duration": label,
        "asset_mwh": lsmc["asset_mwh"],
        "lsmc_annual_mean_gbp": lsmc["mtm_gbp_annualized"]["mean"],
        "merchant_gbp_mw_yr": mtm["merchant"],
        "optimiser_fee_gbp_mw_yr": mtm["optimiser_fee"],
        "opex_gbp_mw_yr": mtm["opex_fixed"],
        "augmentation_gbp_mw_yr": mtm["augmentation"],
        "total_gbp_mw_yr": mtm["total_mean"],
        "pf_da_gbp_mw_yr": pf_da.get("value_gbp_per_mw_year"),
        "pf_sp_gbp_mw_yr": pf_sp.get("value_gbp_per_mw_year"),
    }


def fmt(value: float) -> str:
    return f"{value:,.0f}"


def main() -> int:
    rows = [duration_row(h) for h in DURATIONS]

    print("Duration output coherence check")
    print(f"Source: {OUT}")
    print()
    print(
        "dur  MWh   LSMC annual GBP   merchant/MW/yr   total/MW/yr   PF DA/MW/yr   PF SP/MW/yr"
    )
    for row in rows:
        print(
            f"{row['duration']:>3} "
            f"{row['asset_mwh']:>5.0f} "
            f"{fmt(row['lsmc_annual_mean_gbp']):>17} "
            f"{fmt(row['merchant_gbp_mw_yr']):>16} "
            f"{fmt(row['total_gbp_mw_yr']):>13} "
            f"{fmt(row['pf_da_gbp_mw_yr']):>13} "
            f"{fmt(row['pf_sp_gbp_mw_yr']):>13}"
        )

    print()
    print("MTM component detail")
    print("dur   opt fee   opex   aug capex")
    for row in rows:
        print(
            f"{row['duration']:>3} "
            f"{fmt(row['optimiser_fee_gbp_mw_yr']):>9} "
            f"{fmt(row['opex_gbp_mw_yr']):>7} "
            f"{fmt(row['augmentation_gbp_mw_yr']):>11}"
        )

    merchant = {row["duration"]: row["merchant_gbp_mw_yr"] for row in rows}
    total = {row["duration"]: row["total_gbp_mw_yr"] for row in rows}

    print()
    print("Checks:")
    issues = []
    if merchant["4h"] + 1e-9 < merchant["2h"]:
        issues.append(
            "4h gross merchant value is below 2h. A 4h same-power battery should usually be able to emulate 2h dispatch."
        )
    if total["2h"] > max(total["1h"], total["4h"]) * 1.25:
        issues.append(
            "2h total is more than 25% above both 1h and 4h. The jump is driven by the LSMC/merchant leg, not dashboard arithmetic."
        )

    if issues:
        for issue in issues:
            print(f"FAIL - {issue}")
        return 1

    print("PASS - no duration coherence issue detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
