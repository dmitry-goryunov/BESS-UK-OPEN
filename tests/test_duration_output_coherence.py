import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"


def read_json(name):
    with (OUT / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def duration_metrics(duration_h):
    label = f"{duration_h}h"
    lsmc = read_json(f"lsmc_valuation_summary_{label}.json")
    mtm = read_json(f"mtm_summary_{label}.json")["mtm"]
    return {
        "label": label,
        "lsmc_annual_mean": lsmc["mtm_gbp_annualized"]["mean"],
        "merchant": mtm["merchant"],
        "augmentation": mtm["augmentation"],
        "total": mtm["total_mean"],
    }


def test_duration_outputs_are_available_for_1h_2h_4h():
    for duration_h in (1, 2, 4):
        label = f"{duration_h}h"
        for stem in (
            "lsmc_valuation_summary",
            "mtm_summary",
            "phase6_summary",
            "perfect_foresight_summary",
        ):
            assert (OUT / f"{stem}_{label}.json").exists()


def test_same_power_gross_merchant_value_is_duration_coherent():
    rows = {duration_h: duration_metrics(duration_h) for duration_h in (1, 2, 4)}

    assert rows[4]["merchant"] >= rows[2]["merchant"], (
        "4h gross merchant value is below 2h for the same 100 MW asset. "
        "This suggests the LSMC policy/run is not duration coherent before MTM costs. "
        f"merchant GBP/MW/yr: 1h={rows[1]['merchant']:,.0f}, "
        f"2h={rows[2]['merchant']:,.0f}, 4h={rows[4]['merchant']:,.0f}; "
        f"totals: 1h={rows[1]['total']:,.0f}, "
        f"2h={rows[2]['total']:,.0f}, 4h={rows[4]['total']:,.0f}."
    )
