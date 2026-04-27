"""Generate data/processed/sim_bundle.pkl for Phase 4+ notebook runs.

Usage:
    python scripts/generate_sim_bundle.py --paths 1000 --steps 17520

This script loads calibrated parameter JSON files from data/processed when
available, simulates the joint state paths, and pickles the PathBundle.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.processes.ancillary import AncillaryParams
from src.processes.hpfc import HPFCParams
from src.processes.imbalance import ImbalanceParams
from src.processes.schwartz_smith import SSParams
from src.processes.simulate import default_params_from_config, simulate


def load_params(processed_dir: Path):
    ss_p, hpfc_p, imb_p, anc_p = default_params_from_config()

    ss_path = processed_dir / "ss_params.json"
    hpfc_path = processed_dir / "pca_params.json"
    imb_path = processed_dir / "imbalance_params.json"
    anc_path = processed_dir / "ancillary_params.json"

    if ss_path.exists():
        ss_p = SSParams.from_json(ss_path)
    if hpfc_path.exists():
        hpfc_p = HPFCParams.from_json(hpfc_path)
    if imb_path.exists():
        imb_p = ImbalanceParams.from_json(imb_path)
    if anc_path.exists():
        anc_p = AncillaryParams.from_json(anc_path)

    return ss_p, hpfc_p, imb_p, anc_p


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cached simulation bundle.")
    parser.add_argument("--paths", type=int, default=1000, help="Number of Monte Carlo paths.")
    parser.add_argument("--steps", type=int, default=17_520, help="Number of half-hour steps.")
    parser.add_argument("--seed", type=int, default=42, help="Simulation random seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "sim_bundle.pkl",
        help="Output pickle path.",
    )
    args = parser.parse_args()

    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    dt = 1 / (365 * 48)
    ss_p, hpfc_p, imb_p, anc_p = load_params(processed_dir)

    print(f"Simulating {args.paths:,} paths x {args.steps:,} steps...")
    bundle = simulate(
        ss_p,
        hpfc_p,
        imb_p,
        anc_p,
        n_paths=args.paths,
        n_steps=args.steps,
        dt=dt,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

