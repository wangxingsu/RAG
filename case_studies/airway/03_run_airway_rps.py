from __future__ import annotations

import argparse
from pathlib import Path

from airway_paths import (
    DE_OUTPUT_DIR,
    RPS_OUTPUT_DIR,
    ensure_output_dirs,
    find_latest_all_cluster_diagnostics,
)
from workflow.compute_rps_marker_uniqueness import run_rps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Airway RPS ranking from DE significant markers.")
    parser.add_argument("--de-sig", type=Path, default=DE_OUTPUT_DIR / "DE_sig_up_all_groups.csv")
    parser.add_argument("--cluster-diagnostics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=RPS_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    if not args.de_sig.exists():
        raise FileNotFoundError(f"Missing {args.de_sig}. Run 02_run_airway_de.py first.")

    diagnostics = args.cluster_diagnostics or find_latest_all_cluster_diagnostics()
    if not diagnostics.exists():
        raise FileNotFoundError(diagnostics)

    run_rps(de_sig=args.de_sig, cluster_diagnostics=diagnostics, output_dir=args.output_dir)
    print(f"[RPS] Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
