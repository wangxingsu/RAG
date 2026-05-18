from __future__ import annotations

import argparse
from pathlib import Path

from airway_paths import DE_OUTPUT_DIR, ensure_output_dirs, find_latest_airway_h5ad
from workflow.compute_airway_de import run_airway_de


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run airway-only DE and top-marker table generation.")
    parser.add_argument("--input-h5ad", type=Path, default=None, help="Clustered airway h5ad. Defaults to latest remembered/found.")
    parser.add_argument("--output-dir", type=Path, default=DE_OUTPUT_DIR, help="DE output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    h5ad = args.input_h5ad or find_latest_airway_h5ad()
    if not h5ad.exists():
        raise FileNotFoundError(h5ad)

    run_airway_de(input_h5ad=h5ad, output_dir=args.output_dir)
    print(f"[DE] Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
