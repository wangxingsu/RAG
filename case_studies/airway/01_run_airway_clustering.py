from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from airway_paths import (
    RAW_DATA_DIR,
    CLUSTERING_OUTPUT_DIR,
    RAG_SCRIPT,
    REPO_ROOT,
    ensure_output_dirs,
    find_latest_airway_h5ad,
    find_latest_generated_airway_h5ad,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-aligned RAG clustering pipeline for Airway only.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--reuse-existing", action="store_true", help="Record the latest existing airway h5ad instead of rerunning.")
    return parser.parse_args()


def run_clustering(python_exe: str) -> Path:
    cmd = [
        python_exe,
        str(RAG_SCRIPT),
        "--dataset",
        "airway",
        "--data-dir",
        str(RAW_DATA_DIR),
        "--output-dir",
        str(CLUSTERING_OUTPUT_DIR),
    ]
    print("[Clustering] Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return find_latest_generated_airway_h5ad()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    if args.reuse_existing:
        h5ad = find_latest_airway_h5ad()
    else:
        h5ad = run_clustering(args.python)
    print(f"[Clustering] Airway h5ad: {h5ad}")


if __name__ == "__main__":
    main()
