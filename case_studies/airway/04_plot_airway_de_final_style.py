from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from airway_paths import (
    DE_OUTPUT_DIR,
    FIGURE_OUTPUT_DIR,
    MAIN_FIGURE_CLUSTER_TABLE,
    PLOT_REVIEW_DIR,
    UMAP_OBS_CSV,
    ensure_output_dirs,
    find_latest_airway_h5ad,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render airway DE-final-style UMAP plus marker panels.")
    parser.add_argument("--input-h5ad", type=Path, default=None, help="Clustered airway h5ad. Defaults to latest remembered/found.")
    parser.add_argument("--de-top", type=Path, default=DE_OUTPUT_DIR / "DE_top_up_markers_by_group.csv", help="Top upregulated marker table from 02_run_airway_de.py.")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_OUTPUT_DIR / "airway_de_final_style")
    return parser.parse_args()


def load_plot_module():
    script = PLOT_REVIEW_DIR / "plot_airway_de_final_style.py"
    spec = importlib.util.spec_from_file_location("airway_de_final_style_source", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    h5ad = args.input_h5ad or find_latest_airway_h5ad()
    if not h5ad.exists():
        raise FileNotFoundError(h5ad)
    if not args.de_top.exists():
        raise FileNotFoundError(f"Missing {args.de_top}. Run 02_run_airway_de.py first.")

    module = load_plot_module()
    module.ROOT = Path(__file__).resolve().parent
    module.CLUSTER_TABLE = MAIN_FIGURE_CLUSTER_TABLE
    module.DE_TOP_TABLE = args.de_top
    module.H5AD = h5ad
    module.UMAP_CSV = UMAP_OBS_CSV
    module.OUT = args.output_dir
    module.main()
    print(f"[Plot] Airway DE-final-style figures: {args.output_dir}")


if __name__ == "__main__":
    main()
