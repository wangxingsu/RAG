from __future__ import annotations

from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DATASET = "Airway"
DATASET_FILE = f"{DATASET}.h5"

RAG_SCRIPT = REPO_ROOT / "RAG.py"
PLOT_REVIEW_DIR = THIS_DIR / "plotting"

DATA_DIR = THIS_DIR / "data"
RAW_DATA_DIR = REPO_ROOT / "demo" / "data"
RAW_DATASET_FILE = RAW_DATA_DIR / DATASET_FILE
OUTPUT_DIR = THIS_DIR / "outputs"
CLUSTERING_OUTPUT_DIR = OUTPUT_DIR / "clustering"
DE_OUTPUT_DIR = OUTPUT_DIR / "de"
RPS_OUTPUT_DIR = OUTPUT_DIR / "rps"
FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"

MAIN_FIGURE_CLUSTER_TABLE = DATA_DIR / "airway_main_figure_clusters.csv"
UMAP_OBS_CSV = DATA_DIR / f"{DATASET}_umap_obs.csv"


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RPS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_result_root() -> Path:
    return CLUSTERING_OUTPUT_DIR


def find_latest_airway_h5ad() -> Path:
    return find_latest_generated_airway_h5ad()


def find_latest_generated_airway_h5ad() -> Path:
    result_root = get_result_root()
    candidates = sorted(result_root.glob(f"**/{DATASET}_clustered.h5ad"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No airway clustered h5ad found under {result_root}. "
            "Run 01_run_airway_clustering.py first, or pass --input-h5ad."
        )
    return candidates[0]


def find_latest_all_cluster_diagnostics() -> Path:
    candidates = sorted(
        CLUSTERING_OUTPUT_DIR.glob(f"**/all_clusters_diagnostics_{DATASET}_h5_RAG.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No all-cluster diagnostics found under {CLUSTERING_OUTPUT_DIR}. "
            "Run 01_run_airway_clustering.py first."
        )
    return candidates[0]
