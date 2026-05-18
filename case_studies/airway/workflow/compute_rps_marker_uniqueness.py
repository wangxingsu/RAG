from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MARKER_SUPPORT_DENOM = 10
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute marker-set uniqueness and rare-population prioritisation scores from DE_sig_up_all_groups.csv."
    )
    parser.add_argument("--de-sig", required=True, type=Path, help="DE_sig_up_all_groups.csv from 02_run_airway_de.py.")
    parser.add_argument("--cluster-diagnostics", required=True, type=Path, help="all_clusters_diagnostics_*_RAG.csv from RAG clustering.")
    parser.add_argument("--output-dir", required=True, type=Path, help="RPS output directory.")
    return parser.parse_args()


def natural_key(value: str) -> list[object]:
    import re

    parts = re.split(r"(\d+)", str(value))
    return [int(part) if part.isdigit() else part for part in parts]


def load_cluster_metadata(path: Path) -> pd.DataFrame:
    diag = pd.read_csv(path)
    required = {"cluster_id", "cluster_size", "dominant_label", "dominant_frac"}
    missing = required - set(diag.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")

    total = pd.to_numeric(diag["cluster_size"], errors="coerce").sum()
    out = diag[["cluster_id", "cluster_size", "dominant_label", "dominant_frac"]].copy()
    out["cluster_id"] = out["cluster_id"].astype(str)
    out["n_cells"] = pd.to_numeric(out["cluster_size"], errors="coerce").astype("Int64")
    out["pct_cells"] = pd.to_numeric(out["cluster_size"], errors="coerce") / total
    out["dominant_label"] = out["dominant_label"].fillna("").astype(str)
    out["dominant_frac"] = pd.to_numeric(out["dominant_frac"], errors="coerce")
    return out.drop(columns=["cluster_size"])


def load_marker_sets(de_sig_path: Path) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    de = pd.read_csv(de_sig_path)
    required = {"group_kind", "group_id", "gene", "scores", "log2fc", "fdr", "pct_in", "pct_bg", "delta_pct"}
    missing = required - set(de.columns)
    if missing:
        raise KeyError(f"{de_sig_path} is missing columns: {sorted(missing)}")

    de = de[de["group_kind"].astype(str) == "small_cluster"].copy()
    if de.empty:
        raise ValueError(f"No small_cluster rows found in {de_sig_path}")

    de["cluster_id"] = de["group_id"].astype(str)
    for col in ["scores", "log2fc", "fdr", "pct_in", "pct_bg", "delta_pct"]:
        de[col] = pd.to_numeric(de[col], errors="coerce")

    rows = []
    marker_sets: dict[str, set[str]] = {}
    for cluster_id, sub in de.sort_values(["cluster_id", "scores", "log2fc"], ascending=[True, False, False]).groupby("cluster_id"):
        genes = sub["gene"].astype(str).drop_duplicates().tolist()
        marker_sets[cluster_id] = set(genes)
        ranked = sub.drop_duplicates("gene").copy()
        ranked["marker_rank"] = np.arange(1, len(ranked) + 1)
        rows.append(ranked[["cluster_id", "marker_rank", "gene", "scores", "log2fc", "fdr", "pct_in", "pct_bg", "delta_pct"]])

    marker_long = pd.concat(rows, ignore_index=True)
    return marker_long, marker_sets


def compute_rps(marker_sets: dict[str, set[str]], metadata: pd.DataFrame) -> pd.DataFrame:
    cluster_ids = sorted(marker_sets, key=natural_key)
    rows = []
    for cluster_id in cluster_ids:
        genes_i = marker_sets[cluster_id]
        uniqueness_count_sum = float(
            sum(len(genes_i - marker_sets[other_id]) for other_id in cluster_ids if other_id != cluster_id)
        )
        n_markers = len(genes_i)
        marker_support_score = float(min(n_markers / MARKER_SUPPORT_DENOM, 1.0))
        rps = marker_support_score * uniqueness_count_sum
        rows.append(
            {
                "cluster_id": cluster_id,
                "row_label": f"cluster_{cluster_id}",
                "n_markers": int(n_markers),
                "marker_uniqueness_count_sum": uniqueness_count_sum,
                "marker_support_score": marker_support_score,
                "RPS": rps,
                "RPS_formula": "min(|Gi| / 10, 1) * sum_j |Gi \\ Gj|",
            }
        )

    ranking = pd.DataFrame(rows)
    ranking = ranking.merge(metadata, on="cluster_id", how="left")
    ranking = ranking[
        [
            "cluster_id",
            "row_label",
            "n_cells",
            "pct_cells",
            "dominant_label",
            "dominant_frac",
            "n_markers",
            "marker_uniqueness_count_sum",
            "marker_support_score",
            "RPS",
            "RPS_formula",
        ]
    ]
    ranking = ranking.sort_values(
        ["RPS", "marker_uniqueness_count_sum", "n_markers", "dominant_frac"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    ranking.insert(0, "RPS_rank", np.arange(1, len(ranking) + 1))
    return ranking


def run_rps(de_sig: Path, cluster_diagnostics: Path, output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_long, marker_sets = load_marker_sets(Path(de_sig))
    metadata = load_cluster_metadata(Path(cluster_diagnostics))
    ranking = compute_rps(marker_sets, metadata)

    marker_long.to_csv(output_dir / "candidate_marker_sets_long.csv", index=False)
    ranking.to_csv(output_dir / "RPS_cluster_ranking.csv", index=False)
    print(f"[OK] RPS ranking written to {output_dir}")
    return ranking


def main() -> None:
    args = parse_args()
    run_rps(args.de_sig, args.cluster_diagnostics, args.output_dir)


if __name__ == "__main__":
    main()
