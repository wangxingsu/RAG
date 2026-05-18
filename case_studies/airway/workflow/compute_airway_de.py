import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


sc = None


def require_scanpy():
    global sc
    if sc is None:
        try:
            import scanpy as scanpy_module
        except ImportError as e:
            raise ImportError(
                "scanpy is required for this pipeline. Run it in the same Python "
                "environment used for the RAG demo."
            ) from e
        sc = scanpy_module
    return sc


CL_KEY_DEFAULT = "rc.cluster_init"
TYPE_KEY_DEFAULT = "cell_type"
FDR_MAX = 0.05
L2FC_MIN = 2.0
DELTA_PCT_MIN = 0.20
TOPK = 20
SMALL_CLUSTER_MIN_FRAC = 0.001
SMALL_CLUSTER_MAX_FRAC = 0.03
DOMINANT_FRAC_MIN = 0.70
MIN_SMALL_CLUSTER_DE_CELLS = 15
DE_USE_RAW = True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run RAG small-cluster and cell-type marker differential expression."
        )
    )
    parser.add_argument(
        "--input-h5ad",
        required=True,
        help="Clustered h5ad to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to a timestamped folder next to this script.",
    )
    parser.add_argument("--cluster-key", default=CL_KEY_DEFAULT)
    parser.add_argument("--type-key", default=TYPE_KEY_DEFAULT)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--small-min-frac", type=float, default=SMALL_CLUSTER_MIN_FRAC)
    parser.add_argument("--small-max-frac", type=float, default=SMALL_CLUSTER_MAX_FRAC)
    parser.add_argument(
        "--cluster-order",
        choices=["size_asc", "cluster_id"],
        default="size_asc",
        help="Order of small-cluster rows before type rows.",
    )
    parser.add_argument(
        "--dominant-frac-min",
        type=float,
        default=DOMINANT_FRAC_MIN,
        help="Dominant-label fraction recorded for small-cluster DE diagnostics.",
    )
    parser.add_argument(
        "--min-small-cluster-de-cells",
        type=int,
        default=MIN_SMALL_CLUSTER_DE_CELLS,
        help=(
            "Small clusters below this size are not compared unless they uniquely "
            "map to one dominant_label with dominant_frac >= --dominant-frac-min."
        ),
    )
    return parser.parse_args()


def natural_key(value):
    parts = re.split(r"(\d+)", str(value))
    return [int(p) if p.isdigit() else p for p in parts]


def prepare_adata(adata, cluster_key: str, type_key: str):
    for key in [cluster_key, type_key]:
        if key not in adata.obs:
            raise KeyError(f"Missing adata.obs['{key}']; available columns: {list(adata.obs.columns)}")
        adata.obs[key] = adata.obs[key].astype(str)
    return adata


def select_small_clusters(adata, cluster_key: str, min_frac: float, max_frac: float, order: str):
    n_cells = adata.n_obs
    min_n = math.ceil(min_frac * n_cells)
    max_n = math.ceil(max_frac * n_cells)
    counts = adata.obs[cluster_key].astype(str).value_counts()
    selected = counts[(counts >= min_n) & (counts <= max_n)].copy()
    if order == "size_asc":
        selected = selected.sort_values(kind="mergesort")
    else:
        selected = selected.loc[sorted(selected.index, key=natural_key)]
    return selected, min_n, max_n


def cluster_dominance_table(adata, cluster_key: str, type_key: str, cluster_order: pd.Index):
    rows = []
    obs = adata.obs[[cluster_key, type_key]].astype(str)
    for cluster_id in cluster_order:
        cluster_obs = obs[obs[cluster_key] == str(cluster_id)]
        counts = cluster_obs[type_key].value_counts()
        if counts.empty:
            dominant_label = ""
            dominant_count = 0
            dominant_frac = 0.0
        else:
            dominant_label = str(counts.index[0])
            dominant_count = int(counts.iloc[0])
            dominant_frac = float(dominant_count / counts.sum())
        rows.append(
            {
                "cluster_id": str(cluster_id),
                "n_cells": int(len(cluster_obs)),
                "dominant_label": dominant_label,
                "dominant_count": dominant_count,
                "dominant_frac": dominant_frac,
            }
        )
    return pd.DataFrame(rows)


def load_existing_dominance_table(input_h5ad: Path, cluster_order: pd.Index):
    candidates = sorted(
        input_h5ad.parent.glob("all_clusters_diagnostics_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    diag_path = candidates[0]
    diag = pd.read_csv(diag_path)
    required = {"cluster_id", "cluster_size", "dominant_label", "dominant_count", "dominant_frac"}
    if not required.issubset(diag.columns):
        print(
            f"[Dominance] Ignoring {diag_path}; missing columns: {sorted(required - set(diag.columns))}",
            flush=True,
        )
        return None

    diag["cluster_id"] = diag["cluster_id"].astype(str)
    order = pd.DataFrame({"cluster_id": [str(x) for x in cluster_order], "plot_order": range(len(cluster_order))})
    diag = order.merge(diag, on="cluster_id", how="left").sort_values("plot_order")
    diag = diag.rename(columns={"cluster_size": "n_cells"})
    out = diag[["cluster_id", "n_cells", "dominant_label", "dominant_count", "dominant_frac"]].copy()
    print(f"[Dominance] Loaded existing diagnostics: {diag_path}", flush=True)
    return out


def get_cluster_dominance_table(adata, input_h5ad: Path, cluster_key: str, type_key: str, cluster_order: pd.Index):
    existing = load_existing_dominance_table(input_h5ad, cluster_order)
    if existing is not None:
        missing = existing["dominant_label"].isna().sum()
        if missing:
            print(
                f"[Dominance] Existing diagnostics missed {missing} selected clusters; recomputing from adata.obs.",
                flush=True,
            )
        else:
            return existing
    return cluster_dominance_table(adata, cluster_key, type_key, cluster_order)


def filter_small_clusters_for_de(small_clusters, dominance, min_cells, dominant_frac_min):
    dominance = dominance.copy()
    dominance["cluster_id"] = dominance["cluster_id"].astype(str)
    dominance["dominant_label"] = dominance["dominant_label"].fillna("").astype(str)
    dominance["dominant_frac"] = pd.to_numeric(dominance["dominant_frac"], errors="coerce").fillna(0.0)

    valid_dominance = dominance[
        (dominance["dominant_label"] != "")
        & (dominance["dominant_frac"] >= dominant_frac_min)
    ].copy()
    label_counts = valid_dominance["dominant_label"].value_counts()
    unique_labels = set(label_counts[label_counts == 1].index)

    records = []
    keep_ids = []
    for cluster_id, n_cells in small_clusters.items():
        cluster_id = str(cluster_id)
        dom_row = dominance[dominance["cluster_id"] == cluster_id]
        dominant_label = "" if dom_row.empty else str(dom_row.iloc[0]["dominant_label"])
        dominant_frac = 0.0 if dom_row.empty else float(dom_row.iloc[0]["dominant_frac"])
        unique_type_match = dominant_label in unique_labels and dominant_frac >= dominant_frac_min
        keep = True
        keep_ids.append(cluster_id)
        records.append(
            {
                "cluster_id": cluster_id,
                "n_cells": int(n_cells),
                "dominant_label": dominant_label,
                "dominant_frac": dominant_frac,
                "unique_type_match": bool(unique_type_match),
                "kept_for_de": bool(keep),
                "drop_reason": "",
                "size_note": f"n_cells < {min_cells}" if int(n_cells) < min_cells else "",
            }
        )

    filtered = small_clusters.copy()
    filtered.index = filtered.index.astype(str)
    filtered = filtered.loc[[cid for cid in keep_ids if cid in filtered.index]]
    return filtered, pd.DataFrame(records)


def ensure_log_norm_raw(adata):
    if adata.raw is None:
        raise ValueError(
            "adata.raw is missing. Re-run RAG.py after the preprocessing change "
            "so raw contains full-gene log-normalized expression."
        )


def rank_one_vs_rest(adata, groupby_key: str, target: str, result_key: str):
    sc = require_scanpy()
    ensure_log_norm_raw(adata)
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby_key,
        groups=[str(target)],
        reference="rest",
        method="wilcoxon",
        pts=True,
        use_raw=DE_USE_RAW,
        tie_correct=True,
        key_added=result_key,
    )
    df = sc.get.rank_genes_groups_df(adata, group=str(target), key=result_key).rename(
        columns={
            "names": "gene",
            "pvals_adj": "fdr",
            "logfoldchanges": "log2fc",
            "pct_nz_group": "pct_in",
            "pct_nz_reference": "pct_bg",
            "pts": "pct_in",
            "pts_rest": "pct_bg",
        }
    )
    missing = {"gene", "fdr", "log2fc", "pct_in", "pct_bg"} - set(df.columns)
    if missing:
        raise KeyError(f"rank_genes_groups output is missing columns: {sorted(missing)}")
    for col in ["fdr", "log2fc", "pct_in", "pct_bg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["delta_pct"] = df["pct_in"] - df["pct_bg"]
    return df


def filter_upregulated(df):
    keep = (
        (df["fdr"] <= FDR_MAX)
        & (df["log2fc"] > L2FC_MIN)
        & (df["delta_pct"] >= DELTA_PCT_MIN)
    )
    return df.loc[keep].sort_values(
        ["fdr", "log2fc", "delta_pct"],
        ascending=[True, False, False],
        kind="mergesort",
    )


def run_de(adata, group_specs, output_dir: Path, topk: int):
    sig_rows = []
    top_rows = []
    summary_rows = []

    for i, spec in enumerate(group_specs, start=1):
        target = spec["group_id"]
        groupby_key = spec["groupby_key"]
        result_key = f"de_{spec['group_kind']}_{i}"
        print(f"[DE] {spec['row_label']} vs rest", flush=True)
        full = rank_one_vs_rest(adata, groupby_key, target, result_key)
        sig = filter_upregulated(full)

        full = annotate_de_rows(full, spec)
        sig = annotate_de_rows(sig.copy(), spec)
        top = sig.sort_values(["scores", "log2fc"], ascending=[False, False], kind="mergesort").head(topk).copy()
        top["marker_rank"] = np.arange(1, len(top) + 1)

        sig_rows.append(sig)
        top_rows.append(top)
        summary_rows.append(
            {
                **spec,
                "n_sig_up": int(len(sig)),
                "n_top_markers": int(len(top)),
            }
        )

    sig_all = pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame()
    top_all = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    sig_all.to_csv(output_dir / "DE_sig_up_all_groups.csv", index=False)
    top_all.to_csv(output_dir / "DE_top_up_markers_by_group.csv", index=False)
    return summary, top_all


def annotate_de_rows(df, spec):
    df.insert(0, "row_label", spec["row_label"])
    df.insert(0, "group_id", spec["group_id"])
    df.insert(0, "group_kind", spec["group_kind"])
    return df


def build_group_specs(adata, cluster_key, type_key, small_clusters, dominance):
    dominance_by_cluster = dominance.set_index("cluster_id").to_dict(orient="index")
    specs = []
    for cl, n_cells in small_clusters.items():
        dom = dominance_by_cluster.get(str(cl), {})
        specs.append(
            {
                "group_kind": "small_cluster",
                "group_id": str(cl),
                "row_label": f"cluster_{cl}",
                "groupby_key": cluster_key,
                "n_cells": int(n_cells),
                "pct_cells": float(n_cells / adata.n_obs),
                "dominant_label": dom.get("dominant_label", ""),
                "dominant_frac": float(dom.get("dominant_frac", 0.0)),
            }
        )

    specs.extend(build_type_specs(adata, type_key))
    return specs


def build_type_specs(adata, type_key):
    specs = []
    type_counts = adata.obs[type_key].astype(str).value_counts()
    for cell_type in sorted(type_counts.index, key=natural_key):
        n_cells = int(type_counts.loc[cell_type])
        specs.append(
            {
                "group_kind": "type",
                "group_id": str(cell_type),
                "row_label": f"type_{cell_type}",
                "groupby_key": type_key,
                "n_cells": n_cells,
                "pct_cells": float(n_cells / adata.n_obs),
                "dominant_label": "",
                "dominant_frac": np.nan,
            }
        )
    return specs


def run_airway_de(
    input_h5ad: Path,
    output_dir: Path,
    cluster_key: str = CL_KEY_DEFAULT,
    type_key: str = TYPE_KEY_DEFAULT,
    topk: int = TOPK,
    small_min_frac: float = SMALL_CLUSTER_MIN_FRAC,
    small_max_frac: float = SMALL_CLUSTER_MAX_FRAC,
    cluster_order: str = "size_asc",
    dominant_frac_min: float = DOMINANT_FRAC_MIN,
    min_small_cluster_de_cells: int = MIN_SMALL_CLUSTER_DE_CELLS,
) -> pd.DataFrame:
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Read] {input_h5ad}", flush=True)
    sc = require_scanpy()
    adata = sc.read_h5ad(input_h5ad)
    adata = prepare_adata(adata, cluster_key, type_key)

    small_clusters, min_n, max_n = select_small_clusters(
        adata,
        cluster_key,
        small_min_frac,
        small_max_frac,
        cluster_order,
    )
    print(
        f"[Small clusters] N={adata.n_obs}, range=[{min_n}, {max_n}], selected={list(small_clusters.index)}",
        flush=True,
    )

    dominance = get_cluster_dominance_table(
        adata,
        input_h5ad,
        cluster_key,
        type_key,
        small_clusters.index,
    )

    small_clusters_for_de, small_cluster_de_filter = filter_small_clusters_for_de(
        small_clusters,
        dominance,
        min_small_cluster_de_cells,
        dominant_frac_min,
    )
    print(
        "[Small clusters for DE] "
        f"kept={list(small_clusters_for_de.index)}, "
        f"dropped={small_cluster_de_filter.loc[~small_cluster_de_filter['kept_for_de'], 'cluster_id'].tolist()}",
        flush=True,
    )

    group_specs = build_group_specs(adata, cluster_key, type_key, small_clusters_for_de, dominance)
    summary, _ = run_de(adata, group_specs, output_dir, topk)

    print(f"[OK] Finished. Outputs: {output_dir}", flush=True)
    print(summary[["group_kind", "group_id", "row_label", "n_cells", "n_sig_up", "n_top_markers"]])
    return summary


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_airway_de(
        input_h5ad=Path(args.input_h5ad),
        output_dir=output_dir,
        cluster_key=args.cluster_key,
        type_key=args.type_key,
        topk=args.topk,
        small_min_frac=args.small_min_frac,
        small_max_frac=args.small_max_frac,
        cluster_order=args.cluster_order,
        dominant_frac_min=args.dominant_frac_min,
        min_small_cluster_de_cells=args.min_small_cluster_de_cells,
    )


if __name__ == "__main__":
    main()
