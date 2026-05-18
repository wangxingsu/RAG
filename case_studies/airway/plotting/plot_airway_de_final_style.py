from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parents[1]
CLUSTER_TABLE = ROOT / "data" / "airway_main_figure_clusters.csv"
DE_TOP_TABLE = ROOT / "outputs" / "de" / "DE_top_up_markers_by_group.csv"
H5AD = ROOT / "outputs" / "clustering" / "Airway_h5" / "Airway_clustered.h5ad"
UMAP_CSV = ROOT / "data" / "Airway_umap_obs.csv"
OUT = ROOT / "outputs" / "figures" / "airway_de_final_style"

DATASET = "Airway"

PALETTE = [
    "#3682B4",
    "#45A776",
    "#F05326",
    "#EED777",
    "#334F65",
    "#B397BE",
    "#38CB7D",
    "#DDAE33",
    "#844BB3",
    "#93C555",
    "#5F6694",
    "#DF3881",
]
POINT_SIZE = 4.2


def set_de_final_style() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["figure.titlesize"] = 14
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def unique_keep_order(values: list[str]) -> list[str]:
    return list(OrderedDict((v, None) for v in values if v).keys())


def complement(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r = 255 - int(hex_color[0:2], 16)
    g = 255 - int(hex_color[2:4], 16)
    b = 255 - int(hex_color[4:6], 16)
    return f"#{r:02X}{g:02X}{b:02X}"


def cluster_color_map(rows: list[dict], type_colors: dict[str, str]) -> dict[int, str]:
    by_type: dict[str, list[dict]] = OrderedDict()
    for row in rows:
        by_type.setdefault(row["type"], []).append(row)
    assigned = {}
    for ctype, items in by_type.items():
        base = type_colors.get(ctype, "#3682B4")
        comp = complement(base)
        variants = [base, comp] + [c for c in PALETTE if c.lower() not in {base.lower(), comp.lower()}]
        for i, item in enumerate(items):
            assigned[item["cluster_id"]] = variants[i % len(variants)]
    return assigned


def short_plot_label(cid: int, label: str) -> str:
    if cid == 19:
        return "19_PNEC"
    label = label.replace("Ionocytes", "Ionocyte")
    label = label.replace("Pulmonary neuroendocrine", "PNEC")
    return label


def block_label(row: dict, adata: ad.AnnData) -> str:
    n_cells = int((adata.obs["rc.cluster_init"].astype(str) == str(row["cluster_id"])).sum())
    name = row["plot_label"].split("_", 1)[1] if "_" in row["plot_label"] else row["plot_label"]
    return f"{name} (n={n_cells})"


def load_top_genes() -> dict[int, list[str]]:
    de_top = pd.read_csv(DE_TOP_TABLE)
    required = {"group_kind", "group_id", "gene", "marker_rank"}
    missing = required - set(de_top.columns)
    if missing:
        raise KeyError(f"{DE_TOP_TABLE} is missing columns: {sorted(missing)}")

    de_top = de_top[de_top["group_kind"] == "small_cluster"].copy()
    de_top["cluster_id"] = de_top["group_id"].astype(int)
    de_top["marker_rank"] = pd.to_numeric(de_top["marker_rank"], errors="coerce")
    out = {}
    for cid, sub in de_top.sort_values(["cluster_id", "marker_rank"]).groupby("cluster_id"):
        genes = sub["gene"].astype(str).drop_duplicates().head(5).tolist()
        out[int(cid)] = genes
    return out


def infer_cluster_types(adata: ad.AnnData, cluster_ids: list[int]) -> dict[int, str]:
    clusters = adata.obs["rc.cluster_init"].astype(str)
    cell_types = adata.obs["cell_type"].astype(str)
    out = {}
    for cid in cluster_ids:
        labels = cell_types[clusters == str(cid)]
        if labels.empty:
            raise ValueError(f"Cluster {cid} is not present in adata.obs['rc.cluster_init']")
        out[cid] = labels.value_counts().idxmax()
    return out


def build_rows(adata: ad.AnnData) -> list[dict]:
    source = pd.read_csv(CLUSTER_TABLE)
    required = {"cluster_id", "category", "plot_label"}
    missing = required - set(source.columns)
    if missing:
        raise KeyError(f"{CLUSTER_TABLE} is missing columns: {sorted(missing)}")

    top_genes = load_top_genes()
    cluster_ids = [int(cid) for cid in source["cluster_id"].tolist()]
    cluster_types = infer_cluster_types(adata, cluster_ids)

    rows = []
    for _, row in source.iterrows():
        cid = int(row["cluster_id"])
        category = str(row["category"]).strip().lower()
        if category not in {"known", "extra"}:
            raise ValueError(f"Unsupported category for cluster {cid}: {row['category']}")
        genes = top_genes.get(cid, [])
        if not genes:
            raise ValueError(f"No DE top genes found for cluster {cid} in {DE_TOP_TABLE}")
        label = str(row["plot_label"]).strip()
        rows.append(
            {
                "cluster_id": cid,
                "category": category,
                "type": cluster_types[cid],
                "label": label,
                "plot_label": short_plot_label(cid, label),
                "top5_genes": genes[:5],
            }
        )

    type_order = unique_keep_order([r["type"] for r in rows])
    grouped = []
    for ctype in type_order:
        grouped.extend([r for r in rows if r["type"] == ctype])
    return grouped


def prepare_umap_columns(adata: ad.AnnData, rows: list[dict]) -> None:
    clusters = adata.obs["rc.cluster_init"].astype(str)
    for category in ["known", "extra"]:
        col = f"{category}_plot"
        labels = []
        row_map = {str(r["cluster_id"]): r["plot_label"] for r in rows if r["category"] == category}
        for cid in clusters:
            labels.append(row_map.get(cid, "Abundant"))
        cats = ["Abundant"] + [r["plot_label"] for r in rows if r["category"] == category]
        adata.obs[col] = pd.Categorical(labels, categories=cats, ordered=True)


def attach_umap_from_csv(adata: ad.AnnData) -> None:
    umap = pd.read_csv(UMAP_CSV)
    if len(umap) != adata.n_obs:
        raise ValueError(f"UMAP CSV has {len(umap)} rows, but h5ad has {adata.n_obs} cells")
    if "cell_id" in umap.columns:
        csv_ids = umap["cell_id"].astype(str).tolist()
        obs_ids = adata.obs_names.astype(str).tolist()
        if csv_ids == obs_ids:
            pass
        elif set(csv_ids) == set(obs_ids):
            umap = umap.set_index("cell_id").loc[obs_ids].reset_index()
        else:
            print("Warning: UMAP cell_id values do not match obs_names exactly; using row order.")
    adata.obsm["X_umap"] = umap[["umap_1", "umap_2"]].to_numpy(dtype=np.float32)


def prepare_violin_adata(adata: ad.AnnData, rows: list[dict], group_mode: str) -> tuple[ad.AnnData, list[str]]:
    parts = []
    group_labels = []
    for row in rows:
        mask = adata.obs["rc.cluster_init"].astype(str) == str(row["cluster_id"])
        sub = adata[mask].copy()
        sub.obs["display_group"] = row["plot_label"]
        parts.append(sub)
        group_labels.append(row["plot_label"])

    if group_mode == "basal_club_refs":
        ref_types = ["Basal", "Club"]
    elif group_mode == "matched_type_refs":
        ref_types = unique_keep_order([row["type"] for row in rows])
    else:
        raise ValueError(group_mode)

    for ctype in ref_types:
        mask = adata.obs["cell_type"].astype(str) == ctype
        if mask.any():
            sub = adata[mask].copy()
            label = f"type_{ctype}"
            sub.obs["display_group"] = label
            parts.append(sub)
            group_labels.append(label)

    plot_adata = ad.concat(parts, axis=0, join="outer", merge="same", uns_merge="same", label=None, index_unique="-")
    plot_adata.obs["display_group"] = pd.Categorical(plot_adata.obs["display_group"], categories=group_labels, ordered=True)
    return plot_adata, group_labels


def style_umap_axis(ax) -> None:
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def plot_combined(adata: ad.AnnData, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    group_mode = "matched_type_refs"

    type_order = unique_keep_order(adata.obs["cell_type"].astype(str).tolist())
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category").cat.set_categories(type_order, ordered=True)
    type_colors = {ctype: PALETTE[i % len(PALETTE)] for i, ctype in enumerate(type_order)}
    cluster_colors = cluster_color_map(rows, type_colors)

    prepare_umap_columns(adata, rows)

    marker_blocks = OrderedDict((row["plot_label"], row["top5_genes"]) for row in rows)
    markers = unique_keep_order([gene for genes in marker_blocks.values() for gene in genes])
    var_group_labels = [block_label(row, adata) for row in rows]
    pos = 0
    var_group_positions = []
    for row in rows:
        block_len = len(row["top5_genes"])
        var_group_positions.append((pos, pos + block_len - 1))
        pos += block_len

    plot_adata, group_labels = prepare_violin_adata(adata, rows, group_mode)
    present_markers = [g for g in markers if g in plot_adata.raw.var_names or g in plot_adata.var_names]
    missing = [g for g in markers if g not in present_markers]

    row_palette = []
    label_to_row = {row["plot_label"]: row for row in rows}
    for label in group_labels:
        if label in label_to_row:
            row_palette.append(cluster_colors[label_to_row[label]["cluster_id"]])
        elif label.startswith("type_"):
            row_palette.append(type_colors.get(label.replace("type_", "", 1), "lightgray"))
        else:
            row_palette.append("lightgray")

    fig = plt.figure(figsize=(14, 10), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1, 2.05], hspace=0.08, wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_violin = fig.add_subplot(gs[1, :])

    sc.pl.umap(
        adata,
        color="cell_type",
        palette=[type_colors[c] for c in type_order],
        legend_loc=None,
        title="Original annotation",
        size=POINT_SIZE,
        show=False,
        ax=axes[0],
    )
    style_umap_axis(axes[0])
    handles_original = [
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=5.5, markerfacecolor=type_colors[c], markeredgecolor="none", label=str(c))
        for c in type_order
    ]
    axes[0].legend(handles=handles_original, loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, handletextpad=0.35, columnspacing=0.8, fontsize=7.2)

    for ax, col, title, category in [
        (axes[1], "known_plot", "Known annotated", "known"),
        (axes[2], "extra_plot", "Marker-supported extra", "extra"),
    ]:
        labels = list(adata.obs[col].cat.categories)
        palette = ["lightgray"]
        for row in rows:
            if row["category"] == category:
                palette.append(cluster_colors[row["cluster_id"]])
        sc.pl.umap(adata, color=col, palette=palette, legend_loc=None, title=title, size=POINT_SIZE, show=False, ax=ax)
        style_umap_axis(ax)
        handles = [
            mlines.Line2D([], [], marker="o", linestyle="None", markersize=5.5, markerfacecolor=color, markeredgecolor="none", label=label)
            for label, color in zip(labels, palette)
        ]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=max(1, min(3, len(handles))), frameon=False, handletextpad=0.35, columnspacing=0.8, fontsize=7.2)

    sc.pl.stacked_violin(
        plot_adata,
        present_markers,
        groupby="display_group",
        categories_order=group_labels,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        standard_scale="var",
        row_palette=row_palette,
        swap_axes=False,
        show=False,
        ax=ax_violin,
    )
    for text in ax_violin.texts:
        text.set_rotation(0)
        text.set_ha("center")
        text.set_va("bottom")
        text.set_fontsize(10)
    for text in fig.findobj(Text):
        if "(n=" in text.get_text():
            text.set_rotation(0)
            text.set_ha("center")
            text.set_va("bottom")
            text.set_fontsize(9)
    for axis in fig.axes:
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            if "(n=" in tick.get_text():
                tick.set_rotation(0)
                tick.set_ha("center")
                tick.set_va("bottom")
                tick.set_fontsize(9)
    ax_violin.tick_params(axis="x", labelsize=8)
    ax_violin.tick_params(axis="y", labelsize=10)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.08)

    out_base = OUT / f"airway_main_{group_mode}_top5_genes_DEstyle"
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300)
    plt.close(fig)


def main() -> None:
    set_de_final_style()
    adata = sc.read_h5ad(H5AD)
    rows = build_rows(adata)
    if "X_umap" not in adata.obsm:
        attach_umap_from_csv(adata)

    plot_combined(adata.copy(), rows)


if __name__ == "__main__":
    main()
