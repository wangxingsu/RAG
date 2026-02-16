def get_rare_cell_types(df, true_col="cell_type", rare_frac_threshold=None, verbose=True, point=5):
    total_cells = len(df)
    if rare_frac_threshold is None:
        rare_frac_threshold = 0.1 if total_cells < 1000 else point / 100
    rare_cell_threshold = total_cells * rare_frac_threshold

    cell_counts = df[true_col].value_counts()
    rare_cell_types = cell_counts[cell_counts < rare_cell_threshold].index.tolist()

    if verbose:
        print(f"Total cells: {total_cells}")
        print(f"Rare cell threshold: < {rare_cell_threshold:.0f} cells")
        print(f"Identified rare cell types ({len(rare_cell_types)}): {rare_cell_types}")

    return rare_cell_types


def get_rare_list(e, result):
    if e == "GSE60361_level1_sparse.h5":
        return ["microglia"]
    if e.startswith("GSE103354"):
        return ["Ionocyte", "Neuroendocrine", "Goblet", "Tuft"]
    if e == "GSE84133_mouse.h5":
        return ["T_cell", "macrophage", "B_cell"]
    if e == "GSE84133_human_.h5":
        return ["t_cell", "epsilon", "macrophage", "mast"]
    if e == "GSE103322.h5":
        return ["B cell", "Dendritic", "Endothelial", "Macrophage", "Mast", "T cell", "myocyte"]
    if e == "GSE62270.h5ad":
        return ["Goblet", "EE", "Paneth", "Tuft"]
    if "Jurkat293T_subsets" in e:
        return ["Jurkat"]
    if e.startswith("PBMC68k_subsamples"):
        return ["CD14+ Monocyte"]
    if e.startswith("Hrvatin"):
        return ["4", "5"]
    return get_rare_cell_types(result, true_col="cell_type", verbose=True, point=1)
