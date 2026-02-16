import warnings

import h5py
import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.sparse import issparse

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_adata(filename):
    if filename.endswith("sparse.h5"):
        with h5py.File(filename, "r") as f:
            g = f["expression_data"]
            X = sparse.csc_matrix(
                (g["data"][:], g["indices"][:], g["indptr"][:]),
                shape=tuple(g["shape"][:]),
            )
            cell_id = f["cell_id"][:].astype(str)
            cell_type = f["cell_type"][:].astype(str)
            gene_names = f["gene_names"][:].astype(str)

        adata = sc.AnnData(X.T)
        adata.obs_names = cell_id
        adata.var_names = gene_names
        adata.var["gene_name"] = adata.var_names
        adata.obs["cell_type"] = cell_type
        return adata

    if filename.endswith(".h5"):
        with h5py.File(filename, "r") as f:
            X = f["expression_data"][:]
            cell_id = f["cell_id"][:].astype(str)
            cell_type = f["cell_type"][:].astype(str)
            gene_names = f["gene_names"][:].astype(str)

        adata = sc.AnnData(X)
        adata.obs_names = cell_id
        adata.var_names = gene_names
        adata.obs["cell_type"] = cell_type
        adata.obs["cell_type"] = adata.obs["cell_type"].apply(
            lambda x: "zy" if x in ["zy1", "zy2", "zy3", "zy4"] else x
        )
        adata.var["gene_name"] = adata.var_names
        return adata

    if filename.endswith("GSE62270.h5ad"):
        adata = sc.read_h5ad(filename)
        adata.var_names_make_unique()
        adata.var["gene_name"] = adata.var_names
        if "cell_type" not in adata.obs and "cluster_name" in adata.obs:
            adata.obs["cell_type"] = adata.obs["cluster_name"]
        return adata

    raise ValueError(f"Unsupported file format: {filename}")


def preproMain(filename="E:/datasets/GSE45719.h5"):
    if filename.endswith("p.h5ad"):
        return sc.read_h5ad(filename)

    adata = load_adata(filename)
    adata.var_names_make_unique()

    adata = adata[adata.X.sum(axis=1) > 0]
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    vnames = adata.var_names
    adata.var["mt"] = vnames.str.startswith("MT-") | vnames.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    n_genes = adata.obs["n_genes_by_counts"]
    pct_mt = adata.obs["pct_counts_mt"]
    n_genes_min = n_genes.quantile(0.01)
    n_genes_max = n_genes.quantile(0.99)
    pct_mt_max = pct_mt.quantile(0.95)

    adata = adata[
        (adata.obs["n_genes_by_counts"] > n_genes_min)
        & (adata.obs["n_genes_by_counts"] < n_genes_max)
        & (adata.obs["pct_counts_mt"] < pct_mt_max if pct_mt_max != 0 else True),
        :,
    ]

    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X_dense = adata.X.toarray() if issparse(adata.X) else adata.X
    keep_genes = np.std(X_dense, axis=0) > 0.1
    adata = adata[:, keep_genes]

    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    adata = adata[:, adata.var.highly_variable]
    print(f"[Gene Filter] Retained {adata.shape[1]} highly variable genes")
    print(f"cells: {adata.shape[0]}")
    return adata
