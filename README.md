# RAG

RAG is a Regularised Adaptive Graph-based method for rare-cell identification from single-cell expression data. The method constructs adaptive affinity graphs with cell-specific radius-controlled adjacency and locally scaled affinities, then uses the graphs for Wilcoxon-based representation learning and Leiden clustering.

## Repository Contents

- `RAG.py`: the single runnable entry point for the bundled demo datasets.
- `graphConstruct.py`: implementation of the RAG adaptive affinity graph operator.
- `leiden_clustering.py`: Leiden clustering on a RAG affinity graph.
- `utils/preprocess.py`: data loading, quality control, normalization, log transformation, and highly variable gene selection.
- `demo/`: Deng and airway demo datasets and output directory.
- `eval/`: rare-cell evaluation utilities used by the demo runner.

## Installation

Create a clean Python environment and install the required packages:

```bash
conda create -n rag python=3.12
conda activate rag
pip install -r requirements.txt
```

If `leidenalg` or `python-igraph` cannot be installed by `pip` on your platform, install them with conda:

```bash
conda install -c conda-forge python-igraph leidenalg
```

## Demo Datasets

The repository includes two real scRNA-seq demo datasets:

```text
demo/data/Deng.h5
demo/data/Airway.h5
```

`Deng.h5` is used as the Deng demo dataset, and `Airway.h5` is used as the airway demo dataset. Each file follows the same simple HDF5 layout used by the benchmark loader:

- `expression_data`: cells by genes count matrix
- `cell_id`: cell identifiers
- `cell_type`: reference labels for the demo
- `gene_names`: gene names

## Working Example

Run RAG on both bundled demo datasets:

```bash
python RAG.py
```

Run one dataset only:

```bash
python RAG.py --dataset deng
python RAG.py --dataset airway
```

Outputs are written to `demo/results/`:

- `demo/results/Deng_h5/cluster_assignments.csv`: Deng cell-level RAG cluster assignments
- `demo/results/Airway_h5/cluster_assignments.csv`: airway cell-level RAG cluster assignments
- `demo/results/*/summary_eval.txt`: per-dataset rare-cell identification metrics
- `demo/results/demo_summary_rare.csv`: compact metric table across the demo datasets
- `demo/results/*/rare_diagnostics_*_RAG.csv`: rare-type diagnostics
- `demo/results/*/all_clusters_diagnostics_*_RAG.csv`: all-cluster diagnostics

The public output keeps the metrics, including F1, precision, and RCR (`RareTypeCoverage`).

## Preprocessing and Normalization in the Demo

The demo calls `utils/preprocess.py::preproMain`, which performs the following steps:

1. Load the HDF5 count matrix into an AnnData object.
2. Remove cells with zero total counts.
3. Filter cells with fewer than 200 detected genes.
4. Filter genes detected in fewer than 3 cells.
5. Compute mitochondrial-gene quality metrics when genes start with `MT-` or `mt-`.
6. Remove cells outside the 1st and 99th percentiles of detected-gene counts.
7. Remove cells above the 95th percentile of mitochondrial percentage when mitochondrial counts are present.
8. Normalize each cell to a total count of `1e4` with `scanpy.pp.normalize_total`.
9. Apply `scanpy.pp.log1p`.
10. Keep genes with standard deviation greater than `0.1`.
11. Select up to 5000 highly variable genes with `scanpy.pp.highly_variable_genes`.

After preprocessing, the demo runs PCA, constructs the first RAG graph, builds RAG-derived neighbourhoods for Wilcoxon representation learning, constructs the second RAG graph, and applies Leiden clustering. These steps are printed by `RAG.py` while it runs, so reviewers can see where preprocessing and normalization occur.

## Main Parameters

- `tau_PCA`: cumulative explained-variance threshold for PCA, default `0.9`.
- `rho_M`: candidate-neighbour ratio, default `0.005`.
- `alpha`: Euclidean/cosine hybrid dissimilarity weight, default `0.5`.
- `eta`: radius bias-correction constant, default `1.0`.
- `gamma`: Leiden resolution parameter, default `1.0`.
