# RAG - rare cell identification tool from single-cell RNA sequencing data in Python (名字-功能介绍)

## Introduction
（三句话：RAG算法流程，主要特点，输入输出）
RAG is a regularised adaptive graph-based tool for rare-cell identification from single-cell RNA-seq data.
It first preprocesses and denoises data via PCA, learns an information-theoretic representation using the first regularised adaptive graph, and performs Leiden clustering with the second adaptive graph to obtain robust rare-cell clusters.
Its core module, the RAG operator, constructs regularised adaptive graphs by combining Euclidean/cosine candidate-neighbour construction, cell-specific radius-based adjacency control, and locally scaled hybrid affinity assignment.
It supports standard scRNA-seq expression matrices and AnnData objects as input, outputs cell cluster labels, rare-cell lists and visualisation results, and is compatible with multi-sample and batch-corrected data for large-scale and highly imbalanced single-cell analyses.
![overview](./overview.jpg)

## Repository Contents

- `RAG.py`: entry point for running the demo datasets.
- `graphConstruct.py`: implementation of the RAG graph-construction operator.
- `leiden_clustering.py`: Leiden clustering on the RAG-constructed graph.
- `utils/preprocess.py`: data loading, quality control, normalisation, log transformation, highly variable gene selection, and PCA.
- `eval/`: rare-cell evaluation utilities.
- `demo/`: bundled demo datasets and output directory.

## Installation

Create a clean Python environment and install the dependencies:

```bash
conda create -n rag python=3.12
conda activate rag
pip install -r requirements.txt
```

If `leidenalg` or `python-igraph` cannot be installed through `pip`, install them with conda:

```bash
conda install -c conda-forge python-igraph leidenalg
```

## Demo Datasets

Two real scRNA-seq demo datasets are included:

```text
demo/data/Deng.h5
demo/data/Airway.h5
```

Each HDF5 file follows the same layout:

- `expression_data`: cell-by-gene count matrix
- `cell_id`: cell identifiers
- `cell_type`: reference cell-type labels
- `gene_names`: gene names

## Running the Demo

Run RAG on both demo datasets:

```bash
python RAG.py
```

Run one dataset only:

```bash
python RAG.py --dataset deng
python RAG.py --dataset airway
```

The demo workflow includes data loading, quality control, normalisation, log transformation, highly variable gene selection, PCA, RAG graph construction, Wilcoxon-based representation learning, Leiden clustering, and rare-cell evaluation.

## Outputs

Results are written to `demo/results/`. The main output files include:

```text
demo/results/Deng_h5/cluster_assignments.csv
demo/results/Airway_h5/cluster_assignments.csv
demo/results/*/summary_eval.txt
demo/results/demo_summary_rare.csv
demo/results/*/rare_diagnostics_*_RAG.csv
demo/results/*/all_clusters_diagnostics_*_RAG.csv
```

The evaluation outputs report precision, F1 score, and rare-type coverage rate (RCR / `RareTypeCoverage`).

## Main Parameters

- `tau_PCA`: cumulative explained-variance threshold for PCA, default `0.9`.
- `rho_M`: candidate-neighbour ratio, default `0.005`.
- `M_min`: lower bound of the candidate-neighbour count, default `5`.
- `alpha`: Euclidean/cosine hybrid dissimilarity weight, default `0.5`.
- `eta`: radius bias-correction constant, default `1.0`.
- `gamma`: Leiden resolution parameter, default `1.0`.
