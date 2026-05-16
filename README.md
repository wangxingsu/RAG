# RAG

RAG is a Regularised Adaptive Graph-based method for rare-cell identification from single-cell expression data. It constructs regularised adaptive graphs with cell-specific radius-controlled adjacency and locally scaled affinities, and uses these graphs for Wilcoxon-based representation learning and Leiden clustering.

## Repository Contents

- `RAG.py`: entry point for running the demo datasets.
- `graphConstruct.py`: implementation of the RAG graph-construction operator.
- `leiden_clustering.py`: Leiden clustering on the RAG-constructed graph.
- `utils/preprocess.py`: data loading, quality control, normalisation, log transformation, HVG selection, and PCA.
- `eval/`: rare-cell evaluation utilities.
- `demo/`: bundled demo datasets and output directory.

## Installation

Create a clean Python environment and install the dependencies:

```bash
conda create -n rag python=3.12
conda activate rag
pip install -r requirements.txt