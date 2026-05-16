# Demo Data

This directory contains two real scRNA-seq demo datasets used by `RAG.py`.

- `Deng.h5`: Deng dataset.
- `Airway.h5`: airway epithelial dataset.

Each HDF5 file uses the loader format expected by `utils/preprocess.py`:

- `expression_data`: cells by genes count matrix.
- `cell_id`: cell identifiers.
- `cell_type`: reference cell-type labels.
- `gene_names`: gene names.
