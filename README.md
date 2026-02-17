# RAG

## Overview

RAG is a graph-based pipeline for rare-cell identification from single-cell expression data.
The main entry script is `RAG.py`.

## How To Run

### 1) Environment

Use your environment and make sure the required packages are available, e.g.:

- `numpy`
- `pandas`
- `scipy`
- `scanpy`
- `scikit-learn`
- `igraph`
- `leidenalg`
- `h5py`
- `shannonca`

### 2) Configure dataset path and list

Edit `utils/Path_FileList.py`:

- `pathLocal`: root folder of your datasets
- `fileListAll`: dataset files to run

Then edit `RAG.py` if needed:

- `FILE_LIST`
- `FILE_LIST_NAME`
- `THREADS`

### 3) Run

From project root:

```bash
python RAG.py
```

If you want to use your explicit conda Python path:

```bash
D:/anaconda3/envs/SCA/python.exe RAG.py
```

### 4) Output

Results are written under:

`<pathLocal>/results/<method>/<timestamp+FILE_LIST_NAME>/`

including:

- clustering evaluation text report
- all-summary CSV
- rare-summary CSV
- rare diagnostics CSV

