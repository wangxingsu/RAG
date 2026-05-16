# RAG

## Main interface symbols

- `TAU_PCA`: paper `\tau_{\mathrm{PCA}}`, cumulative explained-variance threshold for PCA.
- `RHO_M`: paper `\rho_M`, candidate-neighbour ratio.
- `ALPHA`: paper `\alpha`, hybrid dissimilarity weight.
- `ETA`: paper `\eta`, radius bias-correction constant.
- `GAMMA`: paper `\gamma`, Leiden resolution parameter.

## RAG operator symbols in code

- `Z`: paper input matrix to the RAG operator.
- `M`: candidate-neighbour count, computed with the same logic as `0429` and exposed with the paper symbol.
- `C_i`: paper candidate-neighbour set `\mathcal{C}_i`.
- `D_H` / `d_H`: paper hybrid dissimilarity `d^{(H)}_{i,j}`.
- `sigma_E`, `sigma_C`: paper local scales `\sigma_i^{(E)}` and `\sigma_i^{(C)}`.
- `r_i`: paper cell-specific radius.
- `S`: paper adaptive affinity matrix.

Implementation-only thresholds for dense/sparse graph construction, bridge pruning, candidate-neighbour capping, and SCA neighbourhood padding are kept private with leading underscores in `graphConstruct.py`. These values preserve the original `0429` calculation logic.
