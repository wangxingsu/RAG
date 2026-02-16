import igraph as ig
import numpy as np
import scipy.sparse as sp


def run_leiden_from_custom_graph(adata, S, resolution=1.0, key_added="rc.cluster_init", seed=0):
    if sp.issparse(S):
        S = S.tocsr()
        S = S.maximum(S.T)
        S.setdiag(0)
        S.eliminate_zeros()

        coo = S.tocoo()
        mask = coo.row < coo.col
        rows = coo.row[mask]
        cols = coo.col[mask]
        data = coo.data[mask].astype(float)

        edges = list(zip(rows.tolist(), cols.tolist()))
        weights = data.tolist()
    else:
        S = np.asarray(S, dtype=float)
        S = np.maximum(S, S.T)
        np.fill_diagonal(S, 0.0)

        rows, cols = np.where(S > 0)
        mask = rows < cols
        edges = list(zip(rows[mask].tolist(), cols[mask].tolist()))
        weights = S[rows[mask], cols[mask]].astype(float).tolist()

    g = ig.Graph(n=S.shape[0], edges=edges, directed=False)
    g.es["weight"] = weights

    try:
        import leidenalg as la

        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=int(seed),
        )
        labels = np.array(part.membership, dtype=int)
    except ImportError:
        labels = np.array(g.community_multilevel(weights=g.es["weight"]).membership, dtype=int)

    adata.obs[key_added] = labels.astype(str)
    return adata
