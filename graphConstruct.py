import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

_DENSE_GRAPH_CELL_LIMIT = 20000
_MAX_NEIGHBOURHOOD_SIZE = 30
_MIN_NEIGHBOURHOOD_SIZE = 2
_REFERENCE_KNN_EXTRA = 32
_BRIDGE_SUPPORT_THRESHOLD = 0.4
_DIRECT_MIN_SHARED_NEIGHBOURS = 1
_M_LOWER_BOUND = 5


def as_csr_graph(S):
    if sparse.issparse(S):
        G = S.tocsr().astype(np.float32)
    else:
        S = np.asarray(S)
        rows, cols = np.where(S > 0)
        data = S[rows, cols].astype(np.float32)
        G = sparse.csr_matrix((data, (rows, cols)), shape=S.shape, dtype=np.float32)
    G = G.maximum(G.T).tocsr()
    G.setdiag(0.0)
    G.eliminate_zeros()
    return G


def _support_score(neigh_sets, u, v, excluded=None):
    excluded = set() if excluded is None else set(excluded)
    nu = neigh_sets[u] - excluded
    nv = neigh_sets[v] - excluded

    if not nu or not nv:
        return 0.0

    inter_size = len(nu & nv)
    if inter_size == 0:
        return 0.0

    denom = min(len(nu), len(nv))
    return inter_size / denom if denom > 0 else 0.0


def prune_bridges(
    S,
    n_obs,
):
    min_degree_to_prune = compute_prune_min_degree_to_prune(n_obs)
    G = as_csr_graph(S)
    n = G.shape[0]
    deg = np.diff(G.indptr)
    neigh_sets = [
        set(G.indices[G.indptr[i] : G.indptr[i + 1]])
        for i in range(n)
    ]
    edge_weights = {}
    coo = G.tocoo()
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if i == j:
            continue
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        edge_weights[(a, b)] = float(w)

    edges_to_remove = set()
    removed_direct = 0
    removed_two_hop = 0
    low_overlap_blocked_by_degree = set()

    def degree_gate(*nodes):
        if min_degree_to_prune is None:
            return True
        return all(deg[int(node)] >= min_degree_to_prune for node in nodes)

    for (u, v), _ in edge_weights.items():
        nu = neigh_sets[u] - {v}
        nv = neigh_sets[v] - {u}
        if len(nu & nv) >= _DIRECT_MIN_SHARED_NEIGHBOURS:
            continue

        if degree_gate(u, v):
            if (u, v) not in edges_to_remove:
                removed_direct += 1
            edges_to_remove.add((u, v))
        else:
            low_overlap_blocked_by_degree.add((u, v))

    for mid in range(n):
        nbrs = sorted(neigh_sets[mid])
        if len(nbrs) != 2:
            continue

        u, v = nbrs
        e1 = (u, mid) if u < mid else (mid, u)
        e2 = (v, mid) if v < mid else (mid, v)
        if e1 not in low_overlap_blocked_by_degree and e2 not in low_overlap_blocked_by_degree:
            continue

        support = _support_score(neigh_sets, u, v, excluded={mid})
        if support >= _BRIDGE_SUPPORT_THRESHOLD:
            continue

        w1 = edge_weights.get(e1)
        w2 = edge_weights.get(e2)
        if w1 is None or w2 is None:
            continue

        weakest = e1 if w1 <= w2 else e2
        if weakest not in edges_to_remove:
            removed_two_hop += 1
        edges_to_remove.add(weakest)

    keep_rows = []
    keep_cols = []
    keep_data = []
    for (u, v), w in edge_weights.items():
        if (u, v) in edges_to_remove:
            continue
        keep_rows.extend([u, v])
        keep_cols.extend([v, u])
        keep_data.extend([w, w])

    out = sparse.csr_matrix(
        (np.asarray(keep_data, dtype=np.float32), (keep_rows, keep_cols)),
        shape=G.shape,
        dtype=np.float32,
    )
    out.setdiag(0.0)
    out.eliminate_zeros()
    return out


def construct_rag_graph(Z, rho_M=0.005, alpha=0.5, eta=1.0):
    Z = np.asarray(Z, dtype=np.float32)
    M = compute_M(Z.shape[0], rho_M)
    if Z.shape[0] > _DENSE_GRAPH_CELL_LIMIT:
        graph = _build_rag_sparse(Z, M=M, alpha=alpha, eta=eta)
    else:
        graph = _build_rag_dense(Z, M=M, alpha=alpha, eta=eta)
    return prune_bridges(graph, Z.shape[0])


def _build_rag_dense(
    Z,
    M,
    alpha,
    eta,
    eps=1e-8,
):
    Z = np.asarray(Z, dtype=np.float32)
    n = Z.shape[0]
    M = int(min(M, n - 1))
    if M <= 0:
        return np.zeros((n, n), dtype=np.float32)

    D_e = pairwise_distances(Z, metric="euclidean").astype(np.float32)
    np.fill_diagonal(D_e, np.inf)
    D_c = pairwise_distances(Z, metric="cosine").astype(np.float32)
    np.fill_diagonal(D_c, np.inf)

    sigma_E = np.partition(D_e, M - 1, axis=1)[:, M - 1] + eps
    sigma_C = np.partition(D_c, M - 1, axis=1)[:, M - 1] + eps
    D_H = alpha * (D_e / sigma_E[:, None]) + (1.0 - alpha) * (D_c / sigma_C[:, None])
    np.fill_diagonal(D_H, np.inf)
    idx_E = np.argpartition(D_e, M - 1, axis=1)[:, :M]
    idx_C = np.argpartition(D_c, M - 1, axis=1)[:, :M]

    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        C_i = np.union1d(idx_E[i], idx_C[i]).astype(np.int32, copy=False)
        d_H = D_H[i, C_i]
        r_i = (d_H.sum() / (len(C_i) - eta + eps)).astype(np.float32)
        mask = d_H <= r_i
        if not np.any(mask):
            j0 = C_i[np.argmin(d_H)]
            S[i, j0] = np.exp(
                -alpha * (D_e[i, j0] ** 2) / (sigma_E[i] * sigma_E[j0] + eps)
                -(1.0 - alpha) * (D_c[i, j0] ** 2) / (sigma_C[i] * sigma_C[j0] + eps)
            )
            continue

        js = C_i[mask]
        de = D_e[i, js]
        dc = D_c[i, js]
        S[i, js] = np.exp(
            -alpha * (de**2) / (sigma_E[i] * sigma_E[js] + eps)
            -(1.0 - alpha) * (dc**2) / (sigma_C[i] * sigma_C[js] + eps)
        ).astype(np.float32)

    S = np.minimum(S, S.T)

    np.fill_diagonal(S, 0.0)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return S


def _build_rag_sparse(
    Z,
    M,
    alpha,
    eta,
    eps=1e-8,
):
    Z = np.asarray(Z, dtype=np.float32)
    n, _ = Z.shape
    M = int(min(M, n - 1))
    if M <= 0:
        return sparse.csr_matrix((n, n), dtype=np.float32)

    nn_E = NearestNeighbors(n_neighbors=M + 1, algorithm="brute", metric="euclidean")
    nn_E.fit(Z)
    dist_E, idx_E = nn_E.kneighbors(return_distance=True)
    neigh_idx_E = idx_E[:, 1:]
    neigh_dist_E = dist_E[:, 1:]

    nn_C = NearestNeighbors(n_neighbors=M + 1, algorithm="brute", metric="cosine")
    nn_C.fit(Z)
    dist_C, idx_C = nn_C.kneighbors(return_distance=True)
    neigh_idx_C = idx_C[:, 1:]
    neigh_dist_C = dist_C[:, 1:]

    sigma_E = (neigh_dist_E[:, -1] + eps).astype(np.float32)
    sigma_C = (neigh_dist_C[:, -1] + eps).astype(np.float32)
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + eps
    Zn = Z / norms

    rows, cols, vals = [], [], []
    for i in range(n):
        C_i = np.union1d(neigh_idx_E[i], neigh_idx_C[i]).astype(np.int32, copy=False)
        de_all = np.linalg.norm(Z[C_i] - Z[i], axis=1).astype(np.float32)
        cos_all = (Zn[C_i] @ Zn[i].reshape(-1, 1)).ravel().astype(np.float32)
        dc_all = 1.0 - cos_all
        d_H_all = alpha * (de_all / sigma_E[i]) + (1.0 - alpha) * (dc_all / sigma_C[i])
        r_i = (d_H_all.sum() / (len(C_i) - eta + eps)).astype(np.float32)
        mask = d_H_all <= r_i
        if not np.any(mask):
            mask[np.argmin(d_H_all)] = True

        js = C_i[mask]
        de = de_all[mask]
        dc = dc_all[mask]
        w = np.exp(
            -alpha * (de**2) / (sigma_E[i] * sigma_E[js] + eps)
            -(1.0 - alpha) * (dc**2) / (sigma_C[i] * sigma_C[js] + eps)
        )
        rows.extend([i] * len(js))
        cols.extend(js.tolist())
        vals.extend(w.tolist())

    S = sparse.csr_matrix(
        (
            np.array(vals, dtype=np.float32),
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=(n, n),
        dtype=np.float32,
    )

    S = S.minimum(S.T).tocsr()
    S.eliminate_zeros()
    return S


def _row_topk_from_weights(row_idx, indptr, indices, data, k):
    if k <= 0:
        return np.empty(0, dtype=np.int32)

    start, end = indptr[row_idx], indptr[row_idx + 1]
    if end <= start:
        return np.empty(0, dtype=np.int32)

    cols = indices[start:end]
    w = data[start:end]

    if cols.size <= k:
        order = np.argsort(-w)
        return cols[order].astype(np.int32, copy=False)

    part = np.argpartition(-w, k - 1)[:k]
    order = part[np.argsort(-w[part])]
    return cols[order].astype(np.int32, copy=False)


def compute_M(N, rho_M):
    M = int(N * rho_M)
    if M > _M_LOWER_BOUND:
        return min(_MAX_NEIGHBOURHOOD_SIZE, M)
    return _M_LOWER_BOUND


def compute_prune_min_degree_to_prune(n_obs):
    min_degree = int(n_obs * 0.0005)
    if min_degree < 4:
        return 4
    if min_degree > 30:
        return 30
    return min_degree


def compute_min_deg(n_obs):
    min_deg = int(n_obs * 0.001)
    if min_deg > _MIN_NEIGHBOURHOOD_SIZE:
        return min(_MAX_NEIGHBOURHOOD_SIZE, min_deg)
    return _MIN_NEIGHBOURHOOD_SIZE


def build_nbhds(S, X_ref):
    n = S.shape[0]
    min_deg = compute_min_deg(n)
    max_deg = _MAX_NEIGHBOURHOOD_SIZE
    assert max_deg >= 1 and min_deg >= 0

    if sparse.issparse(S):
        S_csr = S.tocsr().astype(np.float32)
    else:
        rows, cols = np.where(S > 0)
        data = S[rows, cols].astype(np.float32)
        S_csr = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    S_csr.setdiag(0.0)
    S_csr.eliminate_zeros()
    indptr, indices, data = S_csr.indptr, S_csr.indices, S_csr.data

    row_deg = np.diff(indptr)
    pos = row_deg[row_deg > 0]
    tgt = int(np.clip(np.percentile(pos, 60), min_deg, max_deg)) if pos.size > 0 else min_deg
    k1 = min(tgt, max_deg)

    assert X_ref.shape[0] == n, "X_ref rows must match graph node count"
    k_need = int(min(n - 1, min_deg + _REFERENCE_KNN_EXTRA))
    nn = NearestNeighbors(n_neighbors=k_need + 1, algorithm="brute", metric="cosine")
    nn.fit(X_ref)
    knn_idx = nn.kneighbors(return_distance=False)

    nbhds = []
    for i in range(n):
        base = _row_topk_from_weights(i, indptr, indices, data, k=k1)
        neigh = base.tolist()
        seen = set(neigh)

        if len(neigh) < min_deg:
            for j in knn_idx[i]:
                if j == i or j in seen:
                    continue
                neigh.append(int(j))
                seen.add(int(j))
                if len(neigh) >= min_deg:
                    break

        if i not in seen:
            if len(neigh) >= max_deg:
                neigh = [int(i)] + neigh[: max_deg - 1]
            else:
                neigh = [int(i)] + neigh

        if len(neigh) > max_deg:
            neigh = neigh[:max_deg]

        nbhds.append(neigh)

    return nbhds
