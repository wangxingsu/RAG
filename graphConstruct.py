import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def build_graph_general(
    X,
    k,
    k_max,
    use_rag=True,
    radius_metric="euclidean",
    weight_metric="cosine",
    mutual=True,
    big_threshold=20000,
):
    X = np.asarray(X, dtype=np.float32)
    is_big = X.shape[0] > big_threshold

    if use_rag:
        if is_big:
            return build_rag_sparse(
                X,
                k_max=k_max,
                radius_metric=radius_metric,
                weight_metric=weight_metric,
            )
        return build_rag_dense(
            X,
            k_max=k_max,
            radius_metric=radius_metric,
            weight_metric=weight_metric,
        )

    if is_big:
        return build_kNN_euclidean_sparse(X, k=k, mutual=mutual)
    return build_kNN_euclidean_dense(X, k=k, mutual=mutual)


def build_kNN_euclidean_dense(X, k=20, mutual=True, metric="euclidean", eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]

    D = pairwise_distances(X, metric=metric).astype(np.float32)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]

    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        js = idx[i]
        S[i, js] = (1.0 / (D[i, js] + eps)).astype(np.float32)

    np.fill_diagonal(S, 0.0)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return S


def build_kNN_euclidean_sparse(X, k=20, mutual=True, metric="euclidean", eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape

    algo = "ball_tree" if (metric == "euclidean" and d <= 50) else "brute"
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm=algo, metric=metric)
    nn.fit(X)
    dist, idx = nn.kneighbors(return_distance=True)

    neigh_idx = idx[:, 1:]
    neigh_dist = dist[:, 1:]

    rows, cols, vals = [], [], []
    for i in range(n):
        js = neigh_idx[i]
        w = 1.0 / (neigh_dist[i] + eps)
        rows.extend([i] * len(js))
        cols.extend(js.tolist())
        vals.extend(w.astype(np.float32).tolist())

    S = sparse.csr_matrix(
        (
            np.array(vals, dtype=np.float32),
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=(n, n),
        dtype=np.float32,
    )

    S.setdiag(0.0)
    S.eliminate_zeros()
    return S


def build_rag_dense(
    X,
    k_max=20,
    delta=1.0,
    mutual=True,
    radius_metric="euclidean",
    weight_metric="euclidean",
    eps=1e-8,
):
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]

    D_r = pairwise_distances(X, metric=radius_metric).astype(np.float32)
    np.fill_diagonal(D_r, np.inf)

    if weight_metric == radius_metric:
        D_w = D_r.copy()
    else:
        D_w = pairwise_distances(X, metric=weight_metric).astype(np.float32)
        np.fill_diagonal(D_w, np.inf)

    sigma_w = np.partition(D_w, k_max - 1, axis=1)[:, k_max - 1] + eps
    idx_sorted_r = np.argsort(D_r, axis=1)[:, :k_max]
    Dk_r = np.take_along_axis(D_r, idx_sorted_r, axis=1)
    r = (Dk_r.sum(axis=1) / (k_max - delta + eps)).astype(np.float32)

    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        nbrs = idx_sorted_r[i]
        di = D_r[i, nbrs]
        mask = di <= r[i]
        if not np.any(mask):
            j0 = nbrs[0]
            S[i, j0] = np.exp(-(D_w[i, j0] ** 2) / (sigma_w[i] * sigma_w[j0]))
            continue

        js = nbrs[mask]
        dw = D_w[i, js]
        S[i, js] = np.exp(-(dw**2) / (sigma_w[i] * sigma_w[js])).astype(np.float32)

    if mutual:
        S = np.minimum(S, S.T)

    np.fill_diagonal(S, 0.0)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return S


def build_rag_sparse(
    X,
    k_max=20,
    delta=1.0,
    mutual=True,
    radius_metric="euclidean",
    weight_metric="cosine",
    eps=1e-8,
):
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    k = int(k_max)

    if radius_metric == "euclidean":
        algo_r = "ball_tree" if d <= 50 else "brute"
    else:
        algo_r = "brute"
    nn_r = NearestNeighbors(n_neighbors=k + 1, algorithm=algo_r, metric=radius_metric)
    nn_r.fit(X)
    dist_r, idx_r = nn_r.kneighbors(return_distance=True)
    neigh_idx_r = idx_r[:, 1:]
    neigh_dist_r = dist_r[:, 1:]

    r = (neigh_dist_r.sum(axis=1) / (k - delta + eps)).astype(np.float32)

    if weight_metric == radius_metric:
        sigma_w = (neigh_dist_r[:, -1] + eps).astype(np.float32)
    else:
        if weight_metric == "euclidean":
            algo_w = "ball_tree" if d <= 50 else "brute"
        else:
            algo_w = "brute"
        nn_w = NearestNeighbors(n_neighbors=k + 1, algorithm=algo_w, metric=weight_metric)
        nn_w.fit(X)
        dist_w, _ = nn_w.kneighbors(return_distance=True)
        sigma_w = (dist_w[:, k] + eps).astype(np.float32)

    if weight_metric == "cosine" or radius_metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
        Xn = X / norms

    rows, cols, vals = [], [], []
    for i in range(n):
        js_all = neigh_idx_r[i]
        dr_all = neigh_dist_r[i]
        mask = dr_all <= r[i]
        js = js_all[:1] if not np.any(mask) else js_all[mask]

        if weight_metric == "euclidean":
            if radius_metric == "euclidean" and np.any(mask):
                dw = dr_all[mask]
                if dw.size == 0:
                    dw = np.linalg.norm(X[js] - X[i], axis=1)
            else:
                dw = np.linalg.norm(X[js] - X[i], axis=1)
        else:
            cos = (Xn[js] @ Xn[i].reshape(-1, 1)).ravel().astype(np.float32)
            dw = 1.0 - cos

        w = np.exp(-(dw**2) / (sigma_w[i] * sigma_w[js]))
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

    if mutual:
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


def build_nbhds_from_graph_fast(
    S,
    X_ref=None,
    min_deg: int = 10,
    max_deg: int = 30,
    target="auto",
    ref_knn_extra: int = 32,
    ref_backend: str = "auto",
):
    n = S.shape[0]
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
    if isinstance(target, int):
        tgt = int(np.clip(target, min_deg, max_deg))
    elif target == "auto":
        pos = row_deg[row_deg > 0]
        tgt = int(np.clip(np.percentile(pos, 60), min_deg, max_deg)) if pos.size > 0 else min_deg
    else:
        tgt = min_deg
    k1 = min(tgt, max_deg)

    knn_idx = None
    if X_ref is not None:
        assert X_ref.shape[0] == n, "X_ref rows must match graph node count"
        algo = ref_backend if ref_backend != "auto" else ("ball_tree" if X_ref.shape[1] <= 50 else "brute")
        k_need = int(min(n - 1, min_deg + ref_knn_extra))
        nn = NearestNeighbors(n_neighbors=k_need + 1, algorithm=algo, metric="euclidean")
        nn.fit(X_ref)
        knn_idx = nn.kneighbors(return_distance=False)

    nbhds = []
    for i in range(n):
        base = _row_topk_from_weights(i, indptr, indices, data, k=k1)
        neigh = base.tolist()
        seen = set(neigh)

        if len(neigh) < min_deg and knn_idx is not None:
            for j in knn_idx[i]:
                if j == i or j in seen:
                    continue
                neigh.append(int(j))
                seen.add(int(j))
                if len(neigh) >= min_deg:
                    break

        if len(neigh) > max_deg:
            neigh = neigh[:max_deg]

        if i not in seen:
            neigh = [int(i)] + neigh[:-1] if len(neigh) >= 1 else [int(i)]

        nbhds.append(neigh)

    return nbhds
