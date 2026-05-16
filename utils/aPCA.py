import numpy as np
from sklearn.decomposition import PCA

RANDOM_SEED = 88


def recommend_n_components(X, threshold=0.9):
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.argmax(cum >= threshold) + 1)
    return k, cum


def run_aPCA(X, threshold=0.9):
    k_pca, _ = recommend_n_components(X, threshold=threshold)
    X_pca = PCA(n_components=k_pca, random_state=RANDOM_SEED).fit_transform(X)
    return X_pca
