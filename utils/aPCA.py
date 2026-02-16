import numpy as np
from sklearn.decomposition import PCA

def recommend_n_components(X, threshold=0.9):
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.argmax(cum >= threshold) + 1)
    return k, cum

def run_aPCA(X,cfg):
    k_pca, _ = recommend_n_components(X, threshold=cfg.get('pca_acc_var', 0.9))
    X_pca = PCA(n_components=k_pca, random_state=cfg.get('seed',88)).fit_transform(X)
    return X_pca