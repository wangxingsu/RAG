import gc
import multiprocessing as mp
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from shannonca.dimred import reduce

from eval.evaluate import evaluate
from eval.evaluate_rare import save_rare_summary_to_csv
from eval.saveFile import save_all_summary_to_csv
from graphConstruct import build_graph_general, build_nbhds_from_graph_fast
from leiden_clustering import run_leiden_from_custom_graph
from utils.Path_FileList import fileListDeng, pathLocal
from utils.aPCA import run_aPCA
from utils.controlThreads import prepare_env
from utils.preprocess import preproMain

THREADS = 16
DATA_PATH = pathLocal
FILE_LIST = fileListDeng
FILE_LIST_NAME = "Deng"

def run_one_config(s1_with_rag: bool, s2_with_rag: bool) -> None:
    # Keep only fields used by this script.
    cfg = {
        "pca_acc_var": 0.9,
        "seed": 88,
        "max_limit": 30,
        "min_deg": 12,
        "max_deg": 30,
        "radius_metric": "euclidean",
        "weight_metric": "cosine",
    }

    method = f"RAG_{FILE_LIST_NAME}"
    prepare_env(THREADS)

    now = datetime.now().strftime("%Y-%m-%d-%H%M")
    output_dir = os.path.join(DATA_PATH, "results", method, now + FILE_LIST_NAME)
    os.makedirs(output_dir, exist_ok=True)

    for dataset_rel_path in FILE_LIST:
        adata = preproMain(DATA_PATH + dataset_rel_path)

        cfg["global_min_size"] = int(adata.n_obs * 0.005)
        cfg["global_min_size"] = min(cfg["max_limit"], cfg["global_min_size"]) if cfg["global_min_size"] > 3 else 3
        k_max = cfg["global_min_size"]

        X = adata.X.toarray() if issparse(adata.X) else adata.X
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Stage 1: PCA + first graph.
        X_pca = run_aPCA(X, cfg)
        s1_graph = build_graph_general(
            X_pca,
            15,
            k_max,
            s1_with_rag,
            cfg["radius_metric"],
            cfg["weight_metric"],
            True,
            20000,
        )

        # Stage 2: ShannonCA + second graph + Leiden.
        nbhds = build_nbhds_from_graph_fast(
            s1_graph,
            X_ref=X_pca,
            min_deg=cfg["min_deg"],
            max_deg=cfg["max_deg"],
            target="auto",
        )
        X_sca = reduce(
            X,
            iters=1,
            model="wilcoxon",
            nbhds=nbhds,
            keep_scores=False,
            keep_loadings=False,
            keep_all_iters=False,
        )
        s2_graph = build_graph_general(
            X_sca,
            15,
            k_max,
            s2_with_rag,
            cfg["radius_metric"],
            cfg["weight_metric"],
            True,
            20000,
        )
        adata = run_leiden_from_custom_graph(
            adata,
            s2_graph,
            resolution=1.0,
            key_added="rc.cluster_init",
            seed=cfg["seed"],
        )

        result = pd.concat(
            [adata.obs["cell_type"], adata.obs["rc.cluster_init"]],
            axis=1,
        ).rename(columns={"rc.cluster_init": "cluster_id"})

        evaluate(
            dataset_rel_path,
            method + "Init",
            output_file=f"{output_dir}/evaluation_results_{method}Init.txt",
            result=result,
            output_dir=output_dir,
        )

        # Free memory after each dataset.
        del adata, s1_graph, s2_graph, result, X, X_pca, X_sca, nbhds
        gc.collect()

    save_all_summary_to_csv(f"{output_dir}/evaluation_summary_all_{method}.csv")
    save_rare_summary_to_csv(f"{output_dir}/evaluation_summary_rare_{method}.csv")


if __name__ == "__main__":
    if os.name == "nt":
        mp.set_start_method("spawn", force=True)

    s1_with_rag = True
    s2_with_rag = True
    print(f"=== start process: S1RAG={s1_with_rag}, S2RAG={s2_with_rag} ===")
    process = mp.Process(target=run_one_config, args=(s1_with_rag, s2_with_rag))
    process.start()
    process.join()
    print(f"=== process done: S1RAG={s1_with_rag}, S2RAG={s2_with_rag} ===")
