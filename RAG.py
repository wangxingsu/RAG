import argparse
import gc
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from shannonca.dimred import reduce

from eval.evaluate import evaluate
from eval.evaluate_rare import reset_rare_summary, save_rare_summary_to_csv
from graphConstruct import build_nbhds, construct_rag_graph
from leiden_clustering import run_leiden_from_custom_graph
from utils.aPCA import run_aPCA
from utils.controlThreads import prepare_env
from utils.preprocess import preproMain


PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_DATASETS = {
    "deng": "Deng.h5",
    "airway": "Airway.h5",
}
METHOD = "RAG"


def run_one_dataset(
    input_file,
    output_dir,
    tau_pca=0.9,
    rho_m=0.005,
    alpha=0.5,
    eta=1.0,
    gamma=1.0,
):
    input_file = Path(input_file)
    dataset_name = input_file.name
    dataset_tag = dataset_name.replace(".", "_")
    dataset_output_dir = Path(output_dir) / dataset_tag
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== RAG demo: {dataset_name} ===")
    print("Step 1/6: load data, quality control, total-count normalization, log1p transform, and HVG selection")
    adata = preproMain(str(input_file))

    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print("Step 2/6: PCA representation")
    X_pca = run_aPCA(X, tau_pca)

    print("Step 3/6: first RAG adaptive affinity graph")
    s1_graph = construct_rag_graph(X_pca, rho_M=rho_m, alpha=alpha, eta=eta)

    print("Step 4/6: RAG-derived neighbourhoods and Wilcoxon representation learning")
    nbhds = build_nbhds(s1_graph, X_pca)
    X_sca = reduce(X, iters=1, model="wilcoxon", nbhds=nbhds)

    print("Step 5/6: second RAG adaptive affinity graph and Leiden clustering")
    s2_graph = construct_rag_graph(X_sca, rho_M=rho_m, alpha=alpha, eta=eta)
    adata = run_leiden_from_custom_graph(adata, s2_graph, gamma)

    print("Step 6/6: save cluster labels and rare-cell diagnostics")
    result = adata.obs[["cell_type", "rc.cluster_init"]].rename(columns={"rc.cluster_init": "cluster_id"})
    result.to_csv(dataset_output_dir / "cluster_assignments.csv")

    evaluate(
        dataset_name,
        METHOD,
        output_file=str(dataset_output_dir / "summary_eval.txt"),
        result=result,
        output_dir=str(dataset_output_dir),
    )

    del adata, s1_graph, s2_graph, result, X, X_pca, X_sca, nbhds
    gc.collect()
    print(f"Finished {dataset_name}. Results written to {dataset_output_dir}")


def resolve_demo_files(dataset, data_dir):
    data_dir = Path(data_dir)
    selected = DEMO_DATASETS.keys() if dataset == "all" else [dataset]
    files = []
    missing = []
    for name in selected:
        path = data_dir / DEMO_DATASETS[name]
        if path.exists():
            files.append(path)
        else:
            missing.append(path)

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Demo dataset file(s) not found:\n"
            f"{missing_text}\n"
            "Place Deng.h5 and Airway.h5 in demo/data/ or pass --data-dir."
        )
    return files


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG on the bundled demo datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "deng", "airway"],
        default="all",
        help="Demo dataset to run.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "demo" / "data"),
        help="Directory containing Deng.h5 and Airway.h5.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "demo" / "results"),
        help="Directory for demo outputs.",
    )
    parser.add_argument("--tau-PCA", dest="tau_pca", type=float, default=0.9, help="PCA cumulative explained-variance threshold.")
    parser.add_argument("--rho-M", dest="rho_m", type=float, default=0.005, help="Candidate-neighbour ratio.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid dissimilarity weight.")
    parser.add_argument("--eta", type=float, default=1.0, help="Radius bias-correction constant.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Leiden resolution parameter.")
    return parser.parse_args()


def main():
    args = parse_args()

    prepare_env()
    reset_rare_summary()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_file in resolve_demo_files(args.dataset, args.data_dir):
        run_one_dataset(
            input_file=input_file,
            output_dir=output_dir,
            tau_pca=args.tau_pca,
            rho_m=args.rho_m,
            alpha=args.alpha,
            eta=args.eta,
            gamma=args.gamma,
        )

    save_rare_summary_to_csv(str(output_dir / "demo_summary_rare.csv"))
    print(f"\nDemo summary written to {output_dir / 'demo_summary_rare.csv'}")


if __name__ == "__main__":
    main()
