import os

from eval.evaluate_rare import (
    accumulate_rare_summary,
    build_all_cluster_diagnostics,
    evaluate_rare_main,
    save_rare_diagnostics,
)
from eval.saveFile import save_results_to_file
from utils.getRareType import get_rare_list


def evaluate(e, method, result, output_file, output_dir):
    dataset_name = os.path.basename(e)
    title = f"{dataset_name} with {method}"

    rare_types = get_rare_list(e, result)
    metrics_avg, metrics_per_cluster, identified_rare_info, true_rare_info = evaluate_rare_main(
        cluster_labels=result["cluster_id"].values,
        true_labels=result["cell_type"].values,
        rare_types=rare_types,
        threshold=0.3,
        target_cluster_ids=result["cluster_id"].values,
    )

    true_count = true_rare_info["count"]
    rare_summary = {
        **metrics_avg,
        "IdentifiedRareTypeCount": identified_rare_info["count"],
        "TrueRareTypeCount": true_count,
        "RareTypeCoverage": identified_rare_info["count"] / true_count if true_count else 0.0,
        "IdentifiedRareTypes": identified_rare_info["types"],
        "TrueRareTypes": true_rare_info["types"],
    }
    save_results_to_file(rare_summary, filename=output_file, title=f"{title} - Rare Clustering")

    accumulate_rare_summary(dataset_name=dataset_name, method_name=method, rare_summary=rare_summary)

    dataset_tag = dataset_name.replace(".", "_")
    diag_csv = f"{output_dir}/rare_diagnostics_{dataset_tag}_{method}.csv"
    save_rare_diagnostics(
        dataset_name=dataset_name,
        method_name=method,
        metrics_per_cluster=metrics_per_cluster,
        out_csv=diag_csv,
    )

    all_diag_csv = f"{output_dir}/all_clusters_diagnostics_{dataset_tag}_{method}.csv"
    metrics_all_clusters = build_all_cluster_diagnostics(
        cluster_labels=result["cluster_id"].values,
        true_labels=result["cell_type"].values,
        rare_types=rare_types,
        threshold=0.3,
    )
    save_rare_diagnostics(
        dataset_name=dataset_name,
        method_name=method,
        metrics_per_cluster=metrics_all_clusters,
        out_csv=all_diag_csv,
    )
