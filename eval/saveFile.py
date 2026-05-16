import pandas as pd

OUTPUT_EXCLUDED_KEYS = {
    "recall",
    "gmean",
    "mcc",
    "kappa",
    "accuracy",
    "fmi",
    "coverage",
    "jaccard",
}


def save_results_to_file(result_dict, filename="evaluation_results.txt", title=None):
    with open(filename, 'a') as f:
        if title:
            f.write(f"## {title}\n")
        for k, v in result_dict.items():
            if k in OUTPUT_EXCLUDED_KEYS:
                continue
            f.write(f"{k}: {v}\n")
        f.write("\n")
