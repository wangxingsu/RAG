import pandas as pd

summary_all_rows = []

def accumulate_all_summary(dataset_name, method_name, all_summary_dict):
    row = {"Dataset": dataset_name, "Method": method_name, **all_summary_dict}
    summary_all_rows.append(row)

def save_all_summary_to_csv(filename="evaluation_summary_all.csv"):
    pd.DataFrame(summary_all_rows).to_csv(filename, index=False)

def save_results_to_file(result_dict, filename="evaluation_results.txt", title=None):
    with open(filename, 'a') as f:
        if title:
            f.write(f"## {title}\n")
        for k, v in result_dict.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
