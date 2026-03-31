import pandas as pd

def save_results_to_file(result_dict, filename="evaluation_results.txt", title=None):
    with open(filename, 'a') as f:
        if title:
            f.write(f"## {title}\n")
        for k, v in result_dict.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
