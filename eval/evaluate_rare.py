from collections import Counter
import math

import numpy as np
import pandas as pd

summary_rare_rows = []


def accumulate_rare_summary(dataset_name, method_name, rare_summary):
    summary_rare_rows.append({"Dataset": dataset_name, "Method": method_name, **rare_summary})


def save_rare_summary_to_csv(filename="evaluation_summary_rare.csv"):
    pd.DataFrame(summary_rare_rows).to_csv(filename, index=False)


def get_confusion_matrix(cluster_cell_indices, true_labels, target_label, total_samples):
    tp = np.sum(true_labels[cluster_cell_indices] == target_label)
    fp = len(cluster_cell_indices) - tp
    fn = np.sum(true_labels == target_label) - tp
    tn = total_samples - tp - fp - fn
    return tp, fp, fn, tn


def compute_metrics(tp, fp, fn, tn):
    def safe_div(x, y):
        return x / y if y != 0 else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fmi = (precision * recall) ** 0.5 if precision > 0 and recall > 0 else 0.0
    specificity = safe_div(tn, tn + fp)
    gmean = (recall * specificity) ** 0.5 if recall > 0 and specificity > 0 else 0.0

    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = safe_div(numerator, denominator)

    total = tp + fp + fn + tn
    accuracy = safe_div(tp + tn, total)
    p_e = safe_div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), total**2)
    kappa = safe_div(accuracy - p_e, 1 - p_e)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "gmean": gmean,
        "mcc": mcc,
        "accuracy": accuracy,
        "kappa": kappa,
        "fmi": fmi,
    }


def evaluate_rare_main(cluster_labels, true_labels, rare_types, threshold=0.3, target_cluster_ids=None):
    total_samples = len(true_labels)

    if target_cluster_ids is None:
        cluster_ids = np.unique(cluster_labels)
    else:
        target_set = set(target_cluster_ids)
        cluster_ids = np.array([cid for cid in np.unique(cluster_labels) if cid in target_set])
        if len(cluster_ids) == 0:
            cluster_ids = np.unique(cluster_labels)

    cluster_to_indices = {cid: np.where(cluster_labels == cid)[0] for cid in cluster_ids}
    true_rare_types = sorted(set(rare_types))

    all_metrics_sum = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "gmean": 0.0,
        "mcc": 0.0,
        "accuracy": 0.0,
        "kappa": 0.0,
    }
    metrics_per_cluster = []
    identified_rare_types = set()

    def choose_best_cluster_for_type(target_type):
        best = None
        best_key = None
        for cid, idx in cluster_to_indices.items():
            tp, fp, fn, tn = get_confusion_matrix(idx, true_labels, target_type, total_samples)
            metrics = compute_metrics(tp, fp, fn, tn)
            key = (metrics["f1"], metrics["precision"], metrics["recall"])
            if best is None or key > best_key:
                best = (cid, idx, tp, fp, fn, tn, metrics)
                best_key = key
        return best

    for target_type in true_rare_types:
        cid, idx, tp, fp, fn, tn, metrics = choose_best_cluster_for_type(target_type)

        dominant_label = None
        if len(idx) > 0:
            vals, cnts = np.unique(true_labels[idx], return_counts=True)
            dominant_label = vals[np.argmax(cnts)]

        cluster_size = int(len(idx))
        rare_size = int(np.sum(true_labels == target_type))
        purity = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        coverage = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = cluster_size + rare_size - tp
        jaccard = tp / denom if denom > 0 else 0.0

        is_hit = (rare_size > 0) and (dominant_label == target_type) and (coverage >= threshold)
        if is_hit:
            identified_rare_types.add(target_type)

        for k in all_metrics_sum:
            all_metrics_sum[k] += metrics[k]

        try:
            cid_out = int(cid)
        except Exception:
            cid_out = cid

        metrics_per_cluster.append(
            {
                "cluster_id": cid_out,
                "rare_type": target_type,
                "cluster_dominant": dominant_label,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "cluster_size": cluster_size,
                "rare_size": rare_size,
                "purity": purity,
                "coverage": coverage,
                "jaccard": jaccard,
                **metrics,
            }
        )

    denom = len(true_rare_types)
    metrics_avg = {k: (all_metrics_sum[k] / denom) if denom > 0 else 0.0 for k in all_metrics_sum}

    identified_rare_info = {
        "count": len(identified_rare_types),
        "types": sorted(list(identified_rare_types)),
    }
    true_rare_info = {
        "count": len(true_rare_types),
        "types": true_rare_types,
    }
    return metrics_avg, metrics_per_cluster, identified_rare_info, true_rare_info


def save_rare_diagnostics(dataset_name, method_name, metrics_per_cluster, out_csv):
    df = pd.DataFrame(metrics_per_cluster)
    df.insert(0, "Dataset", dataset_name)
    df.insert(1, "Method", method_name)
    df.to_csv(out_csv, index=False)


def build_all_cluster_diagnostics(cluster_labels, true_labels, rare_types, threshold=0.3):
    total_samples = len(true_labels)
    cluster_ids = np.unique(cluster_labels)
    rare_types_set = set(rare_types)

    cluster_to_indices = {cid: np.where(cluster_labels == cid)[0] for cid in cluster_ids}
    rows = []
    for cid, idx in cluster_to_indices.items():
        counts = Counter(true_labels[idx])
        cluster_size = int(len(idx))
        if cluster_size == 0:
            continue

        dominant_label, dominant_count = counts.most_common(1)[0]
        dominant_frac = dominant_count / cluster_size
        top3 = ";".join([f"{label}:{cnt}" for label, cnt in counts.most_common(3)])

        probs = [cnt / cluster_size for cnt in counts.values()]
        label_entropy = -sum(p * math.log(p + 1e-12) for p in probs)

        tp, fp, fn, tn = get_confusion_matrix(idx, true_labels, dominant_label, total_samples)
        metrics = compute_metrics(tp, fp, fn, tn)

        hit_types = []
        for target_type in rare_types_set:
            tp_t, fp_t, fn_t, tn_t = get_confusion_matrix(idx, true_labels, target_type, total_samples)
            coverage = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
            if coverage >= threshold:
                hit_types.append(target_type)

        rare_total = int(np.sum(true_labels == dominant_label))
        denom = cluster_size + rare_total - tp
        rows.append(
            {
                "cluster_id": int(cid),
                "cluster_size": cluster_size,
                "dominant_label": dominant_label,
                "dominant_count": int(dominant_count),
                "dominant_frac": float(dominant_frac),
                "top3": top3,
                "label_entropy": float(label_entropy),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "purity": float(tp / (tp + fp) if (tp + fp) > 0 else 0.0),
                "coverage": float(tp / (tp + fn) if (tp + fn) > 0 else 0.0),
                "jaccard": float(tp / denom if denom > 0 else 0.0),
                **metrics,
                "is_dominant_rare": bool(dominant_label in rare_types_set),
                "n_rare_types_hit": int(len(hit_types)),
                "hit_types": ";".join(sorted(hit_types)),
            }
        )

    return rows
