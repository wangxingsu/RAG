from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder


def evaluate_all_main(df, true_col="cell_type", pred_col="cluster_id"):
    y_true = LabelEncoder().fit_transform(df[true_col])
    y_pred = LabelEncoder().fit_transform(df[pred_col])

    return {
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
    }
