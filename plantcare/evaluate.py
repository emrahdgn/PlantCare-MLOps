import warnings
from typing import Any, Dict, List, Union

from sklearn.metrics import f1_score, precision_score, recall_score


def multi_label_classification_metrics(
    true_labels: Union[List[int], List[List[int]], Any], pred_labels: Union[List[int], List[List[int]], Any], label_names: List[str]
) -> Dict[str, Any]:
    """
    Compute various evaluation metrics for multi-label classification.

    Args:
        true_labels (Union[List[int], List[List[int]], Any]):
            The true labels. It can be a 1D array-like, label indicator array, or sparse matrix.
        pred_labels (Union[List[int], List[List[int]], Any]):
            The predicted labels. It can be a 1D array-like, label indicator array, or sparse matrix.
        label_names (List[str]):
            The names of the labels.

    Returns:
        dict: A dictionary containing the computed evaluation metrics.
    """
    warnings.filterwarnings("ignore")

    micro_precision = precision_score(true_labels, pred_labels, average="micro")
    micro_recall = recall_score(true_labels, pred_labels, average="micro")
    micro_f1 = f1_score(true_labels, pred_labels, average="micro")

    macro_precision = precision_score(true_labels, pred_labels, average="macro")
    macro_recall = recall_score(true_labels, pred_labels, average="macro")
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")

    label_precision = precision_score(true_labels, pred_labels, average=None)
    label_recall = recall_score(true_labels, pred_labels, average=None)
    label_f1 = f1_score(true_labels, pred_labels, average=None)

    # Create a dictionary to store the metrics
    metrics = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "label_precision": dict(zip(label_names, label_precision)),
        "label_recall": dict(zip(label_names, label_recall)),
        "label_f1": dict(zip(label_names, label_f1)),
    }

    warnings.filterwarnings("default")

    return metrics
