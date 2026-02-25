import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)

def evaluate(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "F1-score": f1_score(y_true, preds),
        "ROC-AUC": roc_auc_score(y_true, probs)
    }


def optimize_threshold(y_true, probs, target_recall=0.75):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    valid = np.where(recall[:-1] >= target_recall)[0]

    if len(valid):
        best = valid[np.argmax(precision[valid])]
    else:
        best = np.argmax(
            2 * precision[:-1] * recall[:-1] /
            (precision[:-1] + recall[:-1] + 1e-9)
        )

    return thresholds[best]
