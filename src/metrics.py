from typing import Dict, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction

def compute_metrics(eval_pred: Union[EvalPrediction, tuple]) -> Dict[str, float]:
    """
    Computes weighted evaluation metrics for multi-class sequence classification.

    Args:
        eval_pred: An instance of transformers.EvalPrediction or a tuple of
                   (predictions, label_ids) passed by the Trainer loop.
                   - predictions: Raw logits of shape (batch_size, num_classes)
                   - label_ids: Ground truth integer class indices (batch_size,)

    Returns:
        dict: Rounded evaluation metrics (accuracy, precision, recall, f1).
    """

    # unpacking predictions & labelss
    if isinstance(eval_pred, EvalPrediction):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=labels,
        y_pred=predictions,
        average="weighted",
        zero_division=0,
    )

    accuracy = accuracy_score(y_true=labels, y_pred=predictions)

    return {
        "accuracy": round(float(accuracy), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(precision)),
        "recall": round(float(recall), 4),
    }
