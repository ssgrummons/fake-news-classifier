from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from scipy.special import softmax
from transformers.trainer_utils import PredictionOutput

def compute_metrics(pred: PredictionOutput) -> dict:
    logits = pred.predictions
    labels = pred.label_ids
    preds = logits.argmax(axis=-1)
    probs = softmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "log_loss": log_loss(labels, probs)
    }

__all__ = ["compute_metrics"]
