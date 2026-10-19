"""Binary metrics v2: tie-aware AP, ROC area and explicit undefined values.

AP uses a step integral, not trapezoidal PR interpolation. At an operating
point with no positive predictions precision is undefined. Only the initial
PR plotting endpoint uses precision=1 by convention.
"""

from itertools import groupby
import math


def validate(labels, scores):
    labels, scores = list(labels), list(scores)
    if not labels or len(labels) != len(scores):
        raise ValueError("Expected nonempty, equally sized labels and scores")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Labels must be binary")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    return labels, scores


def confusion(tp, fp, positives, negatives):
    fn, tn = positives - tp, negatives - fp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": 2 * tp / denominator if denominator else None,
        "accuracy": (tp + tn) / (positives + negatives),
        "balanced_accuracy": (recall + specificity) / 2 if positives and negatives else None,
        "predicted_positive_fraction": (tp + fp) / (positives + negatives),
    }


def operating_point(labels, scores, threshold):
    labels, scores = validate(labels, scores)
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    tp = sum(y == 1 and s >= threshold for y, s in zip(labels, scores))
    fp = sum(y == 0 and s >= threshold for y, s in zip(labels, scores))
    return {"threshold": threshold, **confusion(tp, fp, sum(labels), len(labels) - sum(labels))}


def evaluate(labels, scores, threshold=0.5):
    labels, scores = validate(labels, scores)
    positives, negatives = sum(labels), len(labels) - sum(labels)
    curve = [{"threshold": None, "precision": 1.0,
              "recall": 0.0 if positives else None, "predicted_positive_fraction": 0.0,
              "endpoint_only": True}]
    tp = fp = 0
    ap = auc = previous_recall = previous_fpr = 0.0
    for score, group in groupby(sorted(zip(scores, labels), reverse=True), key=lambda pair: pair[0]):
        values = [y for _, y in group]
        tp += sum(values)
        fp += len(values) - sum(values)
        point = {"threshold": score, **confusion(tp, fp, positives, negatives)}
        recall = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        ap += (recall - previous_recall) * point["precision"]
        auc += (fpr - previous_fpr) * (recall + previous_recall) / 2
        previous_recall, previous_fpr = recall, fpr
        curve.append(point)
    return {
        "metric_version": 2, "n_examples": len(labels), "n_positive": positives,
        "positive_prevalence": positives / len(labels),
        "average_precision": ap if positives else None,
        "roc_auc": auc if positives and negatives else None,
        "operating_point": operating_point(labels, scores, threshold),
        "pr_curve": curve,
    }
