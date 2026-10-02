"""Accuracy, calibration and selective-prediction metrics."""

import numpy as np
from sklearn.metrics import f1_score


def accuracy(probs, y) -> float:
    return float((np.argmax(probs, axis=1) == y).mean())


def macro_f1(probs, y) -> float:
    return float(f1_score(y, np.argmax(probs, axis=1), average="macro",
                          labels=list(range(probs.shape[1])), zero_division=0))


def ece(probs, y, n_bins: int = 10) -> float:
    """Top-label expected calibration error with equal-width bins."""
    conf = probs.max(axis=1)
    correct = np.argmax(probs, axis=1) == y
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(probs, y) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def risk_coverage(conf, correct):
    """Coverage and error rate when keeping the k most confident predictions, k = 1..n."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    errors = np.cumsum(~np.asarray(correct)[order])
    k = np.arange(1, len(order) + 1)
    return k / len(order), errors / k


def aurc(conf, correct) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    return float(np.mean(risk_coverage(conf, correct)[1]))


def threshold_for_target(conf, correct, target: float) -> float:
    """Lowest confidence threshold whose selected set has error <= target (inf if none)."""
    conf, correct = np.asarray(conf), np.asarray(correct)
    order = np.argsort(-conf, kind="stable")
    c = conf[order]
    risk = np.cumsum(~correct[order]) / np.arange(1, len(c) + 1)
    group_end = np.append(c[1:] != c[:-1], True)  # evaluate only at the end of tied groups
    valid = np.flatnonzero(group_end & (risk <= target))
    return float(c[valid[-1]]) if valid.size else float("inf")


def selective(conf, correct, threshold: float) -> dict[str, float]:
    """Automation rate (coverage) and error among automated cases for a fixed threshold."""
    keep = np.asarray(conf) >= threshold
    return {
        "automation_rate": float(keep.mean()),
        "automated_error": float((~np.asarray(correct)[keep]).mean()) if keep.any() else float("nan"),
    }
