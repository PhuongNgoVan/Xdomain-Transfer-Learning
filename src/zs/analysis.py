"""Prediction-based zero-shot metrics and reporting helpers (no model loading)."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.common.constants import HARD_MERGE_RULE

VIEW_LABELS = {"V3-NEU": (0, 1, 2), "V3-VSMEC": (0, 1, 2),
               "V3-VSMEC-strict": (0, 2), "V7-VSMEC": tuple(range(7))}
CLASS_NAMES = {0: "neg", 1: "neu", 2: "pos"}


def metrics_from_predictions(y_true, y_pred, labels: Sequence[int]) -> dict:
    true, pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if true.ndim != 1 or pred.shape != true.shape or not len(true):
        raise ValueError("Metric inputs must be aligned nonempty one-dimensional arrays")
    per_class = f1_score(true, pred, labels=list(labels), average=None, zero_division=0)
    return {"macro_f1": float(np.mean(per_class)),
            "weighted_f1": float(f1_score(true, pred, labels=list(labels), average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(true, pred)),
            "f1_per_class": np.asarray(per_class, dtype=float)}


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence in bits, including zero probability classes."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.ndim != 1 or p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Distributions must have matching finite one-dimensional shapes")
    if (p < 0).any() or (q < 0).any() or p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("Distributions must be nonnegative with positive mass")
    p, q = p / p.sum(), q / q.sum()
    mid = (p + q) / 2
    def kl(x):
        keep = x > 0
        return np.sum(x[keep] * np.log2(x[keep] / mid[keep]))
    return float((kl(p) + kl(q)) / 2)


def prediction_js(y_true, y_pred) -> float:
    return js_divergence(np.bincount(np.asarray(y_true, dtype=int), minlength=3),
                         np.bincount(np.asarray(y_pred, dtype=int), minlength=3))


def transfer_gap(zs_f1, ref_f1):
    return np.asarray(ref_f1) - np.asarray(zs_f1)


def retention(zs_f1, ref_f1):
    zs, ref = np.broadcast_arrays(np.asarray(zs_f1, dtype=float), np.asarray(ref_f1, dtype=float))
    return np.divide(zs, ref, out=np.full(zs.shape, np.nan), where=np.abs(ref) >= 1e-8)


def contrast_stat(values: dict, stat: str):
    if stat == "gap":
        return transfer_gap(values["zs"], values["ref"])
    if stat == "retention":
        return retention(values["zs"], values["ref"])
    raise ValueError(f"Unknown contrast stat {stat!r}")


def confusion_counts(y_true, preds_by_seed, n_true: int, n_pred: int) -> np.ndarray:
    true, preds = np.asarray(y_true, dtype=int), np.asarray(preds_by_seed, dtype=int)
    if preds.ndim == 1:
        preds = preds[None, :]
    if true.ndim != 1 or preds.ndim != 2 or preds.shape[1] != len(true):
        raise ValueError("Confusion inputs must align by test ID")
    if (true < 0).any() or (true >= n_true).any() or (preds < 0).any() or (preds >= n_pred).any():
        raise ValueError("Confusion labels outside declared dimensions")
    indices = np.broadcast_to(true, preds.shape) * n_pred + preds
    return np.bincount(indices.ravel(), minlength=n_true * n_pred).reshape(n_true, n_pred)


def row_normalize(counts) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def merge_labels7(labels):
    values = np.asarray(labels)
    if not np.isin(values, tuple(HARD_MERGE_RULE)).all():
        raise ValueError("VSMEC labels must belong to the frozen 7-class space")
    lookup = np.array([HARD_MERGE_RULE[i] for i in range(7)])
    return lookup[values.astype(int)]


def baseline_floors(y_true, train_labels3, labels, *, n_sim: int = 2000, seed: int = 0) -> dict:
    """Reporting-only floors; training priors never alter model predictions."""
    true, train = np.asarray(y_true, dtype=int), np.asarray(train_labels3, dtype=int)
    if not len(true) or not len(train) or not np.isin(train, [0, 1, 2]).all():
        raise ValueError("Floors require nonempty test and valid training labels")
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    prior = np.bincount(train, minlength=3).astype(float)
    prior /= prior.sum()
    majority = np.full(len(true), int(np.argmax(prior)))
    majority_f1 = metrics_from_predictions(true, majority, labels)["macro_f1"]
    rng = np.random.Generator(np.random.PCG64(seed))
    total = 0.0
    labels = tuple(labels)
    for start in range(0, n_sim, 500):
        n = min(500, n_sim - start)
        pred = rng.choice(3, size=(n, len(true)), p=prior)
        class_f1 = []
        for label in labels:
            actual = true == label
            guessed = pred == label
            tp = (guessed & actual[None, :]).sum(axis=1)
            denominator = guessed.sum(axis=1) + actual.sum()
            class_f1.append(np.divide(2.0 * tp, denominator, out=np.zeros(n), where=denominator > 0))
        total += np.mean(np.stack(class_f1), axis=0).sum()
    return {"majority": majority_f1, "prior_random": float(total / n_sim), "n_sim": n_sim}
