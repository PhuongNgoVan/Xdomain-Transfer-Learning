"""Validated probability and label projections using the frozen shared rule."""
import numpy as np

from src.common.constants import HARD_MERGE_RULE


def validate_probabilities(probs, num_labels=None):
    array = np.asarray(probs, dtype=np.float64)
    if array.ndim != 2 or (num_labels is not None and array.shape[1] != num_labels):
        raise ValueError(f"Expected probability matrix with {num_labels} columns, got {array.shape}")
    if not np.isfinite(array).all() or (array < 0).any() or not np.allclose(array.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities must be finite, non-negative, and sum to one")
    return array


def aggregate_probs_7to3(probs7):
    # Reuse the existing verified probability aggregator rather than another merge rule.
    from scripts.check_step2_data import aggregate_probs_7to3 as shared_aggregate
    return shared_aggregate(validate_probabilities(probs7, 7))


def map_labels7(labels):
    array = np.asarray(labels)
    if not np.isin(array, tuple(HARD_MERGE_RULE)).all():
        raise ValueError("Emotion labels must be in 0..6")
    lookup = np.asarray([HARD_MERGE_RULE[i] for i in range(7)], dtype=np.int64)
    return lookup[array.astype(np.int64)]


def project_probabilities(probs7):
    array = validate_probabilities(probs7, 7)
    probs3 = aggregate_probs_7to3(array)
    return probs3, np.argmax(probs3, axis=1), map_labels7(np.argmax(array, axis=1))
