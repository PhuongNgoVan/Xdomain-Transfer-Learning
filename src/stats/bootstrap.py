"""Two-level nonparametric bootstrap over test sentences and model seeds.

Sentence resamples are shared only by arms evaluated on the same ordered test
set. Model seeds are resampled independently for every arm. Counts are sampled
with a multinomial distribution and evaluated in bounded-size chunks.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _integer_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite integer labels")
    integers = array.astype(np.int64)
    if not np.array_equal(array, integers):
        raise ValueError(f"{name} must contain integer labels")
    return integers


def _validate_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    y_true = _integer_array(arm["y_true"], "y_true")
    predictions = arm["preds_by_seed"]
    if isinstance(predictions, Mapping):
        predictions = [predictions[key] for key in sorted(predictions)]
    preds = _integer_array(predictions, "preds_by_seed")
    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true must be a nonempty vector")
    if preds.ndim == 1:
        preds = preds[None, :]
    if preds.ndim != 2 or preds.shape[0] == 0 or preds.shape[1] != len(y_true):
        raise ValueError("preds_by_seed must have shape (n_seeds, n_test)")
    labels = _integer_array(arm["labels"], "labels")
    if labels.ndim != 1 or labels.size == 0 or len(np.unique(labels)) != len(labels):
        raise ValueError("labels must be a nonempty sequence without duplicates")
    if np.any(y_true < 0) or np.any(preds < 0) or np.any(labels < 0):
        raise ValueError("Label IDs must be nonnegative")
    # A strict view legitimately has neutral predictions outside its score labels.
    return {"y_true": y_true, "preds_by_seed": preds, "labels": labels,
            "stat": str(arm.get("stat", "macro_f1"))}


def _js_bits(true_counts: np.ndarray, predicted_counts: np.ndarray) -> np.ndarray:
    p = true_counts / true_counts.sum(axis=-1, keepdims=True)
    q = predicted_counts / predicted_counts.sum(axis=-1, keepdims=True)
    middle = (p + q) / 2
    p_term = np.zeros_like(p)
    q_term = np.zeros_like(q)
    np.log2(np.divide(p, middle, out=np.ones_like(p), where=p > 0), out=p_term)
    np.log2(np.divide(q, middle, out=np.ones_like(q), where=q > 0), out=q_term)
    return np.clip(0.5 * ((p * p_term).sum(axis=-1) + (q * q_term).sum(axis=-1)), 0, 1)


def weighted_statistics(
    y_true: Sequence[int],
    preds_by_seed: Any,
    labels: Sequence[int],
    weights: np.ndarray,
    stat: str = "macro_f1",
) -> np.ndarray:
    """Evaluate every sentence-weight row and model seed, yielding ``(B,S)``.

    Predictions outside ``labels`` still contribute false negatives to their
    true classes. This is essential for neutral predictions on strict VSMEC.
    ``js`` always retains all three polarity classes for a three-class problem.
    """
    arm = _validate_arm({"y_true": y_true, "preds_by_seed": preds_by_seed, "labels": labels, "stat": stat})
    return _weighted_statistics(arm, weights)


def _weighted_statistics(arm: Mapping[str, Any], weights: np.ndarray) -> np.ndarray:
    truth = arm["y_true"]
    preds = arm["preds_by_seed"]
    labels = arm["labels"]
    stat = arm["stat"]
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w[None, :]
    if w.ndim != 2 or w.shape[1] != len(truth) or not np.all(np.isfinite(w)) or np.any(w < 0) or np.any(w.sum(axis=1) <= 0):
        raise ValueError("weights must have shape (B,n_test), nonnegative finite entries and positive row sums")
    if stat == "accuracy":
        return (w @ (preds == truth).T) / w.sum(axis=1, keepdims=True)
    if stat == "js":
        maximum = int(max(truth.max(), preds.max(), labels.max()))
        space = np.arange(max(3, maximum + 1))
        true_onehot = (truth[:, None] == space).astype(np.float64)
        pred_onehot = (preds.T[:, :, None] == space).astype(np.float64)
        true_counts = (w @ true_onehot)[:, None, :]
        pred_counts = (w @ pred_onehot.reshape(len(truth), -1)).reshape(len(w), len(preds), len(space))
        return _js_bits(np.broadcast_to(true_counts, pred_counts.shape), pred_counts)

    true_onehot = (truth[:, None] == labels).astype(np.float64)
    pred_onehot = (preds.T[:, :, None] == labels).astype(np.float64)
    true_counts = (w @ true_onehot)[:, None, :]
    pred_counts = (w @ pred_onehot.reshape(len(truth), -1)).reshape(len(w), len(preds), len(labels))
    correct_onehot = pred_onehot * true_onehot[:, None, :]
    tp = (w @ correct_onehot.reshape(len(truth), -1)).reshape(len(w), len(preds), len(labels))
    denominator = pred_counts + true_counts
    per_class = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    if stat == "macro_f1":
        return per_class.mean(axis=-1)
    if stat == "weighted_f1":
        support = true_counts.sum(axis=-1)
        return np.divide((per_class * true_counts).sum(axis=-1), support,
                         out=np.zeros((len(w), len(preds))), where=support > 0)
    if stat.startswith("f1_"):
        try:
            label = int(stat[3:])
        except ValueError as exc:
            raise ValueError(f"Unknown statistic {stat!r}") from exc
        positions = np.flatnonzero(labels == label)
        if len(positions) != 1:
            raise ValueError(f"Class {label} is not among score labels {labels.tolist()}")
        return per_class[:, :, int(positions[0])]
    raise ValueError(f"Unknown statistic {stat!r}")


def safe_ratio(numerator: Any, denominator: Any, threshold: float = 1e-8) -> Any:
    """Divide while marking references closer to zero than threshold as NaN."""
    numerator, denominator = np.broadcast_arrays(np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    out = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=out, where=np.abs(denominator) >= threshold)
    return float(out) if out.ndim == 0 else out


def _groups(arms: Mapping[str, Mapping[str, Any]], share_test: Any) -> dict[str, Any]:
    if isinstance(share_test, Mapping):
        if set(share_test) != set(arms):
            raise ValueError("share_test mapping must name every arm exactly once")
        groups = dict(share_test)
    elif isinstance(share_test, (bool, np.bool_)):
        groups = {name: "shared" if share_test else name for name in arms}
    else:
        raise ValueError("share_test must be bool or an arm-to-test-group mapping")
    reference: dict[Any, np.ndarray] = {}
    for name, arm in arms.items():
        group = groups[name]
        if group in reference and not np.array_equal(reference[group], arm["y_true"]):
            raise ValueError("Arms sharing test weights must use identical ordered true labels")
        reference[group] = arm["y_true"]
    return groups


def _draw_shared_test_weights(
    rng: np.random.Generator,
    arms: Mapping[str, Mapping[str, Any]],
    share_test: Any,
    size: int,
) -> dict[str, np.ndarray]:
    """Expose sentence resamples for tests; arms in a group share the same object."""
    groups = _groups(arms, share_test)
    drawn: dict[Any, np.ndarray] = {}
    result = {}
    for name, arm in arms.items():
        group = groups[name]
        if group not in drawn:
            n = len(arm["y_true"])
            drawn[group] = rng.multinomial(n, np.full(n, 1 / n), size=size)
        result[name] = drawn[group]
    return result


def _validate_boot(boot: int, ci: float, chunk_size: int) -> None:
    if not isinstance(boot, (int, np.integer)) or boot < 2:
        raise ValueError("boot must be an integer >= 2")
    if not 0 < ci < 1:
        raise ValueError("ci must lie strictly between 0 and 1")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")


def _summarize(point: float, draws: np.ndarray, ci: float) -> dict[str, Any]:
    draws = np.asarray(draws, dtype=float)
    finite = draws[np.isfinite(draws)]
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(finite, [alpha, 1 - alpha]) if len(finite) else (np.nan, np.nan)
    return {"point": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "se": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
            "n_nan": int(len(draws) - len(finite)), "draws": draws}


def bootstrap_contrast(
    arms: Mapping[str, Mapping[str, Any]],
    fn: Callable[[Mapping[str, Any]], Any],
    share_test: Any = False,
    *,
    boot: int = 10000,
    boot_seed: int = 0,
    ci: float = 0.95,
    chunk_size: int = 500,
    share_seeds: bool = False,
) -> dict[str, Any]:
    """Bootstrap arm contrasts; model seeds are independent unless share_seeds=True.

    ``fn`` accepts a mapping of arm names to scalars for the point estimate,
    and to vectors for resamples. ``share_test`` maps names to test-set group IDs,
    or is True to share all test weights / False to draw each arm independently.
    """
    _validate_boot(boot, ci, chunk_size)
    if not arms:
        raise ValueError("At least one arm is required")
    prepared = {name: _validate_arm(arm) for name, arm in arms.items()}
    if share_seeds:
        identities = []
        for name, arm in arms.items():
            seed_ids = list(arm["seeds"]) if "seeds" in arm else (sorted(arm["preds_by_seed"]) if isinstance(arm["preds_by_seed"], Mapping) else None)
            if seed_ids is None or len(seed_ids) != len(prepared[name]["preds_by_seed"]) or len(set(seed_ids)) != len(seed_ids):
                raise ValueError("Paired bootstrap requires unique explicit seed IDs for every arm")
            identities.append(seed_ids)
        if any(ids != identities[0] for ids in identities[1:]):
            raise ValueError("Paired arms require identical ordered seed IDs")
    _groups(prepared, share_test)
    points = {name: float(_weighted_statistics(arm, np.ones(len(arm["y_true"])))[0].mean())
              for name, arm in prepared.items()}
    point = float(fn(points))
    rng = np.random.Generator(np.random.PCG64(boot_seed))
    draws = np.empty(boot, dtype=np.float64)
    for start in range(0, boot, chunk_size):
        size = min(chunk_size, boot - start)
        weights = _draw_shared_test_weights(rng, prepared, share_test, size)
        values = {}
        shared_indices = None
        for name, arm in prepared.items():
            per_seed = _weighted_statistics(arm, weights[name])
            n_seeds = arm["preds_by_seed"].shape[0]
            if share_seeds:
                if shared_indices is None:
                    shared_indices = rng.integers(0, n_seeds, size=(size, n_seeds))
                seed_indices = shared_indices
            else:
                seed_indices = rng.integers(0, n_seeds, size=(size, n_seeds))
            values[name] = np.take_along_axis(per_seed, seed_indices, axis=1).mean(axis=1)
        result = np.asarray(fn(values), dtype=float)
        if result.ndim == 0:
            result = np.full(size, float(result))
        if result.shape != (size,):
            raise ValueError("Contrast function must return one scalar per bootstrap replicate")
        draws[start:start + size] = result
    return _summarize(point, draws, ci)


def bootstrap_arm(
    y_true: Sequence[int],
    preds_by_seed: Any,
    labels: Sequence[int],
    stat: str = "macro_f1",
    *,
    boot: int = 10000,
    boot_seed: int = 0,
    ci: float = 0.95,
    chunk_size: int = 500,
) -> dict[str, Any]:
    """Bootstrap a mean statistic across independently trained model seeds."""
    arm = _validate_arm({"y_true": y_true, "preds_by_seed": preds_by_seed, "labels": labels, "stat": stat})
    result = bootstrap_contrast({"arm": arm}, lambda values: values["arm"], False,
                                boot=boot, boot_seed=boot_seed, ci=ci, chunk_size=chunk_size)
    points = _weighted_statistics(arm, np.ones(len(arm["y_true"])))[0]
    result["std"] = float(np.std(points, ddof=1)) if len(points) > 1 else 0.0
    result["per_seed"] = points
    return result
