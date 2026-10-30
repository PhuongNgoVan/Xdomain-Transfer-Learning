r"""Evaluation metrics for 3-class sentiment and 7-class emotion (`src/eval/metrics.py`).

Per `docs/contract.md` (Section 1 Row 9–10) and `tasks/M3_eval.md`:
- `zero_division=0` for all F1 computations.
- Fixed 3-class order from `src/common/constants.py`: `NEG=0, NEU=1, POS=2`.
- ECE (Expected Calibration Error): 15 equal-width bins over `[0.0, 1.0]` on the
  maximum softmax probability ($\max_c p_c$). Returns `np.nan` when logits are
  not provided, empty, or contain NaN/Inf (e.g. SVM `decision_function` or NaN logits).
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.common.constants import EMOTIONS7, LABELS3, NEG, NEU, POS

LABEL3_IDS: Tuple[int, int, int] = (NEG, NEU, POS)
EMOTION7_IDS: Tuple[int, ...] = tuple(range(len(EMOTIONS7)))


def softmax_2d(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax over a 2D array `(N, C)`."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"logits must be 2D (N, C), got shape {arr.shape}")
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    sums = np.sum(exp_vals, axis=1, keepdims=True)
    return exp_vals / sums


def compute_ece(
    y_true: Sequence[int],
    logits: Optional[np.ndarray],
    n_bins: int = 15,
    valid_logits: bool = True,
) -> float:
    """Compute Expected Calibration Error (ECE) using `n_bins` equal-width bins on `[0, 1]`.

    Uses maximum softmax probability `conf_i = max_c softmax(logits_i)_c` and
    predicted class `pred_i = argmax_c logits_i`.

    Binning convention (standard equal-width on `[0, 1]`):
    - Bin $b \\in \\{0, \\dots, M-1\\}$ has bounds $[l_b, u_b]$ with $l_b = b/M, u_b = (b+1)/M$.
    - Bin 0 is closed on both ends $[0, 1/M]$; bins $b \\ge 1$ are half-open $(l_b, u_b]$.
    - $\\text{ECE} = \\sum_{b=0}^{M-1} \\frac{|B_b|}{N} \\left| \\text{acc}(B_b) - \\text{conf}(B_b) \\right|$.

    Returns `np.nan` if `valid_logits` is False, `logits` is None, empty, or contains NaN/Inf.
    """
    if not valid_logits or logits is None:
        return float("nan")

    logit_arr = np.asarray(logits, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.int64)

    if logit_arr.ndim != 2 or logit_arr.shape[0] == 0 or logit_arr.shape[0] != y_arr.shape[0]:
        return float("nan")
    if not np.all(np.isfinite(logit_arr)):
        return float("nan")

    probs = softmax_2d(logit_arr)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_arr).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    n_samples = float(len(y_arr))
    ece = 0.0

    for b_idx in range(n_bins):
        low = bin_boundaries[b_idx]
        high = bin_boundaries[b_idx + 1]
        if b_idx == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)

        count = int(np.sum(in_bin))
        if count > 0:
            avg_acc = float(np.mean(accuracies[in_bin]))
            avg_conf = float(np.mean(confidences[in_bin]))
            ece += (count / n_samples) * abs(avg_acc - avg_conf)

    return float(ece)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    logits: Optional[Union[Sequence[Sequence[float]], np.ndarray]] = None,
    labels: Sequence[int] = LABEL3_IDS,
    n_bins: int = 15,
    valid_logits: bool = True,
) -> Dict[str, Any]:
    """Compute all contract evaluation metrics for 3-class sentiment (or 7-class emotion).

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        logits: Optional raw logits `(N, C)` before calibration/adjustment.
        labels: Ordered class IDs (defaults to `(NEG, NEU, POS) = (0, 1, 2)`).
        n_bins: Number of equal-width bins for ECE (default 15 per contract).
        valid_logits: Set False for models without valid probability logits (e.g. SVM),
            causing `ece` to be `np.nan`.

    Returns:
        Dictionary containing:
        - `macro_f1`, `accuracy`, `weighted_f1`, `balanced_acc`
        - `f1_per_class`: tuple of F1 scores aligned with `labels`
        - `f1_neg`, `f1_neu`, `f1_pos` (when `labels == (NEG, NEU, POS)`)
        - `f1_{emotion}` (when `labels == EMOTION7_IDS`)
        - `confusion_matrix`: 2D numpy array `(len(labels), len(labels))` ordered by `labels`
        - `ece`: float or `np.nan`
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true_arr.shape} vs {y_pred_arr.shape}"
        )
    if y_true_arr.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    label_list = list(labels)

    macro_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=label_list,
            average="macro",
            zero_division=0,
        )
    )
    weighted_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=label_list,
            average="weighted",
            zero_division=0,
        )
    )
    per_class_f1 = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=label_list,
        average=None,
        zero_division=0,
    )
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=label_list)

    logit_arr = np.asarray(logits, dtype=np.float64) if logits is not None else None
    ece_val = compute_ece(
        y_true_arr,
        logit_arr,
        n_bins=n_bins,
        valid_logits=valid_logits,
    )

    result: Dict[str, Any] = {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "balanced_acc": bal_acc,
        "f1_per_class": tuple(float(x) for x in per_class_f1),
        "confusion_matrix": cm,
        "ece": ece_val,
    }

    if tuple(label_list) == LABEL3_IDS:
        result["f1_neg"] = float(per_class_f1[0])
        result["f1_neu"] = float(per_class_f1[1])
        result["f1_pos"] = float(per_class_f1[2])
    elif tuple(label_list) == EMOTION7_IDS:
        for idx, emo_name in enumerate(EMOTIONS7):
            result[f"f1_{emo_name.lower()}"] = float(per_class_f1[idx])

    return result
