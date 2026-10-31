"""Evaluate per-sample prediction files (`src/eval/evaluate.py`).

Per `docs/contract.md` (Section 2.2, 2.3) and `tasks/M3_eval.md`:
- Reads per-sample prediction CSV (`id, y_true, y_pred, logit_neg, logit_neu, logit_pos`).
- Supports `inference_mode`:
  - `plain`: argmax over raw logits (or fallback to `y_pred` column when logits are NaN, e.g., Hard_TF_SVM).
  - `prior`: $\\arg\\max(\\text{raw logits} + \\tau \\cdot \\log \\pi)$, where $\\pi = (\\text{prior\\_neg}, \\text{prior\\_neu}, \\text{prior\\_pos})$.
- Shared across `neuesc_dev`, `neuesc_test`, `vsfc_test`.
- Writes result rows strictly conforming to `RESULT_COLUMNS` (Section 2.3) ONLY for
  datasets that have a corresponding prediction file on disk.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.common.constants import NEG, NEU, POS
from src.common.io import append_row_schema, read_csv_schema
from src.common.schema import (
    PRED_COLUMNS,
    RESULT_COLUMNS,
    RESULT_KEY_COLUMNS,
    SPLIT_COLUMNS,
    VALID_INFERENCE_MODES,
    VALID_TARGET_DATASETS,
)
from src.eval.metrics import compute_metrics

DATASET_TO_SPLIT: Dict[str, str] = {
    "neuesc_dev": "dev",
    "neuesc_test": "test",
    "vsfc_test": "test",
}


class MetadataError(ValueError, KeyError):
    """Raised when required run_metadata fields (such as 'precision') are missing or invalid."""


def is_svm_model(model_name: str, loss_name: Optional[str] = None) -> bool:
    """Return True if the model is an SVM (whose `decision_function` is not a probability logit)."""
    m = str(model_name).upper()
    l = str(loss_name or "").upper()
    return "SVM" in m or "HINGE" in l


def compute_priors_from_split(
    split_csv: Union[str, Path],
    subset: str = "train",
) -> Tuple[float, float, float]:
    """Compute `(prior_neg, prior_neu, prior_pos)` from a split CSV file.

    Reads columns according to `SPLIT_COLUMNS` (`id, text, label, subset`) and
    orders class priors by `(NEG, NEU, POS)` from `src/common/constants.py`.
    """
    df_split = read_csv_schema(split_csv, SPLIT_COLUMNS, strict_order=False)
    sub_df = df_split[df_split["subset"] == subset]
    if sub_df.empty:
        raise ValueError(f"No rows found for subset={subset!r} in {split_csv}")

    valid_labels = {NEG, NEU, POS}
    if sub_df["label"].isna().any():
        raise ValueError(f"Column 'label' contains NaN/null values in {split_csv}")
    invalid_labels = set(sub_df["label"].unique()) - valid_labels
    if invalid_labels:
        raise ValueError(
            f"Invalid label values {sorted(invalid_labels)} in {split_csv}; "
            f"allowed values are {sorted(valid_labels)}"
        )

    labels = sub_df["label"].to_numpy(dtype=np.int64)
    n_total = float(len(labels))
    return (
        float(np.sum(labels == NEG) / n_total),
        float(np.sum(labels == NEU) / n_total),
        float(np.sum(labels == POS) / n_total),
    )


def predict_from_logits(
    logits: np.ndarray,
    fallback_y_pred: Sequence[int],
    inference_mode: str = "plain",
    tau: float = 1.0,
    priors: Optional[Tuple[float, float, float]] = None,
    allow_fallback: bool = False,
    allow_nan_fallback: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute predicted labels and effective logits under `plain` or `prior` mode.

    Args:
        logits: Raw `(N, 3)` array `[logit_neg, logit_neu, logit_pos]`.
        fallback_y_pred: `y_pred` column from the prediction file (used when logits contain NaN
            and `allow_fallback=True`).
        inference_mode: `"plain"` or `"prior"`.
        tau: Temperature/scaling parameter $\\tau$ for logit adjustment.
        priors: Tuple `(prior_neg, prior_neu, prior_pos)` of training set class priors $\\pi$.
        allow_fallback: Whether to allow falling back to `fallback_y_pred` when logits contain NaN
            (only True for B2/SVM per contract 2.2).
        allow_nan_fallback: Alias for `allow_fallback` if provided.

    Returns:
        `(y_pred_out, adjusted_logits)`
    """
    if allow_nan_fallback is not None:
        allow_fallback = allow_nan_fallback

    if inference_mode not in VALID_INFERENCE_MODES:
        raise ValueError(
            f"inference_mode must be one of {VALID_INFERENCE_MODES}, got {inference_mode!r}"
        )

    logit_arr = np.asarray(logits, dtype=np.float64)
    if logit_arr.ndim != 2 or logit_arr.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) logits, got shape {logit_arr.shape}")

    # If logits contain NaN/Inf, only allow fallback to y_pred in plain mode when allow_fallback=True (SVM)
    if not np.all(np.isfinite(logit_arr)):
        if inference_mode == "prior":
            raise ValueError("Cannot run 'prior' inference mode when logits contain NaN/Inf")
        if not allow_fallback:
            raise ValueError(
                "Logits contain NaN/Inf; fallback to y_pred is only allowed for SVM models"
            )
        return np.asarray(fallback_y_pred, dtype=np.int64), logit_arr

    if inference_mode == "plain" or float(tau) == 0.0:
        adjusted = logit_arr.copy()
    else:
        if priors is None:
            raise ValueError("priors=(prior_neg, prior_neu, prior_pos) required for inference_mode='prior'")
        pi = np.asarray(priors, dtype=np.float64)
        if pi.shape != (3,) or np.any(pi <= 0.0):
            raise ValueError(f"priors must be 3 positive numbers, got {priors!r}")
        pi = pi / np.sum(pi)
        # Note on sign convention for post-hoc Logit Adjustment (`docs/contract.md` 2.3):
        # Contract explicitly specifies: `prior = argmax(logit thô + τ·log π)`.
        adjusted = logit_arr + float(tau) * np.log(pi)[np.newaxis, :]

    y_pred_out = np.argmax(adjusted, axis=1).astype(np.int64)
    return y_pred_out, adjusted


def evaluate_prediction_file(
    pred_path: Union[str, Path],
    run_metadata: Dict[str, Any],
    dataset: str,
    inference_mode: str = "plain",
    results_csv: Optional[Union[str, Path]] = None,
    overwrite_duplicate_key: bool = False,
    expected_n_rows: Optional[int] = None,
    expected_priors: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """Evaluate a single per-sample prediction CSV and optionally append to `results_csv`.

    Note:
        For non-SVM models (`allow_fallback=False`), `y_pred` in the prediction file is verified
        against `argmax` of raw logits across all `inference_mode`s (allowing ties within `1e-6`).
        B2 (gộp cứng) non-SVM models must also satisfy this check because their logits are the
        log of summed probabilities per merged sentiment group.

    Args:
        pred_path: Path to `results/predictions/{run_id}_{dataset}.csv`.
        run_metadata: Dictionary with run metadata fields (`run_id`, `model`, `variant`,
            `fraction`, `seed`, `lr`, `tau`, `loss`, `precision`, `max_length`, `best_step`,
            `n_train`, `prior_neg`, `prior_neu`, `prior_pos`, `gpu`, `code_version`,
            `data_hash`, `config_path`, `official`).
        dataset: One of `neuesc_dev`, `neuesc_test`, `vsfc_test`.
        inference_mode: `"plain"` or `"prior"`.
        results_csv: Optional path to `results/{member}.csv` to append the evaluated row.
        overwrite_duplicate_key: Whether to update an existing row with the same primary key.
        expected_n_rows: Optional expected number of rows in `pred_path`.
        expected_priors: Optional expected `(prior_neg, prior_neu, prior_pos)` tuple; if provided,
            each component in `run_metadata` must match within `5e-3` absolute tolerance.

    Returns:
        Dictionary matching `RESULT_COLUMNS` in exact order.
    """
    if dataset not in VALID_TARGET_DATASETS:
        raise ValueError(
            f"dataset must be one of {VALID_TARGET_DATASETS}, got {dataset!r}"
        )
    if inference_mode not in VALID_INFERENCE_MODES:
        raise ValueError(
            f"inference_mode must be one of {VALID_INFERENCE_MODES}, got {inference_mode!r}"
        )

    run_id = str(run_metadata["run_id"])

    if (
        "precision" not in run_metadata
        or run_metadata["precision"] is None
        or str(run_metadata["precision"]).strip() == ""
    ):
        raise MetadataError(
            f"[{run_id} | {dataset}] Missing required 'precision' in run_metadata"
        )
    precision_val = str(run_metadata["precision"]).strip()

    expected_split = DATASET_TO_SPLIT[dataset]
    if "split" in run_metadata and run_metadata["split"] is not None:
        provided_split = str(run_metadata["split"])
        if provided_split != expected_split:
            raise ValueError(
                f"[{run_id} | {dataset}] run_metadata['split']={provided_split!r} "
                f"does not match DATASET_TO_SPLIT[{dataset!r}]={expected_split!r}"
            )
    split_name = expected_split

    priors = (
        float(run_metadata["prior_neg"]),
        float(run_metadata["prior_neu"]),
        float(run_metadata["prior_pos"]),
    )
    prior_arr = np.asarray(priors, dtype=np.float64)
    prior_sum = float(np.sum(prior_arr))
    if (
        not np.all(np.isfinite(prior_arr))
        or np.any(prior_arr <= 0.0)
        or not (1.0 - 1e-3 <= prior_sum <= 1.0 + 1e-3)
    ):
        raise ValueError(
            f"[{run_id} | {dataset}] Invalid priors (prior_neg, prior_neu, prior_pos)={priors} "
            f"(sum={prior_sum}); must be finite, > 0, and sum within [1 - 1e-3, 1 + 1e-3]"
        )

    if expected_priors is not None:
        exp_arr = np.asarray(expected_priors, dtype=np.float64)
        if exp_arr.shape != (3,):
            raise ValueError(
                f"[{run_id} | {dataset}] expected_priors must have 3 elements, got {expected_priors!r}"
            )
        max_diff = float(np.max(np.abs(prior_arr - exp_arr)))
        if max_diff > 5e-3:
            raise ValueError(
                f"[{run_id} | {dataset}] run_metadata priors {priors} deviate from "
                f"expected_priors {tuple(float(x) for x in exp_arr)} by {max_diff:.6f} (> 5e-3)"
            )

    df_pred = read_csv_schema(pred_path, PRED_COLUMNS, strict_order=True)

    if expected_n_rows is not None and len(df_pred) != int(expected_n_rows):
        raise ValueError(
            f"[{run_id} | {dataset}] Expected {expected_n_rows} rows in prediction file, "
            f"got {len(df_pred)}"
        )

    if df_pred["id"].isna().any():
        raise ValueError(
            f"[{run_id} | {dataset}] Column 'id' contains NaN/null values"
        )
    if df_pred["id"].duplicated().any():
        dup_ids = df_pred.loc[df_pred["id"].duplicated(), "id"].unique().tolist()
        raise ValueError(
            f"[{run_id} | {dataset}] Duplicate 'id' values found in prediction file: {dup_ids[:10]}"
        )

    valid_labels = {NEG, NEU, POS}
    if df_pred["y_true"].isna().any():
        raise ValueError(
            f"[{run_id} | {dataset}] Column 'y_true' contains NaN/null values"
        )
    if df_pred["y_pred"].isna().any():
        raise ValueError(
            f"[{run_id} | {dataset}] Column 'y_pred' contains NaN/null values"
        )
    invalid_true = set(df_pred["y_true"].unique()) - valid_labels
    if invalid_true:
        raise ValueError(
            f"[{run_id} | {dataset}] Invalid y_true values {sorted(invalid_true)}; "
            f"allowed values are {sorted(valid_labels)}"
        )
    invalid_pred = set(df_pred["y_pred"].unique()) - valid_labels
    if invalid_pred:
        raise ValueError(
            f"[{run_id} | {dataset}] Invalid y_pred values {sorted(invalid_pred)}; "
            f"allowed values are {sorted(valid_labels)}"
        )

    y_true = df_pred["y_true"].to_numpy(dtype=np.int64)
    fallback_y_pred = df_pred["y_pred"].to_numpy(dtype=np.int64)
    raw_logits = df_pred.loc[:, ["logit_neg", "logit_neu", "logit_pos"]].to_numpy(
        dtype=np.float64
    )

    tau = float(run_metadata.get("tau", 1.0))

    model_name = str(run_metadata.get("model", ""))
    loss_name = str(run_metadata.get("loss", ""))
    allow_fallback = is_svm_model(model_name, loss_name)

    try:
        y_pred, eval_logits = predict_from_logits(
            logits=raw_logits,
            fallback_y_pred=fallback_y_pred,
            inference_mode=inference_mode,
            tau=tau,
            priors=priors,
            allow_fallback=allow_fallback,
        )
    except ValueError as exc:
        raise ValueError(f"[{run_id} | {dataset}] {exc}") from exc

    # For non-SVM models, verify file's y_pred matches argmax of raw logits (allowing ties within 1e-6)
    if not allow_fallback:
        row_max = np.max(raw_logits, axis=1)
        pred_logit = raw_logits[np.arange(len(fallback_y_pred)), fallback_y_pred]
        mismatch_mask = pred_logit < (row_max - 1e-6)
        if np.any(mismatch_mask):
            n_mismatch = int(np.sum(mismatch_mask))
            n_total = len(df_pred)
            bad_ids = df_pred.loc[mismatch_mask, "id"].iloc[:5].tolist()
            raise ValueError(
                f"[{run_id} | {dataset}] Column 'y_pred' does not match argmax of raw logits "
                f"on {n_mismatch}/{n_total} rows (first mismatched ids: {bad_ids})"
            )

    has_valid_prob_logits = (not allow_fallback) and bool(
        np.all(np.isfinite(eval_logits))
    )

    m = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        logits=eval_logits,
        labels=(NEG, NEU, POS),
        n_bins=15,
        valid_logits=has_valid_prob_logits,
    )

    row: Dict[str, Any] = {
        "run_id": run_id,
        "model": model_name,
        "variant": str(run_metadata.get("variant", "base")),
        "dataset": dataset,
        "split": split_name,
        "inference_mode": inference_mode,
        "fraction": int(run_metadata["fraction"]),
        "seed": int(run_metadata["seed"]),
        "lr": float(run_metadata["lr"]),
        "tau": tau,
        "loss": loss_name,
        "precision": precision_val,
        "max_length": int(run_metadata.get("max_length", 256)),
        "best_step": int(run_metadata.get("best_step", 0)),
        "n_train": int(run_metadata["n_train"]),
        "prior_neg": priors[0],
        "prior_neu": priors[1],
        "prior_pos": priors[2],
        "macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
        "weighted_f1": m["weighted_f1"],
        "balanced_acc": m["balanced_acc"],
        "f1_neg": m["f1_neg"],
        "f1_neu": m["f1_neu"],
        "f1_pos": m["f1_pos"],
        "ece": m["ece"],
        "gpu": str(run_metadata.get("gpu", "cpu")),
        "code_version": str(run_metadata.get("code_version", "unknown")),
        "data_hash": str(run_metadata.get("data_hash", "unknown")),
        "config_path": str(run_metadata.get("config_path", "")),
        "official": bool(run_metadata.get("official", False)),
    }

    # Ensure exact column order matching RESULT_COLUMNS
    ordered_row = {col: row[col] for col in RESULT_COLUMNS}

    if results_csv is not None:
        append_row_schema(
            ordered_row,
            results_csv,
            RESULT_COLUMNS,
            key_columns=RESULT_KEY_COLUMNS,
            overwrite_duplicate_key=overwrite_duplicate_key,
        )

    return ordered_row


def evaluate_run_predictions(
    predictions_dir: Union[str, Path],
    run_metadata: Dict[str, Any],
    results_csv: Optional[Union[str, Path]] = None,
    datasets: Sequence[str] = VALID_TARGET_DATASETS,
    inference_modes: Optional[Sequence[str]] = None,
    overwrite_duplicate_key: bool = False,
    expected_n_rows: Optional[Union[int, Mapping[str, int]]] = None,
    expected_priors: Optional[Tuple[float, float, float]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate all available prediction files for a run (`run_id`).

    Only writes rows for datasets whose prediction file `{run_id}_{dataset}.csv`
    actually exists in `predictions_dir`.
    If `inference_modes` is None, automatically evaluates `["plain", "prior"]`
    when `loss` is `"la"` or `"balanced_softmax"`, and `["plain"]` otherwise.
    """
    pred_dir = Path(predictions_dir)
    run_id = str(run_metadata["run_id"])
    loss_name = str(run_metadata.get("loss", "ce")).lower()

    if inference_modes is None:
        if loss_name in ("la", "balanced_softmax"):
            modes: Sequence[str] = ("plain", "prior")
        else:
            modes = ("plain",)
    else:
        modes = tuple(inference_modes)

    evaluated_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        pred_file = pred_dir / f"{run_id}_{ds}.csv"
        if not pred_file.is_file():
            continue
        ds_expected_rows: Optional[int]
        if isinstance(expected_n_rows, Mapping):
            ds_expected_rows = expected_n_rows.get(ds)
        else:
            ds_expected_rows = expected_n_rows
        for mode in modes:
            row = evaluate_prediction_file(
                pred_path=pred_file,
                run_metadata=run_metadata,
                dataset=ds,
                inference_mode=mode,
                results_csv=results_csv,
                overwrite_duplicate_key=overwrite_duplicate_key,
                expected_n_rows=ds_expected_rows,
                expected_priors=expected_priors,
            )
            evaluated_rows.append(row)

    return evaluated_rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a prediction file into RESULT_COLUMNS.")
    parser.add_argument("--pred-path", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=list(VALID_TARGET_DATASETS))
    parser.add_argument("--inference-mode", type=str, default="plain", choices=list(VALID_INFERENCE_MODES))
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--variant", type=str, default="base")
    parser.add_argument("--fraction", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--precision", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--best-step", type=int, default=0)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--prior-neg", type=float, required=True)
    parser.add_argument("--prior-neu", type=float, required=True)
    parser.add_argument("--prior-pos", type=float, required=True)
    parser.add_argument("--gpu", type=str, default="cpu")
    parser.add_argument("--code-version", type=str, default="unknown")
    parser.add_argument("--data-hash", type=str, default="unknown")
    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--official", action="store_true", default=False)
    parser.add_argument("--expected-n-rows", type=int, default=None)
    parser.add_argument("--overwrite-duplicate-key", action="store_true", default=False)
    args = parser.parse_args(argv)

    meta = {
        "run_id": args.run_id,
        "model": args.model,
        "variant": args.variant,
        "fraction": args.fraction,
        "seed": args.seed,
        "lr": args.lr,
        "tau": args.tau,
        "loss": args.loss,
        "precision": args.precision,
        "max_length": args.max_length,
        "best_step": args.best_step,
        "n_train": args.n_train,
        "prior_neg": args.prior_neg,
        "prior_neu": args.prior_neu,
        "prior_pos": args.prior_pos,
        "gpu": args.gpu,
        "code_version": args.code_version,
        "data_hash": args.data_hash,
        "config_path": args.config_path,
        "official": args.official,
    }
    row = evaluate_prediction_file(
        pred_path=args.pred_path,
        run_metadata=meta,
        dataset=args.dataset,
        inference_mode=args.inference_mode,
        results_csv=args.results_csv,
        overwrite_duplicate_key=args.overwrite_duplicate_key,
        expected_n_rows=args.expected_n_rows,
    )
    print(f"Evaluated {args.run_id} ({args.dataset}, {args.inference_mode}): macro_f1={row['macro_f1']:.4f}, ece={row['ece']}")


if __name__ == "__main__":
    main()
