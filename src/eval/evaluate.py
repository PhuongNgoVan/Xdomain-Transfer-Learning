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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.common.constants import NEG, NEU, POS
from src.common.io import append_row_schema, read_csv_schema
from src.common.schema import (
    PRED_COLUMNS,
    RESULT_COLUMNS,
    RESULT_KEY_COLUMNS,
    VALID_INFERENCE_MODES,
    VALID_TARGET_DATASETS,
)
from src.eval.metrics import compute_metrics

DATASET_TO_SPLIT: Dict[str, str] = {
    "neuesc_dev": "dev",
    "neuesc_test": "test",
    "vsfc_test": "test",
}


def is_svm_model(model_name: str, loss_name: Optional[str] = None) -> bool:
    """Return True if the model is an SVM (whose `decision_function` is not a probability logit)."""
    m = str(model_name).upper()
    l = str(loss_name or "").upper()
    return "SVM" in m or "HINGE" in l


def predict_from_logits(
    logits: np.ndarray,
    fallback_y_pred: Sequence[int],
    inference_mode: str = "plain",
    tau: float = 1.0,
    priors: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute predicted labels and effective logits under `plain` or `prior` mode.

    Args:
        logits: Raw `(N, 3)` array `[logit_neg, logit_neu, logit_pos]`.
        fallback_y_pred: `y_pred` column from the prediction file (used when logits contain NaN).
        inference_mode: `"plain"` or `"prior"`.
        tau: Temperature/scaling parameter $\\tau$ for logit adjustment.
        priors: Tuple `(prior_neg, prior_neu, prior_pos)` of training set class priors $\\pi$.

    Returns:
        `(y_pred_out, adjusted_logits)`
    """
    if inference_mode not in VALID_INFERENCE_MODES:
        raise ValueError(
            f"inference_mode must be one of {VALID_INFERENCE_MODES}, got {inference_mode!r}"
        )

    logit_arr = np.asarray(logits, dtype=np.float64)
    if logit_arr.ndim != 2 or logit_arr.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) logits, got shape {logit_arr.shape}")

    # If logits contain NaN (e.g. B2 Hard_TF_SVM per contract 2.2), use fallback y_pred in plain mode
    if not np.all(np.isfinite(logit_arr)):
        if inference_mode == "prior":
            raise ValueError("Cannot run 'prior' inference mode when logits contain NaN/Inf")
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
    overwrite_duplicate_key: bool = True,
) -> Dict[str, Any]:
    """Evaluate a single per-sample prediction CSV and optionally append to `results_csv`.

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

    df_pred = read_csv_schema(pred_path, PRED_COLUMNS, strict_order=True)
    y_true = df_pred["y_true"].to_numpy(dtype=np.int64)
    fallback_y_pred = df_pred["y_pred"].to_numpy(dtype=np.int64)
    raw_logits = df_pred.loc[:, ["logit_neg", "logit_neu", "logit_pos"]].to_numpy(
        dtype=np.float64
    )

    tau = float(run_metadata.get("tau", 1.0))
    priors = (
        float(run_metadata["prior_neg"]),
        float(run_metadata["prior_neu"]),
        float(run_metadata["prior_pos"]),
    )

    y_pred, eval_logits = predict_from_logits(
        logits=raw_logits,
        fallback_y_pred=fallback_y_pred,
        inference_mode=inference_mode,
        tau=tau,
        priors=priors,
    )

    model_name = str(run_metadata.get("model", ""))
    loss_name = str(run_metadata.get("loss", ""))
    has_valid_prob_logits = (not is_svm_model(model_name, loss_name)) and bool(
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

    split_name = str(run_metadata.get("split") or DATASET_TO_SPLIT[dataset])

    row: Dict[str, Any] = {
        "run_id": str(run_metadata["run_id"]),
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
        "precision": str(run_metadata.get("precision", "fp16")),
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
    overwrite_duplicate_key: bool = True,
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
        for mode in modes:
            row = evaluate_prediction_file(
                pred_path=pred_file,
                run_metadata=run_metadata,
                dataset=ds,
                inference_mode=mode,
                results_csv=results_csv,
                overwrite_duplicate_key=overwrite_duplicate_key,
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
    parser.add_argument("--precision", type=str, default="fp16")
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
    )
    print(f"Evaluated {args.run_id} ({args.dataset}, {args.inference_mode}): macro_f1={row['macro_f1']:.4f}, ece={row['ece']}")


if __name__ == "__main__":
    main()
