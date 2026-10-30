"""Model and hyperparameter selection strictly on dev split (`src/eval/select.py`).

Per `docs/contract.md` (Section 2.6) and `tasks/M3_eval.md`:
- `select.py` ONLY accepts rows with `split == "dev"`.
- If the input contains ANY row with `split == "test"` (or `split != "dev"`) or
  `dataset == "vsfc_test"` (or `dataset == "neuesc_test"`), immediately raises `ValueError`.
- UIT-VSFC never participates in model selection under any form.
- Default selection criterion (Section 2.6):
  - Filter to `dataset == "neuesc_dev"` and `inference_mode == "plain"`.
  - Maximize `macro_f1`.
  - Tie-breaker 1: lower `val_loss` (if `val_loss` column is provided) or lower `ece`.
  - Tie-breaker 2: smaller `lr`.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.common.io import validate_columns
from src.common.schema import RESULT_COLUMNS


def validate_dev_only(df: pd.DataFrame) -> None:
    """Ensure `df` contains strictly `split == 'dev'` and no test or OOD datasets.

    Raises:
        ValueError: If `df` is empty, missing `split`/`dataset` columns, or contains
            any non-dev row (`split != 'dev'`) or test/VSFC dataset.
    """
    if "split" not in df.columns or "dataset" not in df.columns:
        raise ValueError("Input DataFrame must contain 'split' and 'dataset' columns.")
    if df.empty:
        raise ValueError("Input DataFrame for model selection is empty.")

    bad_splits = set(df["split"].astype(str).unique()) - {"dev"}
    if bad_splits:
        raise ValueError(
            f"Model selection forbidden on non-dev splits: found split={sorted(bad_splits)}. "
            "select.py only accepts split='dev'."
        )

    datasets = set(df["dataset"].astype(str).unique())
    forbidden_datasets = {
        ds for ds in datasets if "test" in ds.lower() or "vsfc" in ds.lower()
    }
    if forbidden_datasets:
        raise ValueError(
            f"Model selection forbidden on test or OOD datasets: found dataset={sorted(forbidden_datasets)}."
        )


def select_best_config(
    df: pd.DataFrame,
    dataset: str = "neuesc_dev",
    inference_mode: str = "plain",
    group_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Select the best configuration(s) from dev result rows per `docs/contract.md` Section 2.6.

    Args:
        df: DataFrame of dev evaluation rows (must NOT contain any test or VSFC rows).
        dataset: Target dev dataset (`"neuesc_dev"` by default).
        inference_mode: Inference mode for selection (`"plain"` by default).
        group_by: Optional columns to group by (e.g. `["model", "variant", "fraction"]`)
            when selecting the best LR per model group. If None, returns a 1-row
            DataFrame with the single overall best row.

    Returns:
        DataFrame containing the selected best row(s).

    Raises:
        ValueError: If any row has `split != 'dev'` or belongs to a test/VSFC dataset,
            or if no rows match `(dataset, inference_mode)`.
    """
    validate_dev_only(df)

    # Validate required columns from RESULT_COLUMNS (allowing optional extra `val_loss` column for tie-breaking)
    validate_columns(df, RESULT_COLUMNS, strict_order=False)

    filtered = df[
        (df["dataset"] == dataset) & (df["inference_mode"] == inference_mode)
    ].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows left after filtering for dataset={dataset!r}, inference_mode={inference_mode!r}"
        )

    # Prepare tie-breaker columns:
    # 1. Primary: higher `macro_f1` -> sort ascending by `-macro_f1`
    # 2. Secondary: lower `val_loss` (if present, filling NaN with +inf);
    #    then lower `ece` (filling NaN with +inf)
    # 3. Tertiary: smaller `lr`
    # 4. Deterministic fallback: smaller `seed`, then `run_id`
    filtered["_neg_macro_f1"] = -filtered["macro_f1"].astype(float)
    if "val_loss" in filtered.columns:
        filtered["_tie_val_loss"] = (
            pd.to_numeric(filtered["val_loss"], errors="coerce")
            .fillna(np.inf)
            .astype(float)
        )
    else:
        filtered["_tie_val_loss"] = np.inf

    filtered["_tie_ece"] = (
        pd.to_numeric(filtered["ece"], errors="coerce")
        .fillna(np.inf)
        .astype(float)
    )
    filtered["_tie_lr"] = filtered["lr"].astype(float)
    filtered["_tie_seed"] = filtered["seed"].astype(int)

    sort_cols = [
        "_neg_macro_f1",
        "_tie_val_loss",
        "_tie_ece",
        "_tie_lr",
        "_tie_seed",
        "run_id",
    ]
    sorted_df = filtered.sort_values(sort_cols, kind="mergesort")

    helper_cols = [
        "_neg_macro_f1",
        "_tie_val_loss",
        "_tie_ece",
        "_tie_lr",
        "_tie_seed",
    ]

    if group_by:
        best_df = (
            sorted_df.groupby(list(group_by), as_index=False, sort=False)
            .first()
            .drop(columns=helper_cols)
            .reset_index(drop=True)
        )
    else:
        best_df = (
            sorted_df.iloc[[0]]
            .drop(columns=helper_cols)
            .reset_index(drop=True)
        )

    return best_df


def select_from_csv(
    csv_path: Union[str, Path],
    dataset: str = "neuesc_dev",
    inference_mode: str = "plain",
    group_by: Optional[Sequence[str]] = None,
    out_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Read a results CSV, enforce dev-only rule, and select the best configuration(s)."""
    fpath = Path(csv_path)
    if not fpath.is_file():
        raise FileNotFoundError(f"Results CSV not found: {fpath}")
    df = pd.read_csv(fpath)
    best_df = select_best_config(
        df=df,
        dataset=dataset,
        inference_mode=inference_mode,
        group_by=group_by,
    )
    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        best_df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return best_df


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Select best model/hyperparameters strictly on dev split."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="neuesc_dev")
    parser.add_argument("--inference-mode", type=str, default="plain")
    parser.add_argument(
        "--group-by",
         nargs="*",
        default=None,
        help="Optional grouping columns, e.g. --group-by model variant fraction",
    )
    args = parser.parse_args(argv)

    best_df = select_from_csv(
        csv_path=args.input_csv,
        dataset=args.dataset,
        inference_mode=args.inference_mode,
        group_by=args.group_by,
        out_csv=args.out_csv,
    )
    print("=== Selected Best Dev Configuration(s) ===")
    cols_to_show = [
        "run_id",
        "model",
        "variant",
        "fraction",
        "lr",
        "macro_f1",
        "ece",
    ]
    if "val_loss" in best_df.columns:
        cols_to_show.insert(-1, "val_loss")
    print(best_df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
