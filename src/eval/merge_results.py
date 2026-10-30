"""Merge and validate result files (`src/eval/merge_results.py`).

Per `docs/contract.md` (Section 2.3) and `tasks/M3_eval.md`:
- Scans `results/*.csv` (excluding `stage1.csv` and `token_stats.csv` / merged output).
- Validates exact column names and order against `RESULT_COLUMNS` (raises `SchemaError` if invalid).
- Detects and raises an error on duplicate primary keys `(run_id, dataset, split, inference_mode)`.
- Supports `--expected <file>` to report any missing expected runs/keys.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from src.common.io import SchemaError, read_csv_schema, validate_columns, write_csv_schema
from src.common.schema import RESULT_COLUMNS, RESULT_KEY_COLUMNS

EXCLUDED_RESULT_FILENAMES: Set[str] = {
    "stage1.csv",
    "token_stats.csv",
    "merged_results.csv",
}


class DuplicateResultKeyError(SchemaError):
    """Raised when duplicate `(run_id, dataset, split, inference_mode)` rows are found."""


class MissingExpectedRunsError(ValueError):
    """Raised when expected runs or keys are missing from merged results."""


def discover_result_files(
    results_dir: Union[str, Path] = "results",
    exclude_filenames: Optional[Set[str]] = None,
) -> List[Path]:
    """List member result CSV files in `results_dir` (excluding `stage1.csv` etc.)."""
    r_dir = Path(results_dir)
    if not r_dir.is_dir():
        return []
    excluded = (
        EXCLUDED_RESULT_FILENAMES
        if exclude_filenames is None
        else EXCLUDED_RESULT_FILENAMES | set(exclude_filenames)
    )
    files = [
        p
        for p in sorted(r_dir.glob("*.csv"))
        if p.is_file() and p.name not in excluded
    ]
    return files


def check_expected_runs(
    merged_df: pd.DataFrame,
    expected_path: Union[str, Path],
) -> List[Tuple[str, ...]]:
    """Compare `merged_df` against an expected runs CSV/TXT file.

    The expected file can either be:
    - A CSV containing key columns (`run_id` and optionally `dataset`, `split`, `inference_mode`), or
    - A plain text / 1-column CSV listing expected `run_id` values.

    Returns:
        List of missing key tuples.
    """
    exp_file = Path(expected_path)
    if not exp_file.is_file():
        raise FileNotFoundError(f"Expected runs file not found: {exp_file}")

    exp_df = pd.read_csv(exp_file)
    if exp_df.empty:
        return []

    # Determine which key columns are specified in expected_path
    present_key_cols = [c for c in RESULT_KEY_COLUMNS if c in exp_df.columns]
    if not present_key_cols:
        # Treat first column as run_id
        first_col = exp_df.columns[0]
        exp_df = exp_df.rename(columns={first_col: "run_id"})
        present_key_cols = ["run_id"]

    if merged_df.empty:
        actual_keys: Set[Tuple[str, ...]] = set()
    else:
        actual_keys = {
            tuple(str(row[c]) for c in present_key_cols)
            for _, row in merged_df.iterrows()
        }

    missing: List[Tuple[str, ...]] = []
    for _, row in exp_df.iterrows():
        key = tuple(str(row[c]) for c in present_key_cols)
        if key not in actual_keys:
            missing.append(key)

    return missing


def merge_result_files(
    results_dir: Union[str, Path] = "results",
    out_csv: Optional[Union[str, Path]] = None,
    expected_path: Optional[Union[str, Path]] = None,
    strict_expected: bool = False,
) -> Dict[str, Any]:
    """Scan, validate, and merge all `results/*.csv` files (excluding `stage1.csv`).

    Raises:
        SchemaError: If any CSV file has wrong columns or column order.
        DuplicateResultKeyError: If any duplicate `(run_id, dataset, split, inference_mode)` exists.
        MissingExpectedRunsError: If `strict_expected=True` and expected runs are missing.
    """
    out_name = Path(out_csv).name if out_csv is not None else None
    extra_exclude = {out_name} if out_name else set()
    csv_files = discover_result_files(results_dir, exclude_filenames=extra_exclude)

    dfs: List[pd.DataFrame] = []
    for fpath in csv_files:
        df = read_csv_schema(fpath, RESULT_COLUMNS, strict_order=True)
        df["_source_file"] = fpath.name
        dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=[*RESULT_COLUMNS, "_source_file"])

    # Check duplicate keys across all files
    if not combined.empty:
        dup_mask = combined.duplicated(subset=list(RESULT_KEY_COLUMNS), keep=False)
        if dup_mask.any():
            dup_rows = combined.loc[dup_mask, [*RESULT_KEY_COLUMNS, "_source_file"]]
            dup_summary = dup_rows.to_dict(orient="records")
            raise DuplicateResultKeyError(
                f"Duplicate primary keys {RESULT_KEY_COLUMNS} detected across result files: {dup_summary}"
            )

    clean_merged = combined.loc[:, list(RESULT_COLUMNS)].copy()
    if not clean_merged.empty:
        clean_merged = clean_merged.sort_values(
            list(RESULT_KEY_COLUMNS), kind="mergesort"
        ).reset_index(drop=True)

    validate_columns(clean_merged, RESULT_COLUMNS, strict_order=True)

    if out_csv is not None:
        write_csv_schema(clean_merged, out_csv, RESULT_COLUMNS, strict_order=True)

    missing_keys: List[Tuple[str, ...]] = []
    if expected_path is not None:
        missing_keys = check_expected_runs(clean_merged, expected_path)
        if missing_keys and strict_expected:
            raise MissingExpectedRunsError(
                f"Missing {len(missing_keys)} expected runs: {missing_keys}"
            )

    return {
        "merged_df": clean_merged,
        "source_files": [p.name for p in csv_files],
        "missing_expected": missing_keys,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate and merge results/*.csv files (excluding stage1.csv)."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/merged_results.csv"),
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help="Optional CSV/TXT file of expected run_ids or keys.",
    )
    parser.add_argument(
        "--strict-expected",
        action="store_true",
        default=False,
        help="Exit with error if any expected run is missing.",
    )
    args = parser.parse_args(argv)

    res = merge_result_files(
        results_dir=args.results_dir,
        out_csv=args.out_csv,
        expected_path=args.expected,
        strict_expected=args.strict_expected,
    )
    print(
        f"=== Merged {len(res['merged_df'])} rows from {len(res['source_files'])} files ==="
    )
    if res["missing_expected"]:
        print(f"WARNING: Missing {len(res['missing_expected'])} expected runs:")
        for k in res["missing_expected"]:
            print(f"  - {k}")


if __name__ == "__main__":
    main()
