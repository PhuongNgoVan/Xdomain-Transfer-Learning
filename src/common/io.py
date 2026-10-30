"""CSV reading, writing, and schema validation utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd


class SchemaError(ValueError):
    """Raised when a DataFrame or CSV file violates the expected column schema."""


def validate_columns(
    df: pd.DataFrame,
    expected_columns: Sequence[str],
    strict_order: bool = True,
) -> None:
    """Validate that `df` contains `expected_columns`.

    Args:
        df: DataFrame to validate.
        expected_columns: Ordered sequence of required column names.
        strict_order: If True (default), `df.columns` must match `expected_columns`
            in exact order and length. If False, checks that all expected columns
            are present (allowing extra columns or different order).

    Raises:
        SchemaError: If validation fails.
    """
    actual = tuple(df.columns)
    expected = tuple(expected_columns)

    if strict_order:
        if actual != expected:
            missing = [c for c in expected if c not in actual]
            extra = [c for c in actual if c not in expected]
            raise SchemaError(
                f"Column schema mismatch. Expected exact columns {expected}, "
                f"got {actual} (missing={missing}, extra={extra})."
            )
    else:
        missing = [c for c in expected if c not in actual]
        if missing:
            raise SchemaError(
                f"Missing required columns {missing}. Expected at least {expected}, "
                f"got {actual}."
            )


def read_csv_schema(
    path: Union[str, Path],
    expected_columns: Sequence[str],
    strict_order: bool = True,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file and validate its columns against `expected_columns`.

    Args:
        path: Path to the CSV file.
        expected_columns: Expected column schema from `src.common.schema`.
        strict_order: Whether to require exact column match and order.
        **read_csv_kwargs: Additional keyword arguments passed to `pd.read_csv`.

    Returns:
        Validated `pd.DataFrame`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        SchemaError: If the CSV columns do not match `expected_columns`.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path, **read_csv_kwargs)
    validate_columns(df, expected_columns, strict_order=strict_order)
    return df


def write_csv_schema(
    df: pd.DataFrame,
    path: Union[str, Path],
    expected_columns: Sequence[str],
    strict_order: bool = True,
    reorder: bool = False,
    **to_csv_kwargs: Any,
) -> Path:
    """Validate `df` against `expected_columns` and write it to `path` as UTF-8 CSV.

    Args:
        df: DataFrame to write.
        path: Destination CSV path (parent directories created automatically).
        expected_columns: Expected column schema from `src.common.schema`.
        strict_order: Whether to enforce exact column order.
        reorder: If True, project/reorder `df` onto `expected_columns` before
            validating strict order (still fails if any expected column is missing
            or if unexpected extra columns exist when `strict_order=True`).
        **to_csv_kwargs: Additional keyword arguments passed to `df.to_csv`.

    Returns:
        Resolved `Path` of the written CSV file.

    Raises:
        SchemaError: If `df` does not satisfy `expected_columns`.
    """
    out_df = df
    if reorder:
        validate_columns(out_df, expected_columns, strict_order=False)
        if strict_order and set(out_df.columns) != set(expected_columns):
            extra = [c for c in out_df.columns if c not in expected_columns]
            raise SchemaError(
                f"Unexpected extra columns when writing schema: {extra}"
            )
        out_df = out_df.loc[:, list(expected_columns)]

    validate_columns(out_df, expected_columns, strict_order=strict_order)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"index": False, "encoding": "utf-8", "lineterminator": "\n"}
    kwargs.update(to_csv_kwargs)
    out_df.to_csv(out_path, **kwargs)
    return out_path


def append_row_schema(
    row: Union[Dict[str, Any], pd.DataFrame],
    path: Union[str, Path],
    expected_columns: Sequence[str],
    key_columns: Optional[Sequence[str]] = None,
    overwrite_duplicate_key: bool = False,
) -> Path:
    """Append a single row (or DataFrame of rows) to a schema-validated CSV file.

    If `path` already exists, validates the existing file's schema first.
    Optionally checks for duplicate keys in `key_columns`.
    """
    if isinstance(row, dict):
        new_df = pd.DataFrame([row])
    elif isinstance(row, pd.DataFrame):
        new_df = row.copy()
    else:
        raise TypeError(f"row must be dict or DataFrame, got {type(row).__name__}")

    validate_columns(new_df, expected_columns, strict_order=False)
    if set(new_df.columns) != set(expected_columns):
        extra = [c for c in new_df.columns if c not in expected_columns]
        raise SchemaError(f"Row has unexpected extra columns: {extra}")
    new_df = new_df.loc[:, list(expected_columns)]

    out_path = Path(path)
    if out_path.is_file():
        existing_df = read_csv_schema(out_path, expected_columns, strict_order=True)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        if key_columns is not None:
            dup_mask = combined.duplicated(subset=list(key_columns), keep=False)
            if dup_mask.any():
                if overwrite_duplicate_key:
                    combined = combined.drop_duplicates(
                        subset=list(key_columns), keep="last"
                    ).reset_index(drop=True)
                else:
                    raise SchemaError(
                        f"Duplicate key detected on {tuple(key_columns)} when appending to {out_path}"
                    )
    else:
        combined = new_df

    return write_csv_schema(combined, out_path, expected_columns, strict_order=True)
