"""Stage A of M7 (CV-P3): Validate independent polarity annotations and compute Cohen's kappa.

Per `tasks/M7_polarity_kappa_wtrans (1).md`:
- Validates `polarity_nhat.csv`, `polarity_long.csv`, and `polarity_key.csv` (with fallback
  to `emotion_id`/`emotion_name` in annotator files if `polarity_key.csv` is absent and
  both annotator files have matching emotion columns).
- Sorts all records by `id` ascending so outputs are 100% independent of input row order.
- Computes observed agreement (`po`), unweighted Cohen's kappa (`kappa`), 95% sentence-level
  bootstrap confidence intervals (`B=2000, seed=42`, reporting dropped NaN iterations),
  linear weighted kappa, Gwet's AC1, and PABAK for:
  1. All 100 sentences (`all`)
  2. 50 Other sentences (`other`)
  3. 50 Surprise sentences (`surprise`)
- Computes 3x3 confusion matrices (row = Nhat, col = Long, ordered by `NEG, NEU, POS`).
- Computes polarity distributions per annotator and original emotion with Wilson 95% CIs.
- Exports deterministic files to `results/polarity/` with no timestamps or absolute paths.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.common.constants import (
    EMO_OTHER,
    EMO_SURPRISE,
    EMOTION7_TO_ID,
    ID_TO_EMOTION7,
    ID_TO_LABEL3,
    LABELS3,
    NEG,
    NEU,
    POS,
)
from src.common.hashing import sha256_file
from src.common.io import write_csv_schema
from src.common.schema import (
    POLARITY_DISAGREEMENT_COLUMNS,
    POLARITY_DISTRIBUTION_COLUMNS,
)

VALID_POLARITIES: Tuple[int, ...] = (NEG, NEU, POS)
VALID_POLARITY_STRS: Tuple[str, ...] = tuple(str(x) for x in VALID_POLARITIES)
TARGET_EMOTION_IDS: Tuple[int, ...] = (EMO_OTHER, EMO_SURPRISE)

# Standard normal 97.5th percentile for 95% Wilson confidence interval
Z_95: float = 1.959963984540054


def round6(val: Optional[float]) -> Optional[float]:
    """Round a float to 6 decimal places, returning None for None or NaN."""
    if val is None or math.isnan(float(val)):
        return None
    return round(float(val), 6)


def fmt6(val: Optional[float]) -> str:
    """Format a float to 6 decimal places as string, or 'NaN' if None/NaN."""
    if val is None or math.isnan(float(val)):
        return "NaN"
    return f"{float(val):.6f}"


def compute_po_and_kappa(
    a: Sequence[int],
    b: Sequence[int],
    labels: Sequence[int] = VALID_POLARITIES,
) -> Tuple[float, float, float]:
    """Compute observed agreement `po`, expected agreement `pe`, and unweighted Cohen's kappa.

    If `pe == 1` (e.g. single-label sample), returns `NaN` for kappa without dividing by zero.
    """
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    if len(a_arr) != len(b_arr) or len(a_arr) == 0:
        raise ValueError("Input label sequences must be non-empty and of equal length.")

    n = float(len(a_arr))
    po = float(np.sum(a_arr == b_arr)) / n
    pe = 0.0
    for k in labels:
        pa_k = float(np.sum(a_arr == k)) / n
        pb_k = float(np.sum(b_arr == k)) / n
        pe += pa_k * pb_k

    if abs(1.0 - pe) < 1e-15:
        return po, pe, float("nan")
    kappa = (po - pe) / (1.0 - pe)
    return po, pe, float(kappa)


def compute_cohen_kappa(
    a: Sequence[int],
    b: Sequence[int],
    labels: Sequence[int] = VALID_POLARITIES,
) -> float:
    """Compute unweighted Cohen's kappa."""
    return compute_po_and_kappa(a, b, labels=labels)[2]


def compute_linear_weighted_kappa(
    a: Sequence[int],
    b: Sequence[int],
    labels: Sequence[int] = VALID_POLARITIES,
) -> float:
    """Compute linear weighted Cohen's kappa with weights `1 - |i - j| / (K - 1)`.

    Assumes `labels` are ordered `NEG (0) < NEU (1) < POS (2)`.
    If expected disagreement is 0 (e.g. single-label sample), returns `NaN`.
    """
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    if len(a_arr) != len(b_arr) or len(a_arr) == 0:
        raise ValueError("Input label sequences must be non-empty and of equal length.")

    k_classes = len(labels)
    max_dist = float(k_classes - 1)
    n = float(len(a_arr))

    # Map label values to positions 0..K-1 in `labels`
    pos_map = {lbl: idx for idx, lbl in enumerate(labels)}
    a_pos = np.array([pos_map[int(x)] for x in a_arr], dtype=float)
    b_pos = np.array([pos_map[int(x)] for x in b_arr], dtype=float)

    obs_disagreement = float(np.sum(np.abs(a_pos - b_pos))) / (n * max_dist)

    pa = np.array([ float(np.sum(a_pos == i)) / n for i in range(k_classes) ], dtype=float)
    pb = np.array([ float(np.sum(b_pos == j)) / n for j in range(k_classes) ], dtype=float)

    exp_disagreement = 0.0
    for i in range(k_classes):
        for j in range(k_classes):
            exp_disagreement += (abs(i - j) / max_dist) * pa[i] * pb[j]

    if abs(exp_disagreement) < 1e-15:
        return float("nan")
    return float(1.0 - (obs_disagreement / exp_disagreement))


def compute_gwet_ac1(
    a: Sequence[int],
    b: Sequence[int],
    labels: Sequence[int] = VALID_POLARITIES,
) -> float:
    """Compute Gwet's AC1 for K=len(labels) categories:

    `pe = sum_k pi_k * (1 - pi_k) / (K - 1)`, where `pi_k = (p_{a,k} + p_{b,k}) / 2`.
    `AC1 = (po - pe) / (1 - pe)`.
    """
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    if len(a_arr) != len(b_arr) or len(a_arr) == 0:
        raise ValueError("Input label sequences must be non-empty and of equal length.")

    k_classes = len(labels)
    n = float(len(a_arr))
    po = float(np.sum(a_arr == b_arr)) / n

    pe_num = 0.0
    for k in labels:
        pa_k = float(np.sum(a_arr == k)) / n
        pb_k = float(np.sum(b_arr == k)) / n
        pi_k = 0.5 * (pa_k + pb_k)
        pe_num += pi_k * (1.0 - pi_k)

    pe = pe_num / float(k_classes - 1)
    if abs(1.0 - pe) < 1e-15:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def compute_pabak(
    a: Sequence[int],
    b: Sequence[int],
    k_classes: int = 3,
) -> float:
    """Compute PABAK = `(K * po - 1) / (K - 1)`."""
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    if len(a_arr) != len(b_arr) or len(a_arr) == 0:
        raise ValueError("Input label sequences must be non-empty and of equal length.")

    po = float(np.sum(a_arr == b_arr)) / float(len(a_arr))
    return float((k_classes * po - 1.0) / float(k_classes - 1))


def bootstrap_kappa_ci(
    a: Sequence[int],
    b: Sequence[int],
    n_boot: int = 2000,
    seed: int = 42,
    labels: Sequence[int] = VALID_POLARITIES,
) -> Dict[str, Any]:
    """Compute 95% sentence-level bootstrap CI (percentile 2.5 to 97.5) for Cohen's kappa.

    Drops bootstrap replicates where kappa is NaN and reports `dropped_count`,
    `dropped_rate`, and `warning_ci_biased` (True when `dropped_rate > 0.05`).
    """
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    if len(a_arr) != len(b_arr) or len(a_arr) == 0:
        raise ValueError("Input label sequences must be non-empty and of equal length.")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")

    n = len(a_arr)
    rng = np.random.default_rng(seed)
    valid_kappas: List[float] = []
    dropped_count = 0

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        k_val = compute_cohen_kappa(a_arr[idx], b_arr[idx], labels=labels)
        if math.isnan(k_val):
            dropped_count += 1
        else:
            valid_kappas.append(k_val)

    dropped_rate = float(dropped_count) / float(n_boot)
    warning_ci_biased = dropped_rate > 0.05

    if valid_kappas:
        low, high = np.percentile(np.asarray(valid_kappas, dtype=float), [2.5, 97.5])
        ci_low = float(low)
        ci_high = float(high)
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot": int(n_boot),
        "dropped_count": int(dropped_count),
        "dropped_rate": dropped_rate,
        "warning_ci_biased": bool(warning_ci_biased),
    }


def wilson_ci(
    count: int,
    total: int,
    z: float = Z_95,
) -> Tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a binomial proportion `count / total`."""
    if total <= 0:
        return float("nan"), float("nan")
    if count < 0 or count > total:
        raise ValueError(f"Invalid count={count} for total={total}.")

    n = float(total)
    p = float(count) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    rad = math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    half = (z * rad) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return float(low), float(high)


def compute_subset_metrics(
    a: Sequence[int],
    b: Sequence[int],
    n_boot: int = 2000,
    seed: int = 42,
    labels: Sequence[int] = VALID_POLARITIES,
) -> Dict[str, Any]:
    """Compute all agreement metrics and bootstrap CI for a pair of label vectors."""
    po, pe, kappa = compute_po_and_kappa(a, b, labels=labels)
    boot = bootstrap_kappa_ci(a, b, n_boot=n_boot, seed=seed, labels=labels)
    kappa_linear = compute_linear_weighted_kappa(a, b, labels=labels)
    ac1 = compute_gwet_ac1(a, b, labels=labels)
    pabak = compute_pabak(a, b, k_classes=len(labels))

    return {
        "n": int(len(a)),
        "po": po,
        "pe": pe,
        "kappa": kappa,
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "n_boot": boot["n_boot"],
        "bootstrap_dropped_count": boot["dropped_count"],
        "bootstrap_dropped_rate": boot["dropped_rate"],
        "warning_ci_biased": boot["warning_ci_biased"],
        "kappa_linear": kappa_linear,
        "gwet_ac1": ac1,
        "pabak": pabak,
    }


def compute_confusion_matrix_df(
    a: Sequence[int],
    b: Sequence[int],
    labels: Sequence[int] = VALID_POLARITIES,
) -> pd.DataFrame:
    """Build 3x3 confusion matrix DataFrame (rows = Nhat `a`, cols = Long `b`, order `NEG, NEU, POS`)."""
    a_arr = np.asarray(a, dtype=int)
    b_arr = np.asarray(b, dtype=int)
    label_names = [ID_TO_LABEL3[k] for k in labels]

    rows: List[Dict[str, Any]] = []
    for r_id, r_name in zip(labels, label_names):
        row_dict: Dict[str, Any] = {"label": r_name}
        for c_id, c_name in zip(labels, label_names):
            row_dict[c_name] = int(np.sum((a_arr == r_id) & (b_arr == c_id)))
        rows.append(row_dict)

    return pd.DataFrame(rows, columns=["label", *label_names])


def _parse_int_id_column(df: pd.DataFrame, file_label: str) -> pd.Series:
    """Parse `id` column as integers and check for duplicates."""
    parsed_ids: List[int] = []
    for idx, val in enumerate(df["id"]):
        s = str(val).strip()
        try:
            parsed_ids.append(int(s))
        except ValueError as exc:
            raise ValueError(
                f"[{file_label}] Invalid non-integer id at row {idx}: {val!r}"
            ) from exc
    series = pd.Series(parsed_ids, index=df.index, dtype=int)
    if series.duplicated().any():
        dups = sorted(series[series.duplicated()].unique().tolist())
        raise ValueError(f"[{file_label}] Duplicate id(s) found: {dups}")
    return series


def _validate_polarity_labels(
    df: pd.DataFrame,
    annotator_name: str,
) -> List[Tuple[str, int, str]]:
    """Return list of `(annotator_name, id, raw_value)` for rows with invalid `polarity_label`."""
    errors: List[Tuple[str, int, str]] = []
    for _, row in df.iterrows():
        rid = int(row["id"])
        raw_val = str(row["polarity_label"]).strip()
        if raw_val not in VALID_POLARITY_STRS:
            errors.append((annotator_name, rid, raw_val))
    return errors


def load_and_validate_polarity_inputs(
    nhat_path: Union[str, Path] = "data/clean/polarity_nhat.csv",
    long_path: Union[str, Path] = "data/clean/polarity_long.csv",
    key_path: Optional[Union[str, Path]] = "data/clean/polarity_key.csv",
    expected_n_per_emotion: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], List[str]]:
    """Validate and merge `polarity_nhat.csv`, `polarity_long.csv`, and `polarity_key.csv`.

    Returns:
        merged_df: Sorted by `id` ascending with columns:
            `id, text, emotion_id, emotion_name, label_nhat, label_long`
        file_hashes: Mapping of filenames to SHA-256 hex digests (`polarity_key.csv` is None
            if fallback was used).
        warnings_list: Input-level warnings (e.g. `"không kiểm chứng được nguồn câu"`).
    """
    p_nhat = Path(nhat_path)
    p_long = Path(long_path)
    p_key = Path(key_path) if key_path is not None else None

    if not p_nhat.is_file():
        raise FileNotFoundError(f"Missing annotator file: {p_nhat}")
    if not p_long.is_file():
        raise FileNotFoundError(f"Missing annotator file: {p_long}")

    df_nhat = pd.read_csv(p_nhat, dtype=str, keep_default_na=False)
    df_long = pd.read_csv(p_long, dtype=str, keep_default_na=False)

    for label, df in (("polarity_nhat.csv", df_nhat), ("polarity_long.csv", df_long)):
        missing = [c for c in ("id", "text", "polarity_label") if c not in df.columns]
        if missing:
            raise ValueError(f"[{label}] Missing required columns: {missing}")

    expected_total = 2 * expected_n_per_emotion
    if len(df_nhat) != expected_total:
        raise ValueError(
            f"[polarity_nhat.csv] Expected {expected_total} rows, got {len(df_nhat)}."
        )
    if len(df_long) != expected_total:
        raise ValueError(
            f"[polarity_long.csv] Expected {expected_total} rows, got {len(df_long)}."
        )

    df_nhat = df_nhat.copy()
    df_long = df_long.copy()
    df_nhat["id"] = _parse_int_id_column(df_nhat, "polarity_nhat.csv")
    df_long["id"] = _parse_int_id_column(df_long, "polarity_long.csv")

    ids_nhat = set(df_nhat["id"].tolist())
    ids_long = set(df_long["id"].tolist())
    if ids_nhat != ids_long:
        only_nhat = sorted(ids_nhat - ids_long)
        only_long = sorted(ids_long - ids_nhat)
        raise ValueError(
            f"ID mismatch between polarity_nhat.csv and polarity_long.csv: "
            f"only_in_nhat={only_nhat[:10]}, only_in_long={only_long[:10]}"
        )

    df_nhat = df_nhat.sort_values("id", kind="mergesort").reset_index(drop=True)
    df_long = df_long.sort_values("id", kind="mergesort").reset_index(drop=True)

    # Validate polarity_label values before or alongside key checks, printing erroneous rows
    label_errors = _validate_polarity_labels(df_nhat, "nhat") + _validate_polarity_labels(
        df_long, "long"
    )
    if label_errors:
        print("=== Invalid polarity_label rows (annotator, id, value) ===")
        for ann, rid, val in label_errors:
            print(f"  annotator={ann}, id={rid}, polarity_label={val!r}")
        preview = ", ".join(
            f"({ann}, id={rid}, val={val!r})" for ann, rid, val in label_errors[:10]
        )
        raise ValueError(
            f"Found {len(label_errors)} invalid or empty polarity_label value(s) "
            f"(must be in {{0, 1, 2}}): {preview}"
        )

    warnings_list: List[str] = []
    file_hashes: Dict[str, Optional[str]] = {
        "polarity_long.csv": sha256_file(p_long),
        "polarity_nhat.csv": sha256_file(p_nhat),
        "polarity_key.csv": None,
    }

    if p_key is not None and p_key.is_file():
        file_hashes["polarity_key.csv"] = sha256_file(p_key)
        df_key = pd.read_csv(p_key, dtype=str, keep_default_na=False)
        missing_key_cols = [
            c for c in ("id", "text", "emotion_id", "emotion_name") if c not in df_key.columns
        ]
        if missing_key_cols:
            raise ValueError(
                f"[polarity_key.csv] Missing required columns: {missing_key_cols}"
            )
        if len(df_key) != expected_total:
            raise ValueError(
                f"[polarity_key.csv] Expected {expected_total} rows, got {len(df_key)}."
            )
        df_key = df_key.copy()
        df_key["id"] = _parse_int_id_column(df_key, "polarity_key.csv")
        ids_key = set(df_key["id"].tolist())
        if ids_key != ids_nhat:
            only_key = sorted(ids_key - ids_nhat)
            only_ann = sorted(ids_nhat - ids_key)
            raise ValueError(
                f"ID mismatch between polarity_key.csv and annotator files: "
                f"only_in_key={only_key[:10]}, only_in_annotators={only_ann[:10]}"
            )
        df_key = df_key.sort_values("id", kind="mergesort").reset_index(drop=True)

        if "split" in df_key.columns:
            bad_splits = [
                (int(r["id"]), str(r["split"]).strip())
                for _, r in df_key.iterrows()
                if str(r["split"]).strip() != "train"
            ]
            if bad_splits:
                raise ValueError(
                    f"[polarity_key.csv] All sentences must come from 'train' split; "
                    f"found non-train rows: {bad_splits[:10]}"
                )
        else:
            warnings_list.append("không kiểm chứng được nguồn câu")
    else:
        # Fallback: both annotator files must contain emotion_id and emotion_name
        has_emo_nhat = {"emotion_id", "emotion_name"}.issubset(df_nhat.columns)
        has_emo_long = {"emotion_id", "emotion_name"}.issubset(df_long.columns)
        if not (has_emo_nhat and has_emo_long):
            raise ValueError(
                "Missing polarity_key.csv and annotator files do not both contain "
                "fallback columns ('emotion_id', 'emotion_name')."
            )
        mismatch_emo_ids: List[int] = []
        for idx in range(len(df_nhat)):
            if (
                str(df_nhat.loc[idx, "emotion_id"]).strip()
                != str(df_long.loc[idx, "emotion_id"]).strip()
                or str(df_nhat.loc[idx, "emotion_name"]).strip()
                != str(df_long.loc[idx, "emotion_name"]).strip()
            ):
                mismatch_emo_ids.append(int(df_nhat.loc[idx, "id"]))
        if mismatch_emo_ids:
            raise ValueError(
                f"Fallback emotion_id/emotion_name mismatch between polarity_nhat.csv "
                f"and polarity_long.csv at id(s): {mismatch_emo_ids}"
            )
        df_key = df_nhat.loc[:, ["id", "text", "emotion_id", "emotion_name"]].copy()
        if "split" in df_nhat.columns and "split" in df_long.columns:
            bad_splits = [
                (int(r["id"]), str(r["split"]).strip())
                for _, r in df_nhat.iterrows()
                if str(r["split"]).strip() != "train"
            ]
            if bad_splits:
                raise ValueError(
                    f"All sentences must come from 'train' split; "
                    f"found non-train rows: {bad_splits[:10]}"
                )
        else:
            warnings_list.append("không kiểm chứng được nguồn câu")

    # Verify text matches across nhat, long, and key by id
    text_mismatches: List[int] = []
    for idx in range(expected_total):
        t_nhat = str(df_nhat.loc[idx, "text"])
        t_long = str(df_long.loc[idx, "text"])
        t_key = str(df_key.loc[idx, "text"])
        if t_nhat != t_long or t_nhat != t_key:
            text_mismatches.append(int(df_nhat.loc[idx, "id"]))
    if text_mismatches:
        raise ValueError(
            f"Text mismatch across files at id(s): {text_mismatches[:10]}"
        )

    # Validate emotion_id and emotion_name mapping and 50/50 counts
    emo_ids: List[int] = []
    emo_names: List[str] = []
    for idx in range(expected_total):
        rid = int(df_key.loc[idx, "id"])
        raw_eid = str(df_key.loc[idx, "emotion_id"]).strip()
        raw_ename = str(df_key.loc[idx, "emotion_name"]).strip()
        try:
            eid = int(raw_eid)
        except ValueError as exc:
            raise ValueError(
                f"Invalid emotion_id={raw_eid!r} at id={rid}"
            ) from exc
        if eid not in ID_TO_EMOTION7:
            raise ValueError(f"Unknown emotion_id={eid} at id={rid}")
        expected_ename = ID_TO_EMOTION7[eid]
        if raw_ename != expected_ename or EMOTION7_TO_ID.get(raw_ename) != eid:
            raise ValueError(
                f"Inconsistent emotion_id={eid} and emotion_name={raw_ename!r} at id={rid} "
                f"(expected {expected_ename!r})"
            )
        if eid not in TARGET_EMOTION_IDS:
            raise ValueError(
                f"Unexpected emotion {raw_ename!r} ({eid}) at id={rid}; "
                f"only Other ({EMO_OTHER}) and Surprise ({EMO_SURPRISE}) are allowed."
            )
        emo_ids.append(eid)
        emo_names.append(raw_ename)

    merged = pd.DataFrame(
        {
            "id": df_nhat["id"].astype(int),
            "text": df_nhat["text"].astype(str),
            "emotion_id": pd.Series(emo_ids, dtype=int),
            "emotion_name": pd.Series(emo_names, dtype=str),
            "label_nhat": df_nhat["polarity_label"].map(lambda x: int(str(x).strip())).astype(int),
            "label_long": df_long["polarity_label"].map(lambda x: int(str(x).strip())).astype(int),
        }
    )

    n_other = int((merged["emotion_id"] == EMO_OTHER).sum())
    n_surprise = int((merged["emotion_id"] == EMO_SURPRISE).sum())
    if n_other != expected_n_per_emotion or n_surprise != expected_n_per_emotion:
        raise ValueError(
            f"Expected {expected_n_per_emotion} Other and {expected_n_per_emotion} Surprise "
            f"sentences, got Other={n_other}, Surprise={n_surprise}."
        )

    return merged, file_hashes, warnings_list


def compute_polarity_distribution_df(
    merged: pd.DataFrame,
    labels: Sequence[int] = VALID_POLARITIES,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute polarity distribution per annotator and emotion group with Wilson 95% CIs."""
    rows: List[Dict[str, Any]] = []
    warnings_list: List[str] = []

    groups: List[Tuple[str, pd.DataFrame]] = [
        ("All", merged),
        (ID_TO_EMOTION7[EMO_OTHER], merged[merged["emotion_id"] == EMO_OTHER]),
        (ID_TO_EMOTION7[EMO_SURPRISE], merged[merged["emotion_id"] == EMO_SURPRISE]),
    ]
    annotators: List[Tuple[str, str]] = [
        ("nhat", "label_nhat"),
        ("long", "label_long"),
    ]

    for emo_label, sub_df in groups:
        total = len(sub_df)
        max_poles: Dict[str, int] = {}
        props_by_ann: Dict[str, Dict[int, float]] = {}

        for ann_name, col_name in annotators:
            vals = sub_df[col_name].to_numpy(dtype=int)
            counts = {k: int(np.sum(vals == k)) for k in labels}
            props_by_ann[ann_name] = {k: counts[k] / float(total) for k in labels}
            max_pole = max(labels, key=lambda k: counts[k])
            max_poles[ann_name] = max_pole
            max_count = counts[max_pole]
            max_low, max_high = wilson_ci(max_count, total)

            if emo_label in (ID_TO_EMOTION7[EMO_OTHER], ID_TO_EMOTION7[EMO_SURPRISE]):
                if max_low <= 0.60 <= max_high:
                    warnings_list.append(
                        f"kết luận yếu: Wilson 95% CI của cực lớn nhất ({ID_TO_LABEL3[max_pole]}) "
                        f"cho {emo_label} ({ann_name}) chứa 0.60 [{fmt6(max_low)}, {fmt6(max_high)}]"
                    )
                if max_low <= 0.50 <= max_high:
                    warnings_list.append(
                        f"cực lớn nhất có thể không chiếm đa số: Wilson 95% CI của cực lớn nhất "
                        f"({ID_TO_LABEL3[max_pole]}) cho {emo_label} ({ann_name}) chứa 0.50 "
                        f"[{fmt6(max_low)}, {fmt6(max_high)}]"
                    )

            for k in labels:
                c = counts[k]
                prop = float(c) / float(total) if total > 0 else float("nan")
                ci_l, ci_h = wilson_ci(c, total)
                rows.append(
                    {
                        "annotator": ann_name,
                        "emotion_name": emo_label,
                        "polarity_id": int(k),
                        "polarity_name": ID_TO_LABEL3[k],
                        "count": int(c),
                        "total": int(total),
                        "proportion": fmt6(prop),
                        "wilson_ci_low": fmt6(ci_l),
                        "wilson_ci_high": fmt6(ci_h),
                    }
                )

        # Check annotator marginal distribution shift within group
        max_diff = max(
            abs(props_by_ann["nhat"][k] - props_by_ann["long"][k]) for k in labels
        )
        if max_poles["nhat"] != max_poles["long"] or round(max_diff, 9) >= 0.15:
            warnings_list.append(
                f"lệch phân bố cực tính giữa Nhật và Long trên nhóm {emo_label} "
                f"(max_diff={fmt6(max_diff)})"
            )

    dist_df = pd.DataFrame(rows, columns=list(POLARITY_DISTRIBUTION_COLUMNS))
    return dist_df, warnings_list


def extract_disagreements_df(merged: pd.DataFrame) -> pd.DataFrame:
    """Extract sentences where `label_nhat != label_long`, sorted by `id` ascending."""
    dis = merged[merged["label_nhat"] != merged["label_long"]].sort_values(
        "id", kind="mergesort"
    )
    out = pd.DataFrame(
        {
            "id": dis["id"].astype(int),
            "text": dis["text"].astype(str),
            "emotion_name": dis["emotion_name"].astype(str),
            "label_nhat": dis["label_nhat"].astype(int),
            "label_long": dis["label_long"].astype(int),
            "final_label": ["" for _ in range(len(dis))],
        },
        columns=list(POLARITY_DISAGREEMENT_COLUMNS),
    )
    return out.reset_index(drop=True)


def _format_subset_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Round all float fields in subset metrics to 6 decimal places for deterministic JSON."""
    return {
        "n": int(metrics["n"]),
        "po": round6(metrics["po"]),
        "pe": round6(metrics["pe"]),
        "kappa": round6(metrics["kappa"]),
        "ci_low": round6(metrics["ci_low"]),
        "ci_high": round6(metrics["ci_high"]),
        "n_boot": int(metrics["n_boot"]),
        "bootstrap_dropped_count": int(metrics["bootstrap_dropped_count"]),
        "bootstrap_dropped_rate": round6(metrics["bootstrap_dropped_rate"]),
        "warning_ci_biased": bool(metrics["warning_ci_biased"]),
        "kappa_linear": round6(metrics["kappa_linear"]),
        "gwet_ac1": round6(metrics["gwet_ac1"]),
        "pabak": round6(metrics["pabak"]),
    }


def render_kappa_report_md(
    report_data: Dict[str, Any],
    dist_df: pd.DataFrame,
) -> str:
    """Render deterministic Markdown report `kappa_report.md` (no timestamps or paths)."""
    lines: List[str] = [
        "# Báo cáo độ đồng thuận gán nhãn cực tính (Giai đoạn A – CV-P3)",
        "",
        "## 1. Tổng quan độ đồng thuận (Cohen's kappa & chỉ số phụ)",
        "",
        "| Tập mẫu | Số câu (n) | po | pe | Cohen's κ | 95% CI (Bootstrap) | Bootstrap bị bỏ | κ trọng số tuyến tính | Gwet AC1 | PABAK |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]

    subset_titles = [
        ("all", "Toàn bộ (All)"),
        ("other", "Other"),
        ("surprise", "Surprise"),
    ]
    for key, title in subset_titles:
        m = report_data["metrics"][key]
        ci_str = f"[{fmt6(m['ci_low'])}, {fmt6(m['ci_high'])}]"
        drop_str = f"{m['bootstrap_dropped_count']}/{m['n_boot']} ({fmt6(m['bootstrap_dropped_rate'])})"
        lines.append(
            f"| {title} | {m['n']} | {fmt6(m['po'])} | {fmt6(m['pe'])} | {fmt6(m['kappa'])} | "
            f"{ci_str} | {drop_str} | {fmt6(m['kappa_linear'])} | {fmt6(m['gwet_ac1'])} | {fmt6(m['pabak'])} |"
        )

    lines.extend(
        [
            "",
            f"- **Tổng số câu bất đồng (`label_nhat != label_long`):** {report_data['disagreement_count']} / {report_data['metrics']['all']['n']}",
            "",
            "## 2. Phân bố cực tính theo người gán và nhãn gốc (kèm 95% Wilson CI)",
            "",
            "| Người gán | Nhãn gốc | Cực tính | Số đếm | Tổng | Tỉ lệ | Wilson 95% CI |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )

    for _, row in dist_df.iterrows():
        lines.append(
            f"| {row['annotator']} | {row['emotion_name']} | {row['polarity_name']} ({row['polarity_id']}) | "
            f"{row['count']} | {row['total']} | {row['proportion']} | "
            f"[{row['wilson_ci_low']}, {row['wilson_ci_high']}] |"
        )

    lines.extend(
        [
            "",
            "## 3. Cảnh báo",
            "",
        ]
    )
    warnings_list: List[str] = report_data.get("warnings", [])
    if warnings_list:
        for w in warnings_list:
            lines.append(f"- {w}")
    else:
        lines.append("- Không có cảnh báo.")

    return "\n".join(lines) + "\n"


def run_polarity_kappa_stage_a(
    nhat_path: Union[str, Path] = "data/clean/polarity_nhat.csv",
    long_path: Union[str, Path] = "data/clean/polarity_long.csv",
    key_path: Optional[Union[str, Path]] = "data/clean/polarity_key.csv",
    out_dir: Union[str, Path] = "results/polarity",
    expected_n_per_emotion: int = 50,
    n_boot: int = 2000,
    seed: int = 42,
    kappa_threshold: float = 0.40,
) -> Dict[str, Any]:
    """Execute Stage A of M7 and write all deterministic output files to `out_dir`."""
    merged, file_hashes, input_warnings = load_and_validate_polarity_inputs(
        nhat_path=nhat_path,
        long_path=long_path,
        key_path=key_path,
        expected_n_per_emotion=expected_n_per_emotion,
    )

    sub_other = merged[merged["emotion_id"] == EMO_OTHER]
    sub_surprise = merged[merged["emotion_id"] == EMO_SURPRISE]

    m_all = compute_subset_metrics(
        merged["label_nhat"].tolist(),
        merged["label_long"].tolist(),
        n_boot=n_boot,
        seed=seed,
    )
    m_other = compute_subset_metrics(
        sub_other["label_nhat"].tolist(),
        sub_other["label_long"].tolist(),
        n_boot=n_boot,
        seed=seed,
    )
    m_surprise = compute_subset_metrics(
        sub_surprise["label_nhat"].tolist(),
        sub_surprise["label_long"].tolist(),
        n_boot=n_boot,
        seed=seed,
    )

    warnings_list: List[str] = list(input_warnings)

    for sub_name, m in (("all", m_all), ("other", m_other), ("surprise", m_surprise)):
        if math.isnan(m["kappa"]):
            warnings_list.append(f"κ NaN trên tập {sub_name}")
        if m["warning_ci_biased"]:
            warnings_list.append(
                f"CI có thể lệch trên tập {sub_name}: tỉ lệ bootstrap bị bỏ = "
                f"{fmt6(m['bootstrap_dropped_rate'])} (> 0.05)"
            )

    # Check if Other and Surprise kappas lie on opposite sides of kappa_threshold (0.40)
    if not math.isnan(m_other["kappa"]) and not math.isnan(m_surprise["kappa"]):
        other_high = round(float(m_other["kappa"]), 9) >= round(float(kappa_threshold), 9)
        surp_high = round(float(m_surprise["kappa"]), 9) >= round(float(kappa_threshold), 9)
        if other_high != surp_high:
            warnings_list.append(
                f"hai κ riêng nằm hai phía của {fmt6(kappa_threshold)}: "
                f"Other κ={fmt6(m_other['kappa'])}, Surprise κ={fmt6(m_surprise['kappa'])}"
            )

    dist_df, dist_warnings = compute_polarity_distribution_df(merged)
    warnings_list.extend(dist_warnings)

    disagreements_df = extract_disagreements_df(merged)

    conf_all = compute_confusion_matrix_df(
        merged["label_nhat"].tolist(), merged["label_long"].tolist()
    )
    conf_other = compute_confusion_matrix_df(
        sub_other["label_nhat"].tolist(), sub_other["label_long"].tolist()
    )
    conf_surprise = compute_confusion_matrix_df(
        sub_surprise["label_nhat"].tolist(), sub_surprise["label_long"].tolist()
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    conf_cols = ("label", *LABELS3)
    write_csv_schema(conf_all, out_path / "confusion_all.csv", conf_cols)
    write_csv_schema(conf_other, out_path / "confusion_other.csv", conf_cols)
    write_csv_schema(conf_surprise, out_path / "confusion_surprise.csv", conf_cols)
    write_csv_schema(
        dist_df,
        out_path / "polarity_distribution.csv",
        POLARITY_DISTRIBUTION_COLUMNS,
    )
    write_csv_schema(
        disagreements_df,
        out_path / "polarity_disagreements.csv",
        POLARITY_DISAGREEMENT_COLUMNS,
    )

    hash_json_path = out_path / "polarity_files_hash.json"
    hash_json_path.write_text(
        json.dumps(file_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_payload: Dict[str, Any] = {
        "disagreement_count": int(len(disagreements_df)),
        "metrics": {
            "all": _format_subset_json(m_all),
            "other": _format_subset_json(m_other),
            "surprise": _format_subset_json(m_surprise),
        },
        "warnings": warnings_list,
    }

    kappa_json_path = out_path / "kappa_report.json"
    kappa_json_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    kappa_md_path = out_path / "kappa_report.md"
    kappa_md_path.write_text(
        render_kappa_report_md(report_payload, dist_df),
        encoding="utf-8",
    )

    return {
        "report": report_payload,
        "raw_metrics": {
            "all": m_all,
            "other": m_other,
            "surprise": m_surprise,
        },
        "distribution_df": dist_df,
        "disagreements_df": disagreements_df,
        "confusion": {
            "all": conf_all,
            "other": conf_other,
            "surprise": conf_surprise,
        },
        "file_hashes": file_hashes,
        "warnings": warnings_list,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage A (CV-P3): Validate polarity annotations and compute Cohen's kappa."
    )
    parser.add_argument(
        "--nhat-path", type=Path, default=Path("data/clean/polarity_nhat.csv")
    )
    parser.add_argument(
        "--long-path", type=Path, default=Path("data/clean/polarity_long.csv")
    )
    parser.add_argument(
        "--key-path", type=Path, default=Path("data/clean/polarity_key.csv")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/polarity")
    )
    parser.add_argument("--n-per-emotion", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run_polarity_kappa_stage_a(
        nhat_path=args.nhat_path,
        long_path=args.long_path,
        key_path=args.key_path,
        out_dir=args.out_dir,
        expected_n_per_emotion=args.n_per_emotion,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
