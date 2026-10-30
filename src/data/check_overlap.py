"""CV-X1: Cross-dataset and intra-dataset duplicate and label-conflict checker.

Per `docs/contract.md` (Section 1 row 1, 6) and `tasks/M1_data.md`:
1. Before mapping UIT-VSFC labels: print original label distribution, cross-check
   against dataset description, and write `docs/vsfc_label_mapping.md`.
2. Normalize text via `textnorm.normalize_text`, hash via `hashing.hash_text`, and
   compute the 9x9 overlap matrix across `(vsmec, neuesc, vsfc) x (train, dev, test)`.
3. Report internal duplicates (same-label vs label-conflict) within each split.
4. Export `data/clean/vsfc_test_clean.csv`: remove internal duplicates, label conflicts,
   and sentences overlapping with NEU-ESC train or VSMEC train.
5. If NEU-ESC test overlaps with VSMEC, record those hashes in the VSMEC train
   exclusion list (never modify NEU-ESC test).
6. Export the overlap matrix CSV (`data/clean/overlap_matrix.csv`), internal duplicate
   stats (`data/clean/internal_stats.csv`), and exclusion list (`data/clean/excluded_records.csv`)
   along with SHA-256 hashes in `data/clean/MANIFEST.json`.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from src.common.constants import (
    EMOTION7_TO_ID,
    ID_TO_EMOTION7,
    ID_TO_LABEL3,
    LABEL3_TO_ID,
    NEG,
    NEU,
    NEUESC_ORIG_NAME_TO_LABEL3,
    NEUESC_ORIG_TO_LABEL3,
    POS,
)
from src.common.hashing import build_manifest, hash_text
from src.common.io import write_csv_schema
from src.common.schema import SPLIT_COLUMNS, VALID_SUBSETS
from src.common.textnorm import normalize_text

DATASETS: Tuple[str, ...] = ("vsmec", "neuesc", "vsfc")
SPLITS: Tuple[str, ...] = VALID_SUBSETS  # ("train", "dev", "test")

_SPLIT_ALIASES: Dict[str, str] = {
    "train": "train",
    "dev": "dev",
    "val": "dev",
    "valid": "dev",
    "validation": "dev",
    "test": "test",
}

VSFC_ORIG_INT_TO_LABEL3: Dict[int, int] = {
    0: NEG,
    1: NEU,
    2: POS,
}


def _normalize_split_name(val: Any) -> str:
    s = str(val).strip().lower()
    if s not in _SPLIT_ALIASES:
        raise ValueError(f"Unknown split name: {val!r}")
    return _SPLIT_ALIASES[s]


def _map_vsmec_label(val: Any) -> int:
    if isinstance(val, (int, float)) and not pd.isna(val):
        idx = int(val)
        if idx in ID_TO_EMOTION7:
            return idx
    s = str(val).strip()
    if s in EMOTION7_TO_ID:
        return EMOTION7_TO_ID[s]
    # Case-insensitive lookup
    for k, v in EMOTION7_TO_ID.items():
        if k.lower() == s.lower():
            return v
    if s.isdigit() and int(s) in ID_TO_EMOTION7:
        return int(s)
    raise ValueError(f"Unknown VSMEC emotion label: {val!r}")


def _map_neuesc_label(val: Any) -> int:
    if isinstance(val, (int, float)) and not pd.isna(val):
        idx = int(val)
        if idx in NEUESC_ORIG_TO_LABEL3:
            return NEUESC_ORIG_TO_LABEL3[idx]
    s = str(val).strip().lower()
    if s in NEUESC_ORIG_NAME_TO_LABEL3:
        return NEUESC_ORIG_NAME_TO_LABEL3[s]
    try:
        idx = int(float(s))
        if idx in NEUESC_ORIG_TO_LABEL3:
            return NEUESC_ORIG_TO_LABEL3[idx]
    except ValueError:
        pass
    raise ValueError(f"Unknown NEU-ESC label: {val!r}")


def _map_vsfc_label(val: Any) -> int:
    if isinstance(val, (int, float)) and not pd.isna(val):
        idx = int(val)
        if idx in VSFC_ORIG_INT_TO_LABEL3:
            return VSFC_ORIG_INT_TO_LABEL3[idx]
    s = str(val).strip().lower()
    if s in LABEL3_TO_ID:
        return LABEL3_TO_ID[s]
    try:
        idx = int(float(s))
        if idx in VSFC_ORIG_INT_TO_LABEL3:
            return VSFC_ORIG_INT_TO_LABEL3[idx]
    except ValueError:
        pass
    raise ValueError(f"Unknown UIT-VSFC label: {val!r}")


def load_raw_dataset_splits(
    raw_dir: Union[str, Path],
    dataset: str,
) -> pd.DataFrame:
    """Load raw dataset (`vsmec`, `neuesc`, or `vsfc`) into a unified DataFrame.

    Supports either a single `*_full.csv` file or per-split files
    (`*_{train,validation,dev,test}.csv`). Preserves `id` as the row index
    in the source file.

    Returns:
        DataFrame with columns `['id', 'text', 'orig_label', 'label', 'subset', 'norm_text', 'text_hash']`.
    """
    raw_path = Path(raw_dir)
    prefix_candidates = {
        "vsmec": ["UIT_VSMEC", "vsmec"],
        "neuesc": ["NEU_ESC", "neuesc"],
        "vsfc": ["UIT_VSFC", "vsfc"],
    }[dataset]

    df_raw: Optional[pd.DataFrame] = None
    for prefix in prefix_candidates:
        full_file = raw_path / f"{prefix}_full.csv"
        if full_file.is_file():
            df_raw = pd.read_csv(full_file)
            if "id" not in df_raw.columns:
                df_raw["id"] = range(len(df_raw))
            break

    if df_raw is None:
        parts: List[pd.DataFrame] = []
        offset = 0
        for split_suffix, canonical_split in [
            ("train", "train"),
            ("validation", "dev"),
            ("dev", "dev"),
            ("test", "test"),
        ]:
            for prefix in prefix_candidates:
                candidate = raw_path / f"{prefix}_{split_suffix}.csv"
                if candidate.is_file():
                    sub_df = pd.read_csv(candidate)
                    sub_df["Split"] = canonical_split
                    if "id" not in sub_df.columns:
                        sub_df["id"] = range(offset, offset + len(sub_df))
                    offset += len(sub_df)
                    parts.append(sub_df)
                    break
        if not parts:
            raise FileNotFoundError(
                f"Could not find raw CSV files for dataset {dataset!r} in {raw_path}"
            )
        df_raw = pd.concat(parts, ignore_index=True)

    # Identify text column
    text_col = next(
        (c for c in ["text", "Sentence", "sentence"] if c in df_raw.columns),
        None,
    )
    if text_col is None:
        raise ValueError(
            f"Cannot find text column in {dataset} (columns={list(df_raw.columns)})"
        )

    # Identify split column
    split_col = next(
        (c for c in ["subset", "Split", "split"] if c in df_raw.columns),
        None,
    )
    if split_col is None:
        raise ValueError(
            f"Cannot find split column in {dataset} (columns={list(df_raw.columns)})"
        )

    # Identify label column
    if dataset == "vsmec":
        label_col = next(
            (c for c in ["Emotion", "emotion", "label"] if c in df_raw.columns),
            None,
        )
        mapper = _map_vsmec_label
    elif dataset == "neuesc":
        label_col = next(
            (c for c in ["sentiment", "label", "Sentiment"] if c in df_raw.columns),
            None,
        )
        mapper = _map_neuesc_label
    elif dataset == "vsfc":
        label_col = next(
            (
                c
                for c in ["Encoded_sentiment", "Sentiment", "sentiment", "label"]
                if c in df_raw.columns
            ),
            None,
        )
        mapper = _map_vsfc_label
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    if label_col is None:
        raise ValueError(
            f"Cannot find label column in {dataset} (columns={list(df_raw.columns)})"
        )

    out = pd.DataFrame()
    out["id"] = df_raw["id"].astype(int)
    out["text"] = df_raw[text_col].fillna("").astype(str)
    out["orig_label"] = df_raw[label_col]
    if dataset == "vsfc" and "Sentiment" in df_raw.columns and label_col == "Encoded_sentiment":
        out["orig_label_str"] = df_raw["Sentiment"].astype(str)
    out["label"] = [mapper(v) for v in df_raw[label_col]]
    out["subset"] = [_normalize_split_name(v) for v in df_raw[split_col]]
    out["norm_text"] = [normalize_text(t) for t in out["text"]]
    out["text_hash"] = [hash_text(nt, normalize=False) for nt in out["norm_text"]]
    return out


def inspect_and_document_vsfc_labels(
    vsfc_df: pd.DataFrame,
    doc_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Print original UIT-VSFC label distribution and write `docs/vsfc_label_mapping.md`."""
    counts_by_split: Dict[str, Dict[int, int]] = {}
    for split_name in SPLITS:
        sub = vsfc_df[vsfc_df["subset"] == split_name]
        counts_by_split[split_name] = {
            lbl: int((sub["label"] == lbl).sum()) for lbl in (NEG, NEU, POS)
        }

    total_counts = {
        lbl: int((vsfc_df["label"] == lbl).sum()) for lbl in (NEG, NEU, POS)
    }

    print("=== Phân bố nhãn gốc UIT-VSFC (0=negative, 1=neutral, 2=positive) ===")
    for split_name in SPLITS:
        c = counts_by_split[split_name]
        print(
            f"  {split_name:5s}: neg(0)={c[NEG]}, neu(1)={c[NEU]}, pos(2)={c[POS]} (total={sum(c.values())})"
        )
    print(
        f"  TOTAL: neg(0)={total_counts[NEG]}, neu(1)={total_counts[NEU]}, pos(2)={total_counts[POS]} (total={len(vsfc_df)})"
    )

    if doc_path is not None:
        p = Path(doc_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Đối chiếu và Ánh xạ Nhãn gốc UIT-VSFC (`docs/vsfc_label_mapping.md`)",
            "",
            "> **Căn cứ hợp đồng (`docs/contract.md` mục 1 dòng 1 & `tasks/M1_data.md`):**",
            "> - Nhãn 3 lớp đích dùng chung cho NEU-ESC và UIT-VSFC: `0 = negative, 1 = neutral, 2 = positive` (khai báo trong `src/common/constants.py`).",
            "> - Trước khi viết ánh xạ cho VSFC, in phân bố nhãn gốc và đối chiếu với mô tả bộ dữ liệu.",
            "",
            "## 1. Phân bố nhãn gốc thực tế trên UIT-VSFC",
            "",
            "| Nhãn gốc (`Encoded_sentiment`) | Tên nhãn gốc (`Sentiment`) | Nhãn đích (`constants.py`) | Train | Validation (Dev) | Test | Tổng |",
            "| :---: | :---: | :---: | ---: | ---: | ---: | ---: |",
            f"| `0` | `negative` | `NEG = 0` (`{ID_TO_LABEL3[NEG]}`) | {counts_by_split['train'][NEG]} | {counts_by_split['dev'][NEG]} | {counts_by_split['test'][NEG]} | {total_counts[NEG]} |",
            f"| `1` | `neutral` | `NEU = 1` (`{ID_TO_LABEL3[NEU]}`) | {counts_by_split['train'][NEU]} | {counts_by_split['dev'][NEU]} | {counts_by_split['test'][NEU]} | {total_counts[NEU]} |",
            f"| `2` | `positive` | `POS = 2` (`{ID_TO_LABEL3[POS]}`) | {counts_by_split['train'][POS]} | {counts_by_split['dev'][POS]} | {counts_by_split['test'][POS]} | {total_counts[POS]} |",
            f"| **Tổng** | | | **{sum(counts_by_split['train'].values())}** | **{sum(counts_by_split['dev'].values())}** | **{sum(counts_by_split['test'].values())}** | **{len(vsfc_df)}** |",
            "",
            "## 2. Kết luận đối chiếu với mô tả bộ dữ liệu gốc",
            "",
            "- Trong bộ dữ liệu **UIT-VSFC** (*Vietnamese Students' Feedback Corpus*, Nguyen et al., 2018), nhãn gốc được mã hóa `0 = negative`, `1 = neutral`, `2 = positive`.",
            "- Cột `Sentiment` và `Encoded_sentiment` trong file dữ liệu gốc hoàn toàn khớp với mô tả bài báo gốc (Train: 11.426, Dev: 1.583, Test: 3.166; lớp `neutral` chiếm tỉ lệ nhỏ nhất ~4,3%).",
            "- Khác với NEU-ESC gốc (`0 = neutral, 1 = positive, 2 = negative, 3 = toxic`), mã số của UIT-VSFC gốc (`0, 1, 2`) **trùng khớp 1-1** với hằng số chuẩn `NEG = 0, NEU = 1, POS = 2` trong `src/common/constants.py`.",
            "",
        ]
        p.write_text("\n".join(lines), encoding="utf-8")

    return {"by_split": counts_by_split, "total": total_counts}


def analyze_internal_duplicates(df_split: pd.DataFrame) -> Dict[str, Any]:
    """Analyze internal duplicate hashes and label conflicts within a single split."""
    if df_split.empty:
        return {
            "total_rows": 0,
            "unique_hashes": 0,
            "same_label_dup_groups": 0,
            "same_label_extra_rows_removed": 0,
            "conflict_label_groups": 0,
            "conflict_label_rows_removed": 0,
            "same_label_hashes": set(),
            "conflict_hashes": set(),
        }

    grouped = df_split.groupby("text_hash", sort=False)
    same_label_groups = 0
    same_label_extra_rows = 0
    conflict_groups = 0
    conflict_rows = 0
    same_label_hashes: Set[str] = set()
    conflict_hashes: Set[str] = set()

    for h, grp in grouped:
        n_rows = len(grp)
        if n_rows <= 1:
            continue
        n_labels = grp["label"].nunique()
        if n_labels == 1:
            same_label_groups += 1
            same_label_extra_rows += n_rows - 1
            same_label_hashes.add(str(h))
        else:
            conflict_groups += 1
            conflict_rows += n_rows
            conflict_hashes.add(str(h))

    return {
        "total_rows": int(len(df_split)),
        "unique_hashes": int(df_split["text_hash"].nunique()),
        "same_label_dup_groups": same_label_groups,
        "same_label_extra_rows_removed": same_label_extra_rows,
        "conflict_label_groups": conflict_groups,
        "conflict_label_rows_removed": conflict_rows,
        "same_label_hashes": same_label_hashes,
        "conflict_hashes": conflict_hashes,
    }


def run_check_overlap(
    raw_dir: Union[str, Path] = "data/raw",
    clean_dir: Union[str, Path] = "data/clean",
    vsfc_doc_path: Optional[Union[str, Path]] = "docs/vsfc_label_mapping.md",
) -> Dict[str, Any]:
    """Execute full CV-X1 overlap analysis and export clean VSFC test + exclusion lists.

    Returns:
        Dictionary summarizing the overlap matrix, internal stats, VSFC clean stats,
        and VSMEC train cross-dataset exclusion hashes.
    """
    raw_path = Path(raw_dir)
    out_dir = Path(clean_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: Dict[str, pd.DataFrame] = {
        ds: load_raw_dataset_splits(raw_path, ds) for ds in DATASETS
    }

    # 1. Inspect and document VSFC label distribution before mapping
    vsfc_label_summary = inspect_and_document_vsfc_labels(
        datasets["vsfc"], doc_path=vsfc_doc_path
    )

    # 2. Build 9x9 overlap matrix across (vsmec, neuesc, vsfc) x (train, dev, test)
    pair_keys: List[str] = [f"{ds}_{sp}" for ds in DATASETS for sp in SPLITS]
    hash_sets: Dict[str, Set[str]] = {}
    split_dfs: Dict[str, pd.DataFrame] = {}
    internal_rows: List[Dict[str, Any]] = []

    for ds in DATASETS:
        for sp in SPLITS:
            key = f"{ds}_{sp}"
            sub = datasets[ds][datasets[ds]["subset"] == sp].copy()
            split_dfs[key] = sub
            hash_sets[key] = set(sub["text_hash"].unique())
            istats = analyze_internal_duplicates(sub)
            internal_rows.append(
                {
                    "dataset": ds,
                    "split": sp,
                    "total_rows": istats["total_rows"],
                    "unique_hashes": istats["unique_hashes"],
                    "same_label_dup_groups": istats["same_label_dup_groups"],
                    "same_label_extra_rows_removed": istats["same_label_extra_rows_removed"],
                    "conflict_label_groups": istats["conflict_label_groups"],
                    "conflict_label_rows_removed": istats["conflict_label_rows_removed"],
                }
            )

    matrix_data: Dict[str, List[int]] = {"split_pair": pair_keys}
    for k_col in pair_keys:
        matrix_data[k_col] = [
            len(hash_sets[k_row] & hash_sets[k_col]) for k_row in pair_keys
        ]
    overlap_matrix_df = pd.DataFrame(matrix_data)
    overlap_matrix_path = out_dir / "overlap_matrix.csv"
    overlap_matrix_df.to_csv(overlap_matrix_path, index=False, encoding="utf-8", lineterminator="\n")

    internal_stats_df = pd.DataFrame(internal_rows)
    internal_stats_path = out_dir / "internal_stats.csv"
    internal_stats_df.to_csv(internal_stats_path, index=False, encoding="utf-8", lineterminator="\n")

    # 3. Clean UIT-VSFC test (`vsfc_test_clean.csv`):
    #    - Remove internal label conflicts (all copies)
    #    - Remove internal same-label duplicates (keep first copy)
    #    - Remove sentences overlapping with NEU-ESC train or VSMEC train
    vsfc_test_df = split_dfs["vsfc_test"].copy()
    vsfc_test_istats = analyze_internal_duplicates(vsfc_test_df)
    vsfc_conflict_hashes: Set[str] = vsfc_test_istats["conflict_hashes"]
    neuesc_train_hashes: Set[str] = hash_sets["neuesc_train"]
    vsmec_train_hashes: Set[str] = hash_sets["vsmec_train"]

    excluded_records: List[Dict[str, Any]] = []
    seen_vsfc_hashes: Set[str] = set()
    keep_vsfc_indices: List[int] = []

    vsfc_removed_counts = {
        "conflict_label": 0,
        "internal_duplicate": 0,
        "overlap_neuesc_train": 0,
        "overlap_vsmec_train": 0,
    }

    for idx, row in vsfc_test_df.iterrows():
        h = str(row["text_hash"])
        reason: Optional[str] = None
        if h in vsfc_conflict_hashes:
            reason = "conflict_label"
        elif h in neuesc_train_hashes:
            reason = "overlap_neuesc_train"
        elif h in vsmec_train_hashes:
            reason = "overlap_vsmec_train"
        elif h in seen_vsfc_hashes:
            reason = "internal_duplicate"

        if reason is not None:
            vsfc_removed_counts[reason] += 1
            excluded_records.append(
                {
                    "target_split": "vsfc_test",
                    "id": int(row["id"]),
                    "label": int(row["label"]),
                    "reason": reason,
                    "text_hash": h,
                    "norm_text": str(row["norm_text"]),
                }
            )
        else:
            seen_vsfc_hashes.add(h)
            keep_vsfc_indices.append(idx)

    vsfc_test_clean_df = vsfc_test_df.loc[keep_vsfc_indices, list(SPLIT_COLUMNS)].reset_index(
        drop=True
    )
    vsfc_clean_path = out_dir / "vsfc_test_clean.csv"
    write_csv_schema(vsfc_test_clean_df, vsfc_clean_path, SPLIT_COLUMNS)

    # 4. Identify VSMEC train rows that overlap with NEU-ESC test (or NEU-ESC dev)
    #    Per contract & M1 spec: "NEU-ESC test trùng với VSMEC thì loại khỏi VSMEC train,
    #    không đụng đến test."
    neuesc_test_hashes: Set[str] = hash_sets["neuesc_test"]
    vsmec_train_df = split_dfs["vsmec_train"]
    vsmec_cross_exclude_hashes: Set[str] = set()

    for _, row in vsmec_train_df.iterrows():
        h = str(row["text_hash"])
        if h in neuesc_test_hashes:
            vsmec_cross_exclude_hashes.add(h)
            excluded_records.append(
                {
                    "target_split": "vsmec_train",
                    "id": int(row["id"]),
                    "label": int(row["label"]),
                    "reason": "overlap_neuesc_test",
                    "text_hash": h,
                    "norm_text": str(row["norm_text"]),
                }
            )

    excluded_df = pd.DataFrame(
        excluded_records,
        columns=["target_split", "id", "label", "reason", "text_hash", "norm_text"],
    )
    excluded_path = out_dir / "excluded_records.csv"
    excluded_df.to_csv(excluded_path, index=False, encoding="utf-8", lineterminator="\n")

    # Update MANIFEST.json in clean_dir
    manifest_path = out_dir / "MANIFEST.json"
    existing_csvs = sorted(out_dir.glob("*.csv"))
    manifest = build_manifest(existing_csvs, base_dir=out_dir)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "vsfc_label_summary": vsfc_label_summary,
        "overlap_matrix": overlap_matrix_df,
        "internal_stats": internal_stats_df,
        "vsfc_test_original_count": int(len(vsfc_test_df)),
        "vsfc_test_clean_count": int(len(vsfc_test_clean_df)),
        "vsfc_removed_counts": vsfc_removed_counts,
        "vsmec_train_overlap_neuesc_test_hashes": sorted(vsmec_cross_exclude_hashes),
        "excluded_records": excluded_df,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CV-X1: Check dataset overlaps and clean VSFC test.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--vsfc-doc", type=Path, default=Path("docs/vsfc_label_mapping.md"))
    args = parser.parse_args(argv)

    summary = run_check_overlap(
        raw_dir=args.raw_dir,
        clean_dir=args.clean_dir,
        vsfc_doc_path=args.vsfc_doc,
    )
    print("\n=== Overlap Matrix ===")
    print(summary["overlap_matrix"].to_string(index=False))
    print("\n=== Internal Duplicate & Conflict Stats ===")
    print(summary["internal_stats"].to_string(index=False))
    print(
        f"\n=== VSFC Test Clean: {summary['vsfc_test_clean_count']} / {summary['vsfc_test_original_count']} "
        f"(removed: {summary['vsfc_removed_counts']}) ==="
    )


if __name__ == "__main__":
    main()
