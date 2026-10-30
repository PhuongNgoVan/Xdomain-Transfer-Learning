"""CV-N1: Clean UIT-VSMEC dataset (`src/data/clean_vsmec.py`).

Runs after `check_overlap.py` and applies `docs/contract.md` Section 1 Row 6:
- Same-label duplicate text within a split: keep one copy (first occurrence).
- Different-label duplicate text (label conflict): remove ALL copies.
- Cross-split duplicate text: remove from train (and remove from dev if overlapping
  with test, ensuring zero overlap across train, dev, and test).
- Plus exclusion list from `check_overlap.py` (e.g., VSMEC train sentences that
  overlap with NEU-ESC test).
- Writes `data/clean/vsmec_{train,dev,test}.csv` and `data/clean/vsmec_clean_report.json`
  and updates `data/clean/MANIFEST.json`.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import pandas as pd

from src.common.hashing import build_manifest
from src.common.io import write_csv_schema
from src.common.schema import SPLIT_COLUMNS, VALID_SUBSETS
from src.data.check_overlap import analyze_internal_duplicates, load_raw_dataset_splits


def _load_external_exclusion_hashes(clean_dir: Path) -> Set[str]:
    """Load `text_hash` values marked for removal from `vsmec_train` by `check_overlap.py`."""
    excluded_path = clean_dir / "excluded_records.csv"
    if not excluded_path.is_file():
        return set()
    df = pd.read_csv(excluded_path)
    if "target_split" not in df.columns or "text_hash" not in df.columns:
        return set()
    sub = df[df["target_split"] == "vsmec_train"]
    return set(sub["text_hash"].astype(str).unique())


def clean_vsmec(
    raw_dir: Union[str, Path] = "data/raw",
    clean_dir: Union[str, Path] = "data/clean",
    extra_train_exclude_hashes: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Clean UIT-VSMEC splits according to contract rules and save `vsmec_{train,dev,test}.csv`.

    Returns:
        Dictionary containing cleaned DataFrames and per-split removal counts by reason.
    """
    raw_path = Path(raw_dir)
    out_dir = Path(clean_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_raw_dataset_splits(raw_path, "vsmec")

    # External exclusion hashes from check_overlap.py
    ext_exclude = _load_external_exclusion_hashes(out_dir)
    if extra_train_exclude_hashes is not None:
        ext_exclude = ext_exclude | set(extra_train_exclude_hashes)

    # Identify global label conflicts across the entire VSMEC corpus or within each split
    # Contract row 6: "Trùng văn bản cùng nhãn: giữ một bản. Trùng khác nhãn: loại tất cả các bản.
    # Trùng chéo split: loại khỏi train. Báo cáo số lượng từng loại"
    split_dfs: Dict[str, pd.DataFrame] = {
        sp: df_all[df_all["subset"] == sp].copy() for sp in VALID_SUBSETS
    }

    # Global hash -> set of labels across all splits (so cross-split label conflicts are also caught)
    global_conflict_hashes: Set[str] = set()
    for h, grp in df_all.groupby("text_hash", sort=False):
        if grp["label"].nunique() > 1:
            global_conflict_hashes.add(str(h))

    # Step 1: Clean `test` split first (internal conflicts + internal same-label duplicates)
    # Step 2: Clean `dev` split (internal conflicts + internal duplicates + cross-split overlap with test)
    # Step 3: Clean `train` split (internal conflicts + internal duplicates + cross-split overlap with dev/test + external overlap from check_overlap.py)
    cleaned_splits: Dict[str, pd.DataFrame] = {}
    stats_by_split: Dict[str, Dict[str, int]] = {}
    removed_log: List[Dict[str, Any]] = []

    # Process in priority order: test -> dev -> train so test and dev hashes are fixed before filtering train
    kept_hashes_by_split: Dict[str, Set[str]] = {"test": set(), "dev": set(), "train": set()}
    raw_hashes_by_split: Dict[str, Set[str]] = {
        sp: set(split_dfs[sp]["text_hash"].astype(str).unique()) for sp in VALID_SUBSETS
    }

    for sp in ("test", "dev", "train"):
        sub = split_dfs[sp]
        istats = analyze_internal_duplicates(sub)
        internal_conflict_hashes: Set[str] = istats["conflict_hashes"] | global_conflict_hashes

        seen_hashes: Set[str] = set()
        keep_indices: List[int] = []
        counts = {
            "original_count": int(len(sub)),
            "conflict_label": 0,
            "same_label_duplicate": 0,
            "cross_split_overlap": 0,
            "external_overlap": 0,
            "kept_count": 0,
        }

        # Determine cross-split reference hashes
        if sp == "train":
            cross_ref_hashes = raw_hashes_by_split["dev"] | raw_hashes_by_split["test"]
        elif sp == "dev":
            cross_ref_hashes = raw_hashes_by_split["test"]
        else:
            cross_ref_hashes = set()

        for idx, row in sub.iterrows():
            h = str(row["text_hash"])
            reason: Optional[str] = None

            if h in internal_conflict_hashes:
                reason = "conflict_label"
            elif sp == "train" and h in ext_exclude:
                reason = "external_overlap"
            elif h in cross_ref_hashes:
                reason = "cross_split_overlap"
            elif h in seen_hashes:
                reason = "same_label_duplicate"

            if reason is not None:
                counts[reason] += 1
                removed_log.append(
                    {
                        "split": sp,
                        "id": int(row["id"]),
                        "label": int(row["label"]),
                        "reason": reason,
                        "text_hash": h,
                        "norm_text": str(row["norm_text"]),
                    }
                )
            else:
                seen_hashes.add(h)
                keep_indices.append(idx)

        clean_sub = sub.loc[keep_indices, list(SPLIT_COLUMNS)].reset_index(drop=True)
        counts["kept_count"] = int(len(clean_sub))
        cleaned_splits[sp] = clean_sub
        kept_hashes_by_split[sp] = seen_hashes
        stats_by_split[sp] = counts

        out_csv = out_dir / f"vsmec_{sp}.csv"
        write_csv_schema(clean_sub, out_csv, SPLIT_COLUMNS)

    # Save detailed removal log and summary JSON
    removed_df = pd.DataFrame(
        removed_log,
        columns=["split", "id", "label", "reason", "text_hash", "norm_text"],
    )
    removed_df.to_csv(
        out_dir / "vsmec_removed.csv", index=False, encoding="utf-8", lineterminator="\n"
    )

    total_summary = {
        "original_total": sum(s["original_count"] for s in stats_by_split.values()),
        "kept_total": sum(s["kept_count"] for s in stats_by_split.values()),
        "conflict_label_total": sum(s["conflict_label"] for s in stats_by_split.values()),
        "same_label_duplicate_total": sum(
            s["same_label_duplicate"] for s in stats_by_split.values()
        ),
        "cross_split_overlap_total": sum(
            s["cross_split_overlap"] for s in stats_by_split.values()
        ),
        "external_overlap_total": sum(
            s["external_overlap"] for s in stats_by_split.values()
        ),
        "by_split": stats_by_split,
    }
    report_path = out_dir / "vsmec_clean_report.json"
    report_path.write_text(
        json.dumps(total_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Update MANIFEST.json in clean_dir
    manifest_path = out_dir / "MANIFEST.json"
    existing_csvs = sorted(out_dir.glob("*.csv"))
    manifest = build_manifest(existing_csvs, base_dir=out_dir)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=== UIT-VSMEC Cleaning Summary ===")
    for sp in VALID_SUBSETS:
        s = stats_by_split[sp]
        print(
            f"  {sp:5s}: original={s['original_count']}, kept={s['kept_count']}, "
            f"same_label_dup={s['same_label_duplicate']}, conflict_label={s['conflict_label']}, "
            f"cross_split={s['cross_split_overlap']}, external={s['external_overlap']}"
        )

    return {
        "splits": cleaned_splits,
        "summary": total_summary,
        "removed": removed_df,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CV-N1: Clean UIT-VSMEC dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    args = parser.parse_args(argv)
    clean_vsmec(raw_dir=args.raw_dir, clean_dir=args.clean_dir)


if __name__ == "__main__":
    main()
