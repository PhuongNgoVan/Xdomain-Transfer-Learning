"""CV-L2: Stratified split generator for NEU-ESC (`src/data/make_splits.py`).

Per `docs/contract.md` (Section 1 Row 1, 4 & Section 2.1) and `tasks/M1_data.md`:
- 100% split: uses official NEU-ESC split (`train`, `dev`, `test`), merging
  `toxic -> negative` according to `src/common/constants.py`, and writes
  `data/splits/neuesc_100.csv`.
- 1%, 5%, 10% splits: stratified sampling by label from `train` per seed
  (seeds 0–4 for 5%; seeds 0–2 for 1% and 10%); `dev` is scaled down by the
  same fraction (with at least 1 sample per class); `test` is kept 100% intact.
- Writes `data/splits/neuesc_{pct}_s{seed}.csv` with schema `SPLIT_COLUMNS`
  (`id, text, label, subset`), where `id` is the row number in the original file.
- Writes `data/splits/MANIFEST.json` containing the SHA-256 of every split file.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.common.constants import ID_TO_LABEL3, NEG, NEU, POS
from src.common.hashing import build_manifest
from src.common.io import write_csv_schema
from src.common.schema import SPLIT_COLUMNS, VALID_SUBSETS
from src.common.seed import set_seed
from src.data.check_overlap import load_raw_dataset_splits

FRACTION_SEEDS: Dict[int, Tuple[int, ...]] = {
    1: (0, 1, 2),
    5: (0, 1, 2, 3, 4),
    10: (0, 1, 2),
}


def compute_stratified_counts(
    label_counts: Dict[int, int],
    fraction_pct: int,
    min_per_class: int = 1,
) -> Dict[int, int]:
    """Compute per-class sample counts preserving class proportions and exact total.

    Uses the largest-remainder method so that `sum(target_counts) == round(N * pct / 100)`
    while guaranteeing at least `min_per_class` samples for every non-empty class.
    """
    classes = sorted(c for c, cnt in label_counts.items() if cnt > 0)
    total_n = sum(label_counts[c] for c in classes)
    if total_n == 0:
        return {c: 0 for c in label_counts}

    ratio = fraction_pct / 100.0
    target_total = max(len(classes) * min_per_class, int(round(total_n * ratio)))
    target_total = min(target_total, total_n)

    exact_quotas = {c: label_counts[c] * (target_total / total_n) for c in classes}
    alloc = {
        c: min(label_counts[c], max(min_per_class, int(np.floor(exact_quotas[c]))))
        for c in classes
    }

    diff = target_total - sum(alloc.values())
    if diff > 0:
        # Distribute remaining slots by largest fractional remainder
        remainders = sorted(
            classes,
            key=lambda c: (exact_quotas[c] - np.floor(exact_quotas[c]), label_counts[c], -c),
            reverse=True,
        )
        idx = 0
        while diff > 0 and any(alloc[c] < label_counts[c] for c in classes):
            c = remainders[idx % len(remainders)]
            if alloc[c] < label_counts[c]:
                alloc[c] += 1
                diff -= 1
            idx += 1
    elif diff < 0:
        # Reduce from classes with largest allocation above min_per_class
        reducible = sorted(
            classes,
            key=lambda c: (alloc[c] - min_per_class, -(exact_quotas[c] - np.floor(exact_quotas[c])), c),
            reverse=True,
        )
        idx = 0
        while diff < 0 and any(alloc[c] > min_per_class for c in classes):
            c = reducible[idx % len(reducible)]
            if alloc[c] > min_per_class:
                alloc[c] -= 1
                diff += 1
            idx += 1

    for c in label_counts:
        if c not in alloc:
            alloc[c] = 0
    return alloc


def stratified_subsample_df(
    df_subset: pd.DataFrame,
    fraction_pct: int,
    seed: int,
    min_per_class: int = 1,
) -> pd.DataFrame:
    """Stratified subsample of `df_subset` (`train` or `dev`) for `(fraction_pct, seed)`.

    Preserves deterministic ordering by original `id`.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    label_counts = {
        int(lbl): int((df_subset["label"] == lbl).sum()) for lbl in (NEG, NEU, POS)
    }
    alloc = compute_stratified_counts(
        label_counts, fraction_pct=fraction_pct, min_per_class=min_per_class
    )

    sampled_parts: List[pd.DataFrame] = []
    for lbl in (NEG, NEU, POS):
        n_take = alloc.get(lbl, 0)
        if n_take <= 0:
            continue
        cls_df = df_subset[df_subset["label"] == lbl].sort_values("id", kind="mergesort")
        chosen_positions = rng.choice(len(cls_df), size=n_take, replace=False)
        chosen_positions.sort()
        sampled_parts.append(cls_df.iloc[chosen_positions])

    result = pd.concat(sampled_parts, ignore_index=True)
    result = result.sort_values("id", kind="mergesort").reset_index(drop=True)
    return result


def make_splits(
    raw_dir: Union[str, Path] = "data/raw",
    splits_dir: Union[str, Path] = "data/splits",
    fraction_seeds: Optional[Dict[int, Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Generate `neuesc_100.csv`, `neuesc_{pct}_s{seed}.csv`, and `data/splits/MANIFEST.json`.

    Returns:
        Summary dictionary with total counts, label counts after merging toxic->negative,
        and manifest hashes.
    """
    raw_path = Path(raw_dir)
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f_seeds = fraction_seeds if fraction_seeds is not None else FRACTION_SEEDS

    df_all = load_raw_dataset_splits(raw_path, "neuesc")

    # Deduplicate / clean any accidental internal or cross-split overlap if present in fixtures
    # (On real NEU-ESC, all 32,966 rows are already unique across train/dev/test).
    test_df = df_all[df_all["subset"] == "test"].copy()
    dev_df = df_all[df_all["subset"] == "dev"].copy()
    train_df = df_all[df_all["subset"] == "train"].copy()

    test_hashes = set(test_df["text_hash"].unique())
    dev_df = dev_df[~dev_df["text_hash"].isin(test_hashes)].drop_duplicates(
        subset=["text_hash"], keep="first"
    )
    dev_hashes = set(dev_df["text_hash"].unique())
    train_df = train_df[
        ~train_df["text_hash"].isin(test_hashes | dev_hashes)
    ].drop_duplicates(subset=["text_hash"], keep="first")

    # Sort each split by original id and combine in canonical order (train, dev, test)
    train_df = train_df.sort_values("id", kind="mergesort").reset_index(drop=True)
    dev_df = dev_df.sort_values("id", kind="mergesort").reset_index(drop=True)
    test_df = test_df.sort_values("id", kind="mergesort").reset_index(drop=True)

    full_100_df = pd.concat(
        [
            train_df.loc[:, list(SPLIT_COLUMNS)],
            dev_df.loc[:, list(SPLIT_COLUMNS)],
            test_df.loc[:, list(SPLIT_COLUMNS)],
        ],
        ignore_index=True,
    )

    written_files: List[Path] = []
    path_100 = out_dir / "neuesc_100.csv"
    write_csv_schema(full_100_df, path_100, SPLIT_COLUMNS)
    written_files.append(path_100)

    # Generate fractional splits (1%, 5%, 10%)
    for pct, seeds in sorted(f_seeds.items()):
        for seed in seeds:
            # Offset seed slightly between train and dev so their RNG streams are independent
            sub_train = stratified_subsample_df(
                train_df, fraction_pct=pct, seed=seed * 1000 + pct, min_per_class=1
            )
            sub_dev = stratified_subsample_df(
                dev_df, fraction_pct=pct, seed=seed * 1000 + pct + 500, min_per_class=1
            )
            split_df = pd.concat(
                [
                    sub_train.loc[:, list(SPLIT_COLUMNS)],
                    sub_dev.loc[:, list(SPLIT_COLUMNS)],
                    test_df.loc[:, list(SPLIT_COLUMNS)],
                ],
                ignore_index=True,
            )
            out_file = out_dir / f"neuesc_{pct}_s{seed}.csv"
            write_csv_schema(split_df, out_file, SPLIT_COLUMNS)
            written_files.append(out_file)

    manifest = build_manifest(written_files, base_dir=out_dir)
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    label_counts_total = {
        ID_TO_LABEL3[lbl]: int((full_100_df["label"] == lbl).sum())
        for lbl in (NEG, NEU, POS)
    }
    label_counts_by_subset = {
        sp: {
            ID_TO_LABEL3[lbl]: int(
                (full_100_df[full_100_df["subset"] == sp]["label"] == lbl).sum()
            )
            for lbl in (NEG, NEU, POS)
        }
        for sp in VALID_SUBSETS
    }

    print(f"=== NEU-ESC 100% Split Summary (Total = {len(full_100_df)}) ===")
    for sp in VALID_SUBSETS:
        c = label_counts_by_subset[sp]
        print(
            f"  {sp:5s}: neg={c['negative']}, neu={c['neutral']}, pos={c['positive']} "
            f"(total={sum(c.values())})"
        )
    print(f"  TOTAL: {label_counts_total}")

    return {
        "total_samples": int(len(full_100_df)),
        "label_counts_total": label_counts_total,
        "label_counts_by_subset": label_counts_by_subset,
        "manifest": manifest,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CV-L2: Generate NEU-ESC splits.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args(argv)
    make_splits(raw_dir=args.raw_dir, splits_dir=args.splits_dir)


if __name__ == "__main__":
    main()
