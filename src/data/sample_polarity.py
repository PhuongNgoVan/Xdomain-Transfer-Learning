"""Sample 100 clean VSMEC sentences (50 Other, 50 Surprise) for independent polarity labeling.

Per `tasks/M1b_polarity_sample.md` (CV-N4, CV-L3):
- Samples 50 `Other` and 50 `Surprise` sentences from clean VSMEC (`vsmec_train.csv`,
  falling back to `vsmec_dev.csv` then `vsmec_test.csv` if fewer than 50 valid unique
  sentences exist in train) with a fixed seed (`seed=42`).
- Uses cleaned `text` column (never `text_seg`).
- Filters out empty sentences, sentences containing only special characters, and
  duplicate texts (replenishing to guarantee `n_per_emotion` per emotion).
- Sorts by `id` before sampling and shuffles the combined 100 sentences deterministically.
- Exports two identical annotator CSV files (`polarity_nhat.csv`, `polarity_long.csv`)
  with columns `id, text, polarity_label` (empty `polarity_label`, no emotion columns).
- Exports `polarity_key.csv` with columns `id, text, emotion_id, emotion_name` in the
  exact same `id` row order for `kappa.py`.
- If `polarity_nhat.csv` and `polarity_long.csv` already exist in `out_dir` (and
  `overwrite=False`), does NOT overwrite them; instead exports only `polarity_key.csv`
  joined by `id` from those two files and updates `MANIFEST.json`.
- Valid polarity labels for manual annotation: `0 = negative, 1 = neutral, 2 = positive`.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.common.constants import (
    EMO_OTHER,
    EMO_SURPRISE,
    ID_TO_EMOTION7,
    ID_TO_LABEL3,
    NEG,
    NEU,
    POS,
)
from src.common.hashing import build_manifest
from src.common.io import read_csv_schema, write_csv_schema
from src.common.schema import (
    POLARITY_KEY_COLUMNS,
    POLARITY_SAMPLE_COLUMNS,
    SPLIT_COLUMNS,
)
from src.common.seed import set_seed
from src.common.textnorm import normalize_text


def is_valid_polarity_text(text: object) -> bool:
    """Return True if `text` is a non-empty string with at least one alphanumeric character."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return False
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if not any(ch.isalnum() for ch in stripped):
        return False
    norm = normalize_text(stripped)
    if not norm or not any(ch.isalnum() for ch in norm):
        return False
    return True


def filter_valid_unique_pool(
    df: pd.DataFrame,
    seen_ids: Optional[Set[int]] = None,
    seen_texts: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Filter out empty, special-char-only, and duplicate sentences after sorting by `id`.

    Uses the cleaned `text` column (never `text_seg`). Sorts by `id` ascending first
    so that deduplication is completely independent of input row order.
    """
    if seen_ids is None:
        seen_ids = set()
    if seen_texts is None:
        seen_texts = set()

    sorted_df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    keep_indices: List[int] = []

    for idx, row in sorted_df.iterrows():
        raw_val = row["text"]
        if not is_valid_polarity_text(raw_val):
            continue
        row_id = int(row["id"])
        if row_id in seen_ids:
            continue
        raw_clean = str(raw_val).strip()
        norm_clean = normalize_text(raw_clean)
        if raw_clean in seen_texts or norm_clean in seen_texts:
            continue

        seen_ids.add(row_id)
        seen_texts.add(raw_clean)
        seen_texts.add(norm_clean)
        keep_indices.append(int(idx))

    if not keep_indices:
        return pd.DataFrame(columns=list(SPLIT_COLUMNS))

    out = sorted_df.loc[keep_indices, list(SPLIT_COLUMNS)].copy()
    out["id"] = out["id"].astype(int)
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(int)
    out["subset"] = out["subset"].astype(str)
    return out.reset_index(drop=True)


def update_clean_manifest(target_dir: Union[str, Path]) -> Dict[str, str]:
    """Compute SHA-256 for all CSV files in `target_dir` and write `MANIFEST.json`."""
    dest = Path(target_dir)
    existing_csvs = sorted(dest.glob("*.csv"))
    manifest = build_manifest(existing_csvs, base_dir=dest)
    manifest_path = dest / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _export_key_from_existing_annotator_files(
    clean_path: Path,
    path1: Path,
    path2: Path,
    key_path: Path,
) -> pd.DataFrame:
    """Build and write `polarity_key.csv` joined by `id` from existing annotator CSVs."""
    df1 = read_csv_schema(
        path1, POLARITY_SAMPLE_COLUMNS, strict_order=False, keep_default_na=False
    )
    df2 = read_csv_schema(
        path2, POLARITY_SAMPLE_COLUMNS, strict_order=False, keep_default_na=False
    )

    ids1 = df1["id"].astype(int).tolist()
    ids2 = df2["id"].astype(int).tolist()
    if ids1 != ids2:
        raise ValueError(
            f"Existing annotator files {path1.name} and {path2.name} have mismatched 'id' sequences."
        )

    # Build id -> (text, label) mapping from clean VSMEC splits
    id_to_label: Dict[int, int] = {}
    id_to_text: Dict[int, str] = {}
    for sp in ("train", "dev", "test"):
        sp_csv = clean_path / f"vsmec_{sp}.csv"
        if sp_csv.is_file():
            df_sp = read_csv_schema(
                sp_csv, SPLIT_COLUMNS, strict_order=False, keep_default_na=False
            )
            for _, row in df_sp.iterrows():
                rid = int(row["id"])
                if rid not in id_to_label:
                    id_to_label[rid] = int(row["label"])
                    id_to_text[rid] = str(row["text"])

    has_inline_emotion = "emotion_id" in df1.columns
    emotion_ids: List[int] = []
    texts: List[str] = []

    for idx, row in df1.iterrows():
        rid = int(row["id"])
        txt = str(row["text"])
        if rid in id_to_label:
            eid = id_to_label[rid]
        elif has_inline_emotion:
            eid = int(row["emotion_id"])
        else:
            raise KeyError(
                f"Sample id={rid} in {path1.name} not found in clean VSMEC files at {clean_path}."
            )
        emotion_ids.append(eid)
        texts.append(txt)

    key_df = pd.DataFrame(
        {
            "id": ids1,
            "text": texts,
            "emotion_id": emotion_ids,
            "emotion_name": [ID_TO_EMOTION7[int(eid)] for eid in emotion_ids],
        }
    )
    write_csv_schema(key_df, key_path, POLARITY_KEY_COLUMNS)
    return key_df


def sample_polarity_sentences(
    clean_dir: Union[str, Path] = "data/clean",
    out_dir: Optional[Union[str, Path]] = None,
    n_per_emotion: int = 50,
    seed: int = 42,
    annotator_filenames: Tuple[str, str] = ("polarity_nhat.csv", "polarity_long.csv"),
    key_filename: str = "polarity_key.csv",
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Sample `n_per_emotion` Other and `n_per_emotion` Surprise sentences from clean VSMEC.

    Writes two identical CSV files with columns `id, text, polarity_label` (empty
    `polarity_label`, no emotion columns) for independent annotation, plus
    `polarity_key.csv` (`id, text, emotion_id, emotion_name`) in matching `id` order,
    and updates `MANIFEST.json`.

    If `annotator_filenames` already exist in `out_dir` and `overwrite=False`,
    does NOT overwrite them and only exports `polarity_key.csv` joined by `id`.
    """
    clean_path = Path(clean_dir)
    dest_dir = Path(out_dir) if out_dir is not None else clean_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    train_csv = clean_path / "vsmec_train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Clean VSMEC file not found: {train_csv}. Run clean_vsmec.py first."
        )

    path1 = dest_dir / annotator_filenames[0]
    path2 = dest_dir / annotator_filenames[1]
    key_path = dest_dir / key_filename

    if not overwrite and (path1.is_file() or path2.is_file()):
        if path1.is_file() != path2.is_file():
            raise FileExistsError(
                f"Inconsistent existing annotator files in {dest_dir}: "
                f"{path1.name} exists={path1.is_file()}, {path2.name} exists={path2.is_file()}."
            )
        key_df = _export_key_from_existing_annotator_files(
            clean_path=clean_path,
            path1=path1,
            path2=path2,
            key_path=key_path,
        )
        update_clean_manifest(dest_dir)
        print(
            f"=== NOTICE: {path1.name} and {path2.name} already exist in {dest_dir}. "
            f"Skipped overwrite and exported only {key_path.name} ({len(key_df)} rows joined by id) ==="
        )
        print(
            f"=== Valid polarity_label values: {NEG} = {ID_TO_LABEL3[NEG]}, "
            f"{NEU} = {ID_TO_LABEL3[NEU]}, {POS} = {ID_TO_LABEL3[POS]} ==="
        )
        return {"annotator1": path1, "annotator2": path2, "key": key_path}

    df_train = read_csv_schema(
        train_csv, SPLIT_COLUMNS, strict_order=False, keep_default_na=False
    )
    seen_ids: Set[int] = set()
    seen_texts: Set[str] = set()
    df_pool = filter_valid_unique_pool(
        df_train, seen_ids=seen_ids, seen_texts=seen_texts
    )

    used_extra_splits: List[str] = []
    for extra_sp in ("dev", "test"):
        need_more = any(
            int((df_pool["label"] == emo).sum()) < n_per_emotion
            for emo in (EMO_OTHER, EMO_SURPRISE)
        )
        if need_more:
            extra_csv = clean_path / f"vsmec_{extra_sp}.csv"
            if extra_csv.is_file():
                df_extra = read_csv_schema(
                    extra_csv, SPLIT_COLUMNS, strict_order=False, keep_default_na=False
                )
                df_extra_clean = filter_valid_unique_pool(
                    df_extra, seen_ids=seen_ids, seen_texts=seen_texts
                )
                df_pool = pd.concat([df_pool, df_extra_clean], ignore_index=True)
                used_extra_splits.append(extra_sp)

    set_seed(seed)
    rng = np.random.default_rng(seed)

    sampled_dfs: List[pd.DataFrame] = []
    for emo_id in (EMO_OTHER, EMO_SURPRISE):
        sub = (
            df_pool[df_pool["label"] == emo_id]
            .sort_values("id", kind="mergesort")
            .reset_index(drop=True)
        )
        if len(sub) < n_per_emotion:
            raise ValueError(
                f"Not enough samples for emotion {ID_TO_EMOTION7[emo_id]} ({emo_id}): "
                f"needed {n_per_emotion}, found {len(sub)}"
            )
        chosen_idx = rng.choice(len(sub), size=n_per_emotion, replace=False)
        chosen_idx.sort()
        picked = sub.iloc[chosen_idx].copy()
        picked["emotion_id"] = picked["label"].astype(int)
        picked["emotion_name"] = [ID_TO_EMOTION7[int(x)] for x in picked["label"]]
        picked["polarity_label"] = ""
        sampled_dfs.append(picked)

    combined = pd.concat(sampled_dfs, ignore_index=True)
    perm = rng.permutation(len(combined))
    combined = combined.iloc[perm].reset_index(drop=True)

    annotator_df = combined.loc[:, list(POLARITY_SAMPLE_COLUMNS)]
    key_df = combined.loc[:, list(POLARITY_KEY_COLUMNS)]

    write_csv_schema(annotator_df, path1, POLARITY_SAMPLE_COLUMNS)
    write_csv_schema(annotator_df, path2, POLARITY_SAMPLE_COLUMNS)
    write_csv_schema(key_df, key_path, POLARITY_KEY_COLUMNS)

    update_clean_manifest(dest_dir)

    extra_note = (
        f" (supplemented from: {', '.join(used_extra_splits)})"
        if used_extra_splits
        else " (100% from train)"
    )
    print(
        f"=== Sampled {len(combined)} VSMEC sentences ({n_per_emotion} Other, {n_per_emotion} Surprise){extra_note} "
        f"-> {path1.name}, {path2.name}, {key_path.name} ==="
    )
    print(
        f"=== Valid polarity_label values: {NEG} = {ID_TO_LABEL3[NEG]}, "
        f"{NEU} = {ID_TO_LABEL3[NEU]}, {POS} = {ID_TO_LABEL3[POS]} ==="
    )
    return {"annotator1": path1, "annotator2": path2, "key": key_path}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample 100 clean VSMEC sentences (50 Other, 50 Surprise) for polarity labeling. "
            "Valid polarity_label values: 0 = negative, 1 = neutral, 2 = positive."
        )
    )
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--n-per-emotion", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing annotator files if present (default: False).",
    )
    args = parser.parse_args(argv)

    sample_polarity_sentences(
        clean_dir=args.clean_dir,
        out_dir=args.out_dir,
        n_per_emotion=args.n_per_emotion,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
