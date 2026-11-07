"""Sample 100 clean VSMEC sentences (50 Other, 50 Surprise) for independent polarity labeling.

Per `tasks/M1b_polarity_sample.md`:
- Samples 50 `Other` and 50 `Surprise` sentences from clean VSMEC (`vsmec_train.csv`,
  falling back to `vsmec_dev.csv` then `vsmec_test.csv` if fewer than 50 valid unique
  sentences exist in train) with a fixed seed (`42`).
- Exports two identical blind CSV files (`polarity_nhat.csv`, `polarity_long.csv`) with
  columns `id, text, polarity_label` (no original emotion columns) and a separate key
  file `polarity_key.csv` (`id, text, emotion_id, emotion_name`).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.common.constants import EMO_OTHER, EMO_SURPRISE, ID_TO_EMOTION7
from src.common.hashing import build_manifest, hash_text
from src.common.io import read_csv_schema, write_csv_schema
from src.common.schema import (
    POLARITY_KEY_COLUMNS,
    POLARITY_SAMPLE_COLUMNS,
    SPLIT_COLUMNS,
)
from src.common.seed import set_seed


def is_valid_polarity_text(text: Any) -> bool:
    """Return True if `text` is a non-empty string containing at least one alphanumeric char."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    return any(ch.isalnum() for ch in stripped)


def filter_valid_unique_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Filter valid sentences, sort by `id` ascending, and deduplicate by normalized text."""
    if df.empty:
        return df.copy()
    valid_mask = df["text"].map(is_valid_polarity_text)
    sub = df.loc[valid_mask].sort_values("id", kind="mergesort").copy()
    seen_hashes: Set[str] = set()
    keep_indices: List[int] = []
    for idx, text_val in zip(sub.index, sub["text"]):
        h = hash_text(str(text_val), normalize=True)
        if h not in seen_hashes:
            seen_hashes.add(h)
            keep_indices.append(idx)
    return sub.loc[keep_indices].reset_index(drop=True)


def update_clean_manifest(clean_dir: Union[str, Path]) -> Dict[str, str]:
    """Compute SHA-256 manifest for all CSV files in `clean_dir` and write `MANIFEST.json`."""
    clean_path = Path(clean_dir)
    csv_files = sorted(clean_path.glob("*.csv"))
    manifest = build_manifest(csv_files, base_dir=clean_path)
    manifest_path = clean_path / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_vsmec_pool(
    clean_path: Path,
    n_per_emotion: int,
) -> Tuple[pd.DataFrame, bool]:
    """Load clean VSMEC pool from train, replenishing from dev/test if needed."""
    train_csv = clean_path / "vsmec_train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Clean VSMEC file not found: {train_csv}. Run clean_vsmec.py first."
        )

    df_train = read_csv_schema(train_csv, SPLIT_COLUMNS, strict_order=False)
    df_pool = filter_valid_unique_pool(df_train)
    replenished = False

    for extra_sp in ("dev", "test"):
        need_more = any(
            int((df_pool["label"] == emo).sum()) < n_per_emotion
            for emo in (EMO_OTHER, EMO_SURPRISE)
        )
        if not need_more:
            break
        extra_csv = clean_path / f"vsmec_{extra_sp}.csv"
        if extra_csv.is_file():
            df_extra = read_csv_schema(extra_csv, SPLIT_COLUMNS, strict_order=False)
            df_extra_valid = filter_valid_unique_pool(df_extra)
            existing_ids = set(df_pool["id"].tolist())
            existing_hashes = {
                hash_text(str(t), normalize=True) for t in df_pool["text"]
            }
            new_rows = []
            for _, row in df_extra_valid.iterrows():
                h = hash_text(str(row["text"]), normalize=True)
                if row["id"] not in existing_ids and h not in existing_hashes:
                    existing_ids.add(row["id"])
                    existing_hashes.add(h)
                    new_rows.append(row)
            if new_rows:
                df_pool = pd.concat(
                    [df_pool, pd.DataFrame(new_rows)], ignore_index=True
                )
                replenished = True

    return df_pool, replenished


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

    Writes two blind CSV files (`POLARITY_SAMPLE_COLUMNS`) and one key CSV file
    (`POLARITY_KEY_COLUMNS`). If annotator files already exist and `overwrite=False`,
    preserves them and only exports `polarity_key.csv` matching their IDs.
    """
    clean_path = Path(clean_dir)
    dest_dir = Path(out_dir) if out_dir is not None else clean_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    path1 = dest_dir / annotator_filenames[0]
    path2 = dest_dir / annotator_filenames[1]
    key_path = dest_dir / key_filename

    df_pool, _ = _load_vsmec_pool(clean_path, n_per_emotion=n_per_emotion)

    if (path1.is_file() or path2.is_file()) and not overwrite:
        ref_path = path1 if path1.is_file() else path2
        df_existing = pd.read_csv(ref_path, keep_default_na=False)
        id_to_row = {int(r["id"]): r for _, r in df_pool.iterrows()}
        key_rows = []
        for _, r in df_existing.iterrows():
            rid = int(r["id"])
            if rid in id_to_row:
                emo_id = int(id_to_row[rid]["label"])
                emo_name = ID_TO_EMOTION7[emo_id]
            elif "emotion_id" in df_existing.columns and "emotion_name" in df_existing.columns:
                emo_id = int(r["emotion_id"])
                emo_name = str(r["emotion_name"])
            else:
                raise ValueError(f"Sentence id={rid} in {ref_path} not found in VSMEC pool.")
            key_rows.append(
                {
                    "id": rid,
                    "text": str(r["text"]),
                    "emotion_id": emo_id,
                    "emotion_name": emo_name,
                }
            )
        key_df = pd.DataFrame(key_rows)
        write_csv_schema(key_df, key_path, POLARITY_KEY_COLUMNS)
        update_clean_manifest(dest_dir)
        return {"annotator1": path1, "annotator2": path2, "key": key_path}

    set_seed(seed)
    rng = np.random.default_rng(seed)

    sampled_dfs: List[pd.DataFrame] = []
    for emo_id in (EMO_OTHER, EMO_SURPRISE):
        sub = df_pool[df_pool["label"] == emo_id].sort_values("id", kind="mergesort")
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

    sample_df = combined.loc[:, list(POLARITY_SAMPLE_COLUMNS)]
    key_df = combined.loc[:, list(POLARITY_KEY_COLUMNS)]

    write_csv_schema(sample_df, path1, POLARITY_SAMPLE_COLUMNS)
    write_csv_schema(sample_df, path2, POLARITY_SAMPLE_COLUMNS)
    write_csv_schema(key_df, key_path, POLARITY_KEY_COLUMNS)
    update_clean_manifest(dest_dir)

    print(
        f"=== Sampled {len(combined)} VSMEC sentences ({n_per_emotion} Other, {n_per_emotion} Surprise) "
        f"-> {path1.name}, {path2.name}, {key_path.name} ==="
    )
    return {"annotator1": path1, "annotator2": path2, "key": key_path}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sample 100 clean VSMEC sentences (50 Other, 50 Surprise) for polarity labeling."
    )
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--n-per-emotion", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
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
