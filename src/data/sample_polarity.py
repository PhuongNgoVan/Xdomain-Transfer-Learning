"""Sample 100 clean VSMEC sentences (50 Other, 50 Surprise) for independent polarity labeling.

Per `tasks/M1_data.md` & `ke_hoach_2_v2.md` (A1 CV-N4, A2 CV-L3):
- Samples 50 `Other` and 50 `Surprise` sentences from clean VSMEC (`vsmec_train.csv`,
  falling back to all clean VSMEC splits if fewer than 50 exist in train) with a fixed seed.
- Exports two identical CSV files with an empty polarity column (`polarity_label`)
  for two independent annotators (`polarity_annotator1.csv` / `polarity_nhat.csv` and
  `polarity_annotator2.csv` / `polarity_long.csv`).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.common.constants import EMO_OTHER, EMO_SURPRISE, ID_TO_EMOTION7
from src.common.io import read_csv_schema, write_csv_schema
from src.common.schema import SPLIT_COLUMNS
from src.common.seed import set_seed

POLARITY_SAMPLE_COLUMNS: Tuple[str, ...] = (
    "id",
    "text",
    "emotion_id",
    "emotion_name",
    "polarity_label",
)


def sample_polarity_sentences(
    clean_dir: Union[str, Path] = "data/clean",
    out_dir: Optional[Union[str, Path]] = None,
    n_per_emotion: int = 50,
    seed: int = 42,
    annotator_filenames: Tuple[str, str] = ("polarity_nhat.csv", "polarity_long.csv"),
) -> Dict[str, Path]:
    """Sample `n_per_emotion` Other and `n_per_emotion` Surprise sentences from clean VSMEC.

    Writes two CSV files with an empty `polarity_label` column for independent annotation.
    """
    clean_path = Path(clean_dir)
    dest_dir = Path(out_dir) if out_dir is not None else clean_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    train_csv = clean_path / "vsmec_train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Clean VSMEC file not found: {train_csv}. Run clean_vsmec.py first."
        )

    df_pool = read_csv_schema(train_csv, SPLIT_COLUMNS, strict_order=False)
    # If train alone has fewer than n_per_emotion in small test fixtures, include dev/test
    for extra_sp in ("dev", "test"):
        extra_csv = clean_path / f"vsmec_{extra_sp}.csv"
        if extra_csv.is_file():
            need_more = any(
                (df_pool["label"] == emo).sum() < n_per_emotion
                for emo in (EMO_OTHER, EMO_SURPRISE)
            )
            if need_more:
                df_extra = read_csv_schema(extra_csv, SPLIT_COLUMNS, strict_order=False)
                df_pool = pd.concat([df_pool, df_extra], ignore_index=True)

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
        sampled_dfs.append(picked.loc[:, list(POLARITY_SAMPLE_COLUMNS)])

    combined = pd.concat(sampled_dfs, ignore_index=True)
    # Shuffle deterministically so annotators don't see all Other then all Surprise in blocks,
    # or keep deterministic order by seed:
    perm = rng.permutation(len(combined))
    combined = combined.iloc[perm].reset_index(drop=True)

    path1 = dest_dir / annotator_filenames[0]
    path2 = dest_dir / annotator_filenames[1]
    write_csv_schema(combined, path1, POLARITY_SAMPLE_COLUMNS)
    write_csv_schema(combined, path2, POLARITY_SAMPLE_COLUMNS)

    print(
        f"=== Sampled {len(combined)} VSMEC sentences ({n_per_emotion} Other, {n_per_emotion} Surprise) "
        f"-> {path1.name}, {path2.name} ==="
    )
    return {"annotator1": path1, "annotator2": path2}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sample 100 clean VSMEC sentences (50 Other, 50 Surprise) for polarity labeling."
    )
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--n-per-emotion", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    sample_polarity_sentences(
        clean_dir=args.clean_dir,
        out_dir=args.out_dir,
        n_per_emotion=args.n_per_emotion,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
