"""Tạo mẫu phiếu gán nhãn tay cho tập phụ VSMEC-strict (`scripts/make_strict_template.py`).

Đề xuất lấy mẫu ngẫu nhiên (với seed cố định) các câu thuộc lớp `Surprise` (5) và `Other` (6)
trong `data/clean/vsmec_test.csv` để 2 người gán độc lập xác định nhãn 3 lớp (0=neg, 1=neu, 2=pos).
Script KHÔNG ghi bất kỳ file nào ra đĩa trừ khi truyền cờ `--generate`.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import numpy as np
import pandas as pd

TARGET_EMOTION_IDS = (5, 6)  # Surprise (5), Other (6)
TEMPLATE_COLUMNS = ("id", "text", "annotator_1", "annotator_2")


def build_strict_template_df(
    vsmec_test_df: pd.DataFrame,
    n_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample up to `n_samples` rows from `Surprise` (5) and `Other` (6) in `vsmec_test_df`."""
    required = {"id", "text", "label"}
    missing = sorted(required - set(vsmec_test_df.columns))
    if missing:
        raise ValueError(f"vsmec_test DataFrame thiếu cột: {missing}")

    pool = vsmec_test_df[vsmec_test_df["label"].astype(int).isin(TARGET_EMOTION_IDS)].copy()
    pool = pool.reset_index(drop=True)
    if len(pool) == 0:
        return pd.DataFrame(columns=list(TEMPLATE_COLUMNS))

    k = min(int(n_samples), len(pool))
    if k <= 0:
        return pd.DataFrame(columns=list(TEMPLATE_COLUMNS))

    rng = np.random.default_rng(int(seed))
    chosen_indices = np.sort(rng.choice(len(pool), size=k, replace=False))
    sampled = pool.iloc[chosen_indices].copy()

    return pd.DataFrame(
        {
            "id": sampled["id"].astype(int).tolist(),
            "text": sampled["text"].astype(str).tolist(),
            "annotator_1": [""] * len(sampled),
            "annotator_2": [""] * len(sampled),
        },
        columns=list(TEMPLATE_COLUMNS),
    )


def generate_strict_template(
    vsmec_test_path: Path = Path("data/clean/vsmec_test.csv"),
    output_path: Path = Path("reports/step2/strict_annotation_template.csv"),
    n_samples: int = 100,
    seed: int = 42,
    generate: bool = False,
) -> Optional[pd.DataFrame]:
    """Build strict annotation template DataFrame and write CSV only when `generate=True`."""
    df_test = pd.read_csv(vsmec_test_path, encoding="utf-8")
    pool_size = int(df_test["label"].astype(int).isin(TARGET_EMOTION_IDS).sum())
    df_template = build_strict_template_df(df_test, n_samples=n_samples, seed=seed)

    if not generate:
        print(
            f"[DRY-RUN] Tìm thấy {pool_size} câu Surprise/Other trong {vsmec_test_path.as_posix()}. "
            f"Sẵn sàng xuất mẫu {len(df_template)} câu (seed={seed}) ra {output_path.as_posix()} "
            f"khi truyền cờ `--generate`."
        )
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_template.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    print(
        f"[GENERATED] Đã ghi {len(df_template)} câu Surprise/Other (seed={seed}) "
        f"ra {output_path.as_posix()}."
    )
    return df_template


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tạo mẫu phiếu gán nhãn VSMEC-strict (chỉ ghi file khi có --generate)."
    )
    parser.add_argument(
        "--vsmec-test",
        type=Path,
        default=Path("data/clean/vsmec_test.csv"),
        help="Đường dẫn tới file vsmec_test.csv (mặc định: data/clean/vsmec_test.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/step2/strict_annotation_template.csv"),
        help="Đường dẫn xuất file mẫu (mặc định: reports/step2/strict_annotation_template.csv).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Số câu Surprise/Other cần lấy mẫu (mặc định: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed lấy mẫu ngẫu nhiên (mặc định: 42).",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Thực sự ghi file CSV mẫu ra đĩa (mặc định chỉ chạy thử không ghi file).",
    )
    args = parser.parse_args(argv)

    generate_strict_template(
        vsmec_test_path=args.vsmec_test,
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        generate=args.generate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
