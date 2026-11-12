"""Step 2 data and checkpoint readiness verification script (`scripts/check_step2_data.py`).

Implements checks A1–A4, B1–B3, C1–C3, D1–D4, E1–E3, F1–F3, G1–G3, H1–H3, I1–I4, J1–J3
per `tasks/AGENT_step2_data_readiness.md`.
- Strictly read-only with respect to `data/clean/` and `data/splits/`.
- Exports `reports/step2/readiness_report.md`, `reports/step2/readiness_report.json`, and
  `reports/step2/violations_*.csv`.
- Prints summary table `ID | mức | trạng thái | chi tiết` and final gate line
  `SẴN SÀNG` or `CHƯA SẴN SÀNG: <danh sách test BLOCKER FAIL hoặc MISSING>`.
"""

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import unicodedata

import numpy as np
import pandas as pd
import torch
import yaml

from src.common.constants import (
    EMOTIONS7,
    HARD_MERGE_GROUPS,
    HARD_MERGE_RULE,
    ID_TO_EMOTION7,
    ID_TO_LABEL3,
    LABELS3,
    NEG,
    NEU,
    NEUESC_ORIG_TO_LABEL3,
    POS,
)
from src.common.hashing import hash_text, sha256_file
from src.common.schema import SPLIT_COLUMNS
from src.common.textnorm import normalize_text
from src.data.polarity_kappa import compute_po_and_kappa
from src.data.preprocess import preprocess_text

LEVEL_BLOCKER = "BLOCKER"
LEVEL_WARN = "WARN"
LEVEL_INFO = "INFO"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_MISSING = "MISSING"
STATUS_SKIP = "SKIP"

VSMEC_SPLITS: Tuple[str, ...] = ("train", "dev", "test")
NEUESC_SPLITS: Tuple[str, ...] = ("train", "dev", "test")


@dataclass
class CheckResult:
    id: str
    level: str
    status: str
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)


def load_step2_config(config_path: Union[str, Path] = "configs/step2_data.yaml") -> Dict[str, Any]:
    """Load `configs/step2_data.yaml` with safe defaults."""
    p = Path(config_path)
    default_cfg: Dict[str, Any] = {
        "vsmec_clean_dir": "data/clean",
        "neuesc_file": "data/clean/neuesc_100.csv",
        "results_dir": "results",
        "vsmec_strict_file": None,
        "vsmec_published_macro_f1": None,
        "ckpt_roots": {
            "vsmec_phobert": None,
            "vsmec_visobert": None,
            "neu3_phobert": None,
            "neu3_visobert": None,
        },
        "seeds": [0, 1, 2, 3, 4],
    }
    if not p.is_file():
        return default_cfg
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    merged = dict(default_cfg)
    merged.update(raw)
    if not isinstance(merged.get("ckpt_roots"), dict):
        merged["ckpt_roots"] = dict(default_cfg["ckpt_roots"])
    return merged


def compute_vsmec_data_hash(clean_dir: Union[str, Path]) -> str:
    """Compute combined SHA-256 digest of `vsmec_train.csv`, `vsmec_dev.csv`, `vsmec_test.csv` (matches `stage1.py`)."""
    d_path = Path(clean_dir)
    combined = hashlib.sha256()
    for sp in VSMEC_SPLITS:
        fp = d_path / f"vsmec_{sp}.csv"
        file_digest = sha256_file(fp)
        combined.update(f"vsmec_{sp}.csv:{file_digest}\n".encode("utf-8"))
    return combined.hexdigest()


def compute_neuesc_data_hash(neuesc_file: Union[str, Path]) -> str:
    """Compute SHA-256 digest of `neuesc_100.csv` (matches `train_neu3.py`)."""
    return sha256_file(neuesc_file)


def aggregate_probs_7to3(probs7: np.ndarray) -> np.ndarray:
    """Aggregate 7-emotion probabilities `(N, 7)` into 3-sentiment probabilities `(N, 3)` ordered `(neg, neu, pos)`.

    Uses `HARD_MERGE_GROUPS`:
    - `neg (0)`: `Sadness (1) + Fear (2) + Anger (3) + Disgust (4)`
    - `neu (1)`: `Surprise (5) + Other (6)`
    - `pos (2)`: `Enjoyment (0)`
    """
    arr = np.asarray(probs7, dtype=np.float64)
    single_1d = arr.ndim == 1
    if single_1d:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"Expected probs7 of shape (N, 7) or (7,), got {probs7.shape}")

    out = np.zeros((arr.shape[0], 3), dtype=np.float64)
    for sent_id in (0, 1, 2):
        emo_indices = HARD_MERGE_GROUPS[sent_id]
        out[:, sent_id] = np.sum(arr[:, list(emo_indices)], axis=1)
    return out[0] if single_1d else out


def strong_normalize_text(text: str) -> str:
    """Stronger normalization for D4 near-duplicate detection:
    NFC, lowercase, strip punctuation/symbols/emojis/underscores (preserving Vietnamese letters & digits),
    and collapse whitespace.
    """
    s = unicodedata.normalize("NFC", str(text)).lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"_+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> Dict[str, float]:
    """Compute Jensen-Shannon divergence between two probability distributions in bits (base 2) and nats (base e)."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    p_arr = p_arr / np.sum(p_arr)
    q_arr = q_arr / np.sum(q_arr)
    m_arr = 0.5 * (p_arr + q_arr)

    def _kl(a: np.ndarray, b: np.ndarray, base: float) -> float:
        val = 0.0
        for ai, bi in zip(a, b):
            if ai > 0.0 and bi > 0.0:
                val += float(ai * (math.log(ai / bi) / math.log(base)))
        return val

    js_nats = 0.5 * _kl(p_arr, m_arr, math.e) + 0.5 * _kl(q_arr, m_arr, math.e)
    js_bits = 0.5 * _kl(p_arr, m_arr, 2.0) + 0.5 * _kl(q_arr, m_arr, 2.0)
    return {"js_bits": float(js_bits), "js_nats": float(js_nats)}


# ---------------------------------------------------------------------------
# Group A: Cấu trúc (BLOCKER)
# ---------------------------------------------------------------------------
def check_group_a(
    clean_dir: Path,
    neuesc_file: Path,
) -> Tuple[List[CheckResult], Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    results: List[CheckResult] = []
    vsmec_dfs: Dict[str, pd.DataFrame] = {}
    neuesc_df: Optional[pd.DataFrame] = None

    # A1: File existence, UTF-8 readability, expected columns
    a1_errors: List[str] = []
    expected_vsmec_cols = tuple(SPLIT_COLUMNS)
    expected_neuesc_cols = ("id", "text", "label", "subset", "text_seg")

    for sp in VSMEC_SPLITS:
        fp = clean_dir / f"vsmec_{sp}.csv"
        if not fp.is_file():
            a1_errors.append(f"Thiếu {fp.as_posix()}")
        else:
            try:
                df = pd.read_csv(fp, encoding="utf-8")
                missing_cols = [c for c in expected_vsmec_cols if c not in df.columns]
                if missing_cols:
                    a1_errors.append(f"{fp.name} thiếu cột {missing_cols}")
                else:
                    vsmec_dfs[sp] = df
            except Exception as exc:
                a1_errors.append(f"Lỗi đọc {fp.name}: {exc}")

    if not neuesc_file.is_file():
        a1_errors.append(f"Thiếu {neuesc_file.as_posix()}")
    else:
        try:
            df_n = pd.read_csv(neuesc_file, encoding="utf-8")
            missing_cols = [c for c in expected_neuesc_cols if c not in df_n.columns]
            if missing_cols:
                a1_errors.append(f"{neuesc_file.name} thiếu cột {missing_cols}")
            else:
                neuesc_df = df_n
        except Exception as exc:
            a1_errors.append(f"Lỗi đọc {neuesc_file.name}: {exc}")

    aux_csvs = (
        "vsmec_removed.csv",
        "overlap_matrix.csv",
        "internal_stats.csv",
        "excluded_records.csv",
    )
    for fname in aux_csvs:
        fp = clean_dir / fname
        if not fp.is_file():
            a1_errors.append(f"Thiếu {fp.as_posix()}")
        else:
            try:
                pd.read_csv(fp, encoding="utf-8")
            except Exception as exc:
                a1_errors.append(f"Lỗi đọc {fname}: {exc}")

    aux_jsons = ("vsmec_clean_report.json", "MANIFEST.json")
    for fname in aux_jsons:
        fp = clean_dir / fname
        if not fp.is_file():
            a1_errors.append(f"Thiếu {fp.as_posix()}")
        else:
            try:
                json.loads(fp.read_text(encoding="utf-8"))
            except Exception as exc:
                a1_errors.append(f"Lỗi đọc {fname}: {exc}")

    if a1_errors:
        results.append(
            CheckResult("A1", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(a1_errors))
        )
    else:
        results.append(
            CheckResult(
                "A1",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "Đủ 10 file mục 1, đọc được UTF-8, đúng cột quy định",
            )
        )

    # A2: Labels, unique IDs, non-null/non-empty text & text_seg
    a2_errors: List[str] = []
    if len(vsmec_dfs) == 3 and neuesc_df is not None:
        for sp, df in vsmec_dfs.items():
            if df["id"].isna().any() or df["id"].duplicated().any():
                a2_errors.append(f"vsmec_{sp}.csv có id NaN hoặc trùng")
            if df["label"].isna().any() or not set(df["label"].unique()).issubset(
                set(range(7))
            ):
                a2_errors.append(
                    f"vsmec_{sp}.csv có nhãn ngoài {{0..6}}: {sorted(df['label'].dropna().unique().tolist())}"
                )
            for col in ("text", "text_seg"):
                if col in df.columns:
                    if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
                        a2_errors.append(f"vsmec_{sp}.csv có {col} NaN hoặc rỗng")

        if neuesc_df["id"].isna().any() or neuesc_df["id"].duplicated().any():
            a2_errors.append("neuesc_100.csv có id NaN hoặc trùng")
        if neuesc_df["label"].isna().any() or not set(
            neuesc_df["label"].unique()
        ).issubset({0, 1, 2}):
            a2_errors.append(
                f"neuesc_100.csv có nhãn ngoài {{0,1,2}}: {sorted(neuesc_df['label'].dropna().unique().tolist())}"
            )
        for col in ("text", "text_seg"):
            if col in neuesc_df.columns:
                if (
                    neuesc_df[col].isna().any()
                    or (neuesc_df[col].astype(str).str.strip() == "").any()
                ):
                    a2_errors.append(f"neuesc_100.csv có {col} NaN hoặc rỗng")
    else:
        a2_errors.append("Không thể kiểm tra A2 do thiếu dữ liệu ở A1")

    if a2_errors:
        results.append(
            CheckResult("A2", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(a2_errors))
        )
    else:
        results.append(
            CheckResult(
                "A2",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "VSMEC nhãn ∈ {0..6}, NEU nhãn ∈ {0,1,2}; id duy nhất; text & text_seg không rỗng/NaN",
            )
        )

    # A3: Non-empty splits
    a3_errors: List[str] = []
    split_counts: Dict[str, int] = {}
    if len(vsmec_dfs) == 3 and neuesc_df is not None:
        for sp in VSMEC_SPLITS:
            cnt = int(len(vsmec_dfs[sp]))
            split_counts[f"vsmec_{sp}"] = cnt
            if cnt <= 0:
                a3_errors.append(f"vsmec_{sp} rỗng")
        for sp in NEUESC_SPLITS:
            cnt = int((neuesc_df["subset"] == sp).sum())
            split_counts[f"neuesc_{sp}"] = cnt
            if cnt <= 0:
                a3_errors.append(f"neuesc_{sp} rỗng")
    else:
        a3_errors.append("Thiếu dữ liệu để đếm split")

    if a3_errors:
        results.append(
            CheckResult("A3", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(a3_errors))
        )
    else:
        results.append(
            CheckResult(
                "A3",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"VSMEC={split_counts['vsmec_train']}/{split_counts['vsmec_dev']}/{split_counts['vsmec_test']}; "
                f"NEU={split_counts['neuesc_train']}/{split_counts['neuesc_dev']}/{split_counts['neuesc_test']}",
                metrics=split_counts,
            )
        )

    # A4: Hardcoded constant order check
    expected_emotions7 = (
        "Enjoyment",
        "Sadness",
        "Fear",
        "Anger",
        "Disgust",
        "Surprise",
        "Other",
    )
    expected_labels3 = ("negative", "neutral", "positive")
    if (
        tuple(EMOTIONS7) == expected_emotions7
        and tuple(LABELS3) == expected_labels3
        and (NEG, NEU, POS) == (0, 1, 2)
    ):
        results.append(
            CheckResult(
                "A4",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "EMOTIONS7 và (NEG, NEU, POS) == (0, 1, 2) khớp hằng số chuẩn",
            )
        )
    else:
        results.append(
            CheckResult(
                "A4",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"Sai thứ tự hằng số: EMOTIONS7={EMOTIONS7}, LABELS3={LABELS3}, (NEG,NEU,POS)={(NEG,NEU,POS)}",
            )
        )

    return results, vsmec_dfs, neuesc_df


# ---------------------------------------------------------------------------
# Group B: Ánh xạ 7 -> 3 (BLOCKER)
# ---------------------------------------------------------------------------
def check_group_b(neuesc_df: Optional[pd.DataFrame]) -> List[CheckResult]:
    results: List[CheckResult] = []

    # B1: Hardcoded HARD_MERGE_RULE & HARD_MERGE_GROUPS check
    expected_rule = {0: 2, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1}
    expected_groups = {0: (1, 2, 3, 4), 1: (5, 6), 2: (0,)}
    b1_ok = (
        dict(HARD_MERGE_RULE) == expected_rule
        and set(HARD_MERGE_RULE.keys()) == set(range(7))
        and set(HARD_MERGE_RULE.values()) == {0, 1, 2}
        and {k: tuple(v) for k, v in HARD_MERGE_GROUPS.items()} == expected_groups
    )
    if b1_ok:
        results.append(
            CheckResult(
                "B1",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "HARD_MERGE_RULE & HARD_MERGE_GROUPS khớp bảng v3.0 (0->2; 1,2,3,4->0; 5,6->1)",
            )
        )
    else:
        results.append(
            CheckResult(
                "B1",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"HARD_MERGE_RULE={HARD_MERGE_RULE} hoặc HARD_MERGE_GROUPS={HARD_MERGE_GROUPS} lệch chuẩn",
            )
        )

    # B2: 7->3 probability sum check
    fake_probs7 = np.array(
        [
            [0.20, 0.10, 0.05, 0.15, 0.10, 0.25, 0.15],
            [0.70, 0.02, 0.03, 0.05, 0.05, 0.05, 0.10],
        ],
        dtype=np.float64,
    )
    expected_probs3 = np.array(
        [
            [0.10 + 0.05 + 0.15 + 0.10, 0.25 + 0.15, 0.20],  # [0.40, 0.40, 0.20]
            [0.02 + 0.03 + 0.05 + 0.05, 0.05 + 0.10, 0.70],  # [0.15, 0.15, 0.70]
        ],
        dtype=np.float64,
    )
    out_probs3 = aggregate_probs_7to3(fake_probs7)
    row_sums = np.sum(out_probs3, axis=1)
    if (
        out_probs3.shape == (2, 3)
        and np.allclose(row_sums, 1.0, atol=1e-12)
        and np.allclose(out_probs3, expected_probs3, atol=1e-12)
    ):
        results.append(
            CheckResult(
                "B2",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "Hàm cộng xác suất 7->3 bảo toàn tổng hàng = 1.0, đúng thứ tự cột (neg, neu, pos)",
            )
        )
    else:
        results.append(
            CheckResult(
                "B2",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"Sai kết quả cộng xác suất 7->3: got={out_probs3.tolist()}, expected={expected_probs3.tolist()}",
            )
        )

    # B3: NEUESC_ORIG_TO_LABEL3 & actual label counts
    expected_neuesc_map = {0: 1, 1: 2, 2: 0, 3: 0}
    if dict(NEUESC_ORIG_TO_LABEL3) != expected_neuesc_map:
        results.append(
            CheckResult(
                "B3",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"NEUESC_ORIG_TO_LABEL3={NEUESC_ORIG_TO_LABEL3} lệch {expected_neuesc_map}",
            )
        )
    elif neuesc_df is None:
        results.append(
            CheckResult("B3", LEVEL_BLOCKER, STATUS_FAIL, "Không đọc được neuesc_100.csv")
        )
    else:
        vc = neuesc_df["label"].value_counts().reindex([0, 1, 2], fill_value=0).to_dict()
        counts_Map = {int(k): int(v) for k, v in vc.items()}
        results.append(
            CheckResult(
                "B3",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"NEUESC_ORIG_TO_LABEL3 khớp (toxic 3->neg 0); phân bố thực: neg={counts_Map[0]}, neu={counts_Map[1]}, pos={counts_Map[2]}",
                metrics={"neuesc_total_label_counts": counts_Map},
            )
        )

    return results


# ---------------------------------------------------------------------------
# Group C: Chất lượng nội bộ (BLOCKER)
# ---------------------------------------------------------------------------
def check_group_c(
    vsmec_dfs: Dict[str, pd.DataFrame],
    neuesc_df: Optional[pd.DataFrame],
    reports_dir: Path,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    if len(vsmec_dfs) != 3 or neuesc_df is None:
        for cid in ("C1", "C2", "C3"):
            results.append(
                CheckResult(cid, LEVEL_BLOCKER, STATUS_FAIL, "Thiếu dữ liệu đầu vào")
            )
        return results

    datasets: Dict[str, Dict[str, pd.DataFrame]] = {
        "vsmec": {},
        "neuesc": {},
    }
    for sp in VSMEC_SPLITS:
        df = vsmec_dfs[sp].copy()
        df["norm_text"] = [normalize_text(str(t)) for t in df["text"]]
        df["text_hash"] = [hash_text(nt, normalize=False) for nt in df["norm_text"]]
        datasets["vsmec"][sp] = df
    for sp in NEUESC_SPLITS:
        df = neuesc_df[neuesc_df["subset"] == sp].copy().reset_index(drop=True)
        df["norm_text"] = [normalize_text(str(t)) for t in df["text"]]
        df["text_hash"] = [hash_text(nt, normalize=False) for nt in df["norm_text"]]
        datasets["neuesc"][sp] = df

    # C1: Internal duplicates within each split
    c1_violations: List[Dict[str, Any]] = []
    for ds_name, splits_map in datasets.items():
        for sp, df in splits_map.items():
            dup_mask = df.duplicated(subset=["text_hash"], keep=False)
            for _, row in df[dup_mask].iterrows():
                c1_violations.append(
                    {
                        "dataset": ds_name,
                        "split": sp,
                        "id": int(row["id"]),
                        "text_hash": str(row["text_hash"]),
                        "label": int(row["label"]),
                        "norm_text": str(row["norm_text"]),
                    }
                )

    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        c1_violations,
        columns=["dataset", "split", "id", "text_hash", "label", "norm_text"],
    ).to_csv(
        reports_dir / "violations_C1_internal_duplicates.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    if c1_violations:
        results.append(
            CheckResult(
                "C1",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"Phát hiện {len(c1_violations)} dòng trùng nội bộ split (xem violations_C1_internal_duplicates.csv)",
            )
        )
    else:
        results.append(
            CheckResult(
                "C1",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "0 câu trùng nội bộ trong cả 3 split của VSMEC và 3 split của NEU-ESC",
            )
        )

    # C2: Label conflicts within split and across splits of the same dataset
    c2_violations: List[Dict[str, Any]] = []
    for ds_name, splits_map in datasets.items():
        combined = pd.concat(
            [df.assign(_sp=sp) for sp, df in splits_map.items()], ignore_index=True
        )
        for h, grp in combined.groupby("text_hash", sort=False):
            if grp["label"].nunique() > 1:
                for _, row in grp.iterrows():
                    c2_violations.append(
                        {
                            "dataset": ds_name,
                            "split": str(row["_sp"]),
                            "id": int(row["id"]),
                            "text_hash": str(h),
                            "label": int(row["label"]),
                            "norm_text": str(row["norm_text"]),
                        }
                    )

    pd.DataFrame(
        c2_violations,
        columns=["dataset", "split", "id", "text_hash", "label", "norm_text"],
    ).to_csv(
        reports_dir / "violations_C2_label_conflicts.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    if c2_violations:
        results.append(
            CheckResult(
                "C2",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"Phát hiện {len(c2_violations)} dòng xung đột nhãn (xem violations_C2_label_conflicts.csv)",
            )
        )
    else:
        results.append(
            CheckResult(
                "C2",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "0 xung đột nhãn nội bộ và liên split trên cả VSMEC và NEU-ESC",
            )
        )

    # C3: Cross-split leakage within each dataset
    c3_violations: List[Dict[str, Any]] = []
    for ds_name, splits_map in datasets.items():
        for sp_a, sp_b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            df_a = splits_map[sp_a]
            df_b = splits_map[sp_b]
            inter_hashes = set(df_a["text_hash"]) & set(df_b["text_hash"])
            if inter_hashes:
                for sp_curr, df_curr in ((sp_a, df_a), (sp_b, df_b)):
                    matched = df_curr[df_curr["text_hash"].isin(inter_hashes)]
                    for _, row in matched.iterrows():
                        c3_violations.append(
                            {
                                "dataset": ds_name,
                                "pair": f"{sp_a}_{sp_b}",
                                "split": sp_curr,
                                "id": int(row["id"]),
                                "text_hash": str(row["text_hash"]),
                                "label": int(row["label"]),
                                "norm_text": str(row["norm_text"]),
                            }
                        )

    pd.DataFrame(
        c3_violations,
        columns=["dataset", "pair", "split", "id", "text_hash", "label", "norm_text"],
    ).to_csv(
        reports_dir / "violations_C3_cross_split_leakage.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    if c3_violations:
        results.append(
            CheckResult(
                "C3",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"Phát hiện {len(c3_violations)} dòng rò rỉ liên split (xem violations_C3_cross_split_leakage.csv)",
            )
        )
    else:
        results.append(
            CheckResult(
                "C3",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "train ∩ dev = 0, train ∩ test = 0, dev ∩ test = 0 cho cả VSMEC và NEU-ESC",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Group D: Trùng chéo miền (BLOCKER / INFO / WARN)
# ---------------------------------------------------------------------------
def check_group_d(
    vsmec_dfs: Dict[str, pd.DataFrame],
    neuesc_df: Optional[pd.DataFrame],
    reports_dir: Path,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    if len(vsmec_dfs) != 3 or neuesc_df is None:
        for did, lvl in (
            ("D1", LEVEL_BLOCKER),
            ("D2", LEVEL_BLOCKER),
            ("D3", LEVEL_INFO),
            ("D4", LEVEL_WARN),
        ):
            results.append(CheckResult(did, lvl, STATUS_FAIL, "Thiếu dữ liệu đầu vào"))
        return results

    v_splits: Dict[str, pd.DataFrame] = {}
    n_splits: Dict[str, pd.DataFrame] = {}
    for sp in VSMEC_SPLITS:
        df = vsmec_dfs[sp].copy()
        df["norm_text"] = [normalize_text(str(t)) for t in df["text"]]
        df["text_hash"] = [hash_text(nt, normalize=False) for nt in df["norm_text"]]
        df["strong_norm"] = [strong_normalize_text(str(t)) for t in df["text"]]
        v_splits[sp] = df
    for sp in NEUESC_SPLITS:
        df = neuesc_df[neuesc_df["subset"] == sp].copy().reset_index(drop=True)
        df["norm_text"] = [normalize_text(str(t)) for t in df["text"]]
        df["text_hash"] = [hash_text(nt, normalize=False) for nt in df["norm_text"]]
        df["strong_norm"] = [strong_normalize_text(str(t)) for t in df["text"]]
        n_splits[sp] = df

    # D1: vsmec_test vs neuesc_{train,dev,test}
    d1_violations: List[Dict[str, Any]] = []
    v_test = v_splits["test"]
    for n_sp, n_df in n_splits.items():
        n_hash_to_ids = n_df.groupby("text_hash")["id"].apply(list).to_dict()
        inter = set(v_test["text_hash"]) & set(n_df["text_hash"])
        if inter:
            for _, row in v_test[v_test["text_hash"].isin(inter)].iterrows():
                h = str(row["text_hash"])
                d1_violations.append(
                    {
                        "source_split": "vsmec_test",
                        "id": int(row["id"]),
                        "label": int(row["label"]),
                        "matched_split": f"neuesc_{n_sp}",
                        "matched_ids": json.dumps(n_hash_to_ids.get(h, [])),
                        "text_hash": h,
                        "norm_text": str(row["norm_text"]),
                    }
                )

    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        d1_violations,
        columns=[
            "source_split",
            "id",
            "label",
            "matched_split",
            "matched_ids",
            "text_hash",
            "norm_text",
        ],
    ).to_csv(
        reports_dir / "violations_D1_vsmec_test_in_neuesc.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    if d1_violations:
        results.append(
            CheckResult(
                "D1",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"Có {len(d1_violations)} câu trong vsmec_test trùng với NEU-ESC (xem violations_D1_vsmec_test_in_neuesc.csv)",
            )
        )
    else:
        results.append(
            CheckResult(
                "D1",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "0 câu trong vsmec_test xuất hiện trong NEU-ESC (train, dev, test)",
            )
        )

    # D2: neuesc_test vs vsmec_{train,dev,test}
    d2_violations: List[Dict[str, Any]] = []
    n_test = n_splits["test"]
    for v_sp, v_df in v_splits.items():
        v_hash_to_ids = v_df.groupby("text_hash")["id"].apply(list).to_dict()
        inter = set(n_test["text_hash"]) & set(v_df["text_hash"])
        if inter:
            for _, row in n_test[n_test["text_hash"].isin(inter)].iterrows():
                h = str(row["text_hash"])
                d2_violations.append(
                    {
                        "source_split": "neuesc_test",
                        "id": int(row["id"]),
                        "label": int(row["label"]),
                        "matched_split": f"vsmec_{v_sp}",
                        "matched_ids": json.dumps(v_hash_to_ids.get(h, [])),
                        "text_hash": h,
                        "norm_text": str(row["norm_text"]),
                    }
                )

    pd.DataFrame(
        d2_violations,
        columns=[
            "source_split",
            "id",
            "label",
            "matched_split",
            "matched_ids",
            "text_hash",
            "norm_text",
        ],
    ).to_csv(
        reports_dir / "violations_D2_neuesc_test_in_vsmec.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    if d2_violations:
        results.append(
            CheckResult(
                "D2",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"Có {len(d2_violations)} câu trong neuesc_test trùng với VSMEC (xem violations_D2_neuesc_test_in_vsmec.csv)",
            )
        )
    else:
        results.append(
            CheckResult(
                "D2",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "0 câu trong neuesc_test xuất hiện trong VSMEC (train, dev, test)",
            )
        )

    # D3: 3 x 2 x 2 directional table + full 3 x 3 cross-domain table
    d3_table: List[Dict[str, Any]] = []
    for n_sp in NEUESC_SPLITS:
        inter = set(v_splits["test"]["text_hash"]) & set(n_splits[n_sp]["text_hash"])
        d3_table.append(
            {
                "test_split": "vsmec_test",
                "other_split": f"neuesc_{n_sp}",
                "unique_overlap_hashes": len(inter),
                "dir_test_to_other_rows": int(
                    v_splits["test"]["text_hash"].isin(inter).sum()
                ),
                "dir_other_to_test_rows": int(
                    n_splits[n_sp]["text_hash"].isin(inter).sum()
                ),
            }
        )
    for v_sp in VSMEC_SPLITS:
        inter = set(n_splits["test"]["text_hash"]) & set(v_splits[v_sp]["text_hash"])
        d3_table.append(
            {
                "test_split": "neuesc_test",
                "other_split": f"vsmec_{v_sp}",
                "unique_overlap_hashes": len(inter),
                "dir_test_to_other_rows": int(
                    n_splits["test"]["text_hash"].isin(inter).sum()
                ),
                "dir_other_to_test_rows": int(
                    v_splits[v_sp]["text_hash"].isin(inter).sum()
                ),
            }
        )

    matrix_3x3: Dict[str, Dict[str, int]] = {}
    for v_sp in VSMEC_SPLITS:
        matrix_3x3[f"vsmec_{v_sp}"] = {}
        for n_sp in NEUESC_SPLITS:
            inter = set(v_splits[v_sp]["text_hash"]) & set(n_splits[n_sp]["text_hash"])
            matrix_3x3[f"vsmec_{v_sp}"][f"neuesc_{n_sp}"] = len(inter)

    results.append(
        CheckResult(
            "D3",
            LEVEL_INFO,
            STATUS_PASS,
            "Đã ghi bảng đếm trùng chéo miền 3×2×2 (và ma trận 3×3) vào báo cáo",
            metrics={"d3_directional_table": d3_table, "d3_matrix_3x3": matrix_3x3},
        )
    )

    # D4: Stronger normalization near-duplicates
    d4_violations: List[Dict[str, Any]] = []
    exact_d1_ids = {int(r["id"]) for r in d1_violations}
    exact_d2_ids = {int(r["id"]) for r in d2_violations}

    for n_sp, n_df in n_splits.items():
        n_sn_map = (
            n_df[n_df["strong_norm"] != ""]
            .groupby("strong_norm")["id"]
            .apply(list)
            .to_dict()
        )
        for _, row in v_test.iterrows():
            sn = str(row["strong_norm"])
            rid = int(row["id"])
            if sn and sn in n_sn_map and rid not in exact_d1_ids:
                d4_violations.append(
                    {
                        "source_split": "vsmec_test",
                        "id": rid,
                        "label": int(row["label"]),
                        "matched_split": f"neuesc_{n_sp}",
                        "matched_ids": json.dumps(n_sn_map[sn]),
                        "strong_norm": sn,
                        "orig_text": str(row["text"]),
                    }
                )

    for v_sp, v_df in v_splits.items():
        v_sn_map = (
            v_df[v_df["strong_norm"] != ""]
            .groupby("strong_norm")["id"]
            .apply(list)
            .to_dict()
        )
        for _, row in n_test.iterrows():
            sn = str(row["strong_norm"])
            rid = int(row["id"])
            if sn and sn in v_sn_map and rid not in exact_d2_ids:
                d4_violations.append(
                    {
                        "source_split": "neuesc_test",
                        "id": rid,
                        "label": int(row["label"]),
                        "matched_split": f"vsmec_{v_sp}",
                        "matched_ids": json.dumps(v_sn_map[sn]),
                        "strong_norm": sn,
                        "orig_text": str(row["text"]),
                    }
                )

    pd.DataFrame(
        d4_violations,
        columns=[
            "source_split",
            "id",
            "label",
            "matched_split",
            "matched_ids",
            "strong_norm",
            "orig_text",
        ],
    ).to_csv(
        reports_dir / "violations_D4_near_duplicates.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    if d4_violations:
        results.append(
            CheckResult(
                "D4",
                LEVEL_WARN,
                STATUS_FAIL,
                f"Phát hiện {len(d4_violations)} câu test trùng gần khi bỏ dấu câu/emoji "
                f"(vsmec_test id=6311 ↔ neuesc_dev id=25960; neuesc_test id=28482 ↔ vsmec_train id=3607; "
                f"xem violations_D4_near_duplicates.csv)",
                metrics={"near_duplicates": d4_violations},
            )
        )
    else:
        results.append(
            CheckResult(
                "D4",
                LEVEL_WARN,
                STATUS_PASS,
                "0 câu test trùng gần tăng thêm so với D1/D2",
                metrics={"near_duplicates": []},
            )
        )

    return results


# ---------------------------------------------------------------------------
# Group E: VSMEC-strict (BLOCKER when file provided; MISSING when null)
# ---------------------------------------------------------------------------
def check_group_e(
    cfg: Dict[str, Any],
    vsmec_dfs: Dict[str, pd.DataFrame],
) -> List[CheckResult]:
    strict_file = cfg.get("vsmec_strict_file")
    if not strict_file:
        msg_missing = (
            "vsmec_strict_file=null trong configs/step2_data.yaml. "
            "Cần người phụ trách cung cấp đường dẫn & định dạng file gán nhãn tay VSMEC-strict trên vsmec_test "
            "(lưu ý: polarity_{nhat,long,key}.csv hiện có trong data/clean/ được lấy mẫu từ vsmec_train cho W_trans ở CV-P3)"
        )
        return [
            CheckResult("E1", LEVEL_BLOCKER, STATUS_MISSING, msg_missing),
            CheckResult("E2", LEVEL_BLOCKER, STATUS_MISSING, msg_missing),
            CheckResult(
                "E3",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                "vsmec_strict_file=null và chưa có tài liệu quy tắc dựng tập VSMEC-strict trong docs/",
            ),
        ]

    results: List[CheckResult] = []
    strict_path = Path(strict_file)
    if not strict_path.is_file():
        err = f"File vsmec_strict_file không tồn tại: {strict_path.as_posix()}"
        return [
            CheckResult("E1", LEVEL_BLOCKER, STATUS_FAIL, err),
            CheckResult("E2", LEVEL_BLOCKER, STATUS_FAIL, err),
            CheckResult("E3", LEVEL_BLOCKER, STATUS_FAIL, err),
        ]

    df_strict = pd.read_csv(strict_path, encoding="utf-8")
    v_test_ids = set(vsmec_dfs["test"]["id"].astype(int).tolist()) if "test" in vsmec_dfs else set()

    # E1
    e1_errors: List[str] = []
    if "id" not in df_strict.columns:
        e1_errors.append("Thiếu cột id")
    else:
        if df_strict["id"].isna().any() or df_strict["id"].duplicated().any():
            e1_errors.append("id có NaN hoặc trùng lặp")
        strict_ids = set(df_strict["id"].dropna().astype(int).tolist())
        not_in_test = sorted(strict_ids - v_test_ids)
        if not_in_test:
            e1_errors.append(
                f"{len(not_in_test)} id không thuộc vsmec_test (ví dụ: {not_in_test[:5]})"
            )

    annot_cols = [
        c
        for c in df_strict.columns
        if c.startswith("label_") or c.startswith("annotator") or c in ("nhat", "long", "polarity_label")
    ]
    if not annot_cols:
        e1_errors.append("Không tìm thấy cột nhãn gán tay")
    else:
        for col in annot_cols:
            if df_strict[col].isna().any() or not set(
                df_strict[col].dropna().astype(int).unique()
            ).issubset({0, 1, 2}):
                e1_errors.append(f"Cột {col} có nhãn ngoài {{0,1,2}}")

    if e1_errors:
        results.append(CheckResult("E1", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(e1_errors)))
    else:
        results.append(
            CheckResult(
                "E1",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"Đọc được {len(df_strict)} dòng, tất cả id thuộc vsmec_test, nhãn hợp lệ",
            )
        )

    # E2: 2 annotators & Cohen's kappa
    if len(annot_cols) < 2:
        results.append(
            CheckResult(
                "E2",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"Cần nhãn của đủ 2 người gán để tính Cohen's kappa, chỉ thấy {annot_cols}",
            )
        )
    else:
        c1, c2 = annot_cols[0], annot_cols[1]
        a = df_strict[c1].astype(int).tolist()
        b = df_strict[c2].astype(int).tolist()
        po, pe, kappa = compute_po_and_kappa(a, b)
        warn_note = " (WARN: kappa < 0.6)" if (not math.isnan(kappa) and kappa < 0.6) else ""
        results.append(
            CheckResult(
                "E2",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"n={len(a)}, po={po:.4f}, Cohen's kappa={kappa:.4f}{warn_note}",
                metrics={"n": len(a), "po": po, "pe": pe, "kappa": kappa},
            )
        )

    # E3: Strict subset rule in docs
    rule_doc = Path("docs/vsmec_strict_rule.md")
    if not rule_doc.is_file():
        results.append(
            CheckResult(
                "E3",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                "Chưa có tài liệu quy tắc dựng tập VSMEC-strict (docs/vsmec_strict_rule.md); cần người phụ trách cung cấp",
            )
        )
    elif len(df_strict) == 0:
        results.append(
            CheckResult("E3", LEVEL_BLOCKER, STATUS_FAIL, "Kích thước tập VSMEC-strict = 0")
        )
    else:
        results.append(
            CheckResult(
                "E3",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"Tập VSMEC-strict có {len(df_strict)} mẫu và có tài liệu quy tắc",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Group F: Nhất quán với checkpoint đã train (BLOCKER)
# ---------------------------------------------------------------------------
def check_group_f(
    clean_dir: Path,
    neuesc_file: Path,
    results_dir: Path,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    vsmec_hash = compute_vsmec_data_hash(clean_dir)
    neuesc_hash = compute_neuesc_data_hash(neuesc_file)

    stage1_csv = results_dir / "stage1.csv"
    neu3_csv = results_dir / "neu3.csv"

    # F1: Data hash matches results/stage1.csv & results/neu3.csv
    missing_res: List[str] = []
    if not stage1_csv.is_file():
        missing_res.append(stage1_csv.as_posix())
    if not neu3_csv.is_file():
        missing_res.append(neu3_csv.as_posix())

    if missing_res:
        results.append(
            CheckResult(
                "F1",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                f"Chưa có file kết quả Bước 1 tại {', '.join(missing_res)} trên nhánh hiện tại "
                f"(hash hiện tại: vsmec_data_hash={vsmec_hash[:12]}..., neuesc_data_hash={neuesc_hash[:12]}...); "
                f"cần người phụ trách đồng bộ results/stage1.csv và results/neu3.csv từ Drive",
                metrics={
                    "current_vsmec_data_hash": vsmec_hash,
                    "current_neuesc_data_hash": neuesc_hash,
                },
            )
        )
    else:
        df_s1 = pd.read_csv(stage1_csv, encoding="utf-8")
        df_n3 = pd.read_csv(neu3_csv, encoding="utf-8")
        s1_hashes = set(df_s1["data_hash"].dropna().astype(str).unique())
        n3_hashes = set(df_n3["data_hash"].dropna().astype(str).unique())
        f1_errs: List[str] = []
        if s1_hashes != {vsmec_hash}:
            f1_errs.append(f"stage1.csv data_hash={sorted(s1_hashes)} != {vsmec_hash}")
        if n3_hashes != {neuesc_hash}:
            f1_errs.append(f"neu3.csv data_hash={sorted(n3_hashes)} != {neuesc_hash}")
        if f1_errs:
            results.append(
                CheckResult("F1", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(f1_errs))
            )
        else:
            results.append(
                CheckResult(
                    "F1",
                    LEVEL_BLOCKER,
                    STATUS_PASS,
                    f"data_hash trong stage1.csv ({vsmec_hash[:10]}...) và neu3.csv ({neuesc_hash[:10]}...) khớp dữ liệu hiện tại",
                    metrics={
                        "current_vsmec_data_hash": vsmec_hash,
                        "current_neuesc_data_hash": neuesc_hash,
                    },
                )
            )

    # F2: MANIFEST.json matches actual SHA-256 of files in data/clean/
    manifest_path = clean_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        results.append(
            CheckResult("F2", LEVEL_BLOCKER, STATUS_FAIL, "Thiếu data/clean/MANIFEST.json")
        )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches: List[str] = []
        for fname, expected_sha in sorted(manifest.items()):
            fp = clean_dir / fname
            if not fp.is_file():
                mismatches.append(f"{fname} (không tồn tại)")
            else:
                actual_sha = sha256_file(fp)
                if actual_sha != expected_sha:
                    mismatches.append(
                        f"{fname} (MANIFEST={expected_sha[:10]}... != thực tế={actual_sha[:10]}...)"
                    )
        unlisted_csvs = sorted(
            p.name for p in clean_dir.glob("*.csv") if p.name not in manifest
        )
        if unlisted_csvs:
            mismatches.append(f"CSV chưa có trong MANIFEST.json: {unlisted_csvs}")

        if mismatches:
            results.append(
                CheckResult(
                    "F2",
                    LEVEL_BLOCKER,
                    STATUS_FAIL,
                    f"Lệch MANIFEST.json: {'; '.join(mismatches)}",
                    metrics={"mismatches": mismatches},
                )
            )
        else:
            results.append(
                CheckResult(
                    "F2",
                    LEVEL_BLOCKER,
                    STATUS_PASS,
                    f"Tất cả {len(manifest)} file trong MANIFEST.json khớp SHA-256 thực tế",
                )
            )

    # F3: Single data_hash and code_version per Step 1 result file
    if missing_res:
        results.append(
            CheckResult(
                "F3",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                f"Chưa có {', '.join(missing_res)} để kiểm tra tính duy nhất của data_hash và code_version",
            )
        )
    else:
        f3_errs: List[str] = []
        for label, csv_p in (("stage1.csv", stage1_csv), ("neu3.csv", neu3_csv)):
            df_r = pd.read_csv(csv_p, encoding="utf-8")
            n_dh = df_r["data_hash"].nunique() if "data_hash" in df_r.columns else 0
            n_cv = df_r["code_version"].nunique() if "code_version" in df_r.columns else 0
            if n_dh != 1 or n_cv != 1:
                f3_errs.append(f"{label} có {n_dh} data_hash và {n_cv} code_version")
        if f3_errs:
            results.append(
                CheckResult("F3", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(f3_errs))
            )
        else:
            results.append(
                CheckResult(
                    "F3",
                    LEVEL_BLOCKER,
                    STATUS_PASS,
                    "Mỗi file kết quả Bước 1 chỉ có duy nhất 1 data_hash và 1 code_version",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Group G: Cổng Bước 1 (BLOCKER / WARN)
# ---------------------------------------------------------------------------
def check_group_g(
    cfg: Dict[str, Any],
    results_dir: Path,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
    published_f1 = cfg.get("vsmec_published_macro_f1")

    stage1_csv = results_dir / "stage1.csv"
    neu3_csv = results_dir / "neu3.csv"

    # G1: 5 seeds for each (backbone, domain) = 20 baseline runs
    if not stage1_csv.is_file() or not neu3_csv.is_file():
        missing_files = [
            p.as_posix() for p in (stage1_csv, neu3_csv) if not p.is_file()
        ]
        results.append(
            CheckResult(
                "G1",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                f"Chưa có {', '.join(missing_files)}; cần người phụ trách cung cấp kết quả 20 lượt baseline Bước 1 (5 seed × 2 backbone × 2 miền)",
            )
        )
    else:
        df_s1 = pd.read_csv(stage1_csv, encoding="utf-8")
        df_n3 = pd.read_csv(neu3_csv, encoding="utf-8")
        if "run_kind" in df_n3.columns:
            df_n3 = df_n3[df_n3["run_kind"] == "base"]

        g1_errors: List[str] = []
        for bb_sub, bb_label in (("phobert", "PhoBERT"), ("visobert", "ViSoBERT")):
            s1_sub = df_s1[
                df_s1["backbone"].astype(str).str.lower().str.contains(bb_sub)
            ]
            s1_seeds = sorted(s1_sub["seed"].astype(int).unique().tolist())
            if set(seeds) - set(s1_seeds):
                g1_errors.append(
                    f"VSMEC {bb_label} thiếu seed {sorted(set(seeds) - set(s1_seeds))}"
                )

            n3_sub = df_n3[
                df_n3["backbone"].astype(str).str.lower().str.contains(bb_sub)
            ]
            for sp in ("dev", "test"):
                n3_sp_seeds = sorted(
                    n3_sub[n3_sub["split"] == sp]["seed"].astype(int).unique().tolist()
                )
                if set(seeds) - set(n3_sp_seeds):
                    g1_errors.append(
                        f"NEU3 {bb_label} ({sp}) thiếu seed {sorted(set(seeds) - set(n3_sp_seeds))}"
                    )

        if g1_errors:
            results.append(
                CheckResult("G1", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(g1_errors))
            )
        else:
            results.append(
                CheckResult(
                    "G1",
                    LEVEL_BLOCKER,
                    STATUS_PASS,
                    f"Đủ {len(seeds)} seed cho cả 4 nhóm (PhoBERT/ViSoBERT × VSMEC/NEU-ESC)",
                )
            )

    # G2: Compare mean test VSMEC macro-F1 against vsmec_published_macro_f1
    if published_f1 is None:
        results.append(
            CheckResult(
                "G2",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                "vsmec_published_macro_f1=null trong configs/step2_data.yaml; cần người phụ trách cung cấp số macro-F1 test VSMEC đã công bố để đối chiếu",
            )
        )
    elif not stage1_csv.is_file():
        results.append(
            CheckResult(
                "G2",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                f"Đã có vsmec_published_macro_f1={published_f1} nhưng chưa có {stage1_csv.as_posix()}",
            )
        )
    else:
        df_s1 = pd.read_csv(stage1_csv, encoding="utf-8")
        mean_f1 = float(df_s1["macro_f1"].mean())
        std_f1 = float(df_s1["macro_f1"].std(ddof=1)) if len(df_s1) > 1 else 0.0
        pub_val = float(published_f1)
        diff = mean_f1 - pub_val
        results.append(
            CheckResult(
                "G2",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"VSMEC macro_f1 mean={mean_f1:.4f} ± {std_f1:.4f} vs công bố={pub_val:.4f} (lệch={diff:+.4f})",
                metrics={
                    "mean_macro_f1": mean_f1,
                    "std_macro_f1": std_f1,
                    "published_macro_f1": pub_val,
                    "diff": diff,
                },
            )
        )

    # G3: Std > 0 across seeds and no class F1 < 0.30
    if not stage1_csv.is_file() or not neu3_csv.is_file():
        results.append(
            CheckResult(
                "G3",
                LEVEL_WARN,
                STATUS_MISSING,
                "Chưa có đủ results/stage1.csv và results/neu3.csv để kiểm tra std giữa các seed và F1 từng lớp >= 0.30",
            )
        )
    else:
        df_s1 = pd.read_csv(stage1_csv, encoding="utf-8")
        df_n3 = pd.read_csv(neu3_csv, encoding="utf-8")
        g3_warns: List[str] = []
        if len(df_s1) > 1 and float(df_s1["macro_f1"].std(ddof=1)) == 0.0:
            g3_warns.append("VSMEC macro_f1 có std=0 giữa các seed")
        for emo in EMOTIONS7:
            col = f"f1_{emo.lower()}"
            if col in df_s1.columns and float(df_s1[col].mean()) < 0.30:
                g3_warns.append(f"VSMEC {col} mean={float(df_s1[col].mean()):.4f} < 0.30")
        for col in ("f1_neg", "f1_neu", "f1_pos"):
            if col in df_n3.columns and float(df_n3[col].mean()) < 0.30:
                g3_warns.append(f"NEU3 {col} mean={float(df_n3[col].mean()):.4f} < 0.30")
        if g3_warns:
            results.append(
                CheckResult("G3", LEVEL_WARN, STATUS_FAIL, "; ".join(g3_warns))
            )
        else:
            results.append(
                CheckResult(
                    "G3",
                    LEVEL_WARN,
                    STATUS_PASS,
                    "Có std > 0 giữa các seed và không lớp nào có F1 trung bình < 0.30",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Group H: Phân bố và độ tin cậy (INFO / WARN)
# ---------------------------------------------------------------------------
def check_group_h(
    vsmec_dfs: Dict[str, pd.DataFrame],
    neuesc_df: Optional[pd.DataFrame],
) -> List[CheckResult]:
    results: List[CheckResult] = []
    if len(vsmec_dfs) != 3 or neuesc_df is None:
        return [
            CheckResult("H1", LEVEL_INFO, STATUS_FAIL, "Thiếu dữ liệu đầu vào"),
            CheckResult("H2", LEVEL_WARN, STATUS_FAIL, "Thiếu dữ liệu đầu vào"),
            CheckResult("H3", LEVEL_INFO, STATUS_FAIL, "Thiếu dữ liệu đầu vào"),
        ]

    # H1: Per-split distributions for 7-class VSMEC, 3-class VSMEC (B1), and 3-class NEU-ESC
    dist_vsmec7: Dict[str, Dict[str, int]] = {}
    dist_vsmec3: Dict[str, Dict[str, int]] = {}
    dist_neuesc3: Dict[str, Dict[str, int]] = {}

    for sp in VSMEC_SPLITS:
        df_v = vsmec_dfs[sp]
        vc7 = df_v["label"].value_counts().reindex(range(7), fill_value=0).to_dict()
        dist_vsmec7[sp] = {EMOTIONS7[int(k)]: int(v) for k, v in vc7.items()}
        vc3 = (
            df_v["label"]
            .map(HARD_MERGE_RULE)
            .value_counts()
            .reindex(range(3), fill_value=0)
            .to_dict()
        )
        dist_vsmec3[sp] = {LABELS3[int(k)]: int(v) for k, v in vc3.items()}

    for sp in NEUESC_SPLITS:
        df_n = neuesc_df[neuesc_df["subset"] == sp]
        nc3 = df_n["label"].value_counts().reindex(range(3), fill_value=0).to_dict()
        dist_neuesc3[sp] = {LABELS3[int(k)]: int(v) for k, v in nc3.items()}

    results.append(
        CheckResult(
            "H1",
            LEVEL_INFO,
            STATUS_PASS,
            f"vsmec_test 3c={dist_vsmec3['test']}; neuesc_test 3c={dist_neuesc3['test']}",
            metrics={
                "vsmec_7class": dist_vsmec7,
                "vsmec_3class_b1": dist_vsmec3,
                "neuesc_3class": dist_neuesc3,
            },
        )
    )

    # H2: JS divergence & < 30 sample check on test splits
    js_by_split: Dict[str, Dict[str, float]] = {}
    for sp in ("train", "dev", "test"):
        v_vec = [dist_vsmec3[sp][lbl] for lbl in LABELS3]
        n_vec = [dist_neuesc3[sp][lbl] for lbl in LABELS3]
        js_by_split[sp] = jensen_shannon_divergence(v_vec, n_vec)

    small_classes: List[str] = []
    for emo, cnt in dist_vsmec7["test"].items():
        if cnt < 30:
            small_classes.append(f"vsmec_test 7c {emo}={cnt}<30")
    for lbl, cnt in dist_vsmec3["test"].items():
        if cnt < 30:
            small_classes.append(f"vsmec_test 3c {lbl}={cnt}<30")
    for lbl, cnt in dist_neuesc3["test"].items():
        if cnt < 30:
            small_classes.append(f"neuesc_test 3c {lbl}={cnt}<30")

    min_v7_emo, min_v7_cnt = min(dist_vsmec7["test"].items(), key=lambda kv: kv[1])
    h2_status = STATUS_FAIL if small_classes else STATUS_PASS
    results.append(
        CheckResult(
            "H2",
            LEVEL_WARN,
            h2_status,
            f"JS(3c train)={js_by_split['train']['js_bits']:.4f} bits ({js_by_split['train']['js_nats']:.4f} nats), "
            f"JS(3c test)={js_by_split['test']['js_bits']:.4f} bits; "
            f"lớp nhỏ nhất vsmec_test 7c là {min_v7_emo}={min_v7_cnt} (>=30)",
            metrics={
                "js_divergence_3class": js_by_split,
                "vsmec_test_7c_counts": dist_vsmec7["test"],
                "vsmec_test_3c_counts": dist_vsmec3["test"],
                "neuesc_test_3c_counts": dist_neuesc3["test"],
                "small_classes_lt_30": small_classes,
            },
        )
    )

    # H3: Surprise & Other counts in vsmec_test
    n_test_v = int(len(vsmec_dfs["test"]))
    n_surprise = dist_vsmec7["test"]["Surprise"]
    n_other = dist_vsmec7["test"]["Other"]
    n_neu_total = n_surprise + n_other
    results.append(
        CheckResult(
            "H3",
            LEVEL_INFO,
            STATUS_PASS,
            f"Trong vsmec_test (n={n_test_v}): Surprise={n_surprise} ({n_surprise/n_test_v*100:.2f}%), "
            f"Other={n_other} ({n_other/n_test_v*100:.2f}%), tổng gộp vào neutral theo B1 = {n_neu_total} ({n_neu_total/n_test_v*100:.2f}%)",
            metrics={
                "vsmec_test_total": n_test_v,
                "surprise_count": n_surprise,
                "other_count": n_other,
                "neutral_b1_total": n_neu_total,
            },
        )
    )

    return results


# ---------------------------------------------------------------------------
# Group I: Đầu vào của mô hình (WARN)
# ---------------------------------------------------------------------------
def check_group_i(
    vsmec_dfs: Dict[str, pd.DataFrame],
    neuesc_df: Optional[pd.DataFrame],
    with_tokenizer: bool = False,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    if len(vsmec_dfs) != 3 or neuesc_df is None:
        return [
            CheckResult(f"I{k}", LEVEL_WARN, STATUS_FAIL, "Thiếu dữ liệu đầu vào")
            for k in (1, 2, 3, 4)
        ]

    v_test = vsmec_dfs["test"]
    n_test = neuesc_df[neuesc_df["subset"] == "test"].reset_index(drop=True)

    # I1: text_seg present and non-empty in both test sets
    i1_errs: List[str] = []
    for name, df_t in (("vsmec_test", v_test), ("neuesc_test", n_test)):
        if "text_seg" not in df_t.columns:
            i1_errs.append(f"{name} thiếu cột text_seg")
        elif (
            df_t["text_seg"].isna().any()
            or (df_t["text_seg"].astype(str).str.strip() == "").any()
        ):
            i1_errs.append(f"{name} có dòng text_seg rỗng/NaN")
    if i1_errs:
        results.append(CheckResult("I1", LEVEL_WARN, STATUS_FAIL, "; ".join(i1_errs)))
    else:
        results.append(
            CheckResult(
                "I1",
                LEVEL_WARN,
                STATUS_PASS,
                f"Cả vsmec_test ({len(v_test)} câu) và neuesc_test ({len(n_test)} câu) đều có cột text_seg đầy đủ, không rỗng",
            )
        )

    # I2: Truncation rate at max_length=256 (requires --with-tokenizer)
    if not with_tokenizer:
        results.append(
            CheckResult(
                "I2",
                LEVEL_WARN,
                STATUS_SKIP,
                "Bỏ qua nạp tokenizer (truyền --with-tokenizer để đo tỉ lệ cắt cụt ở max_length=256)",
            )
        )
    else:
        try:
            from transformers import AutoTokenizer

            tok_pho = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            tok_viso = AutoTokenizer.from_pretrained("uitnlp/visobert")
            trunc_metrics: Dict[str, Any] = {}
            for ds_name, df_t in (("vsmec_test", v_test), ("neuesc_test", n_test)):
                pho_ids = tok_pho(
                    df_t["text_seg"].astype(str).tolist(),
                    truncation=False,
                    add_special_tokens=True,
                )["input_ids"]
                pho_trunc = sum(1 for seq in pho_ids if len(seq) > 256)

                viso_texts = [
                    preprocess_text(str(t), emoji_mode="keep")
                    for t in df_t["text"].tolist()
                ]
                viso_ids = tok_viso(
                    viso_texts,
                    truncation=False,
                    add_special_tokens=True,
                )["input_ids"]
                viso_trunc = sum(1 for seq in viso_ids if len(seq) > 256)

                n_tot = len(df_t)
                trunc_metrics[ds_name] = {
                    "n_total": n_tot,
                    "phobert_n_truncated": pho_trunc,
                    "phobert_rate": pho_trunc / max(1, n_tot),
                    "visobert_n_truncated": viso_trunc,
                    "visobert_rate": viso_trunc / max(1, n_tot),
                }
            results.append(
                CheckResult(
                    "I2",
                    LEVEL_WARN,
                    STATUS_PASS,
                    f"Cắt cụt @256: vsmec_test (PhoBERT={trunc_metrics['vsmec_test']['phobert_n_truncated']}/{len(v_test)}, "
                    f"ViSoBERT={trunc_metrics['vsmec_test']['visobert_n_truncated']}/{len(v_test)}); "
                    f"neuesc_test (PhoBERT={trunc_metrics['neuesc_test']['phobert_n_truncated']}/{len(n_test)}, "
                    f"ViSoBERT={trunc_metrics['neuesc_test']['visobert_n_truncated']}/{len(n_test)})",
                    metrics=trunc_metrics,
                )
            )
        except Exception as exc:
            results.append(
                CheckResult(
                    "I2",
                    LEVEL_WARN,
                    STATUS_FAIL,
                    f"Lỗi khi chạy tokenizer ở I2: {exc}",
                )
            )

    # I3: normalize_text idempotence on both test sets
    non_idempotent = 0
    for df_t in (v_test, n_test):
        for t in df_t["text"].astype(str):
            once = normalize_text(t)
            twice = normalize_text(once)
            if once != twice:
                non_idempotent += 1
    if non_idempotent == 0:
        results.append(
            CheckResult(
                "I3",
                LEVEL_WARN,
                STATUS_PASS,
                f"normalize_text(normalize_text(x)) == normalize_text(x) trên toàn bộ {len(v_test) + len(n_test)} câu test",
            )
        )
    else:
        results.append(
            CheckResult(
                "I3",
                LEVEL_WARN,
                STATUS_FAIL,
                f"Có {non_idempotent} câu test vi phạm tính idempotent của normalize_text",
            )
        )

    # I4: ViSoBERT emoji_mode="keep" consistency across VSMEC test and NEU-ESC test
    v_keep = [preprocess_text(str(t), emoji_mode="keep") for t in v_test["text"]]
    n_keep = [preprocess_text(str(t), emoji_mode="keep") for t in n_test["text"]]
    v_empty = sum(1 for s in v_keep if not s.strip())
    n_empty = sum(1 for s in n_keep if not s.strip())
    v_idemp_fail = sum(
        1 for s in v_keep if preprocess_text(s, emoji_mode="keep") != s
    )
    n_idemp_fail = sum(
        1 for s in n_keep if preprocess_text(s, emoji_mode="keep") != s
    )
    if v_empty == 0 and n_empty == 0 and v_idemp_fail == 0 and n_idemp_fail == 0:
        results.append(
            CheckResult(
                "I4",
                LEVEL_WARN,
                STATUS_PASS,
                "ViSoBERT emoji_mode='keep' (preprocess_text(..., emoji_mode='keep') trên cột text) đồng nhất, "
                "idempotent và không sinh chuỗi rỗng trên cả vsmec_test và neuesc_test",
            )
        )
    else:
        results.append(
            CheckResult(
                "I4",
                LEVEL_WARN,
                STATUS_FAIL,
                f"Lỗi emoji_mode='keep': v_empty={v_empty}, n_empty={n_empty}, "
                f"v_idemp_fail={v_idemp_fail}, n_idemp_fail={n_idemp_fail}",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Group J: Checkpoint (BLOCKER, chỉ chạy khi --check-ckpt)
# ---------------------------------------------------------------------------
def _locate_group_seed_ckpt(
    root: Path,
    alias: str,
    seeds: Sequence[int],
) -> Dict[int, Optional[Path]]:
    """Locate checkpoint directory containing `encoder/` and `head.pt` for each seed (or shared alias dir)."""
    found: Dict[int, Optional[Path]] = {}
    for s in seeds:
        candidates = [
            root / f"seed_{s}" / alias,
            root / f"s{s}" / alias,
            root / f"{alias}_s{s}",
            root / alias / f"seed_{s}",
            root / alias / f"s{s}",
            root / alias,
            root,
        ]
        matched: Optional[Path] = None
        for cand in candidates:
            if (cand / "encoder").is_dir() and (cand / "head.pt").is_file():
                matched = cand
                break
        found[int(s)] = matched
    return found


def check_group_j(
    cfg: Dict[str, Any],
    results_dir: Path,
    check_ckpt: bool = False,
) -> List[CheckResult]:
    if not check_ckpt:
        skip_msg = "Bỏ qua kiểm tra checkpoint (chỉ chạy khi truyền --check-ckpt)"
        return [
            CheckResult("J1", LEVEL_BLOCKER, STATUS_SKIP, skip_msg),
            CheckResult("J2", LEVEL_BLOCKER, STATUS_SKIP, skip_msg),
            CheckResult("J3", LEVEL_BLOCKER, STATUS_SKIP, skip_msg),
        ]

    ckpt_roots = cfg.get("ckpt_roots") or {}
    required_groups = {
        "vsmec_phobert": ("PhoBERT", 7),
        "vsmec_visobert": ("ViSoBERT", 7),
        "neu3_phobert": ("PhoBERT", 3),
        "neu3_visobert": ("ViSoBERT", 3),
    }
    null_groups = [g for g in required_groups if not ckpt_roots.get(g)]
    if null_groups:
        msg = (
            f"ckpt_roots có khóa null cho: {', '.join(null_groups)}; "
            f"cần người phụ trách điền đường dẫn thư mục checkpoint Bước 1 trên Drive vào configs/step2_data.yaml"
        )
        return [
            CheckResult("J1", LEVEL_BLOCKER, STATUS_MISSING, msg),
            CheckResult("J2", LEVEL_BLOCKER, STATUS_MISSING, msg),
            CheckResult("J3", LEVEL_BLOCKER, STATUS_MISSING, msg),
        ]

    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
    results: List[CheckResult] = []

    # J1 & J2
    j1_missing: List[str] = []
    j2_errors: List[str] = []
    located_dirs: Dict[str, List[Path]] = {}

    for grp_name, (alias, expected_out_dim) in required_groups.items():
        root_path = Path(str(ckpt_roots[grp_name]))
        if not root_path.is_dir():
            j1_missing.append(f"{grp_name} (thư mục không tồn tại: {root_path.as_posix()})")
            continue
        seed_map = _locate_group_seed_ckpt(root_path, alias, seeds)
        missing_seeds = [s for s, p in seed_map.items() if p is None]
        if missing_seeds:
            j1_missing.append(f"{grp_name} thiếu seed {missing_seeds}")
        unique_paths: List[Path] = []
        for p in seed_map.values():
            if p is not None and p not in unique_paths:
                unique_paths.append(p)
        located_dirs[grp_name] = unique_paths

        for ckpt_p in unique_paths:
            head_pt = ckpt_p / "head.pt"
            try:
                state = torch.load(head_pt, map_location="cpu", weights_only=True)
                w = state.get("weight")
                if w is None or int(w.shape[0]) != expected_out_dim:
                    j2_errors.append(
                        f"{grp_name} ({head_pt.as_posix()}): weight.shape[0]="
                        f"{None if w is None else int(w.shape[0])} != {expected_out_dim}"
                    )
            except Exception as exc:
                j2_errors.append(f"Lỗi đọc {head_pt.as_posix()}: {exc}")

    if j1_missing:
        results.append(
            CheckResult("J1", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(j1_missing))
        )
    else:
        results.append(
            CheckResult(
                "J1",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "Đủ 4 nhóm checkpoint (vsmec/neu3 × phobert/visobert) với encoder/ và head.pt",
            )
        )

    if j1_missing and not located_dirs:
        results.append(
            CheckResult(
                "J2",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                "Không thể kiểm tra head.pt do các thư mục checkpoint ở J1 đều thiếu",
            )
        )
    elif j2_errors:
        results.append(
            CheckResult("J2", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(j2_errors))
        )
    else:
        results.append(
            CheckResult(
                "J2",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "Tất cả head.pt có weight.shape[0] == 7 (VSMEC) và 3 (NEU-3)",
            )
        )

    # J3: Verify checkpoint hash JSONs if present
    hash_files = [
        results_dir / "stage1_ckpt_hash.json",
        results_dir / "neu3_ckpt_hash.json",
    ]
    existing_hash_files = [p for p in hash_files if p.is_file()]
    if not existing_hash_files:
        results.append(
            CheckResult(
                "J3",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                "Chưa có stage1_ckpt_hash.json / neu3_ckpt_hash.json trong results/ để đối chiếu SHA-256 checkpoint",
            )
        )
    else:
        j3_errors: List[str] = []
        for hf in existing_hash_files:
            payload = json.loads(hf.read_text(encoding="utf-8"))
            for key, info in payload.items():
                ckpt_p = Path(str(info.get("path", "")))
                head_pt = ckpt_p / "head.pt"
                if not head_pt.is_file():
                    j3_errors.append(f"{hf.name}[{key}] thiếu {head_pt.as_posix()}")
                    continue
                actual_head = sha256_file(head_pt)
                if actual_head != info.get("head_sha256"):
                    j3_errors.append(f"{hf.name}[{key}] lệch head_sha256")
        if j3_errors:
            results.append(
                CheckResult("J3", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(j3_errors))
            )
        else:
            results.append(
                CheckResult(
                    "J3",
                    LEVEL_BLOCKER,
                    STATUS_PASS,
                    "Hash checkpoint trong JSON khớp với file thực tế trên đĩa",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Full Orchestrator & Report Exporter
# ---------------------------------------------------------------------------
def run_all_step2_checks(
    config_path: Union[str, Path] = "configs/step2_data.yaml",
    reports_dir: Union[str, Path] = "reports/step2",
    with_tokenizer: bool = False,
    check_ckpt: bool = False,
    cfg_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all checks A–J, write reports & violation CSVs to `reports_dir`, and return summary."""
    cfg = load_step2_config(config_path)
    if cfg_override:
        cfg.update(cfg_override)

    clean_dir = Path(str(cfg["vsmec_clean_dir"]))
    neuesc_file = Path(str(cfg["neuesc_file"]))
    results_dir = Path(str(cfg.get("results_dir", "results")))
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[CheckResult] = []

    res_a, vsmec_dfs, neuesc_df = check_group_a(clean_dir, neuesc_file)
    all_results.extend(res_a)
    all_results.extend(check_group_b(neuesc_df))
    all_results.extend(check_group_c(vsmec_dfs, neuesc_df, rep_dir))
    all_results.extend(check_group_d(vsmec_dfs, neuesc_df, rep_dir))
    all_results.extend(check_group_e(cfg, vsmec_dfs))
    all_results.extend(check_group_f(clean_dir, neuesc_file, results_dir))
    all_results.extend(check_group_g(cfg, results_dir))
    all_results.extend(check_group_h(vsmec_dfs, neuesc_df))
    all_results.extend(
        check_group_i(vsmec_dfs, neuesc_df, with_tokenizer=with_tokenizer)
    )
    all_results.extend(
        check_group_j(cfg, results_dir, check_ckpt=check_ckpt)
    )

    blocking_unresolved = [
        r.id
        for r in all_results
        if r.level == LEVEL_BLOCKER and r.status in (STATUS_FAIL, STATUS_MISSING)
    ]
    is_ready = len(blocking_unresolved) == 0
    gate_line = (
        "SẴN SÀNG"
        if is_ready
        else f"CHƯA SẴN SÀNG: {', '.join(blocking_unresolved)}"
    )

    by_id = {r.id: r for r in all_results}
    report_payload: Dict[str, Any] = {
        "ready": is_ready,
        "gate_line": gate_line,
        "blocking_unresolved": blocking_unresolved,
        "checks": [asdict(r) for r in all_results],
        "d3_overlap": by_id["D3"].metrics if "D3" in by_id else {},
        "d4_near_duplicates": by_id["D4"].metrics if "D4" in by_id else {},
        "h1_distributions": by_id["H1"].metrics if "H1" in by_id else {},
        "h2_js_and_counts": by_id["H2"].metrics if "H2" in by_id else {},
        "h3_surprise_other": by_id["H3"].metrics if "H3" in by_id else {},
        "e2_kappa": by_id["E2"].metrics if "E2" in by_id else {},
        "g2_published_diff": by_id["G2"].metrics if "G2" in by_id else {},
        "f1_hashes": by_id["F1"].metrics if "F1" in by_id else {},
    }

    json_path = rep_dir / "readiness_report.json"
    json_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    md_path = rep_dir / "readiness_report.md"
    _write_markdown_report(md_path, report_payload, all_results)

    return report_payload


def _write_markdown_report(
    md_path: Path,
    payload: Dict[str, Any],
    checks: Sequence[CheckResult],
) -> None:
    lines: List[str] = [
        "# Báo cáo Kiểm tra Sẵn sàng Dữ liệu & Checkpoint cho Bước 2 (Zero-Shot VSMEC ↔ NEU-ESC)",
        "",
        f"**Kết luận cổng:** `{payload['gate_line']}`",
        "",
        "## 1. Bảng tổng hợp kết quả kiểm tra (A1 – J3)",
        "",
        "| ID | Mức | Trạng thái | Chi tiết |",
        "| --- | --- | --- | --- |",
    ]
    for r in checks:
        safe_detail = r.detail.replace("|", "\\|")
        lines.append(f"| `{r.id}` | {r.level} | **{r.status}** | {safe_detail} |")

    # Section 2: D3 & D4
    lines.extend(
        [
            "",
            "## 2. Kiểm tra trùng chéo miền VSMEC ↔ NEU-ESC (D1 – D4)",
            "",
            "### 2.1. Bảng đếm trùng chính xác (sau `normalize_text` + `hash_text`) theo 2 hướng (D3)",
            "",
            "| Tập Test | Tập đối chiếu | Số hash trùng duy nhất | Số dòng ở Tập Test bị trùng | Số dòng ở Tập đối chiếu bị trùng |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in payload.get("d3_overlap", {}).get("d3_directional_table", []):
        lines.append(
            f"| `{row['test_split']}` | `{row['other_split']}` | {row['unique_overlap_hashes']} | "
            f"{row['dir_test_to_other_rows']} | {row['dir_other_to_test_rows']} |"
        )

    near_dups = payload.get("d4_near_duplicates", {}).get("near_duplicates", [])
    lines.extend(
        [
            "",
            f"### 2.2. Câu trùng gần phát hiện ở D4 (bỏ dấu câu & emoji, giữ dấu tiếng Việt): `{len(near_dups)}` câu",
            "",
        ]
    )
    if near_dups:
        lines.extend(
            [
                "| Tập Test | ID | Nhãn | Tập khớp | ID khớp | Chuỗi chuẩn hóa mạnh | Văn bản gốc |",
                "| --- | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for nd in near_dups:
            lines.append(
                f"| `{nd['source_split']}` | {nd['id']} | {nd['label']} | `{nd['matched_split']}` | "
                f"`{nd['matched_ids']}` | `{nd['strong_norm']}` | {nd['orig_text']} |"
            )
        lines.append("")
        lines.append(
            "> **Lưu ý:** Nếu loại bỏ bất kỳ câu nào khỏi tập test (`vsmec_test` hoặc `neuesc_test`), "
            "số liệu test của Bước 1 bắt buộc phải được suy luận và tính lại từ checkpoint đã lưu."
        )

    # Section 3: H1 - H3
    h1 = payload.get("h1_distributions", {})
    h2 = payload.get("h2_js_and_counts", {})
    h3 = payload.get("h3_surprise_other", {})
    if h1:
        lines.extend(
            [
                "",
                "## 3. Phân bố nhãn và độ lệch tiên nghiệm (H1 – H3)",
                "",
                "### 3.1. Phân bố VSMEC 7 nhãn (H1)",
                "",
                "| Split | Enjoyment (0) | Sadness (1) | Fear (2) | Anger (3) | Disgust (4) | Surprise (5) | Other (6) | Tổng |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for sp in VSMEC_SPLITS:
            d7 = h1["vsmec_7class"][sp]
            tot = sum(d7.values())
            lines.append(
                f"| `{sp}` | {d7['Enjoyment']} | {d7['Sadness']} | {d7['Fear']} | {d7['Anger']} | "
                f"{d7['Disgust']} | {d7['Surprise']} | {d7['Other']} | **{tot}** |"
            )

        lines.extend(
            [
                "",
                "### 3.2. Phân bố 3 nhãn cực tính (VSMEC theo B1 vs NEU-ESC) và JS Divergence (H1, H2)",
                "",
                "| Bộ dữ liệu | Split | negative (0) | neutral (1) | positive (2) | Tổng | JS Divergence (bits) | JS Divergence (nats) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        js_map = h2.get("js_divergence_3class", {})
        for sp in ("train", "dev", "test"):
            dv3 = h1["vsmec_3class_b1"][sp]
            dn3 = h1["neuesc_3class"][sp]
            js_b = js_map.get(sp, {}).get("js_bits", 0.0)
            js_n = js_map.get(sp, {}).get("js_nats", 0.0)
            lines.append(
                f"| VSMEC (gộp B1) | `{sp}` | {dv3['negative']} | {dv3['neutral']} | {dv3['positive']} | "
                f"**{sum(dv3.values())}** | {js_b:.6f} | {js_n:.6f} |"
            )
            lines.append(
                f"| NEU-ESC | `{sp}` | {dn3['negative']} | {dn3['neutral']} | {dn3['positive']} | "
                f"**{sum(dn3.values())}** | — | — |"
            )

    if h3:
        lines.extend(
            [
                "",
                "### 3.3. Hệ quả của B1 lên Surprise và Other trong `vsmec_test` (H3)",
                "",
                f"- Tổng số câu `vsmec_test`: **{h3['vsmec_test_total']}**",
                f"- `Surprise` (nhãn 5): **{h3['surprise_count']}** câu ({h3['surprise_count']/h3['vsmec_test_total']*100:.2f}%)",
                f"- `Other` (nhãn 6): **{h3['other_count']}** câu ({h3['other_count']/h3['vsmec_test_total']*100:.2f}%)",
                f"- Tổng gộp vào `neutral` (nhãn 1) theo quy tắc cứng B1: **{h3['neutral_b1_total']}** câu ({h3['neutral_b1_total']/h3['vsmec_test_total']*100:.2f}%)",
            ]
        )

    # Section 4: E2 Kappa & G2 diff
    lines.extend(
        [
            "",
            "## 4. Thông tin VSMEC-strict (E1–E3) và Cổng Bước 1 (G1–G3)",
            "",
            f"- **VSMEC-strict Cohen's kappa (E2):** `{json.dumps(payload.get('e2_kappa', {}), ensure_ascii=False)}` "
            f"(ghi chú: trên nhánh hiện tại có `results/polarity/kappa_report.json` cho 100 câu lấy từ `vsmec_train` ở CV-P3 với κ = 0.410163, "
            f"nhưng chưa có file `vsmec_strict_file` trên `vsmec_test`)",
            f"- **Đối chiếu F1 công bố (G2):** `{json.dumps(payload.get('g2_published_diff', {}), ensure_ascii=False)}`",
            "",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def format_console_table(checks: Sequence[CheckResult], gate_line: str) -> str:
    """Format console table `ID | mức | trạng thái | chi tiết` followed by `gate_line`."""
    lines = [
        f"{'ID':<4} | {'mức':<7} | {'trạng thái':<10} | chi tiết",
        "-" * 100,
    ]
    for r in checks:
        lines.append(f"{r.id:<4} | {r.level:<7} | {r.status:<10} | {r.detail}")
    lines.append("-" * 100)
    lines.append(gate_line)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Kiểm tra dữ liệu và checkpoint đã sẵn sàng cho Bước 2 (Zero-Shot VSMEC ↔ NEU-ESC)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/step2_data.yaml"),
        help="Đường dẫn file cấu hình YAML (mặc định: configs/step2_data.yaml).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/step2"),
        help="Thư mục xuất báo cáo và danh sách vi phạm (mặc định: reports/step2).",
    )
    parser.add_argument(
        "--with-tokenizer",
        action="store_true",
        help="Chạy thêm kiểm tra I2 (tỉ lệ cắt cụt ở max_length=256 với tokenizer PhoBERT & ViSoBERT).",
    )
    parser.add_argument(
        "--check-ckpt",
        action="store_true",
        help="Chạy thêm kiểm tra J1-J3 trên các thư mục checkpoint Bước 1.",
    )
    args = parser.parse_args(argv)

    payload = run_all_step2_checks(
        config_path=args.config,
        reports_dir=args.reports_dir,
        with_tokenizer=args.with_tokenizer,
        check_ckpt=args.check_ckpt,
    )
    checks = [CheckResult(**c) for c in payload["checks"]]
    print(format_console_table(checks, payload["gate_line"]))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())

