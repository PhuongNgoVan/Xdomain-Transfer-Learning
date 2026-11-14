"""Step 2 data and checkpoint readiness verification script (`scripts/check_step2_data.py`).

Implements checks A1–A4, B1–B3, C1–C3, D1–D4, E1–E3, F1–F3 (including F2b),
G1–G3, H1–H4, I1–I4, and J1–J3 per `tasks/AGENT_step2_data_readiness.md` and
updated readiness requirements.

Strict boundaries:
- Read-only with respect to `data/clean/`, `data/splits/`, `MANIFEST.json`,
  and `src/common/constants.py` (`HARD_MERGE_RULE`).
- Does not train models or score zero-shot test predictions.
- Writes all outputs only under `reports/step2/` (or custom `--reports-dir`).
"""

import argparse
from dataclasses import asdict, dataclass, field
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import unicodedata

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

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
from src.data.preprocess import _GENERIC_EMOJI_RE, UNICODE_EMOJI_MAP, preprocess_text

LEVEL_BLOCKER = "BLOCKER"
LEVEL_WARN = "WARN"
LEVEL_INFO = "INFO"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_MISSING = "MISSING"
STATUS_SKIP = "SKIP"

VSMEC_SPLITS: Tuple[str, ...] = ("train", "dev", "test")
NEUESC_SPLITS: Tuple[str, ...] = ("train", "dev", "test")

GROUP_ORDER: Tuple[str, ...] = (
    "vsmec_phobert",
    "vsmec_visobert",
    "neu3_phobert",
    "neu3_visobert",
)
GROUP_LABELS: Dict[str, str] = {
    "vsmec_phobert": "VSMEC PhoBERT",
    "vsmec_visobert": "VSMEC ViSoBERT",
    "neu3_phobert": "NEU-3 PhoBERT",
    "neu3_visobert": "NEU-3 ViSoBERT",
}
GROUP_SPECS: Dict[str, Tuple[str, str, int]] = {
    "vsmec_phobert": ("vsmec", "PhoBERT", 7),
    "vsmec_visobert": ("vsmec", "ViSoBERT", 7),
    "neu3_phobert": ("neuesc", "PhoBERT", 3),
    "neu3_visobert": ("neuesc", "ViSoBERT", 3),
}

_ANNOTATOR_STR_MAP: Dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "neg": 0,
    "negative": 0,
    "neu": 1,
    "neutral": 1,
    "pos": 2,
    "positive": 2,
}

RELOAD_MAX_ABS_DIFF_TOL: float = 2e-3


@dataclass
class CheckResult:
    id: str
    level: str
    status: str
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)


def load_step2_config(
    config_path: Union[str, Path] = "configs/step2_data.yaml",
) -> Dict[str, Any]:
    """Load `configs/step2_data.yaml` with safe defaults."""
    p = Path(config_path)
    default_cfg: Dict[str, Any] = {
        "vsmec_clean_dir": "data/clean",
        "neuesc_file": "data/clean/neuesc_100.csv",
        "results_dir": "results",
        "results_inputs_dir": "reports/step2/inputs",
        "step2_files": [],
        "vsmec_strict_file": None,
        "vsmec_published_macro_f1": {
            "phobert": None,
            "visobert": None,
        },
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
    else:
        cr = dict(default_cfg["ckpt_roots"])
        cr.update(merged["ckpt_roots"])
        merged["ckpt_roots"] = cr

    pub = merged.get("vsmec_published_macro_f1")
    if pub is None:
        merged["vsmec_published_macro_f1"] = dict(default_cfg["vsmec_published_macro_f1"])
    elif isinstance(pub, dict):
        merged["vsmec_published_macro_f1"] = {
            "phobert": pub.get("phobert"),
            "visobert": pub.get("visobert"),
        }
    return merged


def compute_vsmec_data_hash(clean_dir: Union[str, Path]) -> str:
    """Compute combined SHA-256 digest of `vsmec_{train,dev,test}.csv` (matches `src/plm/stage1.py`)."""
    d_path = Path(clean_dir)
    combined = hashlib.sha256()
    for sp in VSMEC_SPLITS:
        fp = d_path / f"vsmec_{sp}.csv"
        file_digest = sha256_file(fp)
        combined.update(f"vsmec_{sp}.csv:{file_digest}\n".encode("utf-8"))
    return combined.hexdigest()


def compute_neuesc_data_hash(neuesc_file: Union[str, Path]) -> str:
    """Compute SHA-256 digest of `neuesc_100.csv` (matches `src/plm/train_neu3.py`)."""
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
        raise ValueError(f"Expected probs7 of shape (N, 7) or (7,), got {arr.shape}")

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


def contains_unicode_emoji(text: str) -> bool:
    """Return True if `text` contains at least one Unicode emoji character."""
    s = str(text)
    if _GENERIC_EMOJI_RE.search(s):
        return True
    return any(ch in UNICODE_EMOJI_MAP for ch in s)


def jensen_shannon_divergence(
    p: Sequence[float], q: Sequence[float]
) -> Dict[str, float]:
    """Compute Jensen-Shannon divergence between two distributions in bits (base 2) and nats (base e)."""
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
# Step 1 Result Loader (supports STAGE1_COLUMNS, RESULT_COLUMNS, NEU3_COLUMNS)
# ---------------------------------------------------------------------------
def discover_result_input_files(
    results_files: Optional[Sequence[Union[str, Path]]] = None,
    results_globs: Optional[Sequence[str]] = None,
    default_inputs_dir: Union[str, Path] = "reports/step2/inputs",
    fallback_results_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Collect candidate Step 1 result CSV paths from CLI flags or default inputs folder."""
    collected: List[Path] = []
    seen_resolved: set[Path] = set()

    def _add(p: Path) -> None:
        if p.is_file():
            rp = p.resolve()
            if rp not in seen_resolved:
                seen_resolved.add(rp)
                collected.append(p)

    has_explicit = bool(results_files) or bool(results_globs)
    if results_files:
        for item in results_files:
            _add(Path(item))
    if results_globs:
        for pat in results_globs:
            for m in sorted(glob.glob(str(pat), recursive=True)):
                _add(Path(m))

    if not has_explicit:
        in_dir = Path(default_inputs_dir)
        if in_dir.is_dir():
            for p in sorted(in_dir.rglob("*.csv")):
                _add(p)
        if not collected and fallback_results_dir is not None:
            fb_dir = Path(fallback_results_dir)
            for fname in ("stage1.csv", "neu3.csv"):
                _add(fb_dir / fname)

    return collected


def _infer_backbone_key(text_val: str) -> Optional[str]:
    low = str(text_val).lower()
    if "phobert" in low:
        return "phobert"
    if "visobert" in low:
        return "visobert"
    return None


def _normalize_split_tag(val: Any) -> str:
    s = str(val).strip().lower()
    if s in ("dev", "val", "valid", "validation", "vsmec_dev", "neuesc_dev"):
        return "dev"
    if s in ("test", "vsmec_test", "neuesc_test"):
        return "test"
    if s in ("train", "vsmec_train", "neuesc_train"):
        return "train"
    return s


def _infer_domain(row: pd.Series, cols: set[str], src_name: str) -> Optional[str]:
    if "eval_domain" in cols and pd.notna(row.get("eval_domain")):
        ed = str(row["eval_domain"]).strip().lower()
        if "neu" in ed:
            return "neuesc"
        if "vsmec" in ed:
            return "vsmec"
    if "stage1" in cols and pd.notna(row.get("stage1")):
        s1 = str(row["stage1"]).strip().lower()
        if "neu" in s1 and not (pd.notna(row.get("stage2")) and str(row["stage2"]).strip()):
            return "neuesc"
    if "dataset" in cols and pd.notna(row.get("dataset")):
        ds = str(row["dataset"]).strip().lower()
        if "vsmec" in ds:
            return "vsmec"
        if "neu" in ds:
            return "neuesc"
    if "f1_enjoyment" in cols:
        return "vsmec"
    run_id = str(row.get("run_id", "")).lower()
    if "stage1" in run_id or "vsmec" in run_id:
        return "vsmec"
    if "neu3" in run_id or "neuesc" in run_id:
        return "neuesc"
    if "stage1" in src_name.lower() or "vsmec" in src_name.lower():
        return "vsmec"
    if "neu3" in src_name.lower() or "neu" in src_name.lower():
        return "neuesc"
    if "f1_neg" in cols:
        return "neuesc"
    return None


def _extract_variant(row: pd.Series, cols: set[str], run_id: str) -> str:
    if "variant" in cols and pd.notna(row.get("variant")) and str(row["variant"]).strip():
        return str(row["variant"]).strip()
    m = re.search(r"(?:Stage1|NEU3)-([^_]+)_", run_id, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    cfg_p = str(row.get("config_path", "")).lower()
    if "sweep" in cfg_p:
        return "sweep"
    return "base"


def _find_stage1_sidecar(csv_path: Path, run_id: str) -> Optional[Dict[str, Any]]:
    candidates = [
        csv_path.parent / "runs" / f"{run_id}.json",
        csv_path.parent.parent / "runs" / f"{run_id}.json",
    ]
    for cand in candidates:
        if cand.is_file():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def _safe_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def load_step1_results(
    results_files: Optional[Sequence[Union[str, Path]]] = None,
    results_globs: Optional[Sequence[str]] = None,
    default_inputs_dir: Union[str, Path] = "reports/step2/inputs",
    fallback_results_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load, normalize schemas, filter LR sweep rows, and deduplicate by `(run_id, split)`.

    Supports:
    - `STAGE1_COLUMNS` (classic VSMEC 7-class CSV, optionally paired with `runs/<run_id>.json` or `split` col)
    - `RESULT_COLUMNS` (legacy unified 31-col CSV using `model`, `dataset`, `split`, `variant`)
    - `NEU3_COLUMNS` (`neu3.csv` using `backbone`, `eval_domain`, `split`, `run_kind`, `variant`, ...)
    """
    paths = discover_result_input_files(
        results_files=results_files,
        results_globs=results_globs,
        default_inputs_dir=default_inputs_dir,
        fallback_results_dir=fallback_results_dir,
    )
    if not paths:
        return {
            "files": [],
            "df": pd.DataFrame(),
            "n_raw_rows": 0,
            "n_sweep_filtered": 0,
            "n_duplicates_dropped": 0,
        }

    normalized_rows: List[Dict[str, Any]] = []
    n_raw_rows = 0
    n_sweep_filtered = 0

    f1_cols_7 = [f"f1_{emo.lower()}" for emo in EMOTIONS7]
    f1_cols_3 = ["f1_neg", "f1_neu", "f1_pos"]

    for csv_path in paths:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as exc:
            raise ValueError(f"Lỗi đọc file kết quả {csv_path.as_posix()}: {exc}") from exc
        if df.empty:
            continue

        cols = set(df.columns)
        has_split_col = "split" in cols
        n_raw_rows += len(df)

        for _, row in df.iterrows():
            run_id = str(row.get("run_id", "")).strip()
            if not run_id:
                continue

            domain = _infer_domain(row, cols, csv_path.name)
            backbone_raw = ""
            if "backbone" in cols and pd.notna(row.get("backbone")):
                backbone_raw = str(row["backbone"]).strip()
            elif "model" in cols and pd.notna(row.get("model")):
                backbone_raw = str(row["model"]).strip()
            backbone_key = _infer_backbone_key(backbone_raw) or _infer_backbone_key(run_id)

            variant = _extract_variant(row, cols, run_id)
            run_kind = (
                str(row["run_kind"]).strip().lower()
                if "run_kind" in cols and pd.notna(row.get("run_kind"))
                else ("sweep" if variant.lower().startswith("sweep") else "base")
            )

            is_sweep = (
                variant.lower().startswith("sweep")
                or run_kind == "sweep"
                or "-sweep" in run_id.lower()
                or "_sweep" in run_id.lower()
            )
            if is_sweep:
                n_sweep_filtered += 1
                continue

            if domain == "vsmec" and backbone_key in ("phobert", "visobert"):
                group_key = f"vsmec_{backbone_key}"
            elif domain == "neuesc" and backbone_key in ("phobert", "visobert"):
                group_key = f"neu3_{backbone_key}"
            else:
                group_key = "unknown"

            seed_val = int(row["seed"]) if "seed" in cols and pd.notna(row.get("seed")) else -1
            data_hash = (
                str(row["data_hash"]).strip()
                if "data_hash" in cols and pd.notna(row.get("data_hash"))
                else ""
            )
            code_version = (
                str(row["code_version"]).strip()
                if "code_version" in cols and pd.notna(row.get("code_version"))
                else ""
            )

            f1_dict: Dict[str, Optional[float]] = {}
            for c in (*f1_cols_7, *f1_cols_3):
                f1_dict[c] = _safe_float(row.get(c)) if c in cols else None

            if (
                f1_dict["f1_neg"] is None
                and "f1_per_class" in cols
                and pd.notna(row.get("f1_per_class"))
            ):
                try:
                    parsed = json.loads(str(row["f1_per_class"]))
                    if isinstance(parsed, list) and len(parsed) == 3:
                        f1_dict["f1_neg"] = float(parsed[0])
                        f1_dict["f1_neu"] = float(parsed[1])
                        f1_dict["f1_pos"] = float(parsed[2])
                except Exception:
                    pass

            base_rec: Dict[str, Any] = {
                "run_id": run_id,
                "domain": domain,
                "backbone_key": backbone_key,
                "group_key": group_key,
                "seed": seed_val,
                "variant": variant,
                "run_kind": run_kind,
                "macro_f1": _safe_float(row.get("macro_f1")),
                "data_hash": data_hash,
                "code_version": code_version,
                "ckpt_path": str(row.get("ckpt_path", "")).strip() if "ckpt_path" in cols and pd.notna(row.get("ckpt_path")) else "",
                "source_file": csv_path.as_posix(),
                **f1_dict,
            }

            if has_split_col and pd.notna(row.get("split")) and str(row.get("split")).strip():
                sp = _normalize_split_tag(row["split"])
                normalized_rows.append({**base_rec, "split": sp})
            else:
                # Classic STAGE1_COLUMNS has 1 row per run_id (dev metrics), with optional sidecar for test
                normalized_rows.append({**base_rec, "split": "dev"})
                sidecar = _find_stage1_sidecar(csv_path, run_id)
                if sidecar and isinstance(sidecar.get("report_only"), dict):
                    test_f1 = _safe_float(sidecar["report_only"].get("test_macro_f1"))
                    if test_f1 is not None:
                        normalized_rows.append(
                            {**base_rec, "split": "test", "macro_f1": test_f1}
                        )

    if not normalized_rows:
        return {
            "files": [p.as_posix() for p in paths],
            "df": pd.DataFrame(),
            "n_raw_rows": n_raw_rows,
            "n_sweep_filtered": n_sweep_filtered,
            "n_duplicates_dropped": 0,
        }

    out_df = pd.DataFrame(normalized_rows)
    before_dedup = len(out_df)
    out_df = out_df.drop_duplicates(subset=["run_id", "split"], keep="last").reset_index(
        drop=True
    )
    n_dups_dropped = before_dedup - len(out_df)

    return {
        "files": [p.as_posix() for p in paths],
        "df": out_df,
        "n_raw_rows": n_raw_rows,
        "n_sweep_filtered": n_sweep_filtered,
        "n_duplicates_dropped": n_dups_dropped,
    }


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
        results.append(CheckResult("A1", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(a1_errors)))
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
        results.append(CheckResult("A2", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(a2_errors)))
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
        results.append(CheckResult("A3", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(a3_errors)))
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

    fake_probs7 = np.array(
        [
            [0.20, 0.10, 0.05, 0.15, 0.10, 0.25, 0.15],
            [0.70, 0.02, 0.03, 0.05, 0.05, 0.05, 0.10],
        ],
        dtype=np.float64,
    )
    expected_probs3 = np.array(
        [
            [0.40, 0.40, 0.20],
            [0.15, 0.15, 0.70],
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
        counts_map = {int(k): int(v) for k, v in vc.items()}
        results.append(
            CheckResult(
                "B3",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"NEUESC_ORIG_TO_LABEL3 khớp (toxic 3->neg 0); phân bố thực: neg={counts_map[0]}, neu={counts_map[1]}, pos={counts_map[2]}",
                metrics={"neuesc_total_label_counts": counts_map},
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

    datasets: Dict[str, Dict[str, pd.DataFrame]] = {"vsmec": {}, "neuesc": {}}
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

    # C1
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

    # C2
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

    # C3
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
# Group D: Trùng chéo miền (BLOCKER / INFO / WARN + Sensitivity CSV)
# ---------------------------------------------------------------------------
def _describe_vsmec_label(label_7: int) -> Tuple[str, int, str]:
    orig_name = ID_TO_EMOTION7[int(label_7)]
    lbl3 = HARD_MERGE_RULE[int(label_7)]
    return orig_name, int(lbl3), ID_TO_LABEL3[int(lbl3)]


def _describe_neuesc_label(label_3: int) -> Tuple[str, int, str]:
    lbl3 = int(label_3)
    name = ID_TO_LABEL3[lbl3]
    return name, lbl3, name


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

    # D1
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

    # D2
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

    # D3
    d3_table: List[Dict[str, Any]] = []
    for n_sp in NEUESC_SPLITS:
        inter = set(v_test["text_hash"]) & set(n_splits[n_sp]["text_hash"])
        d3_table.append(
            {
                "test_split": "vsmec_test",
                "other_split": f"neuesc_{n_sp}",
                "unique_overlap_hashes": len(inter),
                "dir_test_to_other_rows": int(v_test["text_hash"].isin(inter).sum()),
                "dir_other_to_test_rows": int(n_splits[n_sp]["text_hash"].isin(inter).sum()),
            }
        )
    for v_sp in VSMEC_SPLITS:
        inter = set(n_test["text_hash"]) & set(v_splits[v_sp]["text_hash"])
        d3_table.append(
            {
                "test_split": "neuesc_test",
                "other_split": f"vsmec_{v_sp}",
                "unique_overlap_hashes": len(inter),
                "dir_test_to_other_rows": int(n_test["text_hash"].isin(inter).sum()),
                "dir_other_to_test_rows": int(v_splits[v_sp]["text_hash"].isin(inter).sum()),
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

    # D4: Stronger normalization near-duplicates + sensitivity exclude IDs CSV
    d4_violations: List[Dict[str, Any]] = []
    exact_d1_ids = {int(r["id"]) for r in d1_violations}
    exact_d2_ids = {int(r["id"]) for r in d2_violations}

    for n_sp, n_df in n_splits.items():
        n_by_sn: Dict[str, pd.DataFrame] = {
            sn: grp for sn, grp in n_df[n_df["strong_norm"] != ""].groupby("strong_norm", sort=False)
        }
        for _, row in v_test.iterrows():
            sn = str(row["strong_norm"])
            rid = int(row["id"])
            if sn and sn in n_by_sn and rid not in exact_d1_ids:
                m_grp = n_by_sn[sn]
                m_ids = [int(x) for x in m_grp["id"].tolist()]
                m_labels = [int(x) for x in m_grp["label"].tolist()]
                m_texts = [str(x) for x in m_grp["text"].tolist()]
                src_orig_name, src_3c, src_3c_name = _describe_vsmec_label(int(row["label"]))
                m_orig_name, m_3c, m_3c_name = _describe_neuesc_label(m_labels[0])
                d4_violations.append(
                    {
                        "source_split": "vsmec_test",
                        "id": rid,
                        "orig_label": int(row["label"]),
                        "orig_label_name": src_orig_name,
                        "mapped_3class_label": src_3c,
                        "mapped_3class_name": src_3c_name,
                        "matched_split": f"neuesc_{n_sp}",
                        "matched_ids": json.dumps(m_ids),
                        "matched_orig_label": m_labels[0],
                        "matched_orig_label_name": m_orig_name,
                        "matched_3class_label": m_3c,
                        "matched_3class_name": m_3c_name,
                        "label_3class_match": bool(src_3c == m_3c),
                        "strong_norm": sn,
                        "orig_text": str(row["text"]),
                        "matched_text": m_texts[0],
                    }
                )

    for v_sp, v_df in v_splits.items():
        v_by_sn: Dict[str, pd.DataFrame] = {
            sn: grp for sn, grp in v_df[v_df["strong_norm"] != ""].groupby("strong_norm", sort=False)
        }
        for _, row in n_test.iterrows():
            sn = str(row["strong_norm"])
            rid = int(row["id"])
            if sn and sn in v_by_sn and rid not in exact_d2_ids:
                m_grp = v_by_sn[sn]
                m_ids = [int(x) for x in m_grp["id"].tolist()]
                m_labels = [int(x) for x in m_grp["label"].tolist()]
                m_texts = [str(x) for x in m_grp["text"].tolist()]
                src_orig_name, src_3c, src_3c_name = _describe_neuesc_label(int(row["label"]))
                m_orig_name, m_3c, m_3c_name = _describe_vsmec_label(m_labels[0])
                d4_violations.append(
                    {
                        "source_split": "neuesc_test",
                        "id": rid,
                        "orig_label": int(row["label"]),
                        "orig_label_name": src_orig_name,
                        "mapped_3class_label": src_3c,
                        "mapped_3class_name": src_3c_name,
                        "matched_split": f"vsmec_{v_sp}",
                        "matched_ids": json.dumps(m_ids),
                        "matched_orig_label": m_labels[0],
                        "matched_orig_label_name": m_orig_name,
                        "matched_3class_label": m_3c,
                        "matched_3class_name": m_3c_name,
                        "label_3class_match": bool(src_3c == m_3c),
                        "strong_norm": sn,
                        "orig_text": str(row["text"]),
                        "matched_text": m_texts[0],
                    }
                )

    d4_cols = [
        "source_split",
        "id",
        "orig_label",
        "orig_label_name",
        "mapped_3class_label",
        "mapped_3class_name",
        "matched_split",
        "matched_ids",
        "matched_orig_label",
        "matched_orig_label_name",
        "matched_3class_label",
        "matched_3class_name",
        "label_3class_match",
        "strong_norm",
        "orig_text",
        "matched_text",
    ]
    pd.DataFrame(d4_violations, columns=d4_cols).to_csv(
        reports_dir / "violations_D4_near_duplicates.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    sens_rows: List[Dict[str, Any]] = []
    seen_sens: set[Tuple[str, int]] = set()
    for nd in d4_violations:
        key = (str(nd["source_split"]), int(nd["id"]))
        if key not in seen_sens:
            seen_sens.add(key)
            sens_rows.append(
                {
                    "id": int(nd["id"]),
                    "dataset": str(nd["source_split"]),
                    "reason": (
                        f"D4 near-duplicate với {nd['matched_split']} id={nd['matched_ids']} "
                        f"(strong_norm='{nd['strong_norm']}', "
                        f"nhãn gốc={nd['orig_label_name']}->{nd['mapped_3class_name']}, "
                        f"đối chiếu={nd['matched_orig_label_name']}->{nd['matched_3class_name']}, "
                        f"khớp_3_nhãn={nd['label_3class_match']})"
                    ),
                }
            )
    pd.DataFrame(sens_rows, columns=["id", "dataset", "reason"]).to_csv(
        reports_dir / "sensitivity_exclude_ids.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    n_v_affected = sum(1 for ds, _ in seen_sens if ds == "vsmec_test")
    n_n_affected = sum(1 for ds, _ in seen_sens if ds == "neuesc_test")
    v_total = int(len(v_test))
    n_total = int(len(n_test))
    impact = {
        "vsmec_test_affected": n_v_affected,
        "vsmec_test_total": v_total,
        "vsmec_test_rate": n_v_affected / max(1, v_total),
        "neuesc_test_affected": n_n_affected,
        "neuesc_test_total": n_total,
        "neuesc_test_rate": n_n_affected / max(1, n_total),
    }

    if d4_violations:
        pair_descs = [
            (
                f"{nd['source_split']} id={nd['id']} ({nd['orig_label_name']}→{nd['mapped_3class_name']}) ↔ "
                f"{nd['matched_split']} id={nd['matched_ids']} ({nd['matched_orig_label_name']}→{nd['matched_3class_name']}, "
                f"khớp={nd['label_3class_match']})"
            )
            for nd in d4_violations
        ]
        results.append(
            CheckResult(
                "D4",
                LEVEL_WARN,
                STATUS_FAIL,
                f"Giữ dữ liệu đóng băng; {len(d4_violations)} cặp trùng gần: {'; '.join(pair_descs)}. "
                f"Ảnh hưởng độ nhạy: vsmec_test {n_v_affected}/{v_total} ({impact['vsmec_test_rate']*100:.4f}%), "
                f"neuesc_test {n_n_affected}/{n_total} ({impact['neuesc_test_rate']*100:.4f}%) — đã xuất sensitivity_exclude_ids.csv",
                metrics={"near_duplicates": d4_violations, "impact": impact},
            )
        )
    else:
        results.append(
            CheckResult(
                "D4",
                LEVEL_WARN,
                STATUS_PASS,
                "0 câu test trùng gần tăng thêm so với D1/D2",
                metrics={"near_duplicates": [], "impact": impact},
            )
        )

    return results


# ---------------------------------------------------------------------------
# Group E: VSMEC-strict (BLOCKER when file provided; MISSING when null)
# ---------------------------------------------------------------------------
def _parse_annotator_label(val: Any) -> Optional[int]:
    if pd.isna(val):
        return None
    if isinstance(val, (int, np.integer)):
        iv = int(val)
        return iv if iv in (0, 1, 2) else None
    if isinstance(val, (float, np.floating)):
        if math.isnan(float(val)):
            return None
        iv = int(val)
        return iv if float(val) == float(iv) and iv in (0, 1, 2) else None
    s = str(val).strip().lower()
    if s in _ANNOTATOR_STR_MAP:
        return _ANNOTATOR_STR_MAP[s]
    return None


def validate_vsmec_strict_file(
    strict_path: Path,
    vsmec_test_df: pd.DataFrame,
    has_rule: bool = False,
) -> List[CheckResult]:
    """Validate a candidate `VSMEC-strict` CSV against `vsmec_test` (E1, E2, E3)."""
    if not strict_path.is_file():
        err = f"File vsmec_strict_file không tồn tại: {strict_path.as_posix()}"
        return [
            CheckResult("E1", LEVEL_BLOCKER, STATUS_FAIL, err),
            CheckResult("E2", LEVEL_BLOCKER, STATUS_FAIL, err),
            CheckResult("E3", LEVEL_BLOCKER, STATUS_FAIL, err),
        ]

    try:
        df_strict = pd.read_csv(strict_path, encoding="utf-8")
    except Exception as exc:
        err = f"Lỗi đọc {strict_path.as_posix()}: {exc}"
        return [
            CheckResult("E1", LEVEL_BLOCKER, STATUS_FAIL, err),
            CheckResult("E2", LEVEL_BLOCKER, STATUS_FAIL, err),
            CheckResult("E3", LEVEL_BLOCKER, STATUS_FAIL, err),
        ]

    v_test_ids = set(vsmec_test_df["id"].astype(int).tolist())

    annot_cols = [
        c
        for c in df_strict.columns
        if c.startswith("annotator")
        or (c.startswith("label_") and c not in ("label_orig", "label_vsmec"))
        or c in ("nhat", "long", "polarity_nhat", "polarity_long")
    ]

    e1_errors: List[str] = []
    if "id" not in df_strict.columns:
        e1_errors.append("Thiếu cột id")
    else:
        if df_strict["id"].isna().any():
            e1_errors.append("Cột id có giá trị NaN")
        elif df_strict["id"].duplicated().any():
            e1_errors.append("Cột id bị trùng lặp")
        else:
            strict_ids = set(df_strict["id"].astype(int).tolist())
            not_in_test = sorted(strict_ids - v_test_ids)
            if not_in_test:
                e1_errors.append(
                    f"{len(not_in_test)} id không thuộc vsmec_test (ví dụ: {not_in_test[:5]})"
                )

    parsed_cols: Dict[str, List[int]] = {}
    if not annot_cols:
        e1_errors.append("Không tìm thấy cột nhãn của người gán (ví dụ annotator_1, annotator_2)")
    else:
        for col in annot_cols:
            vals = [_parse_annotator_label(v) for v in df_strict[col].tolist()]
            if any(v is None for v in vals):
                e1_errors.append(f"Cột {col} có nhãn rỗng/NaN hoặc ngoài {{0,1,2}}")
            else:
                parsed_cols[col] = [int(v) for v in vals if v is not None]

    results: List[CheckResult] = []
    if e1_errors:
        results.append(CheckResult("E1", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(e1_errors)))
    else:
        results.append(
            CheckResult(
                "E1",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"Đọc được {len(df_strict)} dòng, tất cả id thuộc vsmec_test, nhãn hợp lệ",
                metrics={"n": int(len(df_strict)), "annotator_cols": annot_cols},
            )
        )

    # E2: 2 annotators & Cohen's kappa
    if len(parsed_cols) < 2:
        results.append(
            CheckResult(
                "E2",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"Cần nhãn hợp lệ của đủ 2 người gán để tính Cohen's kappa, chỉ thấy {list(parsed_cols.keys())}",
            )
        )
    else:
        c1, c2 = annot_cols[0], annot_cols[1]
        a, b = parsed_cols[c1], parsed_cols[c2]
        po, pe, kappa = compute_po_and_kappa(a, b)
        kappa_warn = bool(not math.isnan(kappa) and kappa < 0.60)
        warn_note = " (WARN: kappa < 0.60)" if kappa_warn else ""
        results.append(
            CheckResult(
                "E2",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"n={len(a)}, po={po:.4f}, Cohen's kappa={kappa:.4f}{warn_note}",
                metrics={
                    "n": len(a),
                    "po": float(po),
                    "pe": float(pe),
                    "kappa": float(kappa),
                    "kappa_warn": kappa_warn,
                },
            )
        )

    # E3: Strict construction rule documentation
    if not has_rule:
        results.append(
            CheckResult(
                "E3",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                "Chưa có tài liệu quy tắc dựng VSMEC-strict (docs/vsmec_strict_rule.md); cần người phụ trách xác nhận (bỏ Surprise/Other hay gán nhãn tay lại)",
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
                metrics={"n": int(len(df_strict))},
            )
        )

    return results


def check_group_e(
    cfg: Dict[str, Any],
    vsmec_dfs: Dict[str, pd.DataFrame],
) -> List[CheckResult]:
    strict_file = cfg.get("vsmec_strict_file")
    if not strict_file:
        msg_missing = (
            "vsmec_strict_file=null trong configs/step2_data.yaml. "
            "Chưa có file VSMEC-strict trên vsmec_test (các file polarity_{nhat,long,key}.csv cũ lấy từ vsmec_train cho CV-P3 nên KHÔNG dùng được). "
            "Cần người phụ trách cung cấp file gán nhãn tay hoặc quyết định quy tắc strict"
        )
        return [
            CheckResult("E1", LEVEL_BLOCKER, STATUS_MISSING, msg_missing),
            CheckResult("E2", LEVEL_BLOCKER, STATUS_MISSING, msg_missing),
            CheckResult(
                "E3",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                "Chưa chốt quy tắc VSMEC-strict (bỏ các câu Surprise/Other hay gán lại bằng nhãn tay) và chưa có docs/vsmec_strict_rule.md",
            ),
        ]

    if "test" not in vsmec_dfs:
        err = "Thiếu vsmec_test để đối chiếu id"
        return [
            CheckResult("E1", LEVEL_BLOCKER, STATUS_FAIL, err),
            CheckResult("E2", LEVEL_BLOCKER, STATUS_FAIL, err),
            CheckResult("E3", LEVEL_BLOCKER, STATUS_FAIL, err),
        ]

    has_rule = Path("docs/vsmec_strict_rule.md").is_file() or bool(cfg.get("vsmec_strict_rule"))
    return validate_vsmec_strict_file(
        strict_path=Path(str(strict_file)),
        vsmec_test_df=vsmec_dfs["test"],
        has_rule=has_rule,
    )


# ---------------------------------------------------------------------------
# Group F: Mã băm dữ liệu & MANIFEST (F1, F2, F2b, F3)
# ---------------------------------------------------------------------------
def check_group_f(
    cfg: Dict[str, Any],
    clean_dir: Path,
    neuesc_file: Path,
    loaded_results: Dict[str, Any],
) -> List[CheckResult]:
    results: List[CheckResult] = []
    vsmec_hash = compute_vsmec_data_hash(clean_dir)
    neuesc_hash = compute_neuesc_data_hash(neuesc_file)
    res_df: pd.DataFrame = loaded_results["df"]

    # F1: data_hash matches computed hashes for VSMEC & NEU-ESC
    if res_df.empty:
        results.append(
            CheckResult(
                "F1",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                f"Chưa có CSV kết quả Bước 1 trong reports/step2/inputs/ hoặc qua --results-file "
                f"(hash hiện tại: vsmec_data_hash={vsmec_hash[:12]}..., neuesc_data_hash={neuesc_hash[:12]}...)",
                metrics={
                    "current_vsmec_data_hash": vsmec_hash,
                    "current_neuesc_data_hash": neuesc_hash,
                },
            )
        )
    else:
        f1_errs: List[str] = []
        missing_groups: List[str] = []
        for grp in GROUP_ORDER:
            sub = res_df[res_df["group_key"] == grp]
            if sub.empty:
                missing_groups.append(GROUP_LABELS[grp])
                continue
            expected_h = vsmec_hash if grp.startswith("vsmec") else neuesc_hash
            actual_hashes = sorted(
                h for h in sub["data_hash"].dropna().astype(str).unique().tolist() if h
            )
            if not actual_hashes:
                f1_errs.append(f"{GROUP_LABELS[grp]} thiếu cột/giá trị data_hash")
            elif set(actual_hashes) != {expected_h}:
                f1_errs.append(
                    f"{GROUP_LABELS[grp]} lệch data_hash ({[x[:10] for x in actual_hashes]} != {expected_h[:10]})"
                )

        if f1_errs:
            results.append(
                CheckResult(
                    "F1",
                    LEVEL_BLOCKER,
                    STATUS_FAIL,
                    "; ".join(f1_errs),
                    metrics={
                        "current_vsmec_data_hash": vsmec_hash,
                        "current_neuesc_data_hash": neuesc_hash,
                        "missing_groups": missing_groups,
                    },
                )
            )
        elif missing_groups:
            results.append(
                CheckResult(
                    "F1",
                    LEVEL_BLOCKER,
                    STATUS_MISSING,
                    f"Các nhóm đã nạp khớp hash, còn thiếu dữ liệu của: {', '.join(missing_groups)}",
                    metrics={
                        "current_vsmec_data_hash": vsmec_hash,
                        "current_neuesc_data_hash": neuesc_hash,
                        "missing_groups": missing_groups,
                    },
                )
            )
        else:
            results.append(
                CheckResult(
                    "F1",
                    LEVEL_BLOCKER,
                    STATUS_PASS,
                    f"Toàn bộ 4 nhóm khớp data_hash hiện tại (vsmec={vsmec_hash[:10]}..., neuesc={neuesc_hash[:10]}...)",
                    metrics={
                        "current_vsmec_data_hash": vsmec_hash,
                        "current_neuesc_data_hash": neuesc_hash,
                    },
                )
            )

    # F2 (BLOCKER): only verify hash of `step2_files` from config against MANIFEST.json
    manifest_path = clean_dir / "MANIFEST.json"
    step2_files = [str(f) for f in (cfg.get("step2_files") or [])]
    if not manifest_path.is_file():
        results.append(
            CheckResult("F2", LEVEL_BLOCKER, STATUS_FAIL, "Thiếu data/clean/MANIFEST.json")
        )
        manifest: Dict[str, str] = {}
    elif not step2_files:
        results.append(
            CheckResult(
                "F2",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                "Thiếu khóa step2_files trong configs/step2_data.yaml",
            )
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        f2_mismatches: List[str] = []
        for fname in step2_files:
            fp = clean_dir / fname
            if not fp.is_file():
                f2_mismatches.append(f"{fname} (không tồn tại)")
            elif fname not in manifest:
                f2_mismatches.append(f"{fname} (chưa có trong MANIFEST.json)")
            else:
                actual_sha = sha256_file(fp)
                if actual_sha != manifest[fname]:
                    f2_mismatches.append(
                        f"{fname} (MANIFEST={manifest[fname][:10]}... != thực tế={actual_sha[:10]}...)"
                    )

        if f2_mismatches:
            results.append(
                CheckResult(
                    "F2",
                    LEVEL_BLOCKER,
                    STATUS_FAIL,
                    f"Lệch hash các file Step 2: {'; '.join(f2_mismatches)}",
                    metrics={"step2_files": step2_files, "mismatches": f2_mismatches},
                )
            )
        else:
            results.append(
                CheckResult(
                    "F2",
                    LEVEL_BLOCKER,
                    STATUS_PASS,
                    f"Tất cả {len(step2_files)} file Bước 2 trong step2_files khớp SHA-256 trong MANIFEST.json",
                    metrics={"step2_files": step2_files},
                )
            )

    # F2b (WARN): remaining artifacts in data/clean (polarity_*.csv, etc.)
    step2_set = set(step2_files)
    excluded_self = {"MANIFEST.json"}
    remaining_names = sorted(
        (
            {
                p.name
                for p in clean_dir.iterdir()
                if p.is_file() and p.name not in step2_set and p.name not in excluded_self
            }
            | {k for k in manifest.keys() if k not in step2_set and k not in excluded_self}
        )
    )
    artifact_rows: List[Dict[str, str]] = []
    any_f2b_diff = False
    for fname in remaining_names:
        fp = clean_dir / fname
        actual_sha = sha256_file(fp) if fp.is_file() else "không tồn tại"
        man_sha = str(manifest.get(fname, "không có"))
        if actual_sha != man_sha:
            any_f2b_diff = True
            note = "không có trong MANIFEST" if man_sha == "không có" else "lệch MANIFEST"
        else:
            note = "khớp MANIFEST"
        artifact_rows.append(
            {
                "file": fname,
                "actual_sha256": actual_sha,
                "manifest_sha256": man_sha,
                "note": note,
            }
        )

    summary_parts = [
        f"{r['file']}: thực={r['actual_sha256'][:12]}..., MANIFEST={r['manifest_sha256'][:12] + '...' if r['manifest_sha256'] != 'không có' else 'không có'}"
        for r in artifact_rows
    ]
    results.append(
        CheckResult(
            "F2b",
            LEVEL_WARN,
            STATUS_FAIL if any_f2b_diff else STATUS_PASS,
            "Artifact cũ ngoài phạm vi Bước 2 (lệch MANIFEST đã biết, không dùng cho Bước 2): "
            + "; ".join(summary_parts),
            metrics={"artifacts": artifact_rows},
        )
    )

    # F3: unique data_hash & unique code_version per Step 1 group
    if res_df.empty:
        results.append(
            CheckResult(
                "F3",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                "Chưa có kết quả Bước 1 để kiểm tra tính duy nhất của data_hash và code_version",
            )
        )
    else:
        f3_errs: List[str] = []
        f3_missing: List[str] = []
        for grp in GROUP_ORDER:
            sub = res_df[res_df["group_key"] == grp]
            if sub.empty:
                f3_missing.append(GROUP_LABELS[grp])
                continue
            dh_vals = sorted(x for x in sub["data_hash"].dropna().astype(str).unique().tolist() if x)
            cv_vals = sorted(x for x in sub["code_version"].dropna().astype(str).unique().tolist() if x)
            if len(dh_vals) != 1 or len(cv_vals) != 1:
                f3_errs.append(
                    f"{GROUP_LABELS[grp]} có {len(dh_vals)} data_hash và {len(cv_vals)} code_version"
                )

        if f3_errs:
            results.append(CheckResult("F3", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(f3_errs)))
        elif f3_missing:
            results.append(
                CheckResult(
                    "F3",
                    LEVEL_BLOCKER,
                    STATUS_MISSING,
                    f"Các nhóm đã nạp chỉ có 1 data_hash & 1 code_version; còn thiếu: {', '.join(f3_missing)}",
                )
            )
        else:
            results.append(
                CheckResult(
                    "F3",
                    LEVEL_BLOCKER,
                    STATUS_PASS,
                    "Mỗi nhóm trong 4 nhóm chỉ có duy nhất 1 data_hash và 1 code_version",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Group G: Cổng Bước 1 (BLOCKER / WARN)
# ---------------------------------------------------------------------------
def check_group_g(
    cfg: Dict[str, Any],
    loaded_results: Dict[str, Any],
) -> List[CheckResult]:
    results: List[CheckResult] = []
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
    seed_set = set(seeds)
    res_df: pd.DataFrame = loaded_results["df"]

    # G1: Per-group counts (5 seeds each with BOTH dev & test, after sweep filtering)
    group_counts: Dict[str, Dict[str, Any]] = {}
    for grp in GROUP_ORDER:
        if res_df.empty:
            completed_seeds: List[int] = []
        else:
            sub = res_df[res_df["group_key"] == grp]
            completed_seeds = []
            for s in seeds:
                s_rows = sub[sub["seed"].astype(int) == s]
                s_splits = set(s_rows["split"].astype(str).tolist())
                if {"dev", "test"}.issubset(s_splits):
                    completed_seeds.append(s)
        group_counts[grp] = {
            "label": GROUP_LABELS[grp],
            "completed": len(completed_seeds),
            "target": len(seeds),
            "completed_seeds": completed_seeds,
            "missing_seeds": [s for s in seeds if s not in completed_seeds],
        }

    per_group_summary = "; ".join(
        f"{group_counts[g]['label']}: {group_counts[g]['completed']}/{group_counts[g]['target']}"
        for g in GROUP_ORDER
    )
    total_completed = sum(group_counts[g]["completed"] for g in GROUP_ORDER)
    total_target = sum(group_counts[g]["target"] for g in GROUP_ORDER)

    if res_df.empty:
        results.append(
            CheckResult(
                "G1",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                f"Chưa có kết quả Bước 1 (tổng {total_completed}/{total_target} lượt: {per_group_summary}; cần người phụ trách tải stage1.csv và neu3.csv của 2 backbone vào reports/step2/inputs/)",
                metrics={"groups": group_counts, "n_sweep_filtered": loaded_results["n_sweep_filtered"]},
            )
        )
    elif total_completed < total_target:
        missing_detail = "; ".join(
            f"{group_counts[g]['label']} thiếu seed {group_counts[g]['missing_seeds']}"
            for g in GROUP_ORDER
            if group_counts[g]["missing_seeds"]
        )
        results.append(
            CheckResult(
                "G1",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"Chưa đủ {total_target} lượt baseline ({total_completed}/{total_target}): {per_group_summary} ({missing_detail})",
                metrics={"groups": group_counts, "n_sweep_filtered": loaded_results["n_sweep_filtered"]},
            )
        )
    else:
        results.append(
            CheckResult(
                "G1",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"Đủ {total_completed}/{total_target} lượt baseline (mỗi lượt đủ dev & test, đã lọc {loaded_results['n_sweep_filtered']} dòng sweep): {per_group_summary}",
                metrics={"groups": group_counts, "n_sweep_filtered": loaded_results["n_sweep_filtered"]},
            )
        )

    # G2: vsmec_published_macro_f1: {phobert: ..., visobert: ...}
    pub_cfg = cfg.get("vsmec_published_macro_f1") or {}
    if not isinstance(pub_cfg, dict):
        pub_cfg = {"phobert": pub_cfg, "visobert": pub_cfg}

    g2_metrics: Dict[str, Any] = {}
    g2_parts: List[str] = []
    g2_missing_reasons: List[str] = []

    for bb_key, bb_name in (("phobert", "PhoBERT"), ("visobert", "ViSoBERT")):
        pub_val = pub_cfg.get(bb_key)
        grp = f"vsmec_{bb_key}"
        test_sub = (
            res_df[
                (res_df["group_key"] == grp)
                & (res_df["split"] == "test")
                & (res_df["seed"].isin(seed_set))
            ]
            if not res_df.empty
            else pd.DataFrame()
        )
        if pub_val is None:
            g2_missing_reasons.append(f"vsmec_published_macro_f1.{bb_key}=null")
        if test_sub.empty or test_sub["macro_f1"].dropna().empty:
            g2_missing_reasons.append(f"thiếu kết quả test VSMEC {bb_name}")
        if pub_val is not None and not test_sub.empty and not test_sub["macro_f1"].dropna().empty:
            vals = test_sub["macro_f1"].dropna().astype(float)
            mean_f1 = float(vals.mean())
            std_f1 = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            pv = float(pub_val)
            diff = mean_f1 - pv
            g2_metrics[bb_key] = {
                "n_seeds": int(len(vals)),
                "mean_test_macro_f1": mean_f1,
                "std_test_macro_f1": std_f1,
                "published_macro_f1": pv,
                "diff": diff,
            }
            g2_parts.append(
                f"{bb_name}: mean={mean_f1:.4f} ± {std_f1:.4f} vs công bố={pv:.4f} (lệch={diff:+.4f})"
            )

    if g2_missing_reasons:
        prefix = f"{'; '.join(g2_parts)}. " if g2_parts else ""
        results.append(
            CheckResult(
                "G2",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                f"{prefix}Thiếu đầu vào đối chiếu G2: {', '.join(g2_missing_reasons)}",
                metrics=g2_metrics,
            )
        )
    else:
        results.append(
            CheckResult(
                "G2",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "; ".join(g2_parts),
                metrics=g2_metrics,
            )
        )

    # G3 (WARN): std across seeds & mean per-class F1 >= 0.30
    if res_df.empty or any(group_counts[g]["completed"] == 0 for g in GROUP_ORDER):
        missing_g3 = [
            GROUP_LABELS[g] for g in GROUP_ORDER if group_counts[g]["completed"] == 0
        ]
        results.append(
            CheckResult(
                "G3",
                LEVEL_WARN,
                STATUS_MISSING,
                f"Chưa đủ kết quả để kiểm tra std và F1 từng lớp >= 0.30 (thiếu: {', '.join(missing_g3)})",
            )
        )
    else:
        g3_warns: List[str] = []
        g3_stats: Dict[str, Any] = {}
        f1_cols_7 = [f"f1_{emo.lower()}" for emo in EMOTIONS7]
        f1_cols_3 = ["f1_neg", "f1_neu", "f1_pos"]

        for grp in GROUP_ORDER:
            sub_t = res_df[(res_df["group_key"] == grp) & (res_df["split"] == "test")]
            if sub_t.empty:
                sub_t = res_df[res_df["group_key"] == grp]
            vals = sub_t["macro_f1"].dropna().astype(float)
            std_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            if len(vals) > 1 and std_val == 0.0:
                g3_warns.append(f"{GROUP_LABELS[grp]} macro_f1 std=0 giữa các seed")

            cls_means: Dict[str, float] = {}
            target_cols = f1_cols_7 if grp.startswith("vsmec") else f1_cols_3
            for col in target_cols:
                if col in sub_t.columns and not sub_t[col].dropna().empty:
                    mval = float(sub_t[col].dropna().astype(float).mean())
                    cls_means[col] = mval
                    if mval < 0.30:
                        g3_warns.append(f"{GROUP_LABELS[grp]} {col} mean={mval:.4f} < 0.30")
            if grp.startswith("vsmec") and not cls_means:
                for col in f1_cols_3:
                    if col in sub_t.columns and not sub_t[col].dropna().empty:
                        mval = float(sub_t[col].dropna().astype(float).mean())
                        cls_means[col] = mval
                        if mval < 0.30:
                            g3_warns.append(f"{GROUP_LABELS[grp]} {col} mean={mval:.4f} < 0.30")

            g3_stats[grp] = {
                "n_rows": int(len(vals)),
                "macro_f1_mean": float(vals.mean()) if len(vals) else None,
                "macro_f1_std": std_val,
                "class_f1_means": cls_means,
            }

        std_summary = ", ".join(
            f"{GROUP_LABELS[g]} std={g3_stats[g]['macro_f1_std']:.4f}" for g in GROUP_ORDER
        )
        if g3_warns:
            results.append(
                CheckResult(
                    "G3",
                    LEVEL_WARN,
                    STATUS_FAIL,
                    f"Cảnh báo Bước 1 ({std_summary}): {'; '.join(g3_warns)}",
                    metrics=g3_stats,
                )
            )
        else:
            results.append(
                CheckResult(
                    "G3",
                    LEVEL_WARN,
                    STATUS_PASS,
                    f"Các nhóm đều có std > 0 giữa seed ({std_summary}) và mọi lớp có F1 TB >= 0.30",
                    metrics=g3_stats,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Group H: Phân bố, độ lệch tiên nghiệm & độ lệch bề mặt (INFO / WARN)
# ---------------------------------------------------------------------------
def _compute_surface_stats(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["text"].astype(str).tolist()
    n = max(1, len(texts))
    ends_space_dot = sum(1 for t in texts if t.endswith(" .")) / n
    ends_dot = sum(1 for t in texts if t.endswith(".")) / n
    has_emoji = sum(1 for t in texts if contains_unicode_emoji(t)) / n
    word_counts = [len(t.split()) for t in texts]
    all_lower = sum(1 for t in texts if t == t.lower()) / n
    return {
        "n": int(len(texts)),
        "ends_with_space_dot_rate": float(ends_space_dot),
        "ends_with_dot_rate": float(ends_dot),
        "has_emoji_rate": float(has_emoji),
        "word_count_mean": float(np.mean(word_counts)) if word_counts else 0.0,
        "word_count_p95": float(np.percentile(word_counts, 95)) if word_counts else 0.0,
        "all_lowercase_rate": float(all_lower),
    }


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
            CheckResult("H4", LEVEL_INFO, STATUS_FAIL, "Thiếu dữ liệu đầu vào"),
        ]

    # H1
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

    # H2
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

    # H3
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

    # H4 (INFO): surface statistics per split for both domains (numbers only, no conclusions)
    surface_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {
        "vsmec": {sp: _compute_surface_stats(vsmec_dfs[sp]) for sp in VSMEC_SPLITS},
        "neuesc": {
            sp: _compute_surface_stats(neuesc_df[neuesc_df["subset"] == sp])
            for sp in NEUESC_SPLITS
        },
    }
    vt = surface_metrics["vsmec"]["test"]
    nt = surface_metrics["neuesc"]["test"]
    results.append(
        CheckResult(
            "H4",
            LEVEL_INFO,
            STATUS_PASS,
            f"Lệch bề mặt test: vsmec_test (' .'={vt['ends_with_space_dot_rate']*100:.2f}%, "
            f"emoji={vt['has_emoji_rate']*100:.2f}%, words={vt['word_count_mean']:.2f}/p95={vt['word_count_p95']:.1f}, "
            f"lower={vt['all_lowercase_rate']*100:.2f}%) vs neuesc_test (' .'={nt['ends_with_space_dot_rate']*100:.2f}%, "
            f"emoji={nt['has_emoji_rate']*100:.2f}%, words={nt['word_count_mean']:.2f}/p95={nt['word_count_p95']:.1f}, "
            f"lower={nt['all_lowercase_rate']*100:.2f}%)",
            metrics={"surface_stats": surface_metrics},
        )
    )

    return results


# ---------------------------------------------------------------------------
# Group I: Đầu vào của mô hình (WARN)
# ---------------------------------------------------------------------------
def _load_tokenizer_safe(model_name: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name)


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

    # I1
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

    # I2: Truncation rate @256
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
            tok_pho = _load_tokenizer_safe("vinai/phobert-base-v2")
            tok_viso = _load_tokenizer_safe("uitnlp/visobert")
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
            vt = trunc_metrics["vsmec_test"]
            nt = trunc_metrics["neuesc_test"]
            results.append(
                CheckResult(
                    "I2",
                    LEVEL_WARN,
                    STATUS_PASS,
                    f"Cắt cụt @256: vsmec_test (PhoBERT={vt['phobert_n_truncated']}/{vt['n_total']} [{vt['phobert_rate']*100:.2f}%], "
                    f"ViSoBERT={vt['visobert_n_truncated']}/{vt['n_total']} [{vt['visobert_rate']*100:.2f}%]); "
                    f"neuesc_test (PhoBERT={nt['phobert_n_truncated']}/{nt['n_total']} [{nt['phobert_rate']*100:.2f}%], "
                    f"ViSoBERT={nt['visobert_n_truncated']}/{nt['n_total']} [{nt['visobert_rate']*100:.2f}%])",
                    metrics=trunc_metrics,
                )
            )
        except Exception as exc:
            results.append(
                CheckResult(
                    "I2",
                    LEVEL_WARN,
                    STATUS_SKIP,
                    f"Bỏ qua I2 do không tải được tokenizer từ cache/mạng: {exc}",
                )
            )

    # I3
    non_idempotent = 0
    for df_t in (v_test, n_test):
        for t in df_t["text"].astype(str):
            once = normalize_text(t)
            if normalize_text(once) != once:
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

    # I4
    v_keep = [preprocess_text(str(t), emoji_mode="keep") for t in v_test["text"]]
    n_keep = [preprocess_text(str(t), emoji_mode="keep") for t in n_test["text"]]
    v_empty = sum(1 for s in v_keep if not s.strip())
    n_empty = sum(1 for s in n_keep if not s.strip())
    v_idemp_fail = sum(1 for s in v_keep if preprocess_text(s, emoji_mode="keep") != s)
    n_idemp_fail = sum(1 for s in n_keep if preprocess_text(s, emoji_mode="keep") != s)
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
# Group J: Checkpoint & --reload-check (BLOCKER, chỉ chạy khi --check-ckpt)
# ---------------------------------------------------------------------------
def _locate_group_seed_ckpt(
    root: Path,
    alias: str,
    seeds: Sequence[int],
) -> Dict[int, Optional[Path]]:
    """Locate distinct per-seed checkpoint directories containing `encoder/` and `head.pt`."""
    found: Dict[int, Optional[Path]] = {}
    used_paths: set[Path] = set()
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()] if root.is_dir() else []
    alias_dir = root / alias
    alias_subdirs = (
        [d for d in sorted(alias_dir.iterdir()) if d.is_dir()]
        if alias_dir.is_dir()
        else []
    )

    for s in seeds:
        candidates = [
            root / f"seed_{s}" / alias,
            root / f"s{s}" / alias,
            root / f"seed{s}" / alias,
            root / f"seed_{s}",
            root / f"s{s}",
            root / f"seed{s}",
            root / alias / f"seed_{s}",
            root / alias / f"s{s}",
            root / alias / f"seed{s}",
            root / f"{alias}_s{s}",
            root / f"{alias}_seed_{s}",
        ]
        seed_pat = re.compile(rf"(?:^|[_\-])s(?:eed)?_?{s}(?:$|[_\-])", flags=re.IGNORECASE)
        for d in subdirs:
            if seed_pat.search(d.name):
                candidates.append(d / alias)
                candidates.append(d)
        for d in alias_subdirs:
            if seed_pat.search(d.name):
                candidates.append(d)

        matched: Optional[Path] = None
        for cand in candidates:
            if (cand / "encoder").is_dir() and (cand / "head.pt").is_file():
                rc = cand.resolve()
                if rc not in used_paths:
                    matched = cand
                    used_paths.add(rc)
                    break
        found[int(s)] = matched
    return found


def _find_prediction_csv_for_group(
    grp_name: str,
    ckpt_path: Path,
    search_dirs: Iterable[Path],
    loaded_results_df: Optional[pd.DataFrame] = None,
) -> Optional[Path]:
    domain, alias, _ = GROUP_SPECS[grp_name]
    alias_low = alias.lower()
    split_tag = "vsmec_dev" if domain == "vsmec" else "neuesc_dev"
    sub_dir_name = "predictions7" if domain == "vsmec" else "predictions3"

    candidate_dirs: List[Path] = []
    for base in (ckpt_path.parent, ckpt_path.parent.parent, *search_dirs):
        for p in (base, base / sub_dir_name, base.parent / sub_dir_name):
            if p.is_dir() and p not in candidate_dirs:
                candidate_dirs.append(p)

    # First try matching run_ids from loaded_results_df for this group
    run_ids: List[str] = []
    if loaded_results_df is not None and not loaded_results_df.empty:
        sub = loaded_results_df[loaded_results_df["group_key"] == grp_name]
        run_ids = [str(r) for r in sub["run_id"].dropna().unique().tolist()]

    all_csvs: List[Path] = []
    for d in candidate_dirs:
        for fp in sorted(d.glob("*.csv")):
            name_low = fp.name.lower()
            if split_tag in name_low and (alias_low in name_low or any(rid.lower() in name_low for rid in run_ids)):
                return fp
            if split_tag in name_low:
                all_csvs.append(fp)

    for fp in all_csvs:
        if alias_low in str(fp).lower():
            return fp
    return None


def _run_reload_check_for_group(
    grp_name: str,
    ckpt_path: Path,
    vsmec_dfs: Dict[str, pd.DataFrame],
    neuesc_df: Optional[pd.DataFrame],
    pred_csv: Path,
    n_samples: int = 16,
) -> Tuple[bool, float, str]:
    """Reload 1 encoder + head.pt in FP32, run on 16 dev sentences, compare logits vs saved CSV."""
    from transformers import AutoModel, AutoTokenizer

    domain, alias, n_labels = GROUP_SPECS[grp_name]
    if domain == "vsmec":
        dev_df = vsmec_dfs["dev"]
        logit_cols = [f"logit_{i}" for i in range(7)]
    else:
        assert neuesc_df is not None
        dev_df = neuesc_df[neuesc_df["subset"] == "dev"].reset_index(drop=True)
        logit_cols = ["logit_neg", "logit_neu", "logit_pos"]

    pred_df = pd.read_csv(pred_csv, encoding="utf-8")
    missing_lcols = [c for c in logit_cols if c not in pred_df.columns]
    if missing_lcols:
        return False, float("nan"), f"{pred_csv.name} thiếu cột logit {missing_lcols}"

    dev_sub = dev_df.iloc[:n_samples].copy()
    if "id" in pred_df.columns:
        pred_indexed = pred_df.set_index("id")
        try:
            saved_logits = pred_indexed.loc[dev_sub["id"].tolist(), logit_cols].to_numpy(
                dtype=np.float64
            )
        except KeyError:
            saved_logits = pred_df.iloc[: len(dev_sub)][logit_cols].to_numpy(dtype=np.float64)
    else:
        saved_logits = pred_df.iloc[: len(dev_sub)][logit_cols].to_numpy(dtype=np.float64)

    if alias == "PhoBERT":
        texts = dev_sub["text_seg"].astype(str).tolist()
    else:
        texts = [preprocess_text(str(t), emoji_mode="keep") for t in dev_sub["text"].tolist()]

    encoder_dir = ckpt_path / "encoder"
    head_pt = ckpt_path / "head.pt"

    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
    encoder = AutoModel.from_pretrained(str(encoder_dir))
    encoder.eval()
    head_state = torch.load(head_pt, map_location="cpu", weights_only=True)
    weight = head_state["weight"].detach().float()
    bias = head_state["bias"].detach().float()

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = encoder(**enc)
        cls_repr = outputs.last_hidden_state[:, 0, :].float()
        logits = (cls_repr @ weight.T + bias).cpu().numpy().astype(np.float64)

    if logits.shape != saved_logits.shape:
        return (
            False,
            float("nan"),
            f"Lệch shape logit: recomputed={logits.shape} vs saved={saved_logits.shape}",
        )

    max_diff = float(np.max(np.abs(logits - saved_logits)))
    ok = bool(max_diff <= RELOAD_MAX_ABS_DIFF_TOL)
    return ok, max_diff, f"max_abs_diff={max_diff:.2e} (<= {RELOAD_MAX_ABS_DIFF_TOL:.0e})"


def check_group_j(
    cfg: Dict[str, Any],
    results_dir: Path,
    check_ckpt: bool = False,
    reload_check: bool = False,
    vsmec_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    neuesc_df: Optional[pd.DataFrame] = None,
    loaded_results: Optional[Dict[str, Any]] = None,
) -> List[CheckResult]:
    if not check_ckpt:
        skip_msg = "Bỏ qua kiểm tra checkpoint (chỉ chạy khi truyền --check-ckpt)"
        return [
            CheckResult("J1", LEVEL_BLOCKER, STATUS_SKIP, skip_msg),
            CheckResult("J2", LEVEL_BLOCKER, STATUS_SKIP, skip_msg),
            CheckResult("J3", LEVEL_BLOCKER, STATUS_SKIP, skip_msg),
        ]

    ckpt_roots = cfg.get("ckpt_roots") or {}
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]

    j1_group_summary: Dict[str, Any] = {}
    j1_issues: List[str] = []
    j1_nulls: List[str] = []
    j2_errors: List[str] = []
    located_dirs: Dict[str, Dict[int, Path]] = {}

    for grp_name in GROUP_ORDER:
        _, alias, expected_out_dim = GROUP_SPECS[grp_name]
        root_val = ckpt_roots.get(grp_name)
        if not root_val:
            j1_nulls.append(grp_name)
            j1_group_summary[grp_name] = {
                "found": 0,
                "target": len(seeds),
                "missing_seeds": seeds,
                "status": "null",
            }
            continue

        root_path = Path(str(root_val))
        if not root_path.is_dir():
            j1_issues.append(f"{grp_name}: thư mục không tồn tại ({root_path.as_posix()})")
            j1_group_summary[grp_name] = {
                "found": 0,
                "target": len(seeds),
                "missing_seeds": seeds,
                "status": "missing_dir",
            }
            continue

        seed_map = _locate_group_seed_ckpt(root_path, alias, seeds)
        found_map = {s: p for s, p in seed_map.items() if p is not None}
        missing_seeds = [s for s, p in seed_map.items() if p is None]
        located_dirs[grp_name] = found_map
        j1_group_summary[grp_name] = {
            "found": len(found_map),
            "target": len(seeds),
            "missing_seeds": missing_seeds,
            "status": "ok" if not missing_seeds else "incomplete",
        }
        if missing_seeds:
            j1_issues.append(
                f"{GROUP_LABELS[grp_name]}: {len(found_map)}/{len(seeds)} (thiếu seed {missing_seeds})"
            )

        for ckpt_p in found_map.values():
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

    results: List[CheckResult] = []
    counts_str = "; ".join(
        f"{GROUP_LABELS[g]}: {j1_group_summary[g]['found']}/{j1_group_summary[g]['target']}"
        for g in GROUP_ORDER
    )

    if len(j1_nulls) == len(GROUP_ORDER):
        msg_null = (
            f"ckpt_roots đều null ({counts_str}); cần truyền --ckpt-root-<nhóm> hoặc điền configs/step2_data.yaml"
        )
        return [
            CheckResult("J1", LEVEL_BLOCKER, STATUS_MISSING, msg_null, metrics=j1_group_summary),
            CheckResult("J2", LEVEL_BLOCKER, STATUS_MISSING, msg_null, metrics=j1_group_summary),
            CheckResult("J3", LEVEL_BLOCKER, STATUS_MISSING, msg_null, metrics=j1_group_summary),
        ]

    if j1_issues:
        extra_null = f"; nhóm chưa cấu hình: {j1_nulls}" if j1_nulls else ""
        results.append(
            CheckResult(
                "J1",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                f"{counts_str} — {'; '.join(j1_issues)}{extra_null}",
                metrics=j1_group_summary,
            )
        )
    elif j1_nulls:
        results.append(
            CheckResult(
                "J1",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                f"{counts_str} — thiếu --ckpt-root cho: {', '.join(j1_nulls)}",
                metrics=j1_group_summary,
            )
        )
    else:
        results.append(
            CheckResult(
                "J1",
                LEVEL_BLOCKER,
                STATUS_PASS,
                f"Đủ 4 nhóm × {len(seeds)} seed có encoder/ và head.pt ({counts_str})",
                metrics=j1_group_summary,
            )
        )

    # J2
    has_any_ckpt = any(bool(v) for v in located_dirs.values())
    if not has_any_ckpt:
        results.append(
            CheckResult(
                "J2",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                "Không tìm thấy head.pt nào để kiểm tra kích thước lớp đầu ra",
            )
        )
    elif j2_errors:
        results.append(CheckResult("J2", LEVEL_BLOCKER, STATUS_FAIL, "; ".join(j2_errors)))
    else:
        results.append(
            CheckResult(
                "J2",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "Tất cả head.pt đã tìm thấy có weight.shape[0] == 7 (VSMEC) và 3 (NEU-3)",
            )
        )

    # J3: Checkpoint hashes and/or --reload-check (16 dev samples, max_abs_diff <= 2e-3)
    hash_files = [
        results_dir / "stage1_ckpt_hash.json",
        results_dir / "neu3_ckpt_hash.json",
    ]
    existing_hash_files = [p for p in hash_files if p.is_file()]
    j3_hash_errs: List[str] = []
    if existing_hash_files:
        for hf in existing_hash_files:
            payload = json.loads(hf.read_text(encoding="utf-8"))
            for key, info in payload.items():
                ckpt_p = Path(str(info.get("path", "")))
                head_pt = ckpt_p / "head.pt"
                if not head_pt.is_file():
                    j3_hash_errs.append(f"{hf.name}[{key}] thiếu {head_pt.as_posix()}")
                    continue
                if sha256_file(head_pt) != info.get("head_sha256"):
                    j3_hash_errs.append(f"{hf.name}[{key}] lệch head_sha256")

    reload_metrics: Dict[str, Any] = {}
    reload_errs: List[str] = []
    reload_msgs: List[str] = []

    if reload_check:
        if not vsmec_dfs or neuesc_df is None:
            reload_errs.append("Thiếu dữ liệu dev để chạy --reload-check")
        else:
            inputs_dir = Path(str(cfg.get("results_inputs_dir", "reports/step2/inputs")))
            res_df = loaded_results["df"] if loaded_results else pd.DataFrame()
            search_dirs = [results_dir, inputs_dir]
            for grp_name in GROUP_ORDER:
                found_seeds = located_dirs.get(grp_name, {})
                if not found_seeds:
                    reload_msgs.append(f"{GROUP_LABELS[grp_name]}: thiếu checkpoint")
                    continue
                first_seed = sorted(found_seeds.keys())[0]
                ckpt_p = found_seeds[first_seed]
                pred_csv = _find_prediction_csv_for_group(
                    grp_name, ckpt_p, search_dirs, loaded_results_df=res_df
                )
                if pred_csv is None:
                    reload_errs.append(
                        f"{GROUP_LABELS[grp_name]} (s{first_seed}): không tìm thấy file predictions dev để so logit"
                    )
                    continue
                ok, max_diff, note = _run_reload_check_for_group(
                    grp_name, ckpt_p, vsmec_dfs, neuesc_df, pred_csv
                )
                reload_metrics[grp_name] = {
                    "seed": first_seed,
                    "pred_csv": pred_csv.as_posix(),
                    "max_abs_diff": max_diff if not math.isnan(max_diff) else None,
                    "ok": ok,
                }
                if not ok:
                    reload_errs.append(f"{GROUP_LABELS[grp_name]} (s{first_seed}): {note}")
                else:
                    reload_msgs.append(f"{GROUP_LABELS[grp_name]} (s{first_seed}): {note}")

    if j3_hash_errs or reload_errs:
        all_errs = j3_hash_errs + reload_errs
        results.append(
            CheckResult(
                "J3",
                LEVEL_BLOCKER,
                STATUS_FAIL,
                "; ".join(all_errs),
                metrics={"reload_check": reload_metrics},
            )
        )
    elif reload_check and reload_msgs and len(reload_metrics) == len(GROUP_ORDER):
        results.append(
            CheckResult(
                "J3",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "Reload-check 16 câu dev khớp <= 2e-3: " + "; ".join(reload_msgs),
                metrics={"reload_check": reload_metrics},
            )
        )
    elif existing_hash_files and not reload_check:
        results.append(
            CheckResult(
                "J3",
                LEVEL_BLOCKER,
                STATUS_PASS,
                "Hash checkpoint trong JSON khớp đĩa (truyền thêm --reload-check để so logit 16 câu dev)",
            )
        )
    elif reload_check and reload_msgs:
        results.append(
            CheckResult(
                "J3",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                f"Reload-check chưa đủ 4 nhóm: {'; '.join(reload_msgs)}",
                metrics={"reload_check": reload_metrics},
            )
        )
    else:
        results.append(
            CheckResult(
                "J3",
                LEVEL_BLOCKER,
                STATUS_MISSING,
                "Chưa có stage1_ckpt_hash.json / neu3_ckpt_hash.json và chưa chạy --reload-check với file predictions dev",
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
    reload_check: bool = False,
    results_files: Optional[Sequence[Union[str, Path]]] = None,
    results_globs: Optional[Sequence[str]] = None,
    ckpt_roots_override: Optional[Dict[str, Optional[Union[str, Path]]]] = None,
    cfg_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all checks A–J, write reports & violation CSVs to `reports_dir`, and return payload."""
    cfg = load_step2_config(config_path)
    if cfg_override:
        cfg.update(cfg_override)
    if ckpt_roots_override:
        merged_cr = dict(cfg.get("ckpt_roots") or {})
        for k, v in ckpt_roots_override.items():
            if v is not None:
                merged_cr[k] = str(v)
        cfg["ckpt_roots"] = merged_cr

    clean_dir = Path(str(cfg["vsmec_clean_dir"]))
    neuesc_file = Path(str(cfg["neuesc_file"]))
    results_dir = Path(str(cfg.get("results_dir", "results")))
    inputs_dir = Path(str(cfg.get("results_inputs_dir", "reports/step2/inputs")))
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    loaded_results = load_step1_results(
        results_files=results_files,
        results_globs=results_globs,
        default_inputs_dir=inputs_dir,
        fallback_results_dir=results_dir,
    )

    all_results: List[CheckResult] = []
    res_a, vsmec_dfs, neuesc_df = check_group_a(clean_dir, neuesc_file)
    all_results.extend(res_a)
    all_results.extend(check_group_b(neuesc_df))
    all_results.extend(check_group_c(vsmec_dfs, neuesc_df, rep_dir))
    all_results.extend(check_group_d(vsmec_dfs, neuesc_df, rep_dir))
    all_results.extend(check_group_e(cfg, vsmec_dfs))
    all_results.extend(check_group_f(cfg, clean_dir, neuesc_file, loaded_results))
    all_results.extend(check_group_g(cfg, loaded_results))
    all_results.extend(check_group_h(vsmec_dfs, neuesc_df))
    all_results.extend(
        check_group_i(vsmec_dfs, neuesc_df, with_tokenizer=with_tokenizer)
    )
    all_results.extend(
        check_group_j(
            cfg,
            results_dir,
            check_ckpt=check_ckpt,
            reload_check=reload_check,
            vsmec_dfs=vsmec_dfs,
            neuesc_df=neuesc_df,
            loaded_results=loaded_results,
        )
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
        "loaded_result_files": loaded_results["files"],
        "n_sweep_filtered": loaded_results["n_sweep_filtered"],
        "checks": [asdict(r) for r in all_results],
        "d3_overlap": by_id["D3"].metrics if "D3" in by_id else {},
        "d4_near_duplicates": by_id["D4"].metrics if "D4" in by_id else {},
        "f2b_artifacts": by_id["F2b"].metrics if "F2b" in by_id else {},
        "h1_distributions": by_id["H1"].metrics if "H1" in by_id else {},
        "h2_js_and_counts": by_id["H2"].metrics if "H2" in by_id else {},
        "h3_surprise_other": by_id["H3"].metrics if "H3" in by_id else {},
        "h4_surface_stats": by_id["H4"].metrics if "H4" in by_id else {},
        "i2_truncation": by_id["I2"].metrics if "I2" in by_id else {},
        "e2_kappa": by_id["E2"].metrics if "E2" in by_id else {},
        "g1_counts": by_id["G1"].metrics if "G1" in by_id else {},
        "g2_published_diff": by_id["G2"].metrics if "G2" in by_id else {},
        "f1_hashes": by_id["F1"].metrics if "F1" in by_id else {},
    }

    (rep_dir / "readiness_report.json").write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(rep_dir / "readiness_report.md", report_payload, all_results)

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

    d4_data = payload.get("d4_near_duplicates", {})
    near_dups = d4_data.get("near_duplicates", [])
    impact = d4_data.get("impact", {})
    lines.extend(
        [
            "",
            f"### 2.2. Câu trùng gần phát hiện ở D4 (hạn chế đã biết & kiểm tra độ nhạy): `{len(near_dups)}` cặp",
            "",
        ]
    )
    if near_dups:
        lines.extend(
            [
                "| Phía Test | ID | Nhãn gốc (7c/3c) | Nhãn B1 (3c) | Phía đối chiếu | ID khớp | Nhãn gốc đối chiếu | Nhãn B1 đối chiếu | Khớp nhãn 3c? | Chuỗi chuẩn hóa mạnh |",
                "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for nd in near_dups:
            lines.append(
                f"| `{nd['source_split']}` | {nd['id']} | `{nd['orig_label_name']}` ({nd['orig_label']}) | "
                f"`{nd['mapped_3class_name']}` ({nd['mapped_3class_label']}) | `{nd['matched_split']}` | "
                f"`{nd['matched_ids']}` | `{nd['matched_orig_label_name']}` ({nd['matched_orig_label']}) | "
                f"`{nd['matched_3class_name']}` ({nd['matched_3class_label']}) | **{nd['label_3class_match']}** | `{nd['strong_norm']}` |"
            )
        lines.extend(
            [
                "",
                f"- **Tỉ lệ câu bị ảnh hưởng nếu kiểm tra độ nhạy:** `vsmec_test` = **{impact.get('vsmec_test_affected', 0)}/{impact.get('vsmec_test_total', 692)}** "
                f"({impact.get('vsmec_test_rate', 0.0)*100:.4f}%); `neuesc_test` = **{impact.get('neuesc_test_affected', 0)}/{impact.get('neuesc_test_total', 6613)}** "
                f"({impact.get('neuesc_test_rate', 0.0)*100:.4f}%).",
                "- **Quyết định đã chốt:** Giữ nguyên dữ liệu đóng băng, không loại bỏ câu nào khỏi tập test gốc. Hai ID vi phạm được xuất ra [sensitivity_exclude_ids.csv](sensitivity_exclude_ids.csv) để Bước 2 chạy thêm một lượt độ nhạy song song với lượt đầy đủ.",
            ]
        )

    # Section 3: H1 - H4
    h1 = payload.get("h1_distributions", {})
    h2 = payload.get("h2_js_and_counts", {})
    h3 = payload.get("h3_surprise_other", {})
    h4 = payload.get("h4_surface_stats", {}).get("surface_stats", {})
    if h1:
        lines.extend(
            [
                "",
                "## 3. Phân bố nhãn, độ lệch tiên nghiệm và độ lệch bề mặt (H1 – H4)",
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

    if h4:
        lines.extend(
            [
                "",
                "### 3.4. Lệch bề mặt giữa hai miền (H4)",
                "",
                "| Bộ dữ liệu | Split | Số câu | Kết thúc `\" .\"` | Kết thúc `\".\"` | Có emoji Unicode | Số từ TB | Số từ p95 | Toàn chữ thường |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for dom_key, dom_label in (("vsmec", "VSMEC"), ("neuesc", "NEU-ESC")):
            for sp in ("train", "dev", "test"):
                st = h4.get(dom_key, {}).get(sp, {})
                if st:
                    lines.append(
                        f"| {dom_label} | `{sp}` | {st['n']} | {st['ends_with_space_dot_rate']*100:.2f}% | "
                        f"{st['ends_with_dot_rate']*100:.2f}% | {st['has_emoji_rate']*100:.2f}% | "
                        f"{st['word_count_mean']:.2f} | {st['word_count_p95']:.1f} | {st['all_lowercase_rate']*100:.2f}% |"
                    )
        lines.extend(
            [
                "",
                "Hiệu ứng này cần được xét ở phần \"độ lệch nội tại\" của Bước 2.",
            ]
        )

    # Section 4: F2b & E/G/I2
    f2b_rows = payload.get("f2b_artifacts", {}).get("artifacts", [])
    if f2b_rows:
        lines.extend(
            [
                "",
                "## 4. Artifact cũ ngoài phạm vi Bước 2 trong `data/clean/` (F2b)",
                "",
                "> Các file dưới đây là artifact cũ (CV-P3 / mở rộng), **không** được Bước 2 sử dụng. Sai khác với `MANIFEST.json` là đã biết và giữ nguyên `MANIFEST.json` theo quyết định.",
                "",
                "| File | SHA-256 thực tế | SHA-256 trong `MANIFEST.json` | Ghi chú |",
                "| --- | --- | --- | --- |",
            ]
        )
        for ar in f2b_rows:
            man_disp = f"`{ar['manifest_sha256']}`" if ar["manifest_sha256"] != "không có" else "không có"
            lines.append(
                f"| `{ar['file']}` | `{ar['actual_sha256']}` | {man_disp} | {ar['note']} |"
            )

    lines.extend(
        [
            "",
            "## 5. Đoạn mã Colab kiểm tra checkpoint & tokenizer trên Drive (nhóm I2 & J1–J3)",
            "",
            "```python",
            "from google.colab import drive",
            "drive.mount('/content/drive')",
            "",
            "# Chỉnh lại đường dẫn repo & thư mục Drive thực tế của bạn:",
            "%cd /content/Xdomain-Transfer-Learning",
            "!git checkout p1",
            "",
            "!python -m scripts.check_step2_data \\",
            "  --results-glob '/content/drive/MyDrive/xdomain_results/**/*.csv' \\",
            "  --with-tokenizer \\",
            "  --check-ckpt \\",
            "  --reload-check \\",
            "  --ckpt-root-vsmec-phobert '/content/drive/MyDrive/xdomain_ckpts/stage1' \\",
            "  --ckpt-root-vsmec-visobert '/content/drive/MyDrive/xdomain_ckpts/stage1' \\",
            "  --ckpt-root-neu3-phobert '/content/drive/MyDrive/xdomain_ckpts/neu3' \\",
            "  --ckpt-root-neu3-visobert '/content/drive/MyDrive/xdomain_ckpts/neu3'",
            "```",
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
        "--results-file",
        action="append",
        type=Path,
        default=None,
        help="Đường dẫn tới file CSV kết quả Bước 1 (có thể truyền nhiều lần).",
    )
    parser.add_argument(
        "--results-glob",
        action="append",
        type=str,
        default=None,
        help="Mẫu glob tìm file CSV kết quả Bước 1 (có thể truyền nhiều lần; mặc định quét reports/step2/inputs/**/*.csv).",
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
    parser.add_argument(
        "--reload-check",
        action="store_true",
        help="Khi bật cùng --check-ckpt: nạp lại 1 checkpoint mỗi nhóm, chạy trên 16 câu dev và so logit với file predictions.",
    )
    parser.add_argument(
        "--ckpt-root-vsmec-phobert",
        type=Path,
        default=None,
        help="Thư mục chứa checkpoint Bước 1 VSMEC PhoBERT.",
    )
    parser.add_argument(
        "--ckpt-root-vsmec-visobert",
        type=Path,
        default=None,
        help="Thư mục chứa checkpoint Bước 1 VSMEC ViSoBERT.",
    )
    parser.add_argument(
        "--ckpt-root-neu3-phobert",
        type=Path,
        default=None,
        help="Thư mục chứa checkpoint Bước 1 NEU-3 PhoBERT.",
    )
    parser.add_argument(
        "--ckpt-root-neu3-visobert",
        type=Path,
        default=None,
        help="Thư mục chứa checkpoint Bước 1 NEU-3 ViSoBERT.",
    )
    args = parser.parse_args(argv)

    ckpt_overrides = {
        "vsmec_phobert": args.ckpt_root_vsmec_phobert,
        "vsmec_visobert": args.ckpt_root_vsmec_visobert,
        "neu3_phobert": args.ckpt_root_neu3_phobert,
        "neu3_visobert": args.ckpt_root_neu3_visobert,
    }
    payload = run_all_step2_checks(
        config_path=args.config,
        reports_dir=args.reports_dir,
        with_tokenizer=args.with_tokenizer,
        check_ckpt=args.check_ckpt,
        reload_check=args.reload_check,
        results_files=args.results_file,
        results_globs=args.results_glob,
        ckpt_roots_override=ckpt_overrides,
    )
    checks = [CheckResult(**c) for c in payload["checks"]]
    print(format_console_table(checks, payload["gate_line"]))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
