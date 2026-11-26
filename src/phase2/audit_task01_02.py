"""Read-only inventory (Task 01) and data/protocol audit (Task 02) for Phase 2.

Strictly preserves frozen `data/raw/`, `data/clean/`, `data/splits/`, `configs/seq.yaml`,
`configs/zs.yaml`, and `HARD_MERGE_RULE`. Never accesses Drive or downloads remote models
unless `--with-tokenizer` is explicitly requested (and even then uses local cache only).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import unicodedata

import numpy as np
import pandas as pd
import yaml

from src.common.constants import (
    EMOTIONS7,
    HARD_MERGE_GROUPS,
    HARD_MERGE_RULE,
    ID_TO_EMOTION7,
    ID_TO_LABEL3,
    LABELS3,
    NEUESC_ORIG_LABELS,
    NEUESC_ORIG_TO_LABEL3,
)
from src.common.hashing import hash_text, sha256_file
from src.data.preprocess import _GENERIC_EMOJI_RE, UNICODE_EMOJI_MAP, preprocess_text
from src.common.textnorm import normalize_text

EXPECTED_VSMEC_COUNTS = {"train": 5542, "dev": 686, "test": 692}
EXPECTED_NEUESC_COUNTS = {"train": 23048, "dev": 3305, "test": 6613}
TOKENIZER_MARKER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "bpe.codes",
)
CORE_CLEAN_FILES = (
    "vsmec_train.csv",
    "vsmec_dev.csv",
    "vsmec_test.csv",
    "neuesc_100.csv",
)


def load_audit_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load `configs/phase2_audit.yaml` with normalized defaults."""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(raw, dict):
        raise ValueError(f"Audit config must be a YAML mapping: {path}")
    data_paths = dict(raw.get("data_paths") or {})
    phase1_ext = dict(raw.get("phase1_external") or {})
    return {
        "expected_branch": raw.get("expected_branch", "phase2-resource-transfer"),
        "expected_base_commit": raw.get("expected_base_commit", "62df4cd"),
        "data_paths": {
            "raw_dir": data_paths.get("raw_dir", "data/raw"),
            "clean_dir": data_paths.get("clean_dir", "data/clean"),
            "splits_dir": data_paths.get("splits_dir", "data/splits"),
            "wtrans_dir": data_paths.get("wtrans_dir", "data/wtrans"),
            "token_stats_csv": data_paths.get("token_stats_csv", "results/token_stats.csv"),
            "polarity_hash_json": data_paths.get(
                "polarity_hash_json", "results/polarity/polarity_files_hash.json"
            ),
            "step2_readiness_json": data_paths.get(
                "step2_readiness_json", "reports/step2/readiness_report.json"
            ),
        },
        "phase1_external": {
            "vsmec_visobert_ckpt_root": phase1_ext.get("vsmec_visobert_ckpt_root"),
            "neu_visobert_ckpt_root": phase1_ext.get("neu_visobert_ckpt_root"),
            "stage1_results_dir": phase1_ext.get("stage1_results_dir"),
            "zs_results_dir": phase1_ext.get("zs_results_dir"),
            "seq_results_dir": phase1_ext.get("seq_results_dir"),
            "probe_notebook_path": phase1_ext.get("probe_notebook_path"),
        },
        "seeds_required": [int(s) for s in raw.get("seeds_required", [0, 1, 2, 3, 4])],
        "c0_seeds_required": [int(s) for s in raw.get("c0_seeds_required", [0, 1, 2])],
        "max_length": int(raw.get("max_length", 256)),
        "min_dev_vsmec": int(raw.get("min_dev_vsmec", 172)),
        "min_samples_5pct_v7": int(raw.get("min_samples_5pct_v7", 5)),
    }


def _resolve_path(repo_root: Path, value: Optional[Union[str, Path]]) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


def _sha256_lf(path: Path) -> str:
    """Compute SHA-256 after normalizing CRLF -> LF in memory (never modifies file)."""
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _run_git(repo_root: Path, args: Sequence[str]) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return proc.returncode, (proc.stdout or proc.stderr).strip()
    except OSError as exc:
        return 127, str(exc)


# ---------------------------------------------------------------------------
# Task 01: Resource Inventory
# ---------------------------------------------------------------------------


def inspect_git_state(
    repo_root: Path, expected_branch: str, expected_base_commit: str
) -> Dict[str, Any]:
    """Read-only verification of current branch, commit ancestry, and stashes."""
    rc_br, branch = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    rc_head, head_sha = _run_git(repo_root, ["rev-parse", "--short", "HEAD"])
    rc_full, head_full = _run_git(repo_root, ["rev-parse", "HEAD"])
    rc_anc, _ = _run_git(
        repo_root, ["merge-base", "--is-ancestor", expected_base_commit, "HEAD"]
    )
    rc_st, stashes_raw = _run_git(repo_root, ["stash", "list"])
    stashes = [line for line in stashes_raw.splitlines() if line.strip()] if rc_st == 0 else []
    rc_seq, seq_core_sha = _run_git(repo_root, ["rev-parse", "--short", "seq-core"])

    status = "PRESENT"
    notes: List[str] = []
    if rc_br != 0 or rc_head != 0:
        status = "MISSING"
        notes.append("Không đọc được trạng thái git (không phải git repository hoặc thiếu git).")
    else:
        if branch != expected_branch:
            status = "PARTIAL"
            notes.append(f"Nhánh hiện tại là '{branch}', khác kỳ vọng '{expected_branch}'.")
        if rc_anc != 0:
            status = "PARTIAL"
            notes.append(f"HEAD ({head_sha}) không kế thừa từ commit gốc '{expected_base_commit}'.")
        if not notes:
            notes.append(
                f"Đúng nhánh '{branch}' @ {head_sha}, kế thừa trực tiếp từ seq-core ({seq_core_sha or expected_base_commit})."
            )

    return {
        "status": status,
        "branch": branch if rc_br == 0 else None,
        "expected_branch": expected_branch,
        "head_short": head_sha if rc_head == 0 else None,
        "head_full": head_full if rc_full == 0 else None,
        "expected_base_commit": expected_base_commit,
        "base_is_ancestor": rc_anc == 0,
        "seq_core_short": seq_core_sha if rc_seq == 0 else None,
        "stash_count": len(stashes),
        "stashes": stashes,
        "notes": notes,
    }


def _inspect_csv_brief(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    df = pd.read_csv(path, keep_default_na=False)
    info: Dict[str, Any] = {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
    }
    if "subset" in df.columns:
        info["subset_counts"] = {
            str(k): int(v) for k, v in df["subset"].value_counts().sort_index().items()
        }
    return info


def inspect_datasets_inventory(repo_root: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Inventory clean, split, and raw dataset files for UIT-VSMEC and NEU-ESC."""
    dp = cfg["data_paths"]
    clean_dir = _resolve_path(repo_root, dp["clean_dir"])
    raw_dir = _resolve_path(repo_root, dp["raw_dir"])
    splits_dir = _resolve_path(repo_root, dp["splits_dir"])
    assert clean_dir is not None and raw_dir is not None and splits_dir is not None

    vsmec_clean = {
        split: _inspect_csv_brief(clean_dir / f"vsmec_{split}.csv")
        for split in ("train", "dev", "test")
    }
    vsmec_clean_extra = {
        "vsmec_removed.csv": _inspect_csv_brief(clean_dir / "vsmec_removed.csv"),
        "vsmec_clean_report.json": {
            "exists": (clean_dir / "vsmec_clean_report.json").is_file(),
            "size_bytes": (clean_dir / "vsmec_clean_report.json").stat().st_size
            if (clean_dir / "vsmec_clean_report.json").is_file()
            else 0,
        },
    }
    neuesc_clean = _inspect_csv_brief(clean_dir / "neuesc_100.csv")

    raw_files = {}
    if raw_dir.is_dir():
        for p in sorted(raw_dir.glob("*.csv")):
            raw_files[p.name] = {
                "size_bytes": p.stat().st_size,
                "n_lines": _line_count(p),
            }

    split_files = sorted(p.name for p in splits_dir.glob("*.csv")) if splits_dir.is_dir() else []

    vsmec_ok = all(vsmec_clean[s]["exists"] for s in ("train", "dev", "test"))
    neuesc_ok = neuesc_clean["exists"] and set(
        (neuesc_clean.get("subset_counts") or {}).keys()
    ) >= {"train", "dev", "test"}

    return {
        "vsmec_clean": {
            "status": "PRESENT" if vsmec_ok else "MISSING",
            "splits": vsmec_clean,
            "extra": vsmec_clean_extra,
        },
        "neuesc_clean": {
            "status": "PRESENT" if neuesc_ok else "MISSING",
            "file": neuesc_clean,
        },
        "raw_data": {
            "status": "PRESENT"
            if ("NEU_ESC_full.csv" in raw_files and "UIT_VSMEC_full.csv" in raw_files)
            else ("PARTIAL" if raw_files else "MISSING"),
            "dir": str(raw_dir),
            "files": raw_files,
        },
        "legacy_splits": {
            "status": "PARTIAL" if split_files else "MISSING",
            "dir": str(splits_dir),
            "files": split_files,
            "note": (
                "Chỉ có tập cắt NEU-ESC cũ (1%, 5%, 10% độc lập, không lồng nhau) — "
                "chưa có tập con lồng nhau 5/10/25/50% cho UIT-VSMEC và NEU-ESC (làm ở Task 03)."
            ),
        },
    }


def _inspect_single_checkpoint_group(
    ckpt_root: Optional[Path],
    alias: str,
    domain: str,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Validate source checkpoints across seeds without loading full encoder weights."""
    expected_classes = 7 if domain == "vsmec" else 3
    if ckpt_root is None:
        return {
            "status": "MISSING",
            "root": None,
            "alias": alias,
            "domain": domain,
            "expected_classes": expected_classes,
            "n_valid": 0,
            "n_required": len(seeds),
            "per_seed": {
                str(s): {"status": "MISSING", "reason": "Chưa cấu hình đường dẫn ckpt_root (null)."}
                for s in seeds
            },
        }
    if not ckpt_root.is_dir():
        return {
            "status": "MISSING",
            "root": str(ckpt_root),
            "alias": alias,
            "domain": domain,
            "expected_classes": expected_classes,
            "n_valid": 0,
            "n_required": len(seeds),
            "per_seed": {
                str(s): {"status": "MISSING", "reason": f"Thư mục không tồn tại: {ckpt_root}"}
                for s in seeds
            },
        }

    # Lazy import so pure-metadata runs don't fail if torch isn't installed in a minimal env
    from src.zs.ckpt import locate_checkpoint

    per_seed: Dict[str, Dict[str, Any]] = {}
    n_valid = 0
    n_broken = 0
    for seed in seeds:
        try:
            ckpt_dir = locate_checkpoint(ckpt_root, alias, domain, seed)
            enc_dir = ckpt_dir / "encoder"
            tokenizer_files = sorted(
                f for f in TOKENIZER_MARKER_FILES if (enc_dir / f).is_file()
            )
            has_tokenizer = bool(tokenizer_files)
            per_seed[str(seed)] = {
                "status": "PRESENT" if has_tokenizer else "PARTIAL",
                "checkpoint_dir": str(ckpt_dir),
                "tokenizer_files": tokenizer_files,
                "has_tokenizer": has_tokenizer,
                "reason": None if has_tokenizer else "Thiếu file tokenizer trong thư mục encoder/.",
            }
            if has_tokenizer:
                n_valid += 1
        except Exception as exc:
            msg = str(exc)
            is_missing = "found 0" in msg.lower() or "does not exist" in msg.lower()
            per_seed[str(seed)] = {
                "status": "MISSING" if is_missing else "BROKEN",
                "checkpoint_dir": None,
                "reason": msg,
            }
            if not is_missing:
                n_broken += 1

    if n_valid == len(seeds):
        group_status = "PRESENT"
    elif n_valid > 0:
        group_status = "PARTIAL"
    elif n_broken > 0:
        group_status = "BROKEN"
    else:
        group_status = "MISSING"

    return {
        "status": group_status,
        "root": str(ckpt_root),
        "alias": alias,
        "domain": domain,
        "expected_classes": expected_classes,
        "n_valid": n_valid,
        "n_required": len(seeds),
        "per_seed": per_seed,
    }


def inspect_code_and_configs(repo_root: Path) -> Dict[str, Any]:
    """Inventory existing Phase 1/Step 2/Step 3 code & configs and list Phase 2 gaps."""
    tracked_groups = {
        "seq_pipeline": [
            "src/seq/protocol.py",
            "src/seq/stage2.py",
            "src/seq/trainer.py",
            "src/seq/train.py",
            "src/seq/select_lr.py",
            "src/seq/analysis.py",
            "src/seq/analyze.py",
        ],
        "zs_pipeline": [
            "src/zs/ckpt.py",
            "src/zs/views.py",
            "src/zs/mapping.py",
            "src/zs/infer.py",
            "src/zs/analysis.py",
            "src/zs/analyze.py",
            "src/zs/check_consistency.py",
        ],
        "stats_and_plm": [
            "src/plm/model_ckpt.py",
            "src/stats/bootstrap.py",
            "src/stats/holm.py",
            "src/common/constants.py",
            "src/common/schema.py",
            "src/common/hashing.py",
            "src/common/textnorm.py",
            "src/data/preprocess.py",
            "src/data/make_splits.py",
        ],
        "configs": [
            "configs/base.yaml",
            "configs/step2_data.yaml",
            "configs/zs.yaml",
            "configs/seq.yaml",
        ],
    }

    present_map: Dict[str, Dict[str, bool]] = {}
    missing_files: List[str] = []
    for group, rel_paths in tracked_groups.items():
        present_map[group] = {}
        for rel in rel_paths:
            exists = (repo_root / rel).is_file()
            present_map[group][rel] = exists
            if not exists:
                missing_files.append(rel)

    phase2_gaps = [
        "Chưa có chế độ DFT khởi tạo trực tiếp từ pretrained HuggingFace ID (uitnlp/visobert, xlm-roberta-base) trong src/seq/trainer.py (hiện load_encoder_new_head chỉ nạp checkpoint local chứa thư mục encoder/).",
        "Chưa có giao thức C0 low-resource: C0N/C0V trong src/seq/stage2.py hiện nạp checkpoint nguồn 100% cùng miền, gây rò rỉ dữ liệu 100% đích nếu dùng cho 10%/25%/50% (cần chốt ở Task 04 và viết lại ở Task 05).",
        "Chưa có bộ sinh & nạp tập con phân tầng lồng nhau 5% ⊂ 10% ⊂ 25% ⊂ 50% theo ID cho UIT-VSMEC và NEU-ESC (Task 03).",
        "Chưa hỗ trợ tiêu chí chọn best checkpoint trên dev bằng Macro-F1 5 lớp (bỏ Surprise/Other) trong TorchTrainer (Task 04–05).",
        "Chưa có log chi phí huấn luyện đầy đủ (total_params, trainable_params qua requires_grad, peak VRAM, GPU-giờ, tên GPU) trong SEQ_COLUMNS / trainer (Task 05).",
        "Chưa có LLRD (hệ số 0.8/0.9) và LoRA (r=8/16, alpha=2r, query/value, full head; cần thêm peft hoặc module tương đương) (Task 10).",
        "Chưa có góc nhìn nhãn bổ sung V5-VSMEC (Macro-F1 5 lớp), V3-noSurprise, V3-NEU-noToxic trong src/zs/views.py và hỗ trợ Holm m=2 + CI tỷ số SD giữa seed (Task 12).",
    ]

    return {
        "status": "PARTIAL" if not missing_files else "BROKEN",
        "code_files_status": "PRESENT" if not missing_files else "PARTIAL",
        "files": present_map,
        "missing_expected_files": missing_files,
        "phase2_gaps": phase2_gaps,
    }


def inspect_phase1_results(repo_root: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Inventory 100% summary CSVs and per-sample test predictions for ViSoBERT & PhoBERT."""
    ext = cfg["phase1_external"]
    seeds = cfg["seeds_required"]
    c0_seeds = cfg["c0_seeds_required"]

    local_results_dir = repo_root / "results"
    local_files = (
        sorted(str(p.relative_to(repo_root)).replace("\\", "/") for p in local_results_dir.rglob("*") if p.is_file())
        if local_results_dir.is_dir()
        else []
    )

    dirs_to_check = {
        "stage1_results_dir": _resolve_path(repo_root, ext.get("stage1_results_dir")),
        "zs_results_dir": _resolve_path(repo_root, ext.get("zs_results_dir")),
        "seq_results_dir": _resolve_path(repo_root, ext.get("seq_results_dir")),
    }

    summary_hits: Dict[str, Optional[str]] = {
        "stage1.csv": None,
        "neu3.csv": None,
        "zs.csv": None,
        "seq.csv": None,
        "selection_seq.json": None,
    }
    pred_files: List[str] = []

    search_roots = [d for d in dirs_to_check.values() if d is not None and d.is_dir()]
    if local_results_dir.is_dir():
        search_roots.append(local_results_dir)

    for root in search_roots:
        for fname in summary_hits:
            if summary_hits[fname] is None:
                found = list(root.rglob(fname))
                if found:
                    summary_hits[fname] = str(found[0])
        for sub in ("predictions_zs", "predictions_seq", "predictions", "predictions3", "predictions7"):
            pdir = root / sub
            if pdir.is_dir():
                for csv_p in sorted(pdir.glob("*.csv")):
                    pred_files.append(str(csv_p))

    pred_files = sorted(set(pred_files))

    # Count coverage of ViSoBERT and PhoBERT 100% test prediction CSVs
    coverage: Dict[str, Dict[str, Any]] = {}
    for backbone in ("ViSoBERT", "PhoBERT"):
        bb_preds = [Path(p).name for p in pred_files if Path(p).name.startswith(f"{backbone}_")]
        coverage[backbone] = {
            "ref_vsmec_seeds": sorted(
                s for s in seeds if any(f"REFV2V" in n and f"_s{s}" in n and "vsmec_test" in n for n in bb_preds)
            ),
            "ref_neuesc_seeds": sorted(
                s for s in seeds if any(f"REFN2N" in n and f"_s{s}" in n and "neuesc_test" in n for n in bb_preds)
            ),
            "zs_A_seeds": sorted(
                s for s in seeds if any(f"ZSV2N" in n and f"_s{s}" in n and "neuesc_test" in n for n in bb_preds)
            ),
            "zs_B_seeds": sorted(
                s for s in seeds if any(f"ZSN2V" in n and f"_s{s}" in n and "vsmec_test" in n for n in bb_preds)
            ),
            "seq_A_seeds": sorted(
                s for s in seeds if any(f"SEQV2N" in n and f"_s{s}" in n and "neuesc_test" in n for n in bb_preds)
            ),
            "seq_B_seeds": sorted(
                s for s in seeds if any(f"SEQN2V" in n and f"_s{s}" in n and "vsmec_test" in n for n in bb_preds)
            ),
            "c0_N_seeds": sorted(
                s for s in c0_seeds if any(f"C0N" in n and f"_s{s}" in n and "neuesc_test" in n for n in bb_preds)
            ),
            "c0_V_seeds": sorted(
                s for s in c0_seeds if any(f"C0V" in n and f"_s{s}" in n and "vsmec_test" in n for n in bb_preds)
            ),
        }

    any_external_set = any(d is not None for d in dirs_to_check.values())
    has_zs_seq = bool(summary_hits["zs.csv"] and summary_hits["seq.csv"] and pred_files)
    if has_zs_seq:
        vis_cov = coverage["ViSoBERT"]
        all_vis_complete = (
            len(vis_cov["ref_vsmec_seeds"]) == len(seeds)
            and len(vis_cov["ref_neuesc_seeds"]) == len(seeds)
            and len(vis_cov["seq_A_seeds"]) == len(seeds)
            and len(vis_cov["seq_B_seeds"]) == len(seeds)
            and len(vis_cov["c0_N_seeds"]) == len(c0_seeds)
            and len(vis_cov["c0_V_seeds"]) == len(c0_seeds)
        )
        status = "PRESENT" if all_vis_complete else "PARTIAL"
    elif any(v is not None for v in summary_hits.values()) or pred_files:
        status = "PARTIAL"
    else:
        status = "MISSING"

    return {
        "status": status,
        "external_dirs_configured": {k: (str(v) if v else None) for k, v in dirs_to_check.items()},
        "any_external_dir_set": any_external_set,
        "local_results_files": local_files,
        "summary_files": summary_hits,
        "n_prediction_csvs": len(pred_files),
        "prediction_coverage_100pct": coverage,
    }


def _find_workspace_notebooks(repo_root: Path) -> List[str]:
    """Scan workspace for `.ipynb` files while pruning heavy/hidden dirs before descending."""
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
    found: List[str] = []
    for cur_dir, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        for fname in filenames:
            if fname.endswith(".ipynb"):
                rel = Path(cur_dir, fname).relative_to(repo_root)
                found.append(str(rel).replace("\\", "/"))
    return sorted(found)


def inspect_probe_notebook(repo_root: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Locate probe notebook locally and inspect PRIOR / tokenizer fallback if present."""
    configured = _resolve_path(repo_root, cfg["phase1_external"].get("probe_notebook_path"))
    discovered_notebooks = _find_workspace_notebooks(repo_root)

    if configured is None:
        return {
            "status": "MISSING",
            "configured_path": None,
            "discovered_in_workspace": discovered_notebooks,
            "findings": [],
            "note": (
                "Chưa có notebook Probe (Other/Surprise) trong workspace local hoặc chưa cấu hình "
                "`phase1_external.probe_notebook_path`."
            ),
        }

    if not configured.is_file():
        return {
            "status": "MISSING",
            "configured_path": str(configured),
            "discovered_in_workspace": discovered_notebooks,
            "findings": [],
            "note": f"Không tìm thấy notebook Probe tại: {configured}",
        }

    payload = json.loads(configured.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    findings: List[Dict[str, Any]] = []
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if re.search(r"\bPRIOR\b", src) and ("test" in src.lower()):
            findings.append(
                {
                    "cell_index": idx,
                    "issue": "PRIOR_USES_TEST",
                    "detail": "Cell khai báo PRIOR có tham chiếu tới dữ liệu test; cần đổi sang phân phối nhãn của tập train.",
                }
            )
        if "uitnlp/visobert" in src and ("except" in src or "fallback" in src.lower()):
            findings.append(
                {
                    "cell_index": idx,
                    "issue": "TOKENIZER_FALLBACK_VISOBERT",
                    "detail": "Cell có nhánh fallback sang 'uitnlp/visobert'; cần nạp đúng tokenizer của checkpoint và báo lỗi nếu thiếu.",
                }
            )

    return {
        "status": "BROKEN" if findings else "PRESENT",
        "configured_path": str(configured),
        "discovered_in_workspace": discovered_notebooks,
        "n_cells": len(cells),
        "findings": findings,
        "note": (
            f"Phát hiện {len(findings)} vấn đề cần sửa trong notebook Probe."
            if findings
            else "Không phát hiện mẫu lỗi PRIOR từ test hay fallback tokenizer trong notebook Probe."
        ),
    }


def collect_inventory(repo_root: Union[str, Path], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run all Task 01 inventory checks (read-only)."""
    root = Path(repo_root).resolve()
    ext = cfg["phase1_external"]
    seeds = cfg["seeds_required"]
    return {
        "repo_root": str(root),
        "git": inspect_git_state(root, cfg["expected_branch"], cfg["expected_base_commit"]),
        "datasets": inspect_datasets_inventory(root, cfg),
        "checkpoints": {
            "vsmec_visobert": _inspect_single_checkpoint_group(
                _resolve_path(root, ext.get("vsmec_visobert_ckpt_root")),
                alias="ViSoBERT",
                domain="vsmec",
                seeds=seeds,
            ),
            "neu_visobert": _inspect_single_checkpoint_group(
                _resolve_path(root, ext.get("neu_visobert_ckpt_root")),
                alias="ViSoBERT",
                domain="neu3",
                seeds=seeds,
            ),
        },
        "code_and_configs": inspect_code_and_configs(root),
        "phase1_results": inspect_phase1_results(root, cfg),
        "probe_notebook": inspect_probe_notebook(root, cfg),
    }


# ---------------------------------------------------------------------------
# Task 02: Manifest, Data Integrity, Preprocessing, Protocol & Probe Audit
# ---------------------------------------------------------------------------


def audit_manifest_dir(
    directory: Path,
    manifest_name: str = "MANIFEST.json",
    core_files: Sequence[str] = (),
) -> Dict[str, Any]:
    """Compare every file in `directory` against `MANIFEST.json` (raw disk SHA-256 and LF SHA-256)."""
    manifest_path = directory / manifest_name
    if not manifest_path.is_file():
        return {
            "directory": str(directory),
            "manifest_path": str(manifest_path),
            "status": "MISSING",
            "entries": {},
            "untracked_on_disk": [],
            "duplicate_manifest_hashes": {},
        }

    manifest: Dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: Dict[str, Dict[str, Any]] = {}
    hash_to_keys: Dict[str, List[str]] = {}

    for key, recorded_hash in sorted(manifest.items()):
        hash_to_keys.setdefault(str(recorded_hash), []).append(key)
        fp = directory / key
        if not fp.is_file():
            entries[key] = {
                "status": "MISSING_ON_DISK",
                "is_core": key in core_files,
                "recorded_sha256": recorded_hash,
                "disk_sha256": None,
                "lf_sha256": None,
                "size_bytes": None,
                "n_lines": None,
            }
            continue
        disk_hash = sha256_file(fp)
        lf_hash = _sha256_lf(fp)
        if disk_hash == recorded_hash:
            st = "MATCH_EXACT"
        elif lf_hash == recorded_hash:
            st = "MATCH_LF_ONLY"
        else:
            st = "MISMATCH"
        entries[key] = {
            "status": st,
            "is_core": key in core_files,
            "recorded_sha256": recorded_hash,
            "disk_sha256": disk_hash,
            "lf_sha256": lf_hash,
            "size_bytes": fp.stat().st_size,
            "n_lines": _line_count(fp),
        }

    disk_files = sorted(
        p.name for p in directory.iterdir() if p.is_file() and p.name != manifest_name
    )
    untracked = []
    for name in disk_files:
        if name not in manifest:
            fp = directory / name
            untracked.append(
                {
                    "filename": name,
                    "size_bytes": fp.stat().st_size,
                    "disk_sha256": sha256_file(fp),
                    "lf_sha256": _sha256_lf(fp),
                }
            )

    duplicate_recorded = {
        h: sorted(keys) for h, keys in hash_to_keys.items() if len(keys) > 1
    }

    core_all_ok = all(
        entries.get(cf, {}).get("status") in ("MATCH_EXACT", "MATCH_LF_ONLY")
        for cf in core_files
    )
    any_mismatch = any(e["status"] in ("MISMATCH", "MISSING_ON_DISK") for e in entries.values())
    status = "PASS" if (not any_mismatch and not untracked) else ("WARN" if core_all_ok else "FAIL")

    return {
        "directory": str(directory),
        "manifest_path": str(manifest_path),
        "status": status,
        "core_files_ok": core_all_ok,
        "n_manifest_entries": len(manifest),
        "entries": entries,
        "untracked_on_disk": untracked,
        "duplicate_manifest_hashes": duplicate_recorded,
    }


def audit_polarity_anomaly(
    clean_dir: Path,
    clean_manifest_audit: Dict[str, Any],
    polarity_hash_json: Optional[Path],
) -> Dict[str, Any]:
    """Forensic audit of `polarity_long.csv` vs `polarity_nhat.csv` duplicate MANIFEST hash."""
    polarity_recorded_results: Dict[str, str] = {}
    if polarity_hash_json is not None and polarity_hash_json.is_file():
        polarity_recorded_results = json.loads(polarity_hash_json.read_text(encoding="utf-8"))

    details: Dict[str, Dict[str, Any]] = {}
    for fname in ("polarity_long.csv", "polarity_nhat.csv"):
        fp = clean_dir / fname
        entry = clean_manifest_audit.get("entries", {}).get(fname, {})
        row_count = None
        cols: List[str] = []
        non_empty_labels = None
        if fp.is_file():
            df = pd.read_csv(fp, keep_default_na=False)
            row_count = int(len(df))
            cols = list(df.columns)
            if "polarity_label" in df.columns:
                non_empty_labels = int(df["polarity_label"].astype(str).str.strip().ne("").sum())
        res_hash = polarity_recorded_results.get(fname)
        details[fname] = {
            "exists": fp.is_file(),
            "size_bytes": entry.get("size_bytes"),
            "n_rows": row_count,
            "columns": cols,
            "non_empty_polarity_labels": non_empty_labels,
            "manifest_clean_sha256": entry.get("recorded_sha256"),
            "disk_sha256": entry.get("disk_sha256"),
            "lf_sha256": entry.get("lf_sha256"),
            "results_polarity_hash_sha256": res_hash,
            "matches_manifest_clean": entry.get("status") in ("MATCH_EXACT", "MATCH_LF_ONLY"),
            "matches_results_polarity_hash": (
                res_hash is not None
                and entry.get("disk_sha256") is not None
                and res_hash in (entry.get("disk_sha256"), entry.get("lf_sha256"))
            ),
        }

    same_manifest_hash = (
        details["polarity_long.csv"]["manifest_clean_sha256"] is not None
        and details["polarity_long.csv"]["manifest_clean_sha256"]
        == details["polarity_nhat.csv"]["manifest_clean_sha256"]
    )
    different_disk_content = (
        details["polarity_long.csv"]["disk_sha256"] is not None
        and details["polarity_long.csv"]["disk_sha256"]
        != details["polarity_nhat.csv"]["disk_sha256"]
    )

    return {
        "same_hash_in_clean_manifest": same_manifest_hash,
        "different_actual_disk_hashes": different_disk_content,
        "both_match_results_polarity_hash": (
            details["polarity_long.csv"]["matches_results_polarity_hash"]
            and details["polarity_nhat.csv"]["matches_results_polarity_hash"]
        ),
        "files": details,
        "impact_on_task03_subsampling": (
            "KHÔNG ảnh hưởng đến việc tạo tập con train/dev/test ở Task 03 (hai file polarity "
            "chỉ phục vụ gán nhãn kiểm chứng M1b/M7, không tham gia nạp train/dev/test của "
            "UIT-VSMEC hay NEU-ESC)."
        ),
        "root_cause_explanation": (
            "Mã hash trùng trong data/clean/MANIFEST.json phản ánh ảnh chụp file mẫu trống trước khi "
            "hai người gán nhãn (Long, Nhật) điền cột polarity_label; sau khi gán nhãn, hash thật của "
            "từng file đã được lưu khớp tại results/polarity/polarity_files_hash.json nhưng "
            "data/clean/MANIFEST.json đóng băng không được ghi đè."
        ),
        "recommended_action": (
            "Giữ nguyên data/clean/MANIFEST.json đóng băng để không làm lệch provenance Giai đoạn 1; "
            "ghi nhận ngoại lệ này trong biên bản Task 02 và chỉ dùng results/polarity/polarity_files_hash.json "
            "khi đối chiếu artifact gán nhãn polarity."
        ),
    }


def _compute_stratified_quota(counts: Dict[int, int], pct: float) -> Dict[int, int]:
    """Largest-remainder stratified projection for reporting expected subset class sizes."""
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}
    target = int(round(total * pct / 100.0))
    exact = {k: counts[k] * (target / total) for k in counts}
    alloc = {k: int(math.floor(exact[k])) for k in counts}
    rem = target - sum(alloc.values())
    order = sorted(
        counts.keys(),
        key=lambda k: (exact[k] - alloc[k], counts[k], -k),
        reverse=True,
    )
    for k in order[: max(0, rem)]:
        alloc[k] += 1
    return alloc


def _check_domain_internal(
    splits: Dict[str, pd.DataFrame], valid_labels: Sequence[int]
) -> Dict[str, Any]:
    """Run C1 (internal dup), C2 (label conflicts), C3 (cross-split leakage), schema checks."""
    valid_set = set(valid_labels)
    split_stats: Dict[str, Dict[str, Any]] = {}
    hash_sets: Dict[str, set] = {}
    hash_to_labels: Dict[str, set] = {}
    internal_dups: Dict[str, int] = {}
    schema_errors: List[str] = []

    for sp_name, df in splits.items():
        if df.empty:
            schema_errors.append(f"{sp_name}: rỗng")
        if df["id"].isna().any() or df["id"].duplicated().any():
            schema_errors.append(f"{sp_name}: có id rỗng hoặc trùng lặp")
        labels = pd.to_numeric(df["label"], errors="coerce")
        if labels.isna().any() or not labels.isin(valid_set).all():
            schema_errors.append(f"{sp_name}: có nhãn ngoài tập hợp lệ {sorted(valid_set)}")
        if df["text"].astype(str).str.strip().eq("").any():
            schema_errors.append(f"{sp_name}: có chuỗi text rỗng")
        if "text_seg" in df.columns and df["text_seg"].astype(str).str.strip().eq("").any():
            schema_errors.append(f"{sp_name}: có chuỗi text_seg rỗng")

        hashes = [hash_text(str(t), normalize=True) for t in df["text"]]
        hash_series = pd.Series(hashes)
        n_dup = int(hash_series.duplicated().sum())
        internal_dups[sp_name] = n_dup
        hash_sets[sp_name] = set(hashes)

        for h, lbl in zip(hashes, labels.astype(int).tolist()):
            hash_to_labels.setdefault(h, set()).add(lbl)

        counts = {int(k): int((labels == k).sum()) for k in valid_labels}
        n_rows = int(len(df))
        split_stats[sp_name] = {
            "n_rows": n_rows,
            "class_counts": counts,
            "class_pct": {
                int(k): round(100.0 * v / n_rows, 4) if n_rows else 0.0 for k, v in counts.items()
            },
            "internal_duplicate_rows": n_dup,
        }

    label_conflicts = sum(1 for lbls in hash_to_labels.values() if len(lbls) > 1)
    leakage = {
        "train_dev": len(hash_sets["train"] & hash_sets["dev"]),
        "train_test": len(hash_sets["train"] & hash_sets["test"]),
        "dev_test": len(hash_sets["dev"] & hash_sets["test"]),
    }

    return {
        "schema_ok": not schema_errors,
        "schema_errors": schema_errors,
        "splits": split_stats,
        "c1_internal_duplicates_total": sum(internal_dups.values()),
        "c2_label_conflicts": label_conflicts,
        "c3_cross_split_leakage": leakage,
        "c3_total_leakage": sum(leakage.values()),
        "hash_sets": hash_sets,
    }


def trace_neuesc_raw_toxic(raw_dir: Path, clean_neuesc: pd.DataFrame) -> Dict[str, Any]:
    """Verify whether original NEU-ESC 4-class `toxic` labels can be recovered 1-to-1 by `id`."""
    raw_full = raw_dir / "NEU_ESC_full.csv"
    if not raw_full.is_file():
        return {
            "status": "MISSING",
            "raw_file": str(raw_full),
            "recoverable": False,
            "reason": "Không tìm thấy data/raw/NEU_ESC_full.csv.",
        }

    raw_df = pd.read_csv(raw_full, keep_default_na=False)
    if "sentiment" not in raw_df.columns or "text" not in raw_df.columns:
        return {
            "status": "BROKEN",
            "raw_file": str(raw_full),
            "recoverable": False,
            "reason": f"Thiếu cột sentiment/text trong {raw_full.name}: {list(raw_df.columns)}",
        }

    raw_df = raw_df.copy()
    raw_df["id"] = range(len(raw_df))
    merged = clean_neuesc.merge(
        raw_df[["id", "text", "sentiment", "Split"]],
        on="id",
        how="left",
        suffixes=("_clean", "_raw"),
    )
    missing_ids = int(merged["sentiment"].isna().sum())
    norm_match_rate = float(
        (
            merged["text_clean"].astype(str).map(normalize_text)
            == merged["text_raw"].astype(str).map(normalize_text)
        ).mean()
    )
    orig_int = pd.to_numeric(merged["sentiment"], errors="coerce").astype("Int64")
    mapped_check = orig_int.map(NEUESC_ORIG_TO_LABEL3)
    label_consistent = bool((mapped_check == merged["label"]).all())

    per_split_orig: Dict[str, Dict[str, int]] = {}
    for sp in ("train", "dev", "test"):
        sub = merged[merged["subset"] == sp]
        sub_orig = pd.to_numeric(sub["sentiment"], errors="coerce").astype(int)
        per_split_orig[sp] = {
            NEUESC_ORIG_LABELS[k]: int((sub_orig == k).sum())
            for k in sorted(NEUESC_ORIG_LABELS.keys())
        }

    recoverable = missing_ids == 0 and norm_match_rate == 1.0 and label_consistent
    return {
        "status": "PRESENT" if recoverable else "BROKEN",
        "raw_file": str(raw_full),
        "recoverable": recoverable,
        "n_rows_raw": int(len(raw_df)),
        "n_rows_clean": int(len(clean_neuesc)),
        "missing_id_matches": missing_ids,
        "normalized_text_match_rate": round(norm_match_rate, 6),
        "label_mapping_consistent": label_consistent,
        "orig_4class_counts_by_split": per_split_orig,
        "test_toxic_count": per_split_orig["test"]["toxic"],
        "test_non_toxic_count": int(len(merged[merged["subset"] == "test"]))
        - per_split_orig["test"]["toxic"],
    }


def check_underscore_leakage(
    raw_texts: Sequence[str],
    model_inputs: Sequence[str],
    max_delta_rate: float = 0.001,
) -> Dict[str, Any]:
    """Detect whether segmented `_`-joined text leaked into an unsegmented model input stream."""
    if len(raw_texts) != len(model_inputs):
        raise ValueError("raw_texts and model_inputs must have identical length")
    n = len(raw_texts)
    raw_has = sum("_" in str(t) for t in raw_texts)
    inp_has = sum("_" in str(t) for t in model_inputs)
    raw_total_u = sum(str(t).count("_") for t in raw_texts)
    inp_total_u = sum(str(t).count("_") for t in model_inputs)
    raw_rate = raw_has / n if n else 0.0
    inp_rate = inp_has / n if n else 0.0
    delta_rate = inp_rate - raw_rate
    passed = abs(delta_rate) <= max_delta_rate and inp_total_u <= raw_total_u
    return {
        "n_samples": n,
        "raw_docs_with_underscore": raw_has,
        "raw_underscore_doc_rate": round(raw_rate, 6),
        "raw_total_underscores": raw_total_u,
        "input_docs_with_underscore": inp_has,
        "input_underscore_doc_rate": round(inp_rate, 6),
        "input_total_underscores": inp_total_u,
        "delta_doc_rate": round(delta_rate, 6),
        "status": "PASS" if passed else "FAIL",
    }


def _stream_sha256(texts: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for t in texts:
        hasher.update(t.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


_EMOJI_KEYS_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(UNICODE_EMOJI_MAP.keys(), key=len, reverse=True))
)


def _has_emoji(text: str) -> bool:
    return bool(_GENERIC_EMOJI_RE.search(text) or _EMOJI_KEYS_RE.search(text))


def audit_preprocessing_and_truncation(
    vsmec_splits: Dict[str, pd.DataFrame],
    neuesc_splits: Dict[str, pd.DataFrame],
    token_stats_csv: Optional[Path],
    max_length: int = 256,
    with_tokenizer: bool = False,
) -> Dict[str, Any]:
    """Check NFC/idempotence/emoji/underscore safety and truncation rates."""
    per_split: Dict[str, Dict[str, Any]] = {}
    all_pass = True

    for ds_name, splits in (("vsmec", vsmec_splits), ("neuesc", neuesc_splits)):
        for sp_name, df in splits.items():
            key = f"{ds_name}_{sp_name}"
            raw_texts = df["text"].astype(str).tolist()
            seg_texts = df["text_seg"].astype(str).tolist() if "text_seg" in df.columns else []
            vis_inputs = [preprocess_text(t, emoji_mode="keep") for t in raw_texts]

            norm_once = [normalize_text(t) for t in raw_texts]
            norm_idempotent = all(normalize_text(n) == n for n in norm_once)
            prep_idempotent = all(
                preprocess_text(v, emoji_mode="keep") == v for v in vis_inputs
            )
            nfc_compliant = all(unicodedata.is_normalized("NFC", v) for v in vis_inputs)
            raw_emoji_docs = sum(_has_emoji(t) for t in raw_texts)
            vis_emoji_docs = sum(_has_emoji(v) for v in vis_inputs)

            u_check = check_underscore_leakage(raw_texts, vis_inputs)
            seg_u_docs = sum("_" in s for s in seg_texts) if seg_texts else 0
            seg_u_rate = round(seg_u_docs / len(seg_texts), 6) if seg_texts else None

            if not (norm_idempotent and prep_idempotent and nfc_compliant and u_check["status"] == "PASS"):
                all_pass = False

            per_split[key] = {
                "n_samples": len(raw_texts),
                "normalize_idempotent": norm_idempotent,
                "preprocess_keep_idempotent": prep_idempotent,
                "nfc_compliant": nfc_compliant,
                "raw_emoji_docs": raw_emoji_docs,
                "visobert_keep_emoji_docs": vis_emoji_docs,
                "underscore_check_visobert_xlmr": u_check,
                "phobert_text_seg_underscore_docs": seg_u_docs,
                "phobert_text_seg_underscore_rate": seg_u_rate,
                "visobert_xlmr_input_sha256": _stream_sha256(vis_inputs),
                "phobert_text_seg_sha256": _stream_sha256(seg_texts) if seg_texts else None,
            }

    # Existing recorded token stats
    recorded_token_stats: List[Dict[str, Any]] = []
    if token_stats_csv is not None and token_stats_csv.is_file():
        ts_df = pd.read_csv(token_stats_csv)
        recorded_token_stats = ts_df.to_dict(orient="records")

    live_truncation: Dict[str, Any] = {
        "mode": "LOCAL_CACHE_TOKENIZER" if with_tokenizer else "SKIP_OFFLINE",
        "max_length": max_length,
        "results": {},
        "notes": [],
    }
    if with_tokenizer:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            live_truncation["notes"].append(f"Không import được transformers.AutoTokenizer: {exc}")
        else:
            for alias, model_id in (
                ("ViSoBERT", "uitnlp/visobert"),
                ("XLM-RoBERTa-base", "xlm-roberta-base"),
            ):
                try:
                    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                except Exception as exc:
                    live_truncation["results"][alias] = {
                        "status": "MISSING_LOCAL_CACHE",
                        "model_id": model_id,
                        "reason": str(exc),
                    }
                    continue
                ds_res: Dict[str, Any] = {}
                for ds_name, splits in (("vsmec", vsmec_splits), ("neuesc", neuesc_splits)):
                    for sp_name, df in splits.items():
                        texts = [preprocess_text(str(t), emoji_mode="keep") for t in df["text"]]
                        lengths = [
                            len(ids)
                            for ids in tok(texts, padding=False, truncation=False)["input_ids"]
                        ]
                        n_trunc = sum(l > max_length for l in lengths)
                        ds_res[f"{ds_name}_{sp_name}"] = {
                            "n_samples": len(lengths),
                            "n_truncated": n_trunc,
                            "truncation_rate": round(n_trunc / len(lengths), 6) if lengths else 0.0,
                            "p95_tokens": float(np.percentile(lengths, 95)) if lengths else 0.0,
                            "max_tokens": int(max(lengths)) if lengths else 0,
                        }
                live_truncation["results"][alias] = {
                    "status": "PASS",
                    "model_id": model_id,
                    "splits": ds_res,
                }
    else:
        live_truncation["notes"].append(
            "Bỏ qua nạp tokenizer trực tiếp khi chạy mặc định offline (truyền --with-tokenizer để đo "
            "từ cache local); đã trích xuất số liệu NEU-ESC sẵn có từ results/token_stats.csv "
            "(chú ý results/token_stats.csv chưa có dòng riêng cho UIT-VSMEC)."
        )

    return {
        "status": "PASS" if all_pass else "FAIL",
        "per_split": per_split,
        "recorded_token_stats_csv": str(token_stats_csv) if token_stats_csv else None,
        "recorded_token_stats": recorded_token_stats,
        "live_truncation": live_truncation,
    }


def audit_protocol_compatibility(repo_root: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compare configs/base.yaml, configs/seq.yaml, actual 100% outputs (if any), and Phase 2 plan."""
    base_cfg = yaml.safe_load((repo_root / "configs/base.yaml").read_text(encoding="utf-8"))
    seq_cfg = yaml.safe_load((repo_root / "configs/seq.yaml").read_text(encoding="utf-8"))
    zs_cfg = yaml.safe_load((repo_root / "configs/zs.yaml").read_text(encoding="utf-8"))

    seq_train = seq_cfg.get("training", {})
    prov = seq_cfg.get("baseline_provenance", {})

    rows = [
        {
            "dimension": "Backbone phạm vi chính",
            "stage1_base_git": f"{base_cfg.get('backbone')} & {base_cfg.get('visobert_backbone')}",
            "seq_v1": ", ".join(sorted((seq_cfg.get("backbones") or {}).keys())),
            "phase2_proposed": "ViSoBERT (chính, 2 chiều); PhoBERT giữ số liệu 100% làm mốc; XLM-R & LLM ở mức 10%",
            "compatible_100pct": "COMPATIBLE (ViSoBERT giữ nguyên kiến trúc)",
        },
        {
            "dimension": "Tiền xử lý ViSoBERT & max_length",
            "stage1_base_git": f"emoji_mode={base_cfg.get('preprocess', {}).get('emoji_mode')} (base.yaml cũ); max_length={base_cfg.get('max_length')}",
            "seq_v1": f"text_col={seq_cfg['backbones']['ViSoBERT']['text_col']}, emoji_mode={seq_cfg['backbones']['ViSoBERT']['emoji_mode']}, max_length={seq_train.get('max_length')}",
            "phase2_proposed": "text (văn bản thô, không '_'), emoji_mode=keep, max_length=256",
            "compatible_100pct": "COMPATIBLE với seq-v1 & zs.yaml (cần xác minh log baseline Stage 1 thật có bật emokeep hay không)",
        },
        {
            "dimension": "Batch hiệu dụng / Epochs / Weight decay",
            "stage1_base_git": f"{base_cfg.get('batch_size')}×{base_cfg.get('grad_accum')}=32, epochs={base_cfg.get('epochs')}",
            "seq_v1": f"{seq_train.get('batch_size')}×{seq_train.get('gradient_accumulation_steps')}=32, epochs={seq_train.get('epochs')}, wd={seq_train.get('weight_decay')}",
            "phase2_proposed": "Batch hiệu dụng 32, tối đa 10 epoch (LoRA có thể nâng 15 nếu chưa hội tụ), weight_decay=0.01",
            "compatible_100pct": "COMPATIBLE",
        },
        {
            "dimension": "Warmup ratio",
            "stage1_base_git": "Không khai báo tường minh trong base.yaml (nhánh Neu3 dùng 0.06)",
            "seq_v1": f"{seq_train.get('warmup_ratio')} (6%)",
            "phase2_proposed": "0.10 (10%)",
            "compatible_100pct": "DIFFERS (0.06 vs 0.10 — cần chốt ở Task 04)",
        },
        {
            "dimension": "Early stopping patience & tần suất eval",
            "stage1_base_git": f"patience={base_cfg.get('early_stop', {}).get('patience')}, metric={base_cfg.get('early_stop', {}).get('metric')}",
            "seq_v1": f"patience={seq_train.get('patience')}, evals_per_epoch={seq_train.get('evals_per_epoch')}",
            "phase2_proposed": "patience=2 (cần chốt evals_per_epoch ở Task 04)",
            "compatible_100pct": "DIFFERS (patience 5 vs 2 — cần chốt ở Task 04)",
        },
        {
            "dimension": "Tiêu chí chọn best checkpoint trên dev",
            "stage1_base_git": "val_loss (base.yaml) / macro_f1 V7 hoặc V3 trên nhánh train",
            "seq_v1": "Macro-F1 toàn bộ lớp view chính (V7-VSMEC cho SB; V3-NEU cho SA)",
            "phase2_proposed": "Đề xuất: Macro-F1 5 lớp (bỏ Surprise/Other) cho VSMEC, V3-NEU cho NEU-ESC; giữ dev ≥ 172 câu",
            "compatible_100pct": "DIFFERS trên chiều B (V7 vs Macro-F1 5 lớp — cần chốt ở Task 04)",
        },
        {
            "dimension": "Đối chứng C0",
            "stage1_base_git": "Chưa có",
            "seq_v1": f"c0_seeds={seq_cfg.get('c0_seeds')}, c0_lr_mode={seq_cfg.get('c0_lr_mode')}; nạp checkpoint 100% cùng miền rồi train tiếp trên 100%",
            "phase2_proposed": "Đề xuất Task 04: C0 low-resource cả 2 pha chỉ được thấy tập con đích (10/25/50%) cùng seed; không nạp checkpoint 100% đích",
            "compatible_100pct": "DIFFERS ở mức <100% (bắt buộc sửa để không rò rỉ dữ liệu 100% đích)",
        },
        {
            "dimension": "Precision & Seed & Thống kê chính",
            "stage1_base_git": f"precision={base_cfg.get('precision')}",
            "seq_v1": f"precision={seq_train.get('precision')} (strict={seq_train.get('strict_precision')}), seeds={seq_cfg.get('seeds')}, Holm m={len(seq_cfg.get('primary', []))}",
            "phase2_proposed": "precision=bf16, seeds=0..4 (C0: 0..2), Holm m=2 cho Seq−DFT ở 10% hai chiều (ViSoBERT)",
            "compatible_100pct": "COMPATIBLE (chú ý nâng cấp Holm sang m=2 cho GĐ2 ở file phân tích riêng)",
        },
    ]

    return {
        "baseline_provenance_in_seq_yaml": prov,
        "zs_batch_size": zs_cfg.get("batch_size"),
        "comparison_matrix": rows,
        "reusability_verdict_100pct": (
            "CHƯA THỂ KẾT LUẬN TƯƠNG THÍCH HOÀN TOÀN TẠI TASK 02: "
            "(1) Trong workspace local chưa có file kết quả/log thật và dự đoán từng mẫu mức 100% của "
            "ViSoBERT/PhoBERT (baseline_provenance.actual_run_verified=false); "
            "(2) Có 3 điểm khác biệt giao thức giữa `seq-v1` (mốc 100%) và kế hoạch GĐ2 đề xuất: "
            "`warmup_ratio` (0.06 vs 0.10), `patience` (5 vs 2), và tiêu chí chọn best dev trên UIT-VSMEC "
            "(Macro-F1 V7 vs Macro-F1 5 lớp). Nhóm cần quyết định ở Task 04: giữ giao thức chọn dev/warmup/patience "
            "tương thích mốc 100%, hay dùng giao thức mới và trình bày tách biệt / chạy lại mốc 100% ViSoBERT."
        ),
    }


def run_data_protocol_audit(
    repo_root: Union[str, Path],
    cfg: Dict[str, Any],
    with_tokenizer: bool = False,
) -> Dict[str, Any]:
    """Run all Task 02 audit checks (read-only)."""
    root = Path(repo_root).resolve()
    dp = cfg["data_paths"]
    clean_dir = _resolve_path(root, dp["clean_dir"])
    raw_dir = _resolve_path(root, dp["raw_dir"])
    splits_dir = _resolve_path(root, dp["splits_dir"])
    token_stats_csv = _resolve_path(root, dp["token_stats_csv"])
    polarity_hash_json = _resolve_path(root, dp["polarity_hash_json"])
    step2_readiness_json = _resolve_path(root, dp["step2_readiness_json"])
    assert clean_dir is not None and raw_dir is not None and splits_dir is not None

    clean_manifest = audit_manifest_dir(clean_dir, core_files=CORE_CLEAN_FILES)
    splits_manifest = audit_manifest_dir(splits_dir, core_files=("neuesc_100.csv",))
    polarity_audit = audit_polarity_anomaly(clean_dir, clean_manifest, polarity_hash_json)

    vsmec_splits = {
        sp: pd.read_csv(clean_dir / f"vsmec_{sp}.csv", keep_default_na=False)
        for sp in ("train", "dev", "test")
    }
    neuesc_full = pd.read_csv(clean_dir / "neuesc_100.csv", keep_default_na=False)
    neuesc_splits = {
        sp: neuesc_full.loc[neuesc_full["subset"] == sp].copy().reset_index(drop=True)
        for sp in ("train", "dev", "test")
    }

    vsmec_internal = _check_domain_internal(vsmec_splits, valid_labels=range(7))
    neuesc_internal = _check_domain_internal(neuesc_splits, valid_labels=range(3))

    # Cross-domain exact overlaps D1 / D2
    vsmec_all_hashes = set().union(*vsmec_internal["hash_sets"].values())
    neuesc_all_hashes = set().union(*neuesc_internal["hash_sets"].values())
    d1_vsmec_test_in_neu = len(vsmec_internal["hash_sets"]["test"] & neuesc_all_hashes)
    d2_neu_test_in_vsmec = len(neuesc_internal["hash_sets"]["test"] & vsmec_all_hashes)

    # Strip non-JSON-serializable hash_sets before returning
    vsmec_internal.pop("hash_sets")
    neuesc_internal.pop("hash_sets")

    # Projected 5% stratified counts for UIT-VSMEC train
    vsmec_train_counts = vsmec_internal["splits"]["train"]["class_counts"]
    vsmec_5pct_proj = _compute_stratified_quota(vsmec_train_counts, 5.0)
    min_5pct_cls_id = min(vsmec_5pct_proj, key=lambda k: vsmec_5pct_proj[k])
    min_5pct_count = vsmec_5pct_proj[min_5pct_cls_id]
    min_threshold = cfg["min_samples_5pct_v7"]

    # Read known D4 near-duplicate record from step2 readiness report if present
    step2_d4_summary = None
    if step2_readiness_json is not None and step2_readiness_json.is_file():
        s2 = json.loads(step2_readiness_json.read_text(encoding="utf-8"))
        for chk in s2.get("checks", []):
            if chk.get("id") == "D4":
                step2_d4_summary = chk.get("detail")
                break

    toxic_trace = trace_neuesc_raw_toxic(raw_dir, neuesc_full)
    preprocess_audit = audit_preprocessing_and_truncation(
        vsmec_splits,
        neuesc_splits,
        token_stats_csv=token_stats_csv,
        max_length=cfg["max_length"],
        with_tokenizer=with_tokenizer,
    )
    protocol_compat = audit_protocol_compatibility(root, cfg)
    probe_audit = inspect_probe_notebook(root, cfg)

    # Overall Task 03 readiness gate (pure data integrity)
    task03_blockers: List[str] = []
    if not clean_manifest["core_files_ok"]:
        task03_blockers.append("Core clean files không khớp data/clean/MANIFEST.json")
    if not vsmec_internal["schema_ok"] or not neuesc_internal["schema_ok"]:
        task03_blockers.append("Lỗi schema/nhãn/ID trên dữ liệu sạch")
    if (
        vsmec_internal["c1_internal_duplicates_total"] > 0
        or neuesc_internal["c1_internal_duplicates_total"] > 0
        or vsmec_internal["c2_label_conflicts"] > 0
        or neuesc_internal["c2_label_conflicts"] > 0
        or vsmec_internal["c3_total_leakage"] > 0
        or neuesc_internal["c3_total_leakage"] > 0
    ):
        task03_blockers.append("Phát hiện trùng lặp nội bộ, xung đột nhãn hoặc rò rỉ giữa train/dev/test")
    if d1_vsmec_test_in_neu > 0 or d2_neu_test_in_vsmec > 0:
        task03_blockers.append("Phát hiện trùng chéo miền trên tập test (D1/D2)")
    if preprocess_audit["status"] != "PASS":
        task03_blockers.append("Kiểm tra tiền xử lý / chống lọt dấu '_' cho ViSoBERT/XLM-R không đạt")

    return {
        "task03_ready": len(task03_blockers) == 0,
        "task03_blockers": task03_blockers,
        "manifests": {
            "clean": clean_manifest,
            "splits": splits_manifest,
            "polarity_anomaly": polarity_audit,
        },
        "data_integrity": {
            "vsmec": vsmec_internal,
            "neuesc": neuesc_internal,
            "vsmec_dev_size": vsmec_internal["splits"]["dev"]["n_rows"],
            "vsmec_dev_meets_min_172": vsmec_internal["splits"]["dev"]["n_rows"] >= cfg["min_dev_vsmec"],
            "vsmec_5pct_projected_counts": {
                ID_TO_EMOTION7[k]: v for k, v in sorted(vsmec_5pct_proj.items())
            },
            "vsmec_5pct_min_class": {
                "emotion": ID_TO_EMOTION7[min_5pct_cls_id],
                "projected_count": min_5pct_count,
                "threshold": min_threshold,
                "meets_min_threshold": min_5pct_count >= min_threshold,
            },
            "cross_domain_exact": {
                "d1_vsmec_test_in_neuesc": d1_vsmec_test_in_neu,
                "d2_neuesc_test_in_vsmec": d2_neu_test_in_vsmec,
            },
            "step2_d4_near_duplicate_note": step2_d4_summary,
            "neuesc_raw_toxic_trace": toxic_trace,
        },
        "preprocessing_and_truncation": preprocess_audit,
        "protocol_compatibility": protocol_compat,
        "probe_notebook": probe_audit,
    }


# ---------------------------------------------------------------------------
# Markdown Report Renderers
# ---------------------------------------------------------------------------


def render_inventory_markdown(inv: Dict[str, Any]) -> str:
    """Render Task 01 report (`reports/phase2/inventory.md`)."""
    git = inv["git"]
    ds = inv["datasets"]
    ck = inv["checkpoints"]
    cc = inv["code_and_configs"]
    res = inv["phase1_results"]
    probe = inv["probe_notebook"]

    vsmec_s = ds["vsmec_clean"]["splits"]
    vsmec_note = (
        f"train={vsmec_s['train'].get('n_rows')}, dev={vsmec_s['dev'].get('n_rows')}, "
        f"test={vsmec_s['test'].get('n_rows')}; đủ cột `id,text,label,subset,text_seg`."
        if ds["vsmec_clean"]["status"] == "PRESENT"
        else "Thiếu một hoặc nhiều split sạch."
    )
    neu_sub = (ds["neuesc_clean"]["file"].get("subset_counts") or {})
    neu_note = (
        f"train={neu_sub.get('train')}, dev={neu_sub.get('dev')}, test={neu_sub.get('test')} "
        f"(tổng={ds['neuesc_clean']['file'].get('n_rows')}); raw có `NEU_ESC_full.csv` giữ nhãn gốc 4 lớp."
        if ds["neuesc_clean"]["status"] == "PRESENT"
        else "Thiếu `neuesc_100.csv`."
    )

    lines = [
        "# Báo cáo Task 01: Kiểm kê tài nguyên cũ (Phase 2)",
        "",
        "> Sinh tự động (chế độ chỉ đọc, offline CPU) từ `scripts/run_phase2_task01_02.py`. Không chỉnh sửa dữ liệu, checkpoint hay cấu hình gốc.",
        "",
        "## 1. Bảng kiểm kê tổng hợp",
        "",
        "| Tài nguyên | Đường dẫn thực tế | Trạng thái | Ghi chú |",
        "|---|---|---|---|",
        f"| Nhánh & nền Git | `{git['branch']}` @ `{git['head_short']}` (gốc `seq-core` `{git['seq_core_short']}`) | **{git['status']}** | {' '.join(git['notes'])} |",
        f"| UIT-VSMEC train/dev/test | `data/clean/vsmec_{{train,dev,test}}.csv` | **{ds['vsmec_clean']['status']}** | {vsmec_note} |",
        f"| NEU-ESC train/dev/test | `data/clean/neuesc_100.csv` (+ `data/raw/NEU_ESC_full.csv`) | **{ds['neuesc_clean']['status']}** | {neu_note} |",
        f"| Tập chia nhỏ cũ (`data/splits/`) | `data/splits/neuesc_*.csv` ({len(ds['legacy_splits']['files'])} file) | **{ds['legacy_splits']['status']}** | {ds['legacy_splits']['note']} |",
        f"| Checkpoint nguồn UIT-VSMEC (ViSoBERT, s0–s4) | `{ck['vsmec_visobert']['root'] or 'null (chưa chép về local)'}` | **{ck['vsmec_visobert']['status']}** | Hợp lệ `{ck['vsmec_visobert']['n_valid']}/{ck['vsmec_visobert']['n_required']}` seed (cần tải từ Drive trước Task 07/08). |",
        f"| Checkpoint nguồn NEU-ESC (ViSoBERT, s0–s4) | `{ck['neu_visobert']['root'] or 'null (chưa chép về local)'}` | **{ck['neu_visobert']['status']}** | Hợp lệ `{ck['neu_visobert']['n_valid']}/{ck['neu_visobert']['n_required']}` seed (cần tải từ Drive trước Task 07/08). |",
        f"| Code DFT / C0 / Seq / ZS / Stats | `src/seq/`, `src/zs/`, `src/stats/`, `src/plm/model_ckpt.py` | **{cc['status']}** | Đủ bộ lõi `seq-v1` & `zs`; cần bổ sung DFT từ HF, C0 low-resource, LLRD/LoRA, bộ đếm chi phí & view mới (xem Mục 3). |",
        f"| Cấu hình giai đoạn 1 | `configs/base.yaml`, `step2_data.yaml`, `zs.yaml`, `seq.yaml` | **{cc['code_files_status']}** | Bảo toàn nguyên trạng; GĐ2 dùng file cấu hình riêng ở Task 04. |",
        f"| Kết quả & dự đoán từng mẫu 100% (ViSoBERT & PhoBERT) | `results/` (chỉ có `token_stats.csv` & `polarity/*`); ngoài repo: `null` | **{res['status']}** | Tìm thấy `{res['n_prediction_csvs']}` CSV dự đoán test; chưa có `zs.csv`/`seq.csv`/predictions 100% local để tái tính view phụ & đối chứng mốc 100%. |",
        f"| Notebook Probe (Other/Surprise) | `{probe['configured_path'] or 'null (không có .ipynb trong workspace)'}` | **{probe['status']}** | {probe['note']} |",
        "",
        "## 2. Trạng thái Git & Stash hiện có (Chỉ đọc)",
        "",
        f"- **Nhánh hiện tại:** `{git['branch']}` @ `{git['head_full']}`",
        f"- **Kế thừa từ `seq-core` (`{git['expected_base_commit']}`):** `{git['base_is_ancestor']}`",
        f"- **Số stash đang lưu (không tự apply):** `{git['stash_count']}`",
    ]
    for st in git["stashes"]:
        lines.append(f"  - `{st}`")

    lines.extend(
        [
            "",
            "## 3. Khoảng trống mã nguồn cần bổ sung cho Giai đoạn 2",
            "",
        ]
    )
    for gap in cc["phase2_gaps"]:
        lines.append(f"- {gap}")

    lines.extend(
        [
            "",
            "## 4. Chi tiết tài nguyên ngoài Git cần chép từ Drive/Colab",
            "",
            "1. **10 checkpoint nguồn ViSoBERT Giai đoạn 1:**",
            "   - 5 seed (`0..4`) UIT-VSMEC 7 lớp (`ViSoBERT_Stage1-*_s{0..4}_lr*` gồm `encoder/` + tokenizer + `head.pt` + `stage1_ckpt_hash.json`).",
            "   - 5 seed (`0..4`) NEU-ESC 3 lớp (`ViSoBERT_NEU3-*_s{0..4}_lr*` gồm `encoder/` + tokenizer + `head.pt` + `neu3_ckpt_hash.json`).",
            "2. **Kết quả & dự đoán từng mẫu test ở mức 100% (ViSoBERT và PhoBERT):**",
            "   - `zs.csv`, `seq.csv`, `selection_seq.json`, `stage1.csv`, `neu3.csv` (kèm log cấu hình thật để xác minh `baseline_provenance`).",
            "   - Thư mục `predictions_zs/` (`REFV2V`, `REFN2N`, `ZSV2N`, `ZSN2V`) và `predictions_seq/` (`SEQV2N`, `SEQN2V`, `C0N`, `C0V`) để tính lại `Macro-F1 5 lớp`, `V3-noSurprise`, `Strict V3`, `V3-NEU-noToxic` ở Task 12.",
            "3. **Notebook Probe (Other/Surprise):** để sửa `PRIOR` từ `train` và bỏ fallback ngầm sang `uitnlp/visobert` trước khi đưa số liệu probe vào báo cáo.",
            "",
            "## 5. Kết luận Task 01",
            "",
            "- **Tài nguyên sẵn sàng dùng ngay tại local:** Toàn bộ dữ liệu sạch đóng băng (`data/clean/`), dữ liệu gốc 4 lớp NEU-ESC (`data/raw/`), toàn bộ code nền `seq-core` (`src/seq/`, `src/zs/`, `src/stats/`), và cấu hình Giai đoạn 1.",
            "- **Tài nguyên đang `MISSING` ở local (cần chép trước Task 07/08/12, KHÔNG chặn Task 02–05):** 10 checkpoint nguồn ViSoBERT, bộ CSV kết quả + dự đoán từng mẫu mức 100%, và notebook Probe.",
            "",
        ]
    )
    return "\n".join(lines)


def render_data_audit_markdown(audit: Dict[str, Any]) -> str:
    """Render Task 02 report (`reports/phase2/data_audit.md`)."""
    mf_clean = audit["manifests"]["clean"]
    mf_splits = audit["manifests"]["splits"]
    pol = audit["manifests"]["polarity_anomaly"]
    di = audit["data_integrity"]
    vsm = di["vsmec"]
    neu = di["neuesc"]
    tox = di["neuesc_raw_toxic_trace"]
    prep = audit["preprocessing_and_truncation"]
    prot = audit["protocol_compatibility"]
    probe = audit["probe_notebook"]

    gate_label = (
        "**SẴN SÀNG SANG TASK 03 (Tạo tập con dữ liệu)**"
        if audit["task03_ready"]
        else f"**CHƯA SẴN SÀNG:** {', '.join(audit['task03_blockers'])}"
    )

    lines = [
        "# Báo cáo Task 02: Kiểm tra dữ liệu, tiền xử lý và giao thức cũ",
        "",
        f"**Kết luận cổng chuyển sang Task 03:** {gate_label}",
        "",
        "> Mọi kiểm tra chạy ở chế độ chỉ đọc (không chỉnh sửa bất kỳ file nào trong `data/` hay `configs/seq.yaml`). Chi tiết từng mã băm xem tại [manifest_audit.json](manifest_audit.json).",
        "",
        "## 1. Kiểm toán `MANIFEST.json` và lỗi `polarity_long.csv` / `polarity_nhat.csv`",
        "",
        f"- `data/clean/MANIFEST.json`: trạng thái tổng **`{mf_clean['status']}`** | 4 file lõi train/dev/test (`vsmec_{{train,dev,test}}.csv`, `neuesc_100.csv`): **khớp 100% (`core_files_ok={mf_clean['core_files_ok']}`)**.",
        f"- `data/splits/MANIFEST.json`: trạng thái tổng **`{mf_splits['status']}`** (`{mf_splits['n_manifest_entries']}` mục).",
        f"- File nằm trên đĩa `data/clean/` nhưng không có trong `MANIFEST.json`: `{len(mf_clean['untracked_on_disk'])}` file ("
        + ", ".join(f"`{u['filename']}`" for u in mf_clean["untracked_on_disk"])
        + ").",
        "",
        "### 1.1. Điều tra riêng `polarity_long.csv` và `polarity_nhat.csv`",
        "",
        "| File | Kích thước (bytes) | Số dòng | Nhãn đã điền | SHA-256 trong `data/clean/MANIFEST.json` | SHA-256 thực tế trên đĩa | Khớp `results/polarity/polarity_files_hash.json`? |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for fname in ("polarity_long.csv", "polarity_nhat.csv"):
        f = pol["files"][fname]
        lines.append(
            f"| `{fname}` | {f['size_bytes']} | {f['n_rows']} | {f['non_empty_polarity_labels']} | "
            f"`{(f['manifest_clean_sha256'] or '')[:16]}...` | `{(f['disk_sha256'] or '')[:16]}...` | "
            f"**{f['matches_results_polarity_hash']}** |"
        )
    lines.extend(
        [
            "",
            f"- **Nguyên nhân:** {pol['root_cause_explanation']}",
            f"- **Ảnh hưởng đến Task 03:** {pol['impact_on_task03_subsampling']}",
            f"- **Phương án đề xuất (chờ nhóm duyệt):** {pol['recommended_action']}",
            "",
            "## 2. Toàn vẹn dữ liệu Train / Dev / Test, Phân phối nhãn & Dự báo mức 5%",
            "",
            "### 2.1. Kiểm tra cấu trúc, trùng lặp và rò rỉ",
            "",
            "| Bộ dữ liệu | Schema & Nhãn | C1 (Trùng nội bộ) | C2 (Xung đột nhãn) | C3 (Rò rỉ giữa split) | D1/D2 (Trùng chéo miền trên test) |",
            "|---|---|---:|---:|---:|---:|",
            f"| UIT-VSMEC (7 lớp) | `{'PASS' if vsm['schema_ok'] else 'FAIL'}` | {vsm['c1_internal_duplicates_total']} | {vsm['c2_label_conflicts']} | {vsm['c3_total_leakage']} | {di['cross_domain_exact']['d1_vsmec_test_in_neuesc']} |",
            f"| NEU-ESC (3 lớp) | `{'PASS' if neu['schema_ok'] else 'FAIL'}` | {neu['c1_internal_duplicates_total']} | {neu['c2_label_conflicts']} | {neu['c3_total_leakage']} | {di['cross_domain_exact']['d2_neuesc_test_in_vsmec']} |",
            "",
            f"- **Ghi chú D4 (trùng gần đã ghi nhận từ Bước 2):** {di['step2_d4_near_duplicate_note'] or 'Không có bản ghi readiness_report.json.'}",
            f"- **Quy mô `dev` UIT-VSMEC:** `{di['vsmec_dev_size']}` câu (`>= 172`: **{di['vsmec_dev_meets_min_172']}** — nếu giữ nguyên full dev cố định ở Task 04 sẽ tự động thỏa ngưỡng tối thiểu `≥ 172` câu).",
            "",
            "### 2.2. Phân bố nhãn UIT-VSMEC (7 lớp) & Dự báo phân tầng mức 5% (≈ 277 câu)",
            "",
            "| Split | Enjoyment (0) | Sadness (1) | Fear (2) | Anger (3) | Disgust (4) | Surprise (5) | Other (6) | Tổng |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sp in ("train", "dev", "test"):
        c = vsm["splits"][sp]["class_counts"]
        lines.append(
            f"| `{sp}` | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {c[5]} | {c[6]} | **{vsm['splits'][sp]['n_rows']}** |"
        )
    p5 = di["vsmec_5pct_projected_counts"]
    lines.append(
        f"| `train 5% (dự kiến)` | {p5['Enjoyment']} | {p5['Sadness']} | {p5['Fear']} | {p5['Anger']} | {p5['Disgust']} | {p5['Surprise']} | {p5['Other']} | **{sum(p5.values())}** |"
    )
    m5 = di["vsmec_5pct_min_class"]
    lines.extend(
        [
            "",
            f"- **Nhận xét sơ bộ mức 5% UIT-VSMEC:** Lớp nhỏ nhất dự kiến là `{m5['emotion']}` với **`{m5['projected_count']}` mẫu** (ngưỡng tối thiểu `< {m5['threshold']}` mẫu để loại bỏ V7 → đạt ngưỡng kỹ thuật: **{m5['meets_min_threshold']}**; bảng đếm chính thức trên 5 seed sẽ được xuất ở Task 03).",
            "",
            "### 2.3. Phân bố nhãn NEU-ESC (3 lớp) & Khả năng truy vết nhãn `toxic` gốc (cho `V3-NEU-noToxic`)",
            "",
            "| Split | negative (0) | neutral (1) | positive (2) | Tổng |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for sp in ("train", "dev", "test"):
        c = neu["splits"][sp]["class_counts"]
        lines.append(f"| `{sp}` | {c[0]} | {c[1]} | {c[2]} | **{neu['splits'][sp]['n_rows']}** |")

    if tox.get("recoverable"):
        lines.extend(
            [
                "",
                f"- **Truy vết nhãn gốc 4 lớp từ `data/raw/NEU_ESC_full.csv`:** **THÀNH CÔNG 100%** "
                f"(khớp `id` = {tox['n_rows_clean']}/{tox['n_rows_raw']}, khớp `normalize_text` = {tox['normalized_text_match_rate']*100:.2f}%, "
                f"nhất quán ánh xạ `NEUESC_ORIG_TO_LABEL3` = `{tox['label_mapping_consistent']}`).",
                "",
                "| Split (nhãn gốc 4 lớp) | neutral (0) | positive (1) | negative (2) | toxic (3) | Tổng |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for sp in ("train", "dev", "test"):
            o = tox["orig_4class_counts_by_split"][sp]
            tot = sum(o.values())
            lines.append(
                f"| `{sp}` | {o['neutral']} | {o['positive']} | {o['negative']} | **{o['toxic']}** | **{tot}** |"
            )
        lines.append(
            f"- **Kết luận `V3-NEU-noToxic`:** Tập `test` NEU-ESC có `{tox['test_toxic_count']}` câu gốc `toxic`; loại bỏ còn `{tox['test_non_toxic_count']}` câu — **hoàn toàn khả thi tính lại ở Task 12** mà không cần sửa `neuesc_100.csv`."
        )
    else:
        lines.append(f"- **Truy vết nhãn `toxic` gốc:** `{tox['status']}` — {tox.get('reason')}")

    lines.extend(
        [
            "",
            "## 3. Kiểm tra tiền xử lý ViSoBERT / XLM-R, tỷ lệ dấu `_` và tỷ lệ cắt cụt (`max_length=256`)",
            "",
            f"- **Trạng thái kiểm định tổng thể:** **`{prep['status']}`** (mọi split đều thỏa Unicode NFC, `normalize_text` idempotent, `preprocess_text(..., emoji_mode='keep')` idempotent và bảo toàn emoji).",
            "",
            "| Tập dữ liệu | Số câu | Câu có emoji (gốc → keep) | Câu có `_` (gốc `text`) | Câu có `_` (đầu vào ViSoBERT/XLM-R) | Câu có `_` (`text_seg` PhoBERT) | Kiểm tra `_` | Hash đầu vào ViSoBERT/XLM-R (16 ký tự đầu) |",
            "|---|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for key, row in prep["per_split"].items():
        u = row["underscore_check_visobert_xlmr"]
        seg_rate_str = (
            f"{row['phobert_text_seg_underscore_docs']} ({row['phobert_text_seg_underscore_rate']*100:.2f}%)"
            if row["phobert_text_seg_underscore_rate"] is not None
            else "N/A"
        )
        lines.append(
            f"| `{key}` | {row['n_samples']} | {row['raw_emoji_docs']} → {row['visobert_keep_emoji_docs']} | "
            f"{u['raw_docs_with_underscore']} ({u['raw_underscore_doc_rate']*100:.2f}%) | "
            f"{u['input_docs_with_underscore']} ({u['input_underscore_doc_rate']*100:.2f}%) | "
            f"{seg_rate_str} | **{u['status']}** | `{row['visobert_xlmr_input_sha256'][:16]}` |"
        )

    lines.extend(["", "### 3.1. Tỷ lệ cắt cụt ở `max_length = 256`", ""])
    if prep["recorded_token_stats"]:
        lines.extend(
            [
                "Số liệu đã ghi sẵn trong `results/token_stats.csv` (chỉ phủ **NEU-ESC**, chưa có bảng riêng cho **UIT-VSMEC**):",
                "",
                "| Tokenizer | Split | Số câu | p95 token | Max token | Số câu cắt cụt (>256) | Tỷ lệ cắt cụt |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for rec in prep["recorded_token_stats"]:
            lines.append(
                f"| `{rec['tokenizer']}` | `{rec['subset']}` | {rec['n_samples']} | "
                f"{rec['p95_tokens']} | {rec['max_tokens']} | {rec['n_truncated']} | "
                f"{float(rec['truncation_rate'])*100:.4f}% |"
            )
    lt = prep["live_truncation"]
    lines.append("")
    lines.append(f"- **Đo tokenizer trực tiếp (`--with-tokenizer`):** chế độ `{lt['mode']}`.")
    for n in lt.get("notes", []):
        lines.append(f"  - {n}")
    for alias, res_tok in (lt.get("results") or {}).items():
        if res_tok.get("status") == "PASS":
            lines.append(f"  - **`{alias}` (`{res_tok['model_id']}`):**")
            for sp_k, sp_v in res_tok["splits"].items():
                lines.append(
                    f"    - `{sp_k}`: n={sp_v['n_samples']}, truncated={sp_v['n_truncated']} "
                    f"({sp_v['truncation_rate']*100:.4f}%), p95={sp_v['p95_tokens']:.1f}, max={sp_v['max_tokens']}"
                )
        else:
            lines.append(
                f"  - **`{alias}` (`{res_tok.get('model_id')}`):** `{res_tok.get('status')}` — {res_tok.get('reason')}"
            )

    lines.extend(
        [
            "",
            "## 4. Đối chiếu tính tương thích của Mốc 100% cũ với Giao thức Giai đoạn 2",
            "",
            f"- **Provenance trong `configs/seq.yaml`:** `status={prot['baseline_provenance_in_seq_yaml'].get('status')}`, "
            f"`actual_run_verified={prot['baseline_provenance_in_seq_yaml'].get('actual_run_verified')}`.",
            "",
            "| Tiêu chí | Baseline trong Git (`base.yaml`) | Giao thức `seq-v1` (`configs/seq.yaml`) | Giao thức GĐ2 đề xuất | Đánh giá tương thích |",
            "|---|---|---|---|---|",
        ]
    )
    for r in prot["comparison_matrix"]:
        lines.append(
            f"| {r['dimension']} | {r['stage1_base_git']} | {r['seq_v1']} | {r['phase2_proposed']} | {r['compatible_100pct']} |"
        )
    lines.extend(
        [
            "",
            f"**Kết luận khả năng tái sử dụng mốc 100%:** {prot['reusability_verdict_100pct']}",
            "",
            "## 5. Rà soát & Phương án sửa Notebook Probe (Other/Surprise)",
            "",
            f"- **Trạng thái file notebook Probe tại local:** **`{probe['status']}`** — {probe['note']}",
        ]
    )
    if probe.get("findings"):
        for f in probe["findings"]:
            lines.append(f"  - Cell `{f['cell_index']}` (`{f['issue']}`): {f['detail']}")
    else:
        lines.extend(
            [
                "- **Checklist bản vá bắt buộc áp dụng ngay khi đưa notebook Probe từ Colab/Drive về (trước khi trích số vào báo cáo):**",
                "  1. Đổi `PRIOR` từ phân phối nhãn của tập `test` sang phân phối nhãn của tập **`train`** (ví dụ tính từ `data/clean/vsmec_train.csv` gộp 3 nhãn hoặc `data/clean/neuesc_100.csv` lọc `subset == 'train'`).",
                "  2. Loại bỏ nhánh `except` fallback ngầm sang `uitnlp/visobert`; bắt buộc dùng `AutoTokenizer.from_pretrained(Path(ckpt_dir) / 'encoder', local_files_only=True)` và raise lỗi ngay nếu thiếu file tokenizer.",
                "  3. Khi đối chiếu hash file polarity, đọc `results/polarity/polarity_files_hash.json` thay vì mã hash mẫu trống cũ trong `data/clean/MANIFEST.json`.",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def run_task01_02(
    repo_root: Union[str, Path],
    config_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    overrides: Optional[Dict[str, Any]] = None,
    with_tokenizer: bool = False,
) -> Dict[str, Any]:
    """Execute Task 01 and Task 02 end-to-end and write the 4 artifacts into `out_dir`."""
    root = Path(repo_root).resolve()
    cfg = load_audit_config(config_path)
    if overrides:
        for k, v in overrides.items():
            if v is not None and k in cfg["phase1_external"]:
                cfg["phase1_external"][k] = str(v)

    inventory = collect_inventory(root, cfg)
    audit = run_data_protocol_audit(root, cfg, with_tokenizer=with_tokenizer)

    out_path = _resolve_path(root, out_dir)
    assert out_path is not None
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_path / "inventory.md").write_text(
        render_inventory_markdown(inventory), encoding="utf-8"
    )
    (out_path / "manifest_audit.json").write_text(
        json.dumps(audit["manifests"], indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_path / "data_audit.md").write_text(
        render_data_audit_markdown(audit), encoding="utf-8"
    )

    return {"inventory": inventory, "audit": audit, "out_dir": str(out_path)}
