"""Phase 2 executable protocol contract (`phase2-v1`), run-ID builder, dev metrics & stats helpers.

Locked at Task 04 (`configs/phase2.yaml` & `reports/phase2/decisions.md`).
Preserves frozen `src/seq/` and `src/zs/` without modification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score
import yaml

from src.stats.bootstrap import bootstrap_contrast, safe_ratio, weighted_statistics
from src.stats.holm import bootstrap_pvalue, holm_adjust

PROTOCOL_VERSION = "phase2-v1"
PROTOCOL_TAG = "p2v1"

VSMEC_V5_SCORE_LABELS: Tuple[int, ...] = (0, 1, 2, 3, 4)
VSMEC_V7_ALL_LABELS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
NEUESC_V3_SCORE_LABELS: Tuple[int, ...] = (0, 1, 2)

EXPECTED_PRIMARY_FAMILY: Tuple[Tuple[str, str, str, int, str, str, str], ...] = (
    ("P2_SA_10", "ViSoBERT", "SA", 10, "seq_minus_dft", "V3-NEU", "macro_f1"),
    ("P2_SB_10", "ViSoBERT", "SB", 10, "seq_minus_dft", "V7-VSMEC", "macro_f1"),
)

_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9.\-]+$")


def canonical_digest(payload: Any) -> str:
    """Deterministic SHA-256 hex digest over JSON-normalized structure."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_phase2_config(path: Union[str, Path] = "configs/phase2.yaml") -> Dict[str, Any]:
    """Load and strictly validate the locked `phase2-v1` YAML protocol contract."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing Phase 2 config: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Phase 2 config must be a YAML mapping: {cfg_path}")

    if cfg.get("protocol_version") != PROTOCOL_VERSION or cfg.get("protocol_tag") != PROTOCOL_TAG:
        raise ValueError(
            f"Unsupported Phase 2 protocol version/tag: {cfg.get('protocol_version')!r} / {cfg.get('protocol_tag')!r}"
        )

    # Validate backbone preprocessing rules (no segmented text in ViSoBERT or XLMR)
    backbones = cfg.get("backbones") or {}
    for alias in ("ViSoBERT", "XLMR"):
        spec = backbones.get(alias) or {}
        if spec.get("text_col") != "text" or spec.get("emoji_mode") != "keep":
            raise ValueError(f"{alias} must use raw text_col='text' and emoji_mode='keep'")
    if (backbones.get("PhoBERT") or {}).get("role") != "historical_reference_only":
        raise ValueError("PhoBERT is locked as historical_reference_only in phase2-v1")

    # Validate training invariants
    tr = cfg.get("training") or {}
    expected_tr = {
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 32,
        "epochs": 10,
        "evals_per_epoch": 1,
        "patience": 2,
        "warmup_ratio": 0.10,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "dropout": 0.1,
        "max_length": 256,
        "precision": "bf16",
        "strict_precision": True,
    }
    if tr != expected_tr:
        raise ValueError(
            f"Training profile differs from locked phase2-v1; update protocol_version if intentional. Got: {tr}"
        )

    # Validate fixed dev/test & dev selection criteria
    data_cfg = cfg.get("data") or {}
    if data_cfg.get("fixed_dev_test") is not True or data_cfg.get("core_fractions") != [10, 25, 50]:
        raise ValueError("phase2-v1 requires fixed_dev_test=True and core_fractions=[10, 25, 50]")

    dev_sel = cfg.get("dev_selection") or {}
    vsm_dev = dev_sel.get("vsmec") or {}
    neu_dev = dev_sel.get("neuesc") or {}
    if (
        vsm_dev.get("metric") != "macro_f1_v5"
        or tuple(vsm_dev.get("score_labels") or ()) != VSMEC_V5_SCORE_LABELS
        or int(vsm_dev.get("predict_num_labels", 0)) != 7
        or vsm_dev.get("filter_dev_rows") is not False
    ):
        raise ValueError("UIT-VSMEC dev selection must use macro_f1_v5 on full 7-class dev predictions")
    if (
        neu_dev.get("metric") != "macro_f1_v3"
        or tuple(neu_dev.get("score_labels") or ()) != NEUESC_V3_SCORE_LABELS
        or int(neu_dev.get("predict_num_labels", 0)) != 3
        or neu_dev.get("filter_dev_rows") is not False
    ):
        raise ValueError("NEU-ESC dev selection must use macro_f1_v3 on full 3-class dev predictions")

    # Validate C0 low-resource safety invariants
    c0 = cfg.get("c0_protocol") or {}
    if (
        c0.get("mode") != "low_resource_two_stage"
        or c0.get("stage1_source") != "reuse_best_dft_same_target_fraction_seed"
        or c0.get("forbid_100pct_target_checkpoint_below_100") is not True
        or c0.get("stage2_head_init") != "fresh_random_matching_seq_seed"
        or c0.get("reset_optimizer_and_scheduler") is not True
        or c0.get("lr_mode") != "reuse_seq_selected"
    ):
        raise ValueError("C0 protocol violates locked phase2-v1 low-resource control invariants")

    # Validate mandatory 100% SA reruns (5 DFT + 5 Seq = 10 runs)
    anchor = cfg.get("anchor_100pct") or {}
    reruns = anchor.get("mandatory_phase2_v1_reruns") or []
    if (
        len(reruns) != 1
        or reruns[0].get("direction") != "SA"
        or sorted(reruns[0].get("methods") or []) != ["dft", "seq"]
        or reruns[0].get("fraction") != 100
        or reruns[0].get("seeds") != [0, 1, 2, 3, 4]
        or int(anchor.get("mandatory_rerun_count", 0)) != 10
    ):
        raise ValueError("phase2-v1 locks mandatory 5 DFT + 5 Seq reruns at 100% Direction A (SA)")

    # Validate seeds & statistical family (Holm m=2)
    seeds_cfg = cfg.get("seeds") or {}
    if (
        seeds_cfg.get("core") != [0, 1, 2, 3, 4]
        or seeds_cfg.get("c0") != [0, 1, 2]
        or seeds_cfg.get("xlmr") != [0, 1, 2]
        or seeds_cfg.get("sweep_seed") != 0
    ):
        raise ValueError("Seeds configuration differs from locked phase2-v1")

    stats_cfg = cfg.get("statistics") or {}
    if int(stats_cfg.get("holm_family_size", 0)) != 2 or float(stats_cfg.get("alpha", 0.0)) != 0.05:
        raise ValueError("phase2-v1 fixes Holm confirmatory family size m=2 at alpha=0.05")
    actual_primary = tuple(
        (
            p["id"],
            p["backbone"],
            p["direction"],
            int(p["fraction"]),
            p["contrast"],
            p["label_view"],
            p["stat"],
        )
        for p in (stats_cfg.get("primary") or [])
    )
    if actual_primary != EXPECTED_PRIMARY_FAMILY:
        raise ValueError(f"Primary family differs from locked phase2-v1 comparisons: {actual_primary}")

    return cfg


def compute_effective_cfg_hash8(
    protocol_cfg: Mapping[str, Any],
    run_overrides: Optional[Mapping[str, Any]] = None,
) -> str:
    """Compute an 8-hex-char SHA-256 fingerprint of effective training + selection settings."""
    effective = {
        "protocol_version": protocol_cfg["protocol_version"],
        "training": protocol_cfg["training"],
        "dev_selection": protocol_cfg["dev_selection"],
        "c0_protocol": protocol_cfg["c0_protocol"],
        "overrides": dict(run_overrides or {}),
    }
    return canonical_digest(effective)[:8]


def build_phase2_run_id(
    backbone: str,
    direction: str,
    method_tag: str,
    fraction: int,
    seed: int,
    lr: float,
    protocol_cfg: Mapping[str, Any],
    *,
    run_overrides: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build collision-proof Phase 2 run ID including `p2v1` and 8-char config hash.

    Format: `{backbone}_{direction}_{method_tag}_f{fraction}_s{seed}_lr{lr:g}_{protocol_tag}_h{cfg_hash8}`
    Example: `ViSoBERT_SA_seq-lora-r8_f10_s0_lr0.0003_p2v1_h4b91e2a7`
    """
    for label, token in (("backbone", backbone), ("direction", direction), ("method_tag", method_tag)):
        if not token or "_" in token or not _SAFE_TOKEN_RE.fullmatch(token):
            raise ValueError(
                f"{label} must be nonempty alphanumeric/dash/dot without underscores, got {token!r}"
            )
    frac = int(fraction)
    if frac not in (5, 10, 25, 50, 100):
        raise ValueError(f"Invalid fraction: {fraction!r}")
    protocol_tag = str(protocol_cfg.get("protocol_tag", PROTOCOL_TAG))
    cfg_hash8 = compute_effective_cfg_hash8(protocol_cfg, run_overrides=run_overrides)
    lr_str = f"{float(lr):g}"
    return f"{backbone}_{direction}_{method_tag}_f{frac}_s{int(seed)}_lr{lr_str}_{protocol_tag}_h{cfg_hash8}"


def should_extend_lora_epochs(
    best_epoch_1based: int,
    early_stopped: bool,
    protocol_cfg: Mapping[str, Any],
) -> Tuple[bool, int]:
    """Pre-registered objective rule to decide whether LoRA extends from 10 to 15 epochs on dev seed 0."""
    base_epochs = int(protocol_cfg["training"]["epochs"])
    ext_cfg = protocol_cfg["sweeps"]["lora"]["unconverged_extension"]
    if not ext_cfg.get("enabled", False):
        return False, base_epochs
    triggers = {int(e) for e in ext_cfg.get("trigger_best_epochs_at_10", [9, 10])}
    if int(best_epoch_1based) in triggers and (not early_stopped if ext_cfg.get("require_no_early_stop", True) else True):
        return True, int(ext_cfg["extended_max_epochs"])
    return False, base_epochs


def compute_dev_selection_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    target_domain: str,
) -> float:
    """Compute locked Phase 2 dev checkpoint selection metric without filtering dev rows.

    - `vsmec`: predicts 7 classes `(0..6)` on ALL 686 dev sentences; computes per-class F1 on
      each of `{0,1,2,3,4}` (so false positives/negatives involving Surprise=5 or Other=6 still
      penalize precision/recall properly) and averages those 5 F1 scores (`macro_f1_v5`).
    - `neuesc`: computes standard 3-class Macro-F1 (`{0,1,2}`) on ALL 3,305 dev sentences.
    """
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or pred.shape != truth.shape or truth.size == 0:
        raise ValueError("y_true and y_pred must be matching nonempty 1D integer vectors")

    if target_domain == "vsmec":
        if not np.isin(truth, VSMEC_V7_ALL_LABELS).all() or not np.isin(pred, VSMEC_V7_ALL_LABELS).all():
            raise ValueError("UIT-VSMEC dev labels/predictions must lie in 0..6")
        return float(
            f1_score(truth, pred, labels=list(VSMEC_V5_SCORE_LABELS), average="macro", zero_division=0)
        )
    if target_domain == "neuesc":
        if not np.isin(truth, NEUESC_V3_SCORE_LABELS).all() or not np.isin(pred, NEUESC_V3_SCORE_LABELS).all():
            raise ValueError("NEU-ESC dev labels/predictions must lie in 0..2")
        return float(
            f1_score(truth, pred, labels=list(NEUESC_V3_SCORE_LABELS), average="macro", zero_division=0)
        )
    raise ValueError(f"Unknown target_domain: {target_domain!r}")


def apply_phase2_holm(
    rows: List[Dict[str, Any]],
    protocol_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Apply Holm step-down adjustment strictly over the locked m=2 confirmatory family at 10%.

    Never shrinks `m` when a primary comparison is missing or has fewer than 5 paired seeds;
    instead withholds `p_holm` (`np.nan`) and `significant` (`None`). All other comparisons are
    marked `exploratory = True`.
    """
    stats_cfg = protocol_cfg["statistics"]
    expected_m = int(stats_cfg["holm_family_size"])
    alpha = float(stats_cfg["alpha"])
    primary_keys = {
        (p["backbone"], p["direction"], int(p["fraction"]), p["contrast"], p["label_view"], p["stat"])
        for p in stats_cfg["primary"]
    }

    selected: List[Dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("backbone"),
            row.get("direction"),
            int(row.get("fraction", -1)),
            row.get("contrast"),
            row.get("label_view"),
            row.get("stat"),
        )
        is_primary = key in primary_keys
        row["primary"] = is_primary
        row["exploratory"] = not is_primary
        row["p_holm"] = np.nan
        row["significant"] = None
        if is_primary:
            selected.append(row)

    actual_keys = {
        (r["backbone"], r["direction"], int(r["fraction"]), r["contrast"], r["label_view"], r["stat"])
        for r in selected
    }
    if (
        len(selected) == expected_m
        and actual_keys == primary_keys
        and all(int(r.get("n_seeds", 0)) == 5 for r in selected)
        and all(np.isfinite(float(r.get("p_boot", np.nan))) for r in selected)
    ):
        adjusted = holm_adjust([float(r["p_boot"]) for r in selected])
        for row, adj in zip(selected, adjusted):
            row["p_holm"] = float(adj)
            row["significant"] = bool(adj < alpha)
    return rows


def bootstrap_seed_sd_ratio(
    ref_arm: Mapping[str, Any],
    new_arm: Mapping[str, Any],
    labels: Sequence[int],
    seeds: Sequence[int],
    *,
    stat: str = "macro_f1",
    boot: int = 10000,
    boot_seed: int = 0,
    ci: float = 0.95,
    chunk_size: int = 500,
    zero_denom_threshold: float = 1e-8,
) -> Dict[str, Any]:
    """Paired two-level bootstrap CI for between-seed SD ratio `SD(new) / SD(ref)` (exploratory).

    Handles `SD(ref) == 0` safely:
    - If point `SD(ref) < zero_denom_threshold`, returns `defined=False`, `point=np.nan`,
      `ci_lo=np.nan`, `ci_hi=np.nan`.
    - In bootstrap replicates where all resampled seeds happen to be identical (`SD*(ref) == 0`),
      marks those replicates as `NaN` via `safe_ratio`, excludes them from quantiles, and records
      their exact count in `n_nan`.
    """
    if ref_arm["ids"].tolist() != new_arm["ids"].tolist() or not np.array_equal(ref_arm["y_true"], new_arm["y_true"]):
        raise ValueError("SD ratio arms must share identical ordered test IDs and true labels")
    expected_seeds = list(seeds)
    if list(ref_arm["seeds"]) != expected_seeds or list(new_arm["seeds"]) != expected_seeds:
        raise ValueError("SD ratio arms must share identical ordered seed IDs")
    if len(expected_seeds) < 2:
        raise ValueError("Between-seed SD ratio requires at least 2 seeds")

    y_true = np.asarray(ref_arm["y_true"], dtype=np.int64)
    ones = np.ones((1, len(y_true)), dtype=np.float64)
    ref_seed_scores = weighted_statistics(y_true, ref_arm["preds"], labels, ones, stat=stat)[0]
    new_seed_scores = weighted_statistics(y_true, new_arm["preds"], labels, ones, stat=stat)[0]

    ref_sd = float(np.std(ref_seed_scores, ddof=1))
    new_sd = float(np.std(new_seed_scores, ddof=1))
    point_ratio = float(safe_ratio(new_sd, ref_sd, threshold=zero_denom_threshold))

    if not np.isfinite(point_ratio):
        return {
            "defined": False,
            "ref_sd": ref_sd,
            "new_sd": new_sd,
            "point": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "se": np.nan,
            "n_nan": int(boot),
            "n_seeds": len(expected_seeds),
        }

    rng = np.random.Generator(np.random.PCG64(boot_seed))
    n_test = len(y_true)
    n_seeds = len(expected_seeds)
    draws = np.empty(boot, dtype=np.float64)

    for start in range(0, boot, chunk_size):
        size = min(chunk_size, boot - start)
        w = rng.multinomial(n_test, np.full(n_test, 1.0 / n_test), size=size)
        ref_mat = weighted_statistics(y_true, ref_arm["preds"], labels, w, stat=stat)
        new_mat = weighted_statistics(y_true, new_arm["preds"], labels, w, stat=stat)
        seed_idx = rng.integers(0, n_seeds, size=(size, n_seeds))
        ref_sub = np.take_along_axis(ref_mat, seed_idx, axis=1)
        new_sub = np.take_along_axis(new_mat, seed_idx, axis=1)
        ref_sub_sd = np.std(ref_sub, axis=1, ddof=1)
        new_sub_sd = np.std(new_sub, axis=1, ddof=1)
        draws[start : start + size] = safe_ratio(new_sub_sd, ref_sub_sd, threshold=zero_denom_threshold)

    finite = draws[np.isfinite(draws)]
    alpha_tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(finite, [alpha_tail, 1.0 - alpha_tail]) if len(finite) else (np.nan, np.nan)
    return {
        "defined": True,
        "ref_sd": ref_sd,
        "new_sd": new_sd,
        "point": point_ratio,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
        "n_nan": int(len(draws) - len(finite)),
        "n_seeds": n_seeds,
    }
