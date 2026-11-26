"""Phase 2 Task 03: Monotone nested stratified subset generator, verifier, and shared loader.

Guarantees:
- Strictly nested subsets per seed: 5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100% (both total IDs and per-class IDs).
- Monotone per-class allocation avoiding Hamilton / largest-remainder non-monotonicity (Alabama paradox).
- Dev and test splits remain 100% intact (never subsampled).
- Byte-for-byte deterministic CSV/JSON outputs (UTF-8 + LF line endings).
- Read-only on frozen `data/clean/` and `data/splits/`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from src.common.constants import EMOTIONS7, ID_TO_EMOTION7, ID_TO_LABEL3
from src.common.hashing import hash_text, sha256_file

SCHEMA_VERSION = "phase2-subsets-v1"
RNG_BASE_SEED = 20261126
RNG_SPEC = "PCG64([20261126, dataset_code, seed, class_id])"

SUBSET_INDEX_COLUMNS: Tuple[str, ...] = ("id", "label", "text_hash")
DATASET_CODES: Dict[str, int] = {"vsmec": 1, "neuesc": 2}
DATASET_VALID_LABELS: Dict[str, Tuple[int, ...]] = {
    "vsmec": tuple(range(7)),
    "neuesc": tuple(range(3)),
}
DATASET_LABEL_NAMES: Dict[str, Dict[int, str]] = {
    "vsmec": dict(ID_TO_EMOTION7),
    "neuesc": dict(ID_TO_LABEL3),
}


def load_subsets_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load `configs/phase2_subsets.yaml` with normalized defaults."""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(raw, dict):
        raise ValueError(f"Subset config must be a YAML mapping: {path}")
    return {
        "schema_version": str(raw.get("schema_version", SCHEMA_VERSION)),
        "clean_dir": str(raw.get("clean_dir", "data/clean")),
        "subsets_dir": str(raw.get("subsets_dir", "data/subsets_phase2")),
        "reports_dir": str(raw.get("reports_dir", "reports/phase2")),
        "datasets": list(raw.get("datasets", ["vsmec", "neuesc"])),
        "fractions": [int(f) for f in raw.get("fractions", [5, 10, 25, 50])],
        "drift_checked_fractions": [int(f) for f in raw.get("drift_checked_fractions", [10, 25, 50])],
        "seeds": [int(s) for s in raw.get("seeds", [0, 1, 2, 3, 4])],
        "max_drift_pp": float(raw.get("max_drift_pp", 1.0)),
        "min_vsmec_5pct_per_class": int(raw.get("min_vsmec_5pct_per_class", 5)),
        "rng_base_seed": int(raw.get("rng_base_seed", RNG_BASE_SEED)),
    }


def _resolve_path(repo_root: Path, value: Union[str, Path]) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


def _write_csv_lf(df: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        df.loc[:, list(columns)].to_csv(handle, index=False, lineterminator="\n")


def _write_json_lf(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _write_text_lf(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.replace("\r\n", "\n")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(normalized)


def _ids_sha256(ids: Sequence[int]) -> str:
    payload = "\n".join(str(int(x)) for x in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def subset_filename(dataset: str, fraction: int, seed: int) -> str:
    return f"{dataset}_f{int(fraction)}_s{int(seed)}_ids.csv"


def allocate_monotone_counts(
    label_counts: Dict[int, int],
    fractions: Sequence[int],
    min_per_class: int = 1,
) -> Dict[int, Dict[int, int]]:
    """Compute monotone stratified per-class sample counts across ascending fractions.

    Guarantees for every non-empty class `c` and consecutive tiers `p1 < p2`:
        `min_per_class <= n_c(p1) <= n_c(p2) <= label_counts[c]`
    and exact target totals `sum_c n_c(p) == round(N * p / 100)` (clamped to `[min_possible, N]`),
    preventing non-monotonic drops (Alabama paradox) that can happen under independent largest-remainder.
    """
    ordered_fracs = [int(f) for f in fractions]
    if not ordered_fracs or ordered_fracs != sorted(set(ordered_fracs)):
        raise ValueError(f"fractions must be strictly increasing unique ints, got {fractions!r}")
    if any(f <= 0 or f >= 100 for f in ordered_fracs):
        raise ValueError(f"fractions must lie strictly in (0, 100), got {fractions!r}")

    all_classes = sorted(int(k) for k in label_counts.keys())
    active = [c for c in all_classes if int(label_counts[c]) > 0]
    total_n = sum(int(label_counts[c]) for c in active)
    if total_n <= 0:
        raise ValueError("label_counts must have positive total")

    prev: Dict[int, int] = {c: 0 for c in active}
    schedule: Dict[int, Dict[int, int]] = {}

    for pct in ordered_fracs:
        min_floors = {c: min(int(label_counts[c]), max(min_per_class, prev[c])) for c in active}
        min_possible = sum(min_floors.values())
        target_total = int(round(total_n * pct / 100.0))
        target_total = min(total_n, max(min_possible, target_total))

        exact = {c: int(label_counts[c]) * (target_total / total_n) for c in active}
        alloc = {
            c: min(int(label_counts[c]), max(min_floors[c], int(math.floor(exact[c]))))
            for c in active
        }

        diff = target_total - sum(alloc.values())
        while diff > 0:
            eligible = [c for c in active if alloc[c] < int(label_counts[c])]
            if not eligible:
                break
            eligible.sort(
                key=lambda c: (exact[c] - alloc[c], int(label_counts[c]), -c),
                reverse=True,
            )
            step = min(diff, len(eligible))
            for c in eligible[:step]:
                alloc[c] += 1
            diff -= step

        while diff < 0:
            reducible = [c for c in active if alloc[c] > min_floors[c]]
            if not reducible:
                break
            reducible.sort(
                key=lambda c: (alloc[c] - exact[c], alloc[c] - min_floors[c], c),
                reverse=True,
            )
            step = min(-diff, len(reducible))
            for c in reducible[:step]:
                alloc[c] -= 1
            diff += step

        if sum(alloc.values()) != target_total:
            raise AssertionError(
                f"Allocation sum mismatch at {pct}%: got {sum(alloc.values())}, expected {target_total}"
            )
        for c in active:
            if not (prev[c] <= alloc[c] <= int(label_counts[c])):
                raise AssertionError(
                    f"Monotonicity violated for class {c} at {pct}%: prev={prev[c]}, cur={alloc[c]}"
                )

        schedule[pct] = {c: int(alloc.get(c, 0)) for c in all_classes}
        prev = dict(alloc)

    return schedule


def _load_clean_domain_splits(clean_dir: Path, dataset: str, include_test: bool = True) -> Dict[str, pd.DataFrame]:
    """Load canonical train/dev/test frames from `clean_dir`, sorted by integer `id`."""
    if dataset not in DATASET_CODES:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    if dataset == "vsmec":
        splits = {
            sp: pd.read_csv(clean_dir / f"vsmec_{sp}.csv", keep_default_na=False)
            for sp in (("train", "dev", "test") if include_test else ("train", "dev"))
        }
    else:
        full = pd.read_csv(clean_dir / "neuesc_100.csv", keep_default_na=False)
        splits = {
            sp: full.loc[full["subset"].eq(sp)].copy()
            for sp in (("train", "dev", "test") if include_test else ("train", "dev"))
        }

    valid_labels = set(DATASET_VALID_LABELS[dataset])
    cleaned: Dict[str, pd.DataFrame] = {}
    for sp, df in splits.items():
        frame = df.copy()
        if frame.empty:
            raise ValueError(f"{dataset}/{sp} is empty")
        ids = pd.to_numeric(frame["id"], errors="coerce")
        labels = pd.to_numeric(frame["label"], errors="coerce")
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"{dataset}/{sp} has null or duplicate IDs")
        if labels.isna().any() or not labels.isin(valid_labels).all():
            raise ValueError(f"{dataset}/{sp} has labels outside {sorted(valid_labels)}")
        frame["id"] = ids.astype(int)
        frame["label"] = labels.astype(int)
        frame = frame.sort_values("id", kind="mergesort").reset_index(drop=True)
        cleaned[sp] = frame
    return cleaned


def _source_hashes_for_clean_dir(clean_dir: Path, datasets: Sequence[str]) -> Dict[str, str]:
    files: List[str] = []
    if "vsmec" in datasets:
        files.extend(["vsmec_train.csv", "vsmec_dev.csv", "vsmec_test.csv"])
    if "neuesc" in datasets:
        files.append("neuesc_100.csv")
    return {fname: sha256_file(clean_dir / fname) for fname in sorted(set(files))}


def _permuted_class_indices(
    train_df: pd.DataFrame,
    dataset: str,
    seed: int,
    rng_base_seed: int = RNG_BASE_SEED,
) -> Dict[int, pd.DataFrame]:
    """Build one deterministic per-class permutation for `(dataset, seed)`."""
    ds_code = DATASET_CODES[dataset]
    ordered: Dict[int, pd.DataFrame] = {}
    for c in DATASET_VALID_LABELS[dataset]:
        cls_df = train_df.loc[train_df["label"] == c].sort_values("id", kind="mergesort").reset_index(drop=True)
        if cls_df.empty:
            ordered[c] = cls_df
            continue
        rng = np.random.Generator(np.random.PCG64([int(rng_base_seed), ds_code, int(seed), int(c)]))
        perm = rng.permutation(len(cls_df))
        ordered[c] = cls_df.iloc[perm].reset_index(drop=True)
    return ordered


def build_vsmec_5pct_counts_frame(
    subsets_dir: Path,
    seeds: Sequence[int],
    min_per_class_threshold: int = 5,
) -> pd.DataFrame:
    """Build per-seed 7-class count table at 5% UIT-VSMEC to decide V7 5% eligibility."""
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        path = subsets_dir / subset_filename("vsmec", 5, seed)
        idx_df = pd.read_csv(path, keep_default_na=False)
        labels = pd.to_numeric(idx_df["label"], errors="raise").astype(int)
        counts = {c: int((labels == c).sum()) for c in range(7)}
        min_cls = min(counts.keys(), key=lambda c: (counts[c], c))
        min_cnt = counts[min_cls]
        row: Dict[str, Any] = {
            "seed": int(seed),
            "total_rows": int(len(idx_df)),
        }
        for c, emo in enumerate(EMOTIONS7):
            row[f"c{c}_{emo}"] = counts[c]
        row["min_class_name"] = ID_TO_EMOTION7[min_cls]
        row["min_class_count"] = int(min_cnt)
        row["eligible_v7_min5"] = bool(min_cnt >= int(min_per_class_threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def load_phase2_target_splits(
    dataset: str,
    fraction: int,
    seed: int = 0,
    clean_dir: Union[str, Path] = "data/clean",
    subsets_dir: Union[str, Path] = "data/subsets_phase2",
    *,
    include_test: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """Shared strict loader for Task 05–11 (`DFT`, `C0`, `Seq`, `LLRD`, `LoRA`, `XLM-R`, `LLM`).

    By default returns full intact `dev_df` and `test_df`. With include_test=False,
    test is not validated or returned, and VSMEC test is never opened. NEU uses
    a shared CSV: test rows are filtered before validation. Test integrity and
    leakage checks are deferred to main; recorded source hashes identify test.
    For `fraction < 100`, validates
    `MANIFEST.json` SHA-256 digests, index schema, label/text_hash consistency, and zero
    leakage against `dev`/`test` before filtering `train_df` strictly by approved IDs.
    """
    if dataset not in DATASET_CODES:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    frac = int(fraction)
    if frac not in (5, 10, 25, 50, 100):
        raise ValueError(f"Unsupported fraction: {fraction!r}; expected one of 5, 10, 25, 50, 100")

    c_dir = Path(clean_dir)
    splits = _load_clean_domain_splits(c_dir, dataset, include_test=include_test)
    train_full = splits["train"]
    dev_df = splits["dev"]
    test_df = splits.get("test")

    dev_ids = set(dev_df["id"].tolist())
    test_ids = set(test_df["id"].tolist()) if include_test else set()
    dev_hashes = {hash_text(str(t), normalize=True) for t in dev_df["text"]}
    test_hashes = {hash_text(str(t), normalize=True) for t in test_df["text"]} if include_test else set()

    def source_hashes():
        if include_test or dataset != "vsmec":
            return _source_hashes_for_clean_dir(c_dir, [dataset])
        recorded = json.loads((Path(subsets_dir) / "MANIFEST.json").read_text(encoding="utf-8"))
        hashes = {f"vsmec_{sp}.csv": sha256_file(c_dir / f"vsmec_{sp}.csv") for sp in ("train", "dev")}
        hashes["vsmec_test.csv"] = recorded["source_files_sha256"]["vsmec_test.csv"]
        return hashes

    if frac == 100:
        train_ids = train_full["id"].tolist()
        meta = {
            "dataset": dataset,
            "fraction": 100,
            "seed": int(seed),
            "n_train": int(len(train_full)),
            "n_dev": int(len(dev_df)),
            "n_test": int(len(test_df)) if include_test else None,
            "ids_sha256": _ids_sha256(train_ids),
            "source_files_sha256": source_hashes(),
        }
        return train_full.copy(), dev_df.copy(), test_df.copy() if include_test else None, meta

    s_dir = Path(subsets_dir)
    manifest_path = s_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing subset MANIFEST.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Verify source clean files match MANIFEST.json
    recorded_sources = manifest.get("source_files_sha256") or {}
    actual_sources = source_hashes()
    for fname, actual_sha in actual_sources.items():
        if recorded_sources.get(fname) != actual_sha:
            raise ValueError(
                f"Source file hash mismatch for {fname}: manifest={recorded_sources.get(fname)}, actual={actual_sha}"
            )

    fname = subset_filename(dataset, frac, seed)
    entry = (manifest.get("subsets") or {}).get(fname)
    if not isinstance(entry, dict):
        raise ValueError(f"Subset file not registered in MANIFEST.json: {fname}")

    idx_path = s_dir / fname
    if not idx_path.is_file():
        raise ValueError(f"Missing subset index file: {idx_path}")

    actual_file_sha = sha256_file(idx_path)
    if actual_file_sha != entry.get("file_sha256"):
        raise ValueError(
            f"Subset file SHA-256 mismatch for {fname}: expected {entry.get('file_sha256')}, got {actual_file_sha}"
        )

    idx_df = pd.read_csv(idx_path, keep_default_na=False)
    if tuple(idx_df.columns) != SUBSET_INDEX_COLUMNS:
        raise ValueError(
            f"Invalid columns in {fname}: expected {SUBSET_INDEX_COLUMNS}, got {tuple(idx_df.columns)}"
        )
    if idx_df.empty:
        raise ValueError(f"Subset index file is empty: {fname}")

    ids_series = pd.to_numeric(idx_df["id"], errors="coerce")
    lbl_series = pd.to_numeric(idx_df["label"], errors="coerce")
    if ids_series.isna().any() or ids_series.duplicated().any():
        raise ValueError(f"Subset index {fname} contains null or duplicate IDs")
    if lbl_series.isna().any():
        raise ValueError(f"Subset index {fname} contains null labels")

    idx_ids = ids_series.astype(int).tolist()
    if idx_ids != sorted(idx_ids):
        raise ValueError(f"Subset index {fname} IDs must be sorted ascending")
    if len(idx_ids) != int(entry.get("n_rows", -1)):
        raise ValueError(f"Subset index {fname} row count mismatch vs MANIFEST")
    if _ids_sha256(idx_ids) != entry.get("ids_sha256"):
        raise ValueError(f"Subset index {fname} ids_sha256 mismatch vs MANIFEST")

    id_set = set(idx_ids)
    if id_set & dev_ids or id_set & test_ids:
        raise ValueError(f"Leakage detected in {fname}: subset IDs overlap with dev or test")

    train_by_id = train_full.set_index("id", drop=False)
    missing_ids = [i for i in idx_ids if i not in train_by_id.index]
    if missing_ids:
        raise ValueError(f"Subset {fname} references IDs not found in train: {missing_ids[:5]}")

    train_sub = train_by_id.loc[idx_ids].reset_index(drop=True)

    # Verify labels and normalized text hashes 1-to-1 against clean train
    actual_hashes = [hash_text(str(t), normalize=True) for t in train_sub["text"]]
    expected_hashes = idx_df["text_hash"].astype(str).tolist()
    if actual_hashes != expected_hashes:
        raise ValueError(f"text_hash mismatch between {fname} and clean train rows")
    if train_sub["label"].astype(int).tolist() != lbl_series.astype(int).tolist():
        raise ValueError(f"label mismatch between {fname} and clean train rows")

    sub_hash_set = set(actual_hashes)
    if sub_hash_set & dev_hashes or sub_hash_set & test_hashes:
        raise ValueError(f"Text hash leakage detected in {fname}: overlaps with dev or test")

    meta = {
        "dataset": dataset,
        "fraction": frac,
        "seed": int(seed),
        "subset_file": fname,
        "file_sha256": actual_file_sha,
        "ids_sha256": entry["ids_sha256"],
        "n_train": int(len(train_sub)),
        "n_dev": int(len(dev_df)),
        "n_test": int(len(test_df)) if include_test else None,
        "source_files_sha256": actual_sources,
    }
    return train_sub, dev_df.copy(), test_df.copy() if include_test else None, meta


def verify_phase2_subsets(
    clean_dir: Union[str, Path],
    subsets_dir: Union[str, Path],
    vsmec_5pct_csv: Union[str, Path],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify all 5 mandatory Task 03 gates [S1]–[S5] by reading generated CSVs back from disk."""
    c_dir = Path(clean_dir)
    s_dir = Path(subsets_dir)
    fractions = sorted(int(f) for f in cfg["fractions"])
    drift_fracs = set(int(f) for f in cfg["drift_checked_fractions"])
    seeds = [int(s) for s in cfg["seeds"]]
    datasets = list(cfg["datasets"])
    max_drift_limit = float(cfg["max_drift_pp"])

    manifest_path = s_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    s1_violations: List[str] = []
    s2_violations: List[str] = []
    s3_violations: List[str] = []
    s4_violations: List[str] = []
    s5_violations: List[str] = []

    dataset_summaries: Dict[str, Any] = {}

    for ds in datasets:
        splits = _load_clean_domain_splits(c_dir, ds)
        train_full = splits["train"]
        dev_df = splits["dev"]
        test_df = splits["test"]
        full_n = len(train_full)
        full_ids = set(train_full["id"].tolist())
        full_class_counts = {
            c: int((train_full["label"] == c).sum()) for c in DATASET_VALID_LABELS[ds]
        }
        full_class_pct = {c: 100.0 * full_class_counts[c] / full_n for c in DATASET_VALID_LABELS[ds]}

        dev_ids = set(dev_df["id"].tolist())
        test_ids = set(test_df["id"].tolist())
        dev_hashes = {hash_text(str(t), normalize=True) for t in dev_df["text"]}
        test_hashes = {hash_text(str(t), normalize=True) for t in test_df["text"]}
        train_hash_by_id = {
            int(row_id): hash_text(str(txt), normalize=True)
            for row_id, txt in zip(train_full["id"].tolist(), train_full["text"].tolist())
        }
        train_label_by_id = {
            int(row_id): int(lbl)
            for row_id, lbl in zip(train_full["id"].tolist(), train_full["label"].tolist())
        }

        # Verify shared loader cross-method identity [S4] & full integrity on representative 10% seed 0
        rep_pct = 10 if 10 in fractions else fractions[0]
        rep_seed = seeds[0]
        try:
            tr_a, dv_a, te_a, meta_a = load_phase2_target_splits(
                ds, rep_pct, rep_seed, clean_dir=c_dir, subsets_dir=s_dir
            )
            tr_b, dv_b, te_b, meta_b = load_phase2_target_splits(
                ds, rep_pct, rep_seed, clean_dir=c_dir, subsets_dir=s_dir
            )
            if len(dv_a) != len(dev_df) or len(te_a) != len(test_df) or len(dv_b) != len(dev_df) or len(te_b) != len(test_df):
                s2_violations.append(f"{ds} f{rep_pct} s{rep_seed}: loader altered dev/test size")
            if meta_a["ids_sha256"] != meta_b["ids_sha256"] or not tr_a["id"].equals(tr_b["id"]):
                s4_violations.append(f"{ds} f{rep_pct} s{rep_seed}: repeated loader call returned differing IDs")
        except Exception as exc:
            s2_violations.append(f"{ds} f{rep_pct} s{rep_seed}: loader error: {exc}")

        per_fraction_seed_sets: Dict[int, Dict[int, set]] = {f: {} for f in fractions}
        per_fraction_stats: Dict[str, Any] = {}

        for pct in fractions:
            max_drift_pct = 0.0
            n_rows_seen: List[int] = []
            sample_counts_s0: Dict[str, int] = {}
            for seed in seeds:
                fname = subset_filename(ds, pct, seed)
                idx_path = s_dir / fname
                entry = (manifest.get("subsets") or {}).get(fname)
                if not idx_path.is_file() or not isinstance(entry, dict):
                    s2_violations.append(f"{ds} f{pct} s{seed}: missing index file or MANIFEST entry")
                    continue
                if sha256_file(idx_path) != entry.get("file_sha256"):
                    s2_violations.append(f"{ds} f{pct} s{seed}: file SHA-256 mismatch")
                    continue

                idx_df = pd.read_csv(idx_path, keep_default_na=False)
                if tuple(idx_df.columns) != SUBSET_INDEX_COLUMNS or idx_df.empty:
                    s2_violations.append(f"{ds} f{pct} s{seed}: invalid schema or empty CSV")
                    continue

                idx_ids = pd.to_numeric(idx_df["id"], errors="coerce")
                idx_lbls = pd.to_numeric(idx_df["label"], errors="coerce")
                if idx_ids.isna().any() or idx_ids.duplicated().any() or idx_lbls.isna().any():
                    s2_violations.append(f"{ds} f{pct} s{seed}: null/duplicate IDs or null labels")
                    continue

                id_list = idx_ids.astype(int).tolist()
                lbl_list = idx_lbls.astype(int).tolist()
                hash_list = idx_df["text_hash"].astype(str).tolist()
                if id_list != sorted(id_list) or _ids_sha256(id_list) != entry.get("ids_sha256"):
                    s2_violations.append(f"{ds} f{pct} s{seed}: unsorted IDs or ids_sha256 mismatch")
                    continue

                # Explicit [S2] checks: membership in train, 1-to-1 label & text_hash, zero dev/test overlap
                sub_ids = set(id_list)
                sub_hashes = set(hash_list)
                if not sub_ids.issubset(full_ids):
                    s2_violations.append(f"{ds} f{pct} s{seed}: IDs outside clean train")
                    continue
                if any(train_label_by_id[i] != l or train_hash_by_id[i] != h for i, l, h in zip(id_list, lbl_list, hash_list)):
                    s2_violations.append(f"{ds} f{pct} s{seed}: label or text_hash mismatch vs clean train")
                    continue
                if (sub_ids & dev_ids) or (sub_ids & test_ids) or (sub_hashes & dev_hashes) or (sub_hashes & test_hashes):
                    s2_violations.append(f"{ds} f{pct} s{seed}: overlap with dev/test")

                per_fraction_seed_sets[pct][seed] = sub_ids
                n_rows = len(id_list)
                n_rows_seen.append(n_rows)

                # Stratification drift [S3]
                cls_counts = {c: 0 for c in DATASET_VALID_LABELS[ds]}
                for l in lbl_list:
                    cls_counts[l] = cls_counts.get(l, 0) + 1
                if seed == seeds[0]:
                    sample_counts_s0 = {str(c): cls_counts[c] for c in DATASET_VALID_LABELS[ds]}
                for c in DATASET_VALID_LABELS[ds]:
                    drift = abs((100.0 * cls_counts[c] / n_rows) - full_class_pct[c])
                    if drift > max_drift_pct:
                        max_drift_pct = drift
                    if pct in drift_fracs and drift > max_drift_limit:
                        s3_violations.append(
                            f"{ds} f{pct} s{seed} class={c}: drift={drift:.4f}pp > {max_drift_limit}pp"
                        )

            # Seed diversity [S4]: all seeds at same (ds, pct) must be pairwise distinct
            seen_seeds = sorted(per_fraction_seed_sets[pct].keys())
            for i in range(len(seen_seeds)):
                for j in range(i + 1, len(seen_seeds)):
                    sa, sb = seen_seeds[i], seen_seeds[j]
                    if per_fraction_seed_sets[pct][sa] == per_fraction_seed_sets[pct][sb]:
                        s4_violations.append(f"{ds} f{pct}: seed {sa} and seed {sb} produced identical ID sets")

            per_fraction_stats[str(pct)] = {
                "n_rows": n_rows_seen[0] if n_rows_seen and len(set(n_rows_seen)) == 1 else n_rows_seen,
                "max_drift_pp_across_seeds": round(max_drift_pct, 6),
                "within_1pp_limit": max_drift_pct <= max_drift_limit,
                "class_counts_s0": sample_counts_s0,
            }

        # Strict nested inclusion [S1]: 5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100% (total and per-class)
        for seed in seeds:
            prev_ids: Optional[set] = None
            prev_pct: Optional[int] = None
            prev_cls_sets: Optional[Dict[int, set]] = None
            for pct in fractions:
                cur_ids = per_fraction_seed_sets[pct].get(seed)
                if cur_ids is None:
                    s1_violations.append(f"{ds} s{seed}: missing fraction {pct}%")
                    continue
                if not cur_ids.issubset(full_ids):
                    s1_violations.append(f"{ds} s{seed} f{pct}: IDs not subset of 100% train")
                idx_df = pd.read_csv(s_dir / subset_filename(ds, pct, seed), keep_default_na=False)
                cur_cls_sets = {
                    c: set(idx_df.loc[idx_df["label"].astype(int) == c, "id"].astype(int).tolist())
                    for c in DATASET_VALID_LABELS[ds]
                }
                if prev_ids is not None:
                    if not prev_ids.issubset(cur_ids) or len(prev_ids) >= len(cur_ids):
                        s1_violations.append(
                            f"{ds} s{seed}: {prev_pct}% ({len(prev_ids)}) is not a strict subset of {pct}% ({len(cur_ids)})"
                        )
                    assert prev_cls_sets is not None
                    for c in DATASET_VALID_LABELS[ds]:
                        if not prev_cls_sets[c].issubset(cur_cls_sets[c]):
                            s1_violations.append(
                                f"{ds} s{seed} class={c}: {prev_pct}% IDs not subset of {pct}% IDs"
                            )
                prev_ids = cur_ids
                prev_pct = pct
                prev_cls_sets = cur_cls_sets

        dataset_summaries[ds] = {
            "n_train_100": full_n,
            "n_dev_fixed": int(len(dev_df)),
            "n_test_fixed": int(len(test_df)),
            "train_100_class_counts": {str(k): v for k, v in full_class_counts.items()},
            "train_100_class_pct": {str(k): round(v, 4) for k, v in full_class_pct.items()},
            "fractions": per_fraction_stats,
        }

    # [S5]: MANIFEST completeness & vsmec_5pct_class_counts.csv verification
    expected_n_files = len(datasets) * len(fractions) * len(seeds)
    registered_subsets = manifest.get("subsets") or {}
    if len(registered_subsets) != expected_n_files:
        s5_violations.append(
            f"MANIFEST subset count={len(registered_subsets)}, expected {expected_n_files}"
        )
    for fname, entry in sorted(registered_subsets.items()):
        fp = s_dir / fname
        if not fp.is_file():
            s5_violations.append(f"Missing subset file listed in MANIFEST: {fname}")
        elif sha256_file(fp) != entry.get("file_sha256"):
            s5_violations.append(f"SHA-256 mismatch for {fname}")

    v5_path = Path(vsmec_5pct_csv)
    v5_summary: Dict[str, Any] = {}
    if not v5_path.is_file():
        s5_violations.append(f"Missing {v5_path}")
    else:
        v5_df = pd.read_csv(v5_path)
        if len(v5_df) != len(seeds):
            s5_violations.append(f"vsmec_5pct_class_counts.csv has {len(v5_df)} rows, expected {len(seeds)}")
        min_overall = int(v5_df["min_class_count"].min()) if not v5_df.empty else 0
        all_eligible = bool(v5_df["eligible_v7_min5"].all()) if not v5_df.empty else False
        v5_summary = {
            "n_seeds": int(len(v5_df)),
            "total_rows_per_seed": sorted(set(int(x) for x in v5_df["total_rows"].tolist())),
            "min_class_name": str(v5_df.loc[v5_df["min_class_count"].idxmin(), "min_class_name"]),
            "min_class_count_across_seeds": min_overall,
            "threshold": int(cfg["min_vsmec_5pct_per_class"]),
            "eligible_v7_min5_all_seeds": all_eligible,
        }

    gates = {
        "S1_nested_inclusion": {"status": "PASS" if not s1_violations else "FAIL", "violations": s1_violations},
        "S2_zero_split_leakage": {"status": "PASS" if not s2_violations else "FAIL", "violations": s2_violations},
        "S3_stratification_drift_1pp": {"status": "PASS" if not s3_violations else "FAIL", "violations": s3_violations},
        "S4_seed_diversity_and_identity": {"status": "PASS" if not s4_violations else "FAIL", "violations": s4_violations},
        "S5_manifest_and_5pct_table": {"status": "PASS" if not s5_violations else "FAIL", "violations": s5_violations},
    }
    all_passed = all(g["status"] == "PASS" for g in gates.values())

    return {
        "schema_version": cfg["schema_version"],
        "all_gates_passed": all_passed,
        "expected_subset_files": expected_n_files,
        "gates": gates,
        "datasets": dataset_summaries,
        "vsmec_5pct_summary": v5_summary,
    }


def render_subset_audit_markdown(audit: Dict[str, Any], v5_df: pd.DataFrame) -> str:
    """Render human-readable Task 03 report (`reports/phase2/subset_audit.md`)."""
    gate_badge = "**PASS (ĐẠT TOÀN BỘ 5 CỔNG S1–S5 — SẴN SÀNG SANG TASK 04 & TASK 05)**" if audit["all_gates_passed"] else "**FAIL (CÓ CỔNG CHƯA ĐẠT)**"
    lines = [
        "# Báo cáo Task 03: Kiểm định tập con phân tầng lồng nhau (Phase 2)",
        "",
        f"**Kết luận cổng Task 03:** {gate_badge}",
        "",
        f"> Sinh tự động từ `scripts/make_phase2_subsets.py` (`{audit['schema_version']}`, RNG `{RNG_SPEC}`). Dữ liệu đóng băng `data/clean/` và `data/splits/` giữ nguyên 100%; `dev` và `test` cố định không rút mẫu.",
        "",
        "## 1. Tổng hợp 5 cổng kiểm định bắt buộc (`[S1]–[S5]`)",
        "",
        "| Cổng | Nội dung kiểm định | Trạng thái | Số vi phạm |",
        "|---|---|---|---:|",
    ]
    gate_labels = {
        "S1_nested_inclusion": "Tính lồng nhau nghiêm ngặt `5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100%` (tổng ID & từng lớp, 5 seed)",
        "S2_zero_split_leakage": "0 giao ID và `text_hash` với `dev`/`test`; giữ nguyên kích thước `dev`/`test`",
        "S3_stratification_drift_1pp": "Lệch phân phối từng lớp tối đa `≤ 1,0` điểm phần trăm ở `10%, 25%, 50%`",
        "S4_seed_diversity_and_identity": "5 seed đôi một khác nhau; mọi phương pháp gọi loader trả về cùng tập ID",
        "S5_manifest_and_5pct_table": "Đủ 40 file chỉ mục khớp SHA-256 trong `MANIFEST.json` & bảng đếm 5% UIT-VSMEC",
    }
    for gkey, glabel in gate_labels.items():
        g = audit["gates"][gkey]
        lines.append(f"| `{gkey}` | {glabel} | **{g['status']}** | {len(g['violations'])} |")

    lines.extend(
        [
            "",
            "## 2. Quy mô mẫu thực tế & Độ lệch phân phối lớp tối đa (so với `train` 100%)",
            "",
            "| Bộ dữ liệu | Mức | Số câu `train` | `dev` cố định | `test` cố định | Lệch phân phối lớp lớn nhất (`|Δ%|`) | Đạt `≤ 1,0` điểm %? |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for ds_name, ds_label in (("vsmec", "UIT-VSMEC (7 lớp)"), ("neuesc", "NEU-ESC (3 lớp)")):
        if ds_name not in audit["datasets"]:
            continue
        d = audit["datasets"][ds_name]
        for pct_str in ("5", "10", "25", "50"):
            if pct_str not in d["fractions"]:
                continue
            f = d["fractions"][pct_str]
            lines.append(
                f"| {ds_label} | `{pct_str}%` | **{f['n_rows']}** | {d['n_dev_fixed']} | {d['n_test_fixed']} | "
                f"{f['max_drift_pp_across_seeds']:.4f} pp | **{'PASS' if f['within_1pp_limit'] else 'FAIL'}** |"
            )
        lines.append(
            f"| {ds_label} | `100%` | **{d['n_train_100']}** | {d['n_dev_fixed']} | {d['n_test_fixed']} | 0.0000 pp | **PASS** |"
        )

    lines.extend(
        [
            "",
            "## 3. Phân bố từng lớp theo mức (tại `seed 0`, đồng nhất số lượng trên cả 5 seed)",
            "",
            "### 3.1. UIT-VSMEC (`train`, 7 lớp cảm xúc)",
            "",
            "| Mức | Enjoyment (0) | Sadness (1) | Fear (2) | Anger (3) | Disgust (4) | Surprise (5) | Other (6) | Tổng |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if "vsmec" in audit["datasets"]:
        vd = audit["datasets"]["vsmec"]
        for pct_str in ("5", "10", "25", "50"):
            if pct_str in vd["fractions"]:
                c = vd["fractions"][pct_str]["class_counts_s0"]
                lines.append(
                    f"| `{pct_str}%` | {c['0']} | {c['1']} | {c['2']} | {c['3']} | {c['4']} | {c['5']} | {c['6']} | **{vd['fractions'][pct_str]['n_rows']}** |"
                )
        c100 = vd["train_100_class_counts"]
        lines.append(
            f"| `100%` | {c100['0']} | {c100['1']} | {c100['2']} | {c100['3']} | {c100['4']} | {c100['5']} | {c100['6']} | **{vd['n_train_100']}** |"
        )

    lines.extend(
        [
            "",
            "### 3.2. NEU-ESC (`train`, 3 nhãn cực tính)",
            "",
            "| Mức | negative (0) | neutral (1) | positive (2) | Tổng |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    if "neuesc" in audit["datasets"]:
        nd = audit["datasets"]["neuesc"]
        for pct_str in ("5", "10", "25", "50"):
            if pct_str in nd["fractions"]:
                c = nd["fractions"][pct_str]["class_counts_s0"]
                lines.append(
                    f"| `{pct_str}%` | {c['0']} | {c['1']} | {c['2']} | **{nd['fractions'][pct_str]['n_rows']}** |"
                )
        c100 = nd["train_100_class_counts"]
        lines.append(
            f"| `100%` | {c100['0']} | {c100['1']} | {c100['2']} | **{nd['n_train_100']}** |"
        )

    lines.extend(
        [
            "",
            "## 4. Bảng đếm từng lớp mức 5% của UIT-VSMEC trên 5 seed (`vsmec_5pct_class_counts.csv`)",
            "",
            "| Seed | Tổng | Enjoyment | Sadness | Fear | Anger | Disgust | Surprise | Other | Lớp nhỏ nhất | Số mẫu min | Đạt ngưỡng `≥ 5`? |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for _, r in v5_df.iterrows():
        lines.append(
            f"| `s{int(r['seed'])}` | {int(r['total_rows'])} | {int(r['c0_Enjoyment'])} | {int(r['c1_Sadness'])} | "
            f"{int(r['c2_Fear'])} | {int(r['c3_Anger'])} | {int(r['c4_Disgust'])} | {int(r['c5_Surprise'])} | "
            f"{int(r['c6_Other'])} | `{r['min_class_name']}` | **{int(r['min_class_count'])}** | **{bool(r['eligible_v7_min5'])}** |"
        )

    v5s = audit["vsmec_5pct_summary"]
    lines.extend(
        [
            "",
            f"- **Đánh giá điều kiện mức 5% UIT-VSMEC (V7):** Lớp nhỏ nhất trên toàn bộ 5 seed là `{v5s.get('min_class_name')}` với **`{v5s.get('min_class_count_across_seeds')}` mẫu** (`≥ {v5s.get('threshold')}` mẫu → **đủ điều kiện kỹ thuật về số mẫu tối thiểu: `{v5s.get('eligible_v7_min5_all_seeds')}`**).",
            "- **Chuyển tiếp sang Task 04:** Mức `5%` của cả UIT-VSMEC (`277` câu) và NEU-ESC (`1.152` câu) vẫn là hạng mục **tùy chọn theo ngân sách GPU** (+26 cấu hình lõi), quyết định bật/tắt chính thức tại **Task 04**.",
            "- **Chuyển tiếp sang Task 05:** Thành viên 2 sử dụng trực tiếp hàm `src.phase2.subsets.load_phase2_target_splits(dataset, fraction, seed)` đọc từ `data/subsets_phase2/` + `MANIFEST.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_phase2_subsets(
    repo_root: Union[str, Path],
    config_path: Union[str, Path],
    *,
    clean_dir_override: Optional[Union[str, Path]] = None,
    subsets_dir_override: Optional[Union[str, Path]] = None,
    reports_dir_override: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generate all nested subset index CSVs, `MANIFEST.json`, 5% count table, and audit reports."""
    root = Path(repo_root).resolve()
    cfg = load_subsets_config(config_path)

    clean_dir = _resolve_path(root, clean_dir_override or cfg["clean_dir"])
    subsets_dir = _resolve_path(root, subsets_dir_override or cfg["subsets_dir"])
    reports_dir = _resolve_path(root, reports_dir_override or cfg["reports_dir"])
    subsets_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(cfg["datasets"])
    fractions = sorted(int(f) for f in cfg["fractions"])
    seeds = [int(s) for s in cfg["seeds"]]
    rng_base = int(cfg["rng_base_seed"])

    source_hashes = _source_hashes_for_clean_dir(clean_dir, datasets)
    subsets_manifest: Dict[str, Any] = {}

    for ds in datasets:
        splits = _load_clean_domain_splits(clean_dir, ds)
        train_df = splits["train"].copy()
        train_df["text_hash"] = [hash_text(str(t), normalize=True) for t in train_df["text"]]

        label_counts = {
            c: int((train_df["label"] == c).sum()) for c in DATASET_VALID_LABELS[ds]
        }
        alloc_schedule = allocate_monotone_counts(label_counts, fractions, min_per_class=1)

        for seed in seeds:
            ordered_by_class = _permuted_class_indices(train_df, ds, seed, rng_base_seed=rng_base)
            for pct in fractions:
                parts = [
                    ordered_by_class[c].iloc[: alloc_schedule[pct][c]]
                    for c in DATASET_VALID_LABELS[ds]
                    if alloc_schedule[pct][c] > 0
                ]
                sub_df = (
                    pd.concat(parts, ignore_index=True)
                    .sort_values("id", kind="mergesort")
                    .reset_index(drop=True)
                )
                fname = subset_filename(ds, pct, seed)
                out_csv = subsets_dir / fname
                _write_csv_lf(sub_df, out_csv, SUBSET_INDEX_COLUMNS)

                ids_list = sub_df["id"].astype(int).tolist()
                subsets_manifest[fname] = {
                    "dataset": ds,
                    "fraction": int(pct),
                    "seed": int(seed),
                    "n_rows": int(len(sub_df)),
                    "file_sha256": sha256_file(out_csv),
                    "ids_sha256": _ids_sha256(ids_list),
                    "class_counts": {str(c): int(alloc_schedule[pct][c]) for c in DATASET_VALID_LABELS[ds]},
                }

    manifest_payload = {
        "schema_version": cfg["schema_version"],
        "datasets": datasets,
        "fractions": fractions,
        "seeds": seeds,
        "rng_spec": RNG_SPEC,
        "source_files_sha256": source_hashes,
        "subsets": subsets_manifest,
    }
    _write_json_lf(subsets_dir / "MANIFEST.json", manifest_payload)

    # Export 5% UIT-VSMEC class count table
    v5_csv_path = reports_dir / "vsmec_5pct_class_counts.csv"
    v5_df = build_vsmec_5pct_counts_frame(
        subsets_dir,
        seeds=seeds,
        min_per_class_threshold=int(cfg["min_vsmec_5pct_per_class"]),
    )
    _write_csv_lf(v5_df, v5_csv_path, list(v5_df.columns))

    # Run post-generation verification reading back from disk
    audit = verify_phase2_subsets(clean_dir, subsets_dir, v5_csv_path, cfg)
    _write_json_lf(reports_dir / "subset_audit.json", audit)
    _write_text_lf(reports_dir / "subset_audit.md", render_subset_audit_markdown(audit, v5_df))

    return {
        "subsets_dir": str(subsets_dir),
        "reports_dir": str(reports_dir),
        "manifest": manifest_payload,
        "audit": audit,
        "vsmec_5pct_counts": v5_df.to_dict(orient="records"),
    }
