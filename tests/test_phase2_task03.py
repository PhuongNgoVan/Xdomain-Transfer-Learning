"""Offline unit tests for Phase 2 Task 03 (nested stratified subsets & shared loader)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from src.common.hashing import hash_text, sha256_file
from src.phase2.subsets import (
    SUBSET_INDEX_COLUMNS,
    allocate_monotone_counts,
    build_vsmec_5pct_counts_frame,
    generate_phase2_subsets,
    load_phase2_target_splits,
    subset_filename,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_clean_dir(base: Path) -> Path:
    """Create tiny valid clean datasets (7-class VSMEC + 3-class NEU-ESC) with disjoint splits."""
    c_dir = base / "clean"
    c_dir.mkdir(parents=True, exist_ok=True)

    # VSMEC: 700 train rows across 7 classes (10% = 70 rows so rounding <= 0.71pp < 1.0pp;
    # class 5 has 24 rows -> 5% of 700 = 35 rows -> class 5 gets 1 sample < 5 -> eligible_v7_min5=False)
    v_train_counts = {0: 196, 1: 120, 2: 50, 3: 60, 4: 130, 5: 24, 6: 120}
    rows = []
    cur_id = 0
    for lbl, cnt in v_train_counts.items():
        for i in range(cnt):
            rows.append(
                {
                    "id": cur_id,
                    "text": f"vsmec train sample {cur_id} class {lbl}",
                    "label": lbl,
                    "subset": "train",
                    "text_seg": f"vsmec_train sample_{cur_id} class_{lbl}",
                }
            )
            cur_id += 1
    pd.DataFrame(rows).to_csv(c_dir / "vsmec_train.csv", index=False, lineterminator="\n")

    for sp, n_sp in (("dev", 28), ("test", 28)):
        sp_rows = []
        for i in range(n_sp):
            sp_rows.append(
                {
                    "id": cur_id,
                    "text": f"vsmec {sp} sample {cur_id}",
                    "label": i % 7,
                    "subset": sp,
                    "text_seg": f"vsmec_{sp} sample_{cur_id}",
                }
            )
            cur_id += 1
        pd.DataFrame(sp_rows).to_csv(c_dir / f"vsmec_{sp}.csv", index=False, lineterminator="\n")

    # NEU-ESC: 600 train, 30 dev, 30 test across 3 classes
    n_rows = []
    nid = 0
    for lbl, cnt in {0: 120, 1: 390, 2: 90}.items():
        for _ in range(cnt):
            n_rows.append(
                {
                    "id": nid,
                    "text": f"neuesc train sample {nid} lbl {lbl}",
                    "label": lbl,
                    "subset": "train",
                    "text_seg": f"neuesc_train sample_{nid} lbl_{lbl}",
                }
            )
            nid += 1
    for sp, n_sp in (("dev", 30), ("test", 30)):
        for i in range(n_sp):
            n_rows.append(
                {
                    "id": nid,
                    "text": f"neuesc {sp} sample {nid}",
                    "label": i % 3,
                    "subset": sp,
                    "text_seg": f"neuesc_{sp} sample_{nid}",
                }
            )
            nid += 1
    pd.DataFrame(n_rows).to_csv(c_dir / "neuesc_100.csv", index=False, lineterminator="\n")
    return c_dir


def test_monotone_allocation_prevents_alabama_paradox() -> None:
    # Classic Alabama paradox setup: N=140 with counts {0: 60, 1: 60, 2: 20}
    # At 7% (M=10), independent largest-remainder gives {0:4, 1:4, 2:2};
    # at 8% (M=11), independent largest-remainder drops class 2 to 1 ({0:5, 1:5, 2:1}).
    paradox_counts = {0: 60, 1: 60, 2: 20}
    sched = allocate_monotone_counts(paradox_counts, [7, 8, 25, 50], min_per_class=1)
    assert sum(sched[7].values()) == 10
    assert sum(sched[8].values()) == 11
    assert sched[7][2] == 2
    assert sched[8][2] >= sched[7][2]  # monotonicity strictly preserved!

    # Real UIT-VSMEC & NEU-ESC train class distributions
    vsmec_counts = {0: 1558, 1: 947, 2: 316, 3: 391, 4: 1069, 5: 241, 6: 1020}
    v_sched = allocate_monotone_counts(vsmec_counts, [5, 10, 25, 50])
    assert [sum(v_sched[p].values()) for p in (5, 10, 25, 50)] == [277, 554, 1386, 2771]
    for c in range(7):
        assert v_sched[5][c] <= v_sched[10][c] <= v_sched[25][c] <= v_sched[50][c] <= vsmec_counts[c]

    neuesc_counts = {0: 4215, 1: 15936, 2: 2897}
    n_sched = allocate_monotone_counts(neuesc_counts, [5, 10, 25, 50])
    assert [sum(n_sched[p].values()) for p in (5, 10, 25, 50)] == [1152, 2305, 5762, 11524]
    for c in range(3):
        assert n_sched[5][c] <= n_sched[10][c] <= n_sched[25][c] <= n_sched[50][c] <= neuesc_counts[c]


def test_nested_inclusion_seed_diversity_and_5pct_eligibility_on_synthetic(tmp_path: Path) -> None:
    clean_dir = _make_synthetic_clean_dir(tmp_path)
    subsets_dir = tmp_path / "subsets"
    reports_dir = tmp_path / "reports"

    res = generate_phase2_subsets(
        repo_root=REPO_ROOT,
        config_path=REPO_ROOT / "configs/phase2_subsets.yaml",
        clean_dir_override=clean_dir,
        subsets_dir_override=subsets_dir,
        reports_dir_override=reports_dir,
    )
    audit = res["audit"]
    assert audit["all_gates_passed"] is True

    # On this synthetic fixture, class 5 has 14 train rows -> 1 sample at 5% (< 5) -> eligible_v7_min5 must be False
    v5_df = pd.read_csv(reports_dir / "vsmec_5pct_class_counts.csv")
    assert len(v5_df) == 5
    assert (v5_df["min_class_count"] < 5).all()
    assert (~v5_df["eligible_v7_min5"]).all()
    assert audit["vsmec_5pct_summary"]["eligible_v7_min5_all_seeds"] is False

    # Verify strict nesting and seed diversity directly via shared loader
    for ds in ("vsmec", "neuesc"):
        for seed in range(5):
            sets = []
            for pct in (5, 10, 25, 50, 100):
                tr, dv, te, _ = load_phase2_target_splits(ds, pct, seed, clean_dir=clean_dir, subsets_dir=subsets_dir)
                assert len(dv) == (28 if ds == "vsmec" else 30)
                assert len(te) == (28 if ds == "vsmec" else 30)
                sets.append(set(tr["id"].tolist()))
            for i in range(len(sets) - 1):
                assert sets[i] < sets[i + 1]
        s0_ids = set(load_phase2_target_splits(ds, 10, 0, clean_dir=clean_dir, subsets_dir=subsets_dir)[0]["id"])
        s1_ids = set(load_phase2_target_splits(ds, 10, 1, clean_dir=clean_dir, subsets_dir=subsets_dir)[0]["id"])
        assert s0_ids != s1_ids


def test_loader_rejects_leakage_duplicates_and_tampered_hashes(tmp_path: Path) -> None:
    clean_dir = _make_synthetic_clean_dir(tmp_path)
    subsets_dir = tmp_path / "subsets"
    reports_dir = tmp_path / "reports"
    generate_phase2_subsets(
        repo_root=REPO_ROOT,
        config_path=REPO_ROOT / "configs/phase2_subsets.yaml",
        clean_dir_override=clean_dir,
        subsets_dir_override=subsets_dir,
        reports_dir_override=reports_dir,
    )

    fname = subset_filename("vsmec", 10, 0)
    idx_path = subsets_dir / fname
    manifest_path = subsets_dir / "MANIFEST.json"
    orig_bytes = idx_path.read_bytes()
    orig_manifest = manifest_path.read_text(encoding="utf-8")

    # 1. Tamper file without updating MANIFEST -> SHA-256 mismatch
    df = pd.read_csv(idx_path, keep_default_na=False)
    df.iloc[:-1].to_csv(idx_path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_phase2_target_splits("vsmec", 10, 0, clean_dir=clean_dir, subsets_dir=subsets_dir)

    # 2. Inject a dev ID AND recompute MANIFEST to bypass hash check -> must STILL reject as leakage!
    dev_df = pd.read_csv(clean_dir / "vsmec_dev.csv", keep_default_na=False)
    leaked_row = pd.DataFrame(
        [
            {
                "id": int(dev_df.iloc[0]["id"]),
                "label": int(dev_df.iloc[0]["label"]),
                "text_hash": hash_text(str(dev_df.iloc[0]["text"]), normalize=True),
            }
        ]
    )
    tampered = pd.concat([df.iloc[:-1], leaked_row], ignore_index=True).sort_values("id")
    tampered.to_csv(idx_path, index=False, lineterminator="\n")
    mf = json.loads(orig_manifest)
    mf["subsets"][fname]["file_sha256"] = sha256_file(idx_path)
    mf["subsets"][fname]["n_rows"] = len(tampered)
    mf["subsets"][fname]["ids_sha256"] = (
        __import__("hashlib")
        .sha256("\n".join(str(int(x)) for x in tampered["id"]).encode("utf-8"))
        .hexdigest()
    )
    manifest_path.write_text(json.dumps(mf, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="Leakage detected"):
        load_phase2_target_splits("vsmec", 10, 0, clean_dir=clean_dir, subsets_dir=subsets_dir)

    # 3. Duplicate ID with updated MANIFEST -> must reject duplicate IDs
    dup = pd.concat([df.iloc[:2], df.iloc[:1]], ignore_index=True)
    dup.to_csv(idx_path, index=False, lineterminator="\n")
    mf2 = json.loads(orig_manifest)
    mf2["subsets"][fname]["file_sha256"] = sha256_file(idx_path)
    mf2["subsets"][fname]["n_rows"] = len(dup)
    manifest_path.write_text(json.dumps(mf2, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate IDs"):
        load_phase2_target_splits("vsmec", 10, 0, clean_dir=clean_dir, subsets_dir=subsets_dir)

    # Restore clean state
    idx_path.write_bytes(orig_bytes)
    manifest_path.write_text(orig_manifest, encoding="utf-8")


def test_real_repo_subsets_reproducibility_drift_and_frozen_data_untouched(tmp_path: Path) -> None:
    clean_dir = REPO_ROOT / "data/clean"
    splits_dir = REPO_ROOT / "data/splits"
    if not (clean_dir / "vsmec_train.csv").is_file():
        pytest.skip("Clean datasets not present")

    clean_before: Dict[str, str] = {p.name: sha256_file(p) for p in sorted(clean_dir.iterdir()) if p.is_file()}
    splits_before: Dict[str, str] = {p.name: sha256_file(p) for p in sorted(splits_dir.iterdir()) if p.is_file()}

    run1_sub = tmp_path / "r1_sub"
    run1_rep = tmp_path / "r1_rep"
    run2_sub = tmp_path / "r2_sub"
    run2_rep = tmp_path / "r2_rep"

    r1 = generate_phase2_subsets(
        REPO_ROOT,
        REPO_ROOT / "configs/phase2_subsets.yaml",
        subsets_dir_override=run1_sub,
        reports_dir_override=run1_rep,
    )
    r2 = generate_phase2_subsets(
        REPO_ROOT,
        REPO_ROOT / "configs/phase2_subsets.yaml",
        subsets_dir_override=run2_sub,
        reports_dir_override=run2_rep,
    )

    assert r1["audit"]["all_gates_passed"] is True
    assert r2["audit"]["all_gates_passed"] is True
    assert r1["audit"]["expected_subset_files"] == 40
    assert r1["audit"]["vsmec_5pct_summary"]["min_class_count_across_seeds"] == 12
    assert r1["audit"]["vsmec_5pct_summary"]["eligible_v7_min5_all_seeds"] is True

    # Byte-for-byte reproducibility across every generated file
    r1_files = sorted(p.name for p in run1_sub.iterdir())
    r2_files = sorted(p.name for p in run2_sub.iterdir())
    assert r1_files == r2_files and len(r1_files) == 41  # 40 CSVs + MANIFEST.json
    for fname in r1_files:
        assert (run1_sub / fname).read_bytes() == (run2_sub / fname).read_bytes(), fname
    for fname in ("vsmec_5pct_class_counts.csv", "subset_audit.json", "subset_audit.md"):
        assert (run1_rep / fname).read_bytes() == (run2_rep / fname).read_bytes(), fname

    # Frozen directories untouched
    clean_after = {p.name: sha256_file(p) for p in sorted(clean_dir.iterdir()) if p.is_file()}
    splits_after = {p.name: sha256_file(p) for p in sorted(splits_dir.iterdir()) if p.is_file()}
    assert clean_before == clean_after
    assert splits_before == splits_after
