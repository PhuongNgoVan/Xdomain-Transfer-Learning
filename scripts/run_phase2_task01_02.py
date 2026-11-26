"""CLI runner for Phase 2 Task 01 (Inventory) and Task 02 (Data & Protocol Audit).

Usage (offline default, read-only on frozen data/configs):
    .venv/Scripts/python.exe -X utf8 -m scripts.run_phase2_task01_02 \\
        --config configs/phase2_audit.yaml --out-dir reports/phase2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from src.phase2.audit_task01_02 import run_task01_02


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 Task 01 (resource inventory) and Task 02 (data/protocol audit)."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root directory (default: current directory).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_audit.yaml"),
        help="Path to Phase 2 audit YAML config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/phase2"),
        help="Directory where inventory.md/json, data_audit.md, manifest_audit.json are written.",
    )
    parser.add_argument("--vsmec-ckpt-root", type=Path, default=None)
    parser.add_argument("--neu-ckpt-root", type=Path, default=None)
    parser.add_argument("--stage1-results-dir", type=Path, default=None)
    parser.add_argument("--zs-results-dir", type=Path, default=None)
    parser.add_argument("--seq-results-dir", type=Path, default=None)
    parser.add_argument("--probe-notebook", type=Path, default=None)
    parser.add_argument(
        "--with-tokenizer",
        action="store_true",
        help="Also compute max_length=256 truncation using locally cached HF tokenizers only.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    overrides = {
        "vsmec_visobert_ckpt_root": args.vsmec_ckpt_root,
        "neu_visobert_ckpt_root": args.neu_ckpt_root,
        "stage1_results_dir": args.stage1_results_dir,
        "zs_results_dir": args.zs_results_dir,
        "seq_results_dir": args.seq_results_dir,
        "probe_notebook_path": args.probe_notebook,
    }
    res = run_task01_02(
        repo_root=args.repo_root,
        config_path=args.config,
        out_dir=args.out_dir,
        overrides=overrides,
        with_tokenizer=args.with_tokenizer,
    )
    inv = res["inventory"]
    audit = res["audit"]

    print("=== Phase 2 Task 01: Resource Inventory Summary ===")
    print(f"  Git branch        : {inv['git']['branch']} @ {inv['git']['head_short']} [{inv['git']['status']}]")
    print(f"  UIT-VSMEC clean   : {inv['datasets']['vsmec_clean']['status']}")
    print(f"  NEU-ESC clean     : {inv['datasets']['neuesc_clean']['status']}")
    print(f"  Raw data          : {inv['datasets']['raw_data']['status']}")
    print(f"  VSMEC ViSoBERT ck : {inv['checkpoints']['vsmec_visobert']['status']} ({inv['checkpoints']['vsmec_visobert']['n_valid']}/{inv['checkpoints']['vsmec_visobert']['n_required']})")
    print(f"  NEU ViSoBERT ckpt : {inv['checkpoints']['neu_visobert']['status']} ({inv['checkpoints']['neu_visobert']['n_valid']}/{inv['checkpoints']['neu_visobert']['n_required']})")
    print(f"  100% results/pred : {inv['phase1_results']['status']} (pred CSVs={inv['phase1_results']['n_prediction_csvs']})")
    print(f"  Probe notebook    : {inv['probe_notebook']['status']}")

    print("\n=== Phase 2 Task 02: Data & Protocol Audit Summary ===")
    print(f"  Clean MANIFEST    : {audit['manifests']['clean']['status']} (core_files_ok={audit['manifests']['clean']['core_files_ok']})")
    print(f"  Splits MANIFEST   : {audit['manifests']['splits']['status']}")
    print(f"  Polarity anomaly  : same_manifest_hash={audit['manifests']['polarity_anomaly']['same_hash_in_clean_manifest']}, diff_disk_hash={audit['manifests']['polarity_anomaly']['different_actual_disk_hashes']}")
    print(f"  Preprocess / '_'  : {audit['preprocessing_and_truncation']['status']}")
    print(f"  NEU raw toxic     : {audit['data_integrity']['neuesc_raw_toxic_trace']['status']} (test_toxic={audit['data_integrity']['neuesc_raw_toxic_trace'].get('test_toxic_count')})")
    if audit["task03_ready"]:
        print("\nGATE: SẴN SÀNG SANG TASK 03 (Tạo tập con dữ liệu)")
        print(f"Reports written to: {res['out_dir']}")
        return 0

    print(f"\nGATE: CHƯA SẴN SÀNG SANG TASK 03 — Blockers: {audit['task03_blockers']}")
    print(f"Reports written to: {res['out_dir']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
