"""CLI to generate and verify Phase 2 nested stratified train subsets (Task 03).

Usage:
    .venv/Scripts/python.exe -X utf8 -m scripts.make_phase2_subsets --config configs/phase2_subsets.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

from src.phase2.subsets import generate_phase2_subsets


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 2 nested stratified subsets (5% ⊂ 10% ⊂ 25% ⊂ 50%) and audit reports."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_subsets.yaml"))
    parser.add_argument("--clean-dir", type=Path, default=None)
    parser.add_argument("--subsets-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    res = generate_phase2_subsets(
        repo_root=args.repo_root,
        config_path=args.config,
        clean_dir_override=args.clean_dir,
        subsets_dir_override=args.subsets_dir,
        reports_dir_override=args.reports_dir,
    )
    audit = res["audit"]
    v5 = audit["vsmec_5pct_summary"]

    print("=== Phase 2 Task 03: Nested Stratified Subsets Summary ===")
    print(f"  Subsets written : {res['subsets_dir']} ({audit['expected_subset_files']} CSV files + MANIFEST.json)")
    print(f"  Reports written : {res['reports_dir']}")
    for gkey, gval in audit["gates"].items():
        print(f"  {gkey:32s}: {gval['status']} (violations={len(gval['violations'])})")
    print(
        f"  VSMEC 5% min cls: {v5.get('min_class_name')} = {v5.get('min_class_count_across_seeds')} "
        f"(eligible_v7_min5={v5.get('eligible_v7_min5_all_seeds')})"
    )

    if audit["all_gates_passed"]:
        print("\nGATE: PASS (Tất cả 5 cổng S1-S5 đạt — Sẵn sàng sang Task 04 & Task 05)")
        return 0

    print("\nGATE: FAIL — Xem chi tiết vi phạm trong reports/phase2/subset_audit.json")
    return 1


if __name__ == "__main__":
    sys.exit(main())
