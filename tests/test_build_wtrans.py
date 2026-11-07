"""Unit and acceptance tests for Stage B of M7 (`src/data/build_wtrans.py`)."""

from pathlib import Path
import json
import shutil
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.constants import (
    EMO_ANGER,
    EMO_DISGUST,
    EMO_ENJOYMENT,
    EMO_FEAR,
    EMO_OTHER,
    EMO_SADNESS,
    EMO_SURPRISE,
    EMOTIONS7,
    HARD_MERGE_RULE,
    LABELS3,
    NEG,
    NEU,
    POS,
)
from src.data.build_wtrans import (
    decide_emotion_column_rule,
    main as build_wtrans_main,
    run_build_wtrans_stage_b,
)
from src.data.polarity_kappa import compute_cohen_kappa, run_polarity_kappa_stage_a
from tests.test_polarity_kappa import _write_synthetic_stage_a_files


def _prepare_stage_b_fixture(
    base_dir: Path,
    other_nhat: List[int],
    other_long: List[int],
    surp_nhat: List[int],
    surp_long: List[int],
    adjudicated_default: int = NEU,
) -> Tuple[Path, Path, Path, Path, Path]:
    """Run Stage A on synthetic files and populate `final_label` in `polarity_disagreements.csv`."""
    in_dir = base_dir / "clean"
    stage_a_dir = base_dir / "results_polarity"
    p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
        in_dir,
        other_nhat,
        other_long,
        surp_nhat,
        surp_long,
        include_key=True,
        include_split_col=True,
        split_val="train",
    )
    run_polarity_kappa_stage_a(
        nhat_path=p_nhat,
        long_path=p_long,
        key_path=p_key,
        out_dir=stage_a_dir,
        n_boot=100,
        seed=42,
    )
    p_dis = stage_a_dir / "polarity_disagreements.csv"
    p_hash = stage_a_dir / "polarity_files_hash.json"

    df_dis = pd.read_csv(p_dis, dtype=str, keep_default_na=False)
    if len(df_dis) > 0:
        df_dis["final_label"] = str(adjudicated_default)
        df_dis.to_csv(p_dis, index=False, encoding="utf-8", lineterminator="\n")

    return p_nhat, p_long, p_key, p_dis, p_hash


# ---------------------------------------------------------------------------
# 1 & 2. Decision table, boundary conditions, and approval threshold precision
# ---------------------------------------------------------------------------
class TestDecisionTableAndThresholds:
    def test_decision_table_branches_and_boundaries(self) -> None:
        """1. Bảng quyết định: NaN dừng; <0.40 HARD_MERGE; >=0.40 & >=30/50 HARD; <30/50 SOFT; biên 30/50, 29/50, kappa=0.40, n!=50."""
        # kappa NaN -> stop
        with pytest.raises(ValueError, match="kappa is NaN"):
            decide_emotion_column_rule(
                kappa_all=float("nan"),
                counts={NEG: 10, NEU: 20, POS: 20},
            )

        # kappa < 0.40 -> HARD_MERGE (neutral = (0, 1, 0))
        dec_low = decide_emotion_column_rule(
            kappa_all=0.399999,
            counts={NEG: 45, NEU: 3, POS: 2},
        )
        assert dec_low["rule"] == "HARD_MERGE"
        assert dec_low["vector"] == (0.0, 1.0, 0.0)

        # Boundary: kappa == 0.40 exactly and max pole == 30/50 -> HARD (one-hot)
        dec_hard_30 = decide_emotion_column_rule(
            kappa_all=0.40,
            counts={NEG: 10, NEU: 10, POS: 30},
        )
        assert dec_hard_30["rule"] == "HARD"
        assert dec_hard_30["vector"] == (0.0, 0.0, 1.0)

        # Boundary: kappa == 0.40 and max pole == 29/50 -> SOFT (proportion vector)
        dec_soft_29 = decide_emotion_column_rule(
            kappa_all=0.40,
            counts={NEG: 11, NEU: 10, POS: 29},
        )
        assert dec_soft_29["rule"] == "SOFT"
        assert dec_soft_29["vector"] == pytest.approx((11 / 50, 10 / 50, 29 / 50), abs=1e-12)

        # Total sentences != 50 -> stop
        with pytest.raises(ValueError, match="Expected exactly 50 sentences"):
            decide_emotion_column_rule(
                kappa_all=0.65,
                counts={NEG: 10, NEU: 20, POS: 19},
            )

    def test_approval_thresholds_pole_056_gives_28_not_29_and_kappa_round9(
        self, tmp_path: Path
    ) -> None:
        """2. Ngưỡng từ wtrans_approval.md: pole_threshold=0,56 với n=50 phải ra 28 câu (ceil(round(x*n, 9))); kappa so sau round(kappa, 9)."""
        # In IEEE 754 float, 0.56 * 50 == 28.000000000000004; ceil(round(0.56 * 50, 9)) must be 28
        dec_28 = decide_emotion_column_rule(
            kappa_all=0.40 - 1e-12,  # rounds to 0.400000000 at 9 decimal places
            counts={NEG: 12, NEU: 10, POS: 28},
            kappa_threshold=0.40,
            pole_threshold=0.56,
            expected_n=50,
        )
        assert dec_28["min_pole_count"] == 28
        assert dec_28["rule"] == "HARD"
        assert dec_28["vector"] == (0.0, 0.0, 1.0)

        dec_27 = decide_emotion_column_rule(
            kappa_all=0.40 - 1e-12,
            counts={NEG: 13, NEU: 10, POS: 27},
            kappa_threshold=0.40,
            pole_threshold=0.56,
            expected_n=50,
        )
        assert dec_27["min_pole_count"] == 28
        assert dec_27["rule"] == "SOFT"

        # Verify through wtrans_approval.md in run_build_wtrans_stage_b as well
        other_l = [0] * 12 + [1] * 10 + [2] * 28
        surp_l = [0] * 13 + [1] * 10 + [2] * 27
        p_nhat, p_long, p_key, p_dis, p_hash = _prepare_stage_b_fixture(
            tmp_path / "pole_056", other_l, other_l, surp_l, surp_l
        )
        approval_md = tmp_path / "wtrans_approval.md"
        approval_md.write_text(
            "approved: true\n"
            "kappa_threshold: 0.40\n"
            "pole_threshold: 0.56\n"
            "approver: Chủ nhiệm\n"
            "date: 2026-11-07\n",
            encoding="utf-8",
        )
        res = run_build_wtrans_stage_b(
            date="2026-11-07",
            nhat_path=p_nhat,
            long_path=p_long,
            key_path=p_key,
            disagreements_path=p_dis,
            hash_path=p_hash,
            approval_path=approval_md,
            out_wtrans_dir=tmp_path / "wtrans_056",
            out_rule_doc=tmp_path / "docs_056" / "wtrans_rule.md",
            constants_path=None,
            n_boot=100,
            seed=42,
        )
        assert res["decisions"]["Other"]["min_pole_count"] == 28
        assert res["decisions"]["Other"]["rule"] == "HARD"
        assert res["decisions"]["Surprise"]["rule"] == "SOFT"


# ---------------------------------------------------------------------------
# 3. Stage B stop conditions (hash mismatch, modified disagreement, bad final_label, missing --date)
# ---------------------------------------------------------------------------
class TestStageBIntegrityStops:
    def test_stop_when_hash_changes_for_annotators_or_key(self, tmp_path: Path) -> None:
        other_nhat = [0] * 10 + [1] * 25 + [2] * 15
        other_long = [0] * 10 + [1] * 20 + [2] * 20
        surp_nhat = [0] * 5 + [1] * 10 + [2] * 35
        surp_long = [0] * 5 + [1] * 10 + [2] * 35

        for target_idx, label in enumerate(("nhat", "long", "key")):
            p_nhat, p_long, p_key, p_dis, p_hash = _prepare_stage_b_fixture(
                tmp_path / f"hash_{label}",
                other_nhat,
                other_long,
                surp_nhat,
                surp_long,
            )
            target_file = (p_nhat, p_long, p_key)[target_idx]
            target_file.write_text(
                target_file.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with pytest.raises(ValueError, match="SHA-256 integrity check failed"):
                run_build_wtrans_stage_b(
                    date="2026-11-07",
                    nhat_path=p_nhat,
                    long_path=p_long,
                    key_path=p_key,
                    disagreements_path=p_dis,
                    hash_path=p_hash,
                    approval_path=tmp_path / "none.md",
                    out_wtrans_dir=tmp_path / f"out_{label}",
                    out_rule_doc=tmp_path / f"doc_{label}" / "wtrans_rule.md",
                    constants_path=None,
                )

    def test_stop_when_disagreement_text_or_label_nhat_modified(self, tmp_path: Path) -> None:
        other_nhat = [0] * 10 + [1] * 25 + [2] * 15
        other_long = [0] * 10 + [1] * 20 + [2] * 20
        surp_nhat = [0] * 5 + [1] * 10 + [2] * 35
        surp_long = [0] * 5 + [1] * 10 + [2] * 35

        p_nhat, p_long, p_key, p_dis, p_hash = _prepare_stage_b_fixture(
            tmp_path / "mod_dis", other_nhat, other_long, surp_nhat, surp_long
        )
        df_dis = pd.read_csv(p_dis, dtype=str, keep_default_na=False)
        orig_text = df_dis.loc[0, "text"]

        df_dis.loc[0, "text"] = "câu đã bị sửa trái phép"
        df_dis.to_csv(p_dis, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="Modified 'text'"):
            run_build_wtrans_stage_b(
                date="2026-11-07",
                nhat_path=p_nhat,
                long_path=p_long,
                key_path=p_key,
                disagreements_path=p_dis,
                hash_path=p_hash,
                approval_path=tmp_path / "none.md",
                out_wtrans_dir=tmp_path / "w1",
                out_rule_doc=tmp_path / "d1" / "wtrans_rule.md",
                constants_path=None,
            )

        df_dis.loc[0, "text"] = orig_text
        df_dis.loc[0, "label_nhat"] = "0"
        df_dis.to_csv(p_dis, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="Modified 'label_nhat'"):
            run_build_wtrans_stage_b(
                date="2026-11-07",
                nhat_path=p_nhat,
                long_path=p_long,
                key_path=p_key,
                disagreements_path=p_dis,
                hash_path=p_hash,
                approval_path=tmp_path / "none.md",
                out_wtrans_dir=tmp_path / "w2",
                out_rule_doc=tmp_path / "d2" / "wtrans_rule.md",
                constants_path=None,
            )

    def test_stop_when_disagreement_id_set_mismatches(self, tmp_path: Path) -> None:
        other_nhat = [0] * 10 + [1] * 25 + [2] * 15
        other_long = [0] * 10 + [1] * 20 + [2] * 20
        surp_nhat = [0] * 5 + [1] * 10 + [2] * 35
        surp_long = [0] * 5 + [1] * 10 + [2] * 35

        p_nhat, p_long, p_key, p_dis, p_hash = _prepare_stage_b_fixture(
            tmp_path / "bad_ids", other_nhat, other_long, surp_nhat, surp_long
        )
        df_dis = pd.read_csv(p_dis, dtype=str, keep_default_na=False)
        df_dis.loc[0, "id"] = "10"  # id=10 is an agreed row, not a disagreed row
        df_dis.to_csv(p_dis, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="Disagreed IDs do not match"):
            run_build_wtrans_stage_b(
                date="2026-11-07",
                nhat_path=p_nhat,
                long_path=p_long,
                key_path=p_key,
                disagreements_path=p_dis,
                hash_path=p_hash,
                approval_path=tmp_path / "none.md",
                out_wtrans_dir=tmp_path / "w",
                out_rule_doc=tmp_path / "d" / "wtrans_rule.md",
                constants_path=None,
            )

    def test_stop_when_final_label_empty_or_outside_0_1_2(self, tmp_path: Path) -> None:
        other_nhat = [0] * 10 + [1] * 25 + [2] * 15
        other_long = [0] * 10 + [1] * 20 + [2] * 20
        surp_nhat = [0] * 5 + [1] * 10 + [2] * 35
        surp_long = [0] * 5 + [1] * 10 + [2] * 35

        p_nhat, p_long, p_key, p_dis, p_hash = _prepare_stage_b_fixture(
            tmp_path / "bad_final", other_nhat, other_long, surp_nhat, surp_long
        )
        df_dis = pd.read_csv(p_dis, dtype=str, keep_default_na=False)

        for bad_val in ("", "3", "-1", "abc"):
            df_dis.loc[0, "final_label"] = bad_val
            df_dis.to_csv(p_dis, index=False, encoding="utf-8", lineterminator="\n")
            with pytest.raises(ValueError, match="Invalid or empty final_label"):
                run_build_wtrans_stage_b(
                    date="2026-11-07",
                    nhat_path=p_nhat,
                    long_path=p_long,
                    key_path=p_key,
                    disagreements_path=p_dis,
                    hash_path=p_hash,
                    approval_path=tmp_path / "none.md",
                    out_wtrans_dir=tmp_path / "w",
                    out_rule_doc=tmp_path / "d" / "wtrans_rule.md",
                    constants_path=None,
                )

    def test_stop_when_missing_date_argument(self, tmp_path: Path) -> None:
        # CLI without --date must exit with non-zero code
        with pytest.raises(SystemExit) as exc_info:
            build_wtrans_main([])
        assert exc_info.value.code != 0

        # Direct call with emptydate string raises ValueError
        with pytest.raises(ValueError, match="Invalid --date"):
            run_build_wtrans_stage_b(date="", constants_path=None)


# ---------------------------------------------------------------------------
# 4, 5, 6, 7, 8, 9. Matrix, HARD_MERGE equality, independent kappa, constants.py, rule doc, determinism
# ---------------------------------------------------------------------------
class TestMatrixApprovalReportAndDeterminism:
    def test_matrix_3x7_structure_fixed_columns_and_sums(self, tmp_path: Path) -> None:
        """4. Ma trận 3x7: hàng theo LABELS3, cột theo EMOTIONS7; mỗi cột tổng 1; Enjoyment=(0,0,1); Sadness,Fear,Anger,Disgust=(1,0,0)."""
        other_l = [0] * 18 + [1] * 17 + [2] * 15  # SOFT: (0.36, 0.34, 0.30)
        surp_l = [0] * 5 + [1] * 10 + [2] * 35  # HARD: (0, 0, 1)

        p_nhat, p_long, p_key, p_dis, p_hash = _prepare_stage_b_fixture(
            tmp_path / "mat", other_l, other_l, surp_l, surp_l
        )
        res = run_build_wtrans_stage_b(
            date="2026-11-07",
            nhat_path=p_nhat,
            long_path=p_long,
            key_path=p_key,
            disagreements_path=p_dis,
            hash_path=p_hash,
            approval_path=tmp_path / "none.md",
            out_wtrans_dir=tmp_path / "wtrans_out",
            out_rule_doc=tmp_path / "docs" / "wtrans_rule.md",
            constants_path=None,
            n_boot=100,
            seed=42,
        )

        df_w = pd.read_csv(res["wtrans_csv_path"])
        assert df_w.shape == (3, 8)  # 'label' + 7 emotion columns
        assert tuple(df_w.columns) == ("label", *EMOTIONS7)
        assert tuple(df_w["label"]) == LABELS3

        for emo in EMOTIONS7:
            col = df_w[emo].to_numpy(dtype=float)
            assert float(np.sum(col)) == pytest.approx(1.0, abs=1e-9)

        assert df_w["Enjoyment"].to_numpy(dtype=float).tolist() == pytest.approx([0.0, 0.0, 1.0])
        for neg_emo in ("Sadness", "Fear", "Anger", "Disgust"):
            assert df_w[neg_emo].to_numpy(dtype=float).tolist() == pytest.approx([1.0, 0.0, 0.0])

        assert df_w["Surprise"].to_numpy(dtype=float).tolist() == pytest.approx([0.0, 0.0, 1.0])
        assert df_w["Other"].to_numpy(dtype=float).tolist() == pytest.approx([0.36, 0.34, 0.30])

    def test_hard_merge_both_columns_equals_wtrans_hardmerge(self, tmp_path: Path) -> None:
        """5. Luật HARD_MERGE cho cả hai cột thì wtrans.csv bằng wtrans_hardmerge.csv."""
        other_nhat = [0] * 20 + [1] * 20 + [2] * 10
        other_long = [1] * 20 + [2] * 20 + [0] * 10
        surp_nhat = [0] * 20 + [1] * 20 + [2] * 10
        surp_long = [2] * 20 + [0] * 20 + [1] * 10

        p_nhat, p_long, p_key, p_dis, p_hash = _prepare_stage_b_fixture(
            tmp_path / "hm", other_nhat, other_long, surp_nhat, surp_long, adjudicated_default=POS
        )
        res = run_build_wtrans_stage_b(
            date="2026-11-07",
            nhat_path=p_nhat,
            long_path=p_long,
            key_path=p_key,
            disagreements_path=p_dis,
            hash_path=p_hash,
            approval_path=tmp_path / "none.md",
            out_wtrans_dir=tmp_path / "out_hm",
            out_rule_doc=tmp_path / "docs_hm" / "wtrans_rule.md",
            constants_path=None,
            n_boot=100,
            seed=42,
        )
        assert res["decisions"]["Surprise"]["rule"] == "HARD_MERGE"
        assert res["decisions"]["Other"]["rule"] == "HARD_MERGE"
        assert res["wtrans_csv_path"].read_bytes() == res["hardmerge_csv_path"].read_bytes()

    def test_stage_b_kappa_is_computed_from_independent_annotations_not_final_label(
        self, tmp_path: Path
    ) -> None:
        """6. kappa dùng ở giai đoạn B tính từ hai file nhãn độc lập, KHÔNG dùng final_label."""
        # Nhat and Long strongly disagree (kappa < 0.40), but final_label is set identical to Nhat (POS=45/50).
        # If final_label were mistakenly used to compute kappa against Nhat or Long, kappa would be >= 0.40
        # and Surprise/Other would become HARD (POS). Because independent kappa < 0.40, rule MUST be HARD_MERGE.
        other_nhat = [POS] * 45 + [NEG] * 5
        other_long = [NEG] * 45 + [POS] * 5
        surp_nhat = [POS] * 45 + [NEU] * 5
        surp_long = [NEU] * 45 + [POS] * 5

        p_nhat, p_long, p_key, p_dis, p_hash = _prepare_stage_b_fixture(
            tmp_path / "indep_kappa",
            other_nhat,
            other_long,
            surp_nhat,
            surp_long,
            adjudicated_default=POS,
        )
        expected_indep_kappa = compute_cohen_kappa(
            other_nhat + surp_nhat, other_long + surp_long
        )
        assert expected_indep_kappa < 0.40

        res = run_build_wtrans_stage_b(
            date="2026-11-07",
            nhat_path=p_nhat,
            long_path=p_long,
            key_path=p_key,
            disagreements_path=p_dis,
            hash_path=p_hash,
            approval_path=tmp_path / "none.md",
            out_wtrans_dir=tmp_path / "out_indep",
            out_rule_doc=tmp_path / "docs_indep" / "wtrans_rule.md",
            constants_path=None,
            n_boot=100,
            seed=42,
        )
        assert res["metrics"]["all"]["kappa"] == pytest.approx(expected_indep_kappa, abs=1e-12)
        assert res["decisions"]["Surprise"]["rule"] == "HARD_MERGE"
        assert res["decisions"]["Other"]["rule"] == "HARD_MERGE"

    def test_wtrans_init_rule_constants_protection_and_declaration(
        self, tmp_path: Path
    ) -> None:
        """7. WTRANS_INIT_RULE: thiếu/không hợp lệ wtrans_approval.md thì không ghi constants.py; hợp lệ thì ghi đúng ánh xạ và giữ nguyên HARD_MERGE_RULE."""
        other_l = [0] * 18 + [1] * 17 + [2] * 15  # SOFT: (0.36, 0.34, 0.30)
        surp_l = [0] * 5 + [1] * 10 + [2] * 35  # HARD: (0, 0, 1)

        p_nhat, p_long, p_key, p_dis, p_hash = _prepare_stage_b_fixture(
            tmp_path / "const_test", other_l, other_l, surp_l, surp_l
        )

        orig_constants = _REPO_ROOT / "src" / "common" / "constants.py"
        tmp_const = tmp_path / "constants_copy.py"
        shutil.copyfile(orig_constants, tmp_const)
        orig_bytes = tmp_const.read_bytes()

        # (a) Non-existent approval file -> untouched
        res_missing = run_build_wtrans_stage_b(
            date="2026-11-07",
            nhat_path=p_nhat,
            long_path=p_long,
            key_path=p_key,
            disagreements_path=p_dis,
            hash_path=p_hash,
            approval_path=tmp_path / "does_not_exist.md",
            out_wtrans_dir=tmp_path / "w_a",
            out_rule_doc=tmp_path / "d_a" / "wtrans_rule.md",
            constants_path=tmp_const,
            n_boot=100,
            seed=42,
        )
        assert res_missing["constants_updated"] is False
        assert tmp_const.read_bytes() == orig_bytes

        # (b) Missing required key ('approver' missing) -> untouched
        bad_approval_missing_key = tmp_path / "approval_missing_key.md"
        bad_approval_missing_key.write_text(
            "approved: true\nkappa_threshold: 0.40\npole_threshold: 0.60\ndate: 2026-11-07\n",
            encoding="utf-8",
        )
        res_missing_key = run_build_wtrans_stage_b(
            date="2026-11-07",
            nhat_path=p_nhat,
            long_path=p_long,
            key_path=p_key,
            disagreements_path=p_dis,
            hash_path=p_hash,
            approval_path=bad_approval_missing_key,
            out_wtrans_dir=tmp_path / "w_b",
            out_rule_doc=tmp_path / "d_b" / "wtrans_rule.md",
            constants_path=tmp_const,
            n_boot=100,
            seed=42,
        )
        assert res_missing_key["constants_updated"] is False
        assert tmp_const.read_bytes() == orig_bytes

        # (c) approved != true -> untouched
        bad_approval_false = tmp_path / "approval_false.md"
        bad_approval_false.write_text(
            "approved: false\n"
            "kappa_threshold: 0.40\n"
            "pole_threshold: 0.60\n"
            "approver: PhuongNV\n"
            "date: 2026-11-07\n",
            encoding="utf-8",
        )
        res_false = run_build_wtrans_stage_b(
            date="2026-11-07",
            nhat_path=p_nhat,
            long_path=p_long,
            key_path=p_key,
            disagreements_path=p_dis,
            hash_path=p_hash,
            approval_path=bad_approval_false,
            out_wtrans_dir=tmp_path / "w_c",
            out_rule_doc=tmp_path / "d_c" / "wtrans_rule.md",
            constants_path=tmp_const,
            n_boot=100,
            seed=42,
        )
        assert res_false["constants_updated"] is False
        assert tmp_const.read_bytes() == orig_bytes

        # (d) Valid approval -> updates tmp_const with WTRANS_INIT_RULE and preserves HARD_MERGE_RULE
        good_approval = tmp_path / "approval_valid.md"
        good_approval.write_text(
            "approved: true\n"
            "kappa_threshold: 0.40\n"
            "pole_threshold: 0.60\n"
            "approver: PhuongNV\n"
            "date: 2026-11-07\n",
            encoding="utf-8",
        )
        res_valid = run_build_wtrans_stage_b(
            date="2026-11-07",
            nhat_path=p_nhat,
            long_path=p_long,
            key_path=p_key,
            disagreements_path=p_dis,
            hash_path=p_hash,
            approval_path=good_approval,
            out_wtrans_dir=tmp_path / "w_d",
            out_rule_doc=tmp_path / "d_d" / "wtrans_rule.md",
            constants_path=tmp_const,
            n_boot=100,
            seed=42,
        )
        assert res_valid["constants_updated"] is True
        ns: Dict[str, object] = {}
        exec(tmp_const.read_text(encoding="utf-8"), ns)
        assert ns["HARD_MERGE_RULE"] == HARD_MERGE_RULE
        wtrans_rule = ns["WTRANS_INIT_RULE"]
        assert isinstance(wtrans_rule, dict)
        assert wtrans_rule[EMO_ENJOYMENT] == (0.0, 0.0, 1.0)
        for neg_id in (EMO_SADNESS, EMO_FEAR, EMO_ANGER, EMO_DISGUST):
            assert wtrans_rule[neg_id] == (1.0, 0.0, 0.0)
        assert wtrans_rule[EMO_SURPRISE] == (0.0, 0.0, 1.0)
        assert wtrans_rule[EMO_OTHER] == pytest.approx((0.36, 0.34, 0.30), abs=1e-6)

    def test_wtrans_rule_md_contains_all_required_fields_and_byte_determinism(
        self, tmp_path: Path
    ) -> None:
        """8 & 9. wtrans_rule.md có đủ thông tin; chạy hai lần cùng --date cho 3 file đầu ra giống hệt từng byte."""
        other_nhat = [0] * 15 + [1] * 30 + [2] * 5  # low kappa on Other
        other_long = [1] * 15 + [2] * 15 + [0] * 15 + [1] * 5
        surp_nhat = [0] * 10 + [1] * 10 + [2] * 30  # high kappa on Surprise, max pole 30/50
        surp_long = [0] * 10 + [1] * 10 + [2] * 30

        p_nhat, p_long, p_key, p_dis, p_hash = _prepare_stage_b_fixture(
            tmp_path / "doc_det", other_nhat, other_long, surp_nhat, surp_long, adjudicated_default=NEU
        )

        res1 = run_build_wtrans_stage_b(
            date="2026-11-08",
            nhat_path=p_nhat,
            long_path=p_long,
            key_path=p_key,
            disagreements_path=p_dis,
            hash_path=p_hash,
            approval_path=tmp_path / "no_approval.md",
            out_wtrans_dir=tmp_path / "out1",
            out_rule_doc=tmp_path / "docs1" / "wtrans_rule.md",
            constants_path=None,
            n_boot=100,
            seed=42,
            adjudicator="Chủ nhiệm đề tài",
        )
        res2 = run_build_wtrans_stage_b(
            date="2026-11-08",
            nhat_path=p_nhat,
            long_path=p_long,
            key_path=p_key,
            disagreements_path=p_dis,
            hash_path=p_hash,
            approval_path=tmp_path / "no_approval.md",
            out_wtrans_dir=tmp_path / "out2",
            out_rule_doc=tmp_path / "docs2" / "wtrans_rule.md",
            constants_path=None,
            n_boot=100,
            seed=42,
            adjudicator="Chủ nhiệm đề tài",
        )

        # 8. Check required content in wtrans_rule.md
        md_text = res1["rule_doc_path"].read_text(encoding="utf-8")
        assert "2026-11-08" in md_text  # --date
        assert "mặc định" in md_text  # threshold source
        assert "0.400000" in md_text and "0.600000" in md_text  # thresholds
        assert "Chủ nhiệm đề tài" in md_text  # adjudicator
        assert "Surprise" in md_text and "Other" in md_text
        assert res1["decisions"]["Surprise"]["rule"] in md_text
        assert res1["decisions"]["Other"]["rule"] in md_text
        assert "Cohen's κ" in md_text and "95% CI (Bootstrap)" in md_text
        assert "κ trọng số tuyến tính" in md_text and "Gwet AC1" in md_text
        assert "Wilson 95% CI" in md_text
        hashes_obj = json.loads(p_hash.read_text(encoding="utf-8"))
        assert hashes_obj["polarity_nhat.csv"] in md_text
        assert hashes_obj["polarity_long.csv"] in md_text
        assert hashes_obj["polarity_key.csv"] in md_text
        assert "## 5. Cảnh báo" in md_text
        assert "hai κ riêng nằm hai phía" in md_text
        assert "chứa ngưỡng 0.600000" in md_text

        # 9. Byte-for-byte determinism across two runs with same --date
        assert res1["wtrans_csv_path"].read_bytes() == res2["wtrans_csv_path"].read_bytes()
        assert res1["hardmerge_csv_path"].read_bytes() == res2["hardmerge_csv_path"].read_bytes()
        assert res1["rule_doc_path"].read_bytes() == res2["rule_doc_path"].read_bytes()
