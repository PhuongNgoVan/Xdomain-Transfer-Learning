"""Unit and acceptance tests for Stage A of M7 (`src/data/polarity_kappa.py`)."""

import json
import math
from pathlib import Path
import re
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.constants import (
    EMO_OTHER,
    EMO_SURPRISE,
    ID_TO_EMOTION7,
    LABELS3,
    NEG,
    NEU,
    POS,
)
from src.data.polarity_kappa import (
    bootstrap_kappa_ci,
    compute_cohen_kappa,
    compute_confusion_matrix_df,
    compute_gwet_ac1,
    compute_linear_weighted_kappa,
    compute_pabak,
    compute_po_and_kappa,
    load_and_validate_polarity_inputs,
    run_polarity_kappa_stage_a,
    wilson_ci,
)


def _write_synthetic_stage_a_files(
    base_dir: Path,
    labels_other_nhat: List[int],
    labels_other_long: List[int],
    labels_surp_nhat: List[int],
    labels_surp_long: List[int],
    include_key: bool = True,
    include_split_col: bool = True,
    split_val: str = "train",
    include_fallback_cols_in_annotators: bool = False,
    shuffle_nhat: bool = False,
    shuffle_long: bool = False,
    shuffle_key: bool = False,
    seed: int = 42,
) -> Tuple[Path, Path, Path]:
    """Write synthetic `polarity_nhat.csv`, `polarity_long.csv`, and `polarity_key.csv` in `base_dir`."""
    base_dir.mkdir(parents=True, exist_ok=True)
    rows_nhat = []
    rows_long = []
    rows_key = []

    for i, (ln, ll) in enumerate(zip(labels_other_nhat, labels_other_long)):
        rid = 10 + i
        txt = f"câu other mẫu số {i}"
        r_n = {"id": rid, "text": txt, "polarity_label": ln}
        r_l = {"id": rid, "text": txt, "polarity_label": ll}
        if include_fallback_cols_in_annotators:
            r_n["emotion_id"] = EMO_OTHER
            r_n["emotion_name"] = ID_TO_EMOTION7[EMO_OTHER]
            r_l["emotion_id"] = EMO_OTHER
            r_l["emotion_name"] = ID_TO_EMOTION7[EMO_OTHER]
        rows_nhat.append(r_n)
        rows_long.append(r_l)
        r_k = {
            "id": rid,
            "text": txt,
            "emotion_id": EMO_OTHER,
            "emotion_name": ID_TO_EMOTION7[EMO_OTHER],
        }
        if include_split_col:
            r_k["split"] = split_val
        rows_key.append(r_k)

    for i, (ln, ll) in enumerate(zip(labels_surp_nhat, labels_surp_long)):
        rid = 500 + i
        txt = f"câu surprise mẫu số {i}"
        r_n = {"id": rid, "text": txt, "polarity_label": ln}
        r_l = {"id": rid, "text": txt, "polarity_label": ll}
        if include_fallback_cols_in_annotators:
            r_n["emotion_id"] = EMO_SURPRISE
            r_n["emotion_name"] = ID_TO_EMOTION7[EMO_SURPRISE]
            r_l["emotion_id"] = EMO_SURPRISE
            r_l["emotion_name"] = ID_TO_EMOTION7[EMO_SURPRISE]
        rows_nhat.append(r_n)
        rows_long.append(r_l)
        r_k = {
            "id": rid,
            "text": txt,
            "emotion_id": EMO_SURPRISE,
            "emotion_name": ID_TO_EMOTION7[EMO_SURPRISE],
        }
        if include_split_col:
            r_k["split"] = split_val
        rows_key.append(r_k)

    df_nhat = pd.DataFrame(rows_nhat)
    df_long = pd.DataFrame(rows_long)
    df_key = pd.DataFrame(rows_key)

    rng = np.random.default_rng(seed)
    if shuffle_nhat:
        perm = rng.permutation(len(df_nhat))
        df_nhat = df_nhat.iloc[perm].reset_index(drop=True)
    if shuffle_long:
        perm = rng.permutation(len(df_long))
        df_long = df_long.iloc[perm].reset_index(drop=True)
    if shuffle_key:
        perm = rng.permutation(len(df_key))
        df_key = df_key.iloc[perm].reset_index(drop=True)

    p_nhat = base_dir / "polarity_nhat.csv"
    p_long = base_dir / "polarity_long.csv"
    p_key = base_dir / "polarity_key.csv"

    df_nhat.to_csv(p_nhat, index=False, encoding="utf-8", lineterminator="\n")
    df_long.to_csv(p_long, index=False, encoding="utf-8", lineterminator="\n")
    if include_key:
        df_key.to_csv(p_key, index=False, encoding="utf-8", lineterminator="\n")

    return p_nhat, p_long, p_key


def _default_50_labels() -> Tuple[List[int], List[int]]:
    other = [0, 1, 2] * 16 + [0, 1]
    surp = [1, 2, 0] * 16 + [1, 2]
    return other, surp


# ---------------------------------------------------------------------------
# 1, 2, 3. Hand-calculated examples, edge cases, and sklearn equivalence
# ---------------------------------------------------------------------------
class TestHandExamplesAndMetrics:
    def test_hand_example_section_4(self) -> None:
        """1. Ví dụ tính tay a=[0,0,1,1,2,2,0,1,2,1], b=[0,1,1,1,2,0,0,1,2,2]."""
        a = [0, 0, 1, 1, 2, 2, 0, 1, 2, 1]
        b = [0, 1, 1, 1, 2, 0, 0, 1, 2, 2]

        po, pe, kappa = compute_po_and_kappa(a, b)
        assert po == pytest.approx(0.7, abs=1e-12)
        assert pe == pytest.approx(0.34, abs=1e-12)
        assert kappa == pytest.approx(6.0 / 11.0, abs=1e-12)
        assert round(kappa, 4) == pytest.approx(0.5455, abs=1e-4)

        ac1 = compute_gwet_ac1(a, b)
        # pe for AC1 = 0.33 -> AC1 = (0.7 - 0.33) / (1 - 0.33) = 0.37 / 0.67
        assert ac1 == pytest.approx(0.37 / 0.67, abs=1e-12)
        assert round(ac1, 4) == pytest.approx(0.5522, abs=1e-4)

        pabak = compute_pabak(a, b)
        assert pabak == pytest.approx(0.55, abs=1e-12)

        kappa_lin = compute_linear_weighted_kappa(a, b)
        assert kappa_lin == pytest.approx(1.0 - (0.4 / 0.84), abs=1e-12)
        assert round(kappa_lin, 4) == pytest.approx(0.5238, abs=1e-4)

    def test_identical_three_labels_and_single_label_nan(self) -> None:
        """2. Hai danh sách giống nhau đủ ba nhãn: kappa=1. Giống nhau, một nhãn: kappa=NaN."""
        a_three = [0, 1, 2, 0, 1, 2]
        assert compute_cohen_kappa(a_three, a_three) == pytest.approx(1.0, abs=1e-12)
        assert compute_linear_weighted_kappa(a_three, a_three) == pytest.approx(1.0, abs=1e-12)

        a_single = [1, 1, 1, 1, 1, 1]
        po, pe, kappa = compute_po_and_kappa(a_single, a_single)
        assert po == pytest.approx(1.0, abs=1e-12)
        assert pe == pytest.approx(1.0, abs=1e-12)
        assert math.isnan(kappa)
        assert math.isnan(compute_linear_weighted_kappa(a_single, a_single))

    @pytest.mark.parametrize("seed", [0, 7, 42, 99, 1234])
    def test_matches_sklearn_cohen_kappa_unweighted_and_linear(self, seed: int) -> None:
        """3. kappa và kappa trọng số khớp sklearn.metrics.cohen_kappa_score trên dữ liệu ngẫu nhiên."""
        rng = np.random.default_rng(seed)
        a = rng.integers(0, 3, size=80).tolist()
        b = rng.integers(0, 3, size=80).tolist()

        sk_unweighted = cohen_kappa_score(a, b, labels=[0, 1, 2], weights=None)
        sk_linear = cohen_kappa_score(a, b, labels=[0, 1, 2], weights="linear")

        assert compute_cohen_kappa(a, b) == pytest.approx(sk_unweighted, abs=1e-12)
        assert compute_linear_weighted_kappa(a, b) == pytest.approx(sk_linear, abs=1e-12)


# ---------------------------------------------------------------------------
# 4, 5, 6. Bootstrap CI, Confusion matrix 3x3, and Wilson 95% CI
# ---------------------------------------------------------------------------
class TestBootstrapConfusionAndWilson:
    def test_bootstrap_reproducibility_nan_dropping_and_warning(self) -> None:
        """4. Bootstrap: cùng seed cho cùng CI; kappa NaN được đếm và bỏ; cảnh báo khi bỏ >5%."""
        a = [0, 0, 1, 1, 2, 2, 0, 1, 2, 1] * 5
        b = [0, 1, 1, 1, 2, 0, 0, 1, 2, 2] * 5

        b1 = bootstrap_kappa_ci(a, b, n_boot=500, seed=42)
        b2 = bootstrap_kappa_ci(a, b, n_boot=500, seed=42)
        assert b1 == b2
        assert b1["dropped_count"] == 0
        assert b1["dropped_rate"] == pytest.approx(0.0)
        assert b1["warning_ci_biased"] is False
        assert b1["ci_low"] < compute_cohen_kappa(a, b) < b1["ci_high"]

        # All-identical single-class -> 100% NaN dropped
        single = [1] * 20
        b_nan = bootstrap_kappa_ci(single, single, n_boot=200, seed=42)
        assert b_nan["dropped_count"] == 200
        assert b_nan["dropped_rate"] == pytest.approx(1.0)
        assert b_nan["warning_ci_biased"] is True
        assert math.isnan(b_nan["ci_low"]) and math.isnan(b_nan["ci_high"])

        # Rare second label -> >5% of bootstrap resamples are single-label (NaN)
        a_rare = [1, 1, 1, 1, 1, 1, 1, 2]
        b_rare = [1, 1, 1, 1, 1, 1, 1, 2]
        b_warn = bootstrap_kappa_ci(a_rare, b_rare, n_boot=500, seed=42)
        assert b_warn["dropped_count"] > 0
        assert b_warn["dropped_rate"] == pytest.approx(b_warn["dropped_count"] / 500.0)
        assert b_warn["dropped_rate"] > 0.05
        assert b_warn["warning_ci_biased"] is True

    def test_confusion_matrix_3x3_rows_nhat_cols_long_and_subsets(self, tmp_path: Path) -> None:
        """5. Ma trận nhầm lẫn 3x3: hàng là Nhật, cột là Long, thứ tự NEG, NEU, POS; toàn bộ và từng nhãn gốc."""
        # Other: 20 (NEG, NEG), 10 (NEG, NEU), 15 (NEU, POS), 5 (POS, POS)
        other_nhat = [NEG] * 30 + [NEU] * 15 + [POS] * 5
        other_long = [NEG] * 20 + [NEU] * 10 + [POS] * 20
        # Surprise: 10 (NEG, POS), 25 (NEU, NEU), 15 (POS, NEG)
        surp_nhat = [NEG] * 10 + [NEU] * 25 + [POS] * 15
        surp_long = [POS] * 10 + [NEU] * 25 + [NEG] * 15

        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path / "in",
            other_nhat,
            other_long,
            surp_nhat,
            surp_long,
        )
        out_dir = tmp_path / "out"
        res = run_polarity_kappa_stage_a(p_nhat, p_long, p_key, out_dir, n_boot=100, seed=42)

        for name in ("all", "other", "surprise"):
            df_c = pd.read_csv(out_dir / f"confusion_{name}.csv")
            assert list(df_c.columns) == ["label", *LABELS3]
            assert list(df_c["label"]) == list(LABELS3)
            assert df_c.equals(res["confusion"][name])

        c_other = res["confusion"]["other"].set_index("label")
        # Row = Nhat (negative), Col = Long (neutral) has 10; transpose (neutral, negative) has 0
        assert c_other.loc["negative", "negative"] == 20
        assert c_other.loc["negative", "neutral"] == 10
        assert c_other.loc["neutral", "negative"] == 0
        assert c_other.loc["neutral", "positive"] == 15
        assert c_other.loc["positive", "positive"] == 5

        c_surp = res["confusion"]["surprise"].set_index("label")
        assert c_surp.loc["negative", "positive"] == 10
        assert c_surp.loc["positive", "negative"] == 15
        assert c_surp.loc["neutral", "neutral"] == 25

        c_all = res["confusion"]["all"].set_index("label")
        assert (c_all.to_numpy() == c_other.to_numpy() + c_surp.to_numpy()).all()

    def test_wilson_95_ci_30_50_and_29_50(self) -> None:
        """6. Wilson 95%: 30/50 xấp xỉ [0,462; 0,724], 29/50 xấp xỉ [0,442; 0,706]."""
        low_30, high_30 = wilson_ci(30, 50)
        assert low_30 == pytest.approx(0.462, abs=1e-3)
        assert high_30 == pytest.approx(0.724, abs=1e-3)

        low_29, high_29 = wilson_ci(29, 50)
        assert low_29 == pytest.approx(0.442, abs=1e-3)
        assert high_29 == pytest.approx(0.706, abs=1e-3)


# ---------------------------------------------------------------------------
# 7. Input validation must stop (one test per case)
# ---------------------------------------------------------------------------
class TestInputValidationStops:
    def test_stop_when_row_count_not_100(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path, other_ok[:-1], other_ok[:-1], surp_ok, surp_ok
        )
        with pytest.raises(ValueError, match="Expected 100 rows"):
            load_and_validate_polarity_inputs(p_nhat, p_long, p_key)

    def test_stop_when_duplicate_id(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path, other_ok, other_ok, surp_ok, surp_ok
        )
        df_n = pd.read_csv(p_nhat)
        df_n.loc[1, "id"] = df_n.loc[0, "id"]
        df_n.to_csv(p_nhat, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="Duplicate id"):
            load_and_validate_polarity_inputs(p_nhat, p_long, p_key)

    def test_stop_when_id_sets_differ_across_three_files(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path / "case1", other_ok, other_ok, surp_ok, surp_ok
        )
        df_l = pd.read_csv(p_long)
        df_l.loc[0, "id"] = 99999
        df_l.to_csv(p_long, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="ID mismatch"):
            load_and_validate_polarity_inputs(p_nhat, p_long, p_key)

        p_nhat2, p_long2, p_key2 = _write_synthetic_stage_a_files(
            tmp_path / "case2", other_ok, other_ok, surp_ok, surp_ok
        )
        df_k = pd.read_csv(p_key2)
        df_k.loc[0, "id"] = 88888
        df_k.to_csv(p_key2, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="ID mismatch"):
            load_and_validate_polarity_inputs(p_nhat2, p_long2, p_key2)

    def test_stop_when_polarity_label_out_of_range(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        bad_other = list(other_ok)
        bad_other[0] = 3
        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path, bad_other, other_ok, surp_ok, surp_ok
        )
        with pytest.raises(ValueError, match="invalid or empty polarity_label"):
            load_and_validate_polarity_inputs(p_nhat, p_long, p_key)

    def test_stop_when_polarity_label_empty(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path, other_ok, other_ok, surp_ok, surp_ok
        )
        df_l = pd.read_csv(p_long, dtype=str)
        df_l.loc[5, "polarity_label"] = ""
        df_l.to_csv(p_long, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="invalid or empty polarity_label"):
            load_and_validate_polarity_inputs(p_nhat, p_long, p_key)

    def test_stop_when_text_differs_between_two_annotators(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path, other_ok, other_ok, surp_ok, surp_ok
        )
        df_l = pd.read_csv(p_long, dtype=str)
        df_l.loc[0, "text"] = "văn bản bị sửa ở file của Long"
        df_l.to_csv(p_long, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="Text mismatch"):
            load_and_validate_polarity_inputs(p_nhat, p_long, p_key)

    def test_stop_when_text_differs_from_polarity_key(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path, other_ok, other_ok, surp_ok, surp_ok
        )
        df_k = pd.read_csv(p_key, dtype=str)
        df_k.loc[0, "text"] = "văn bản khác trong polarity_key"
        df_k.to_csv(p_key, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="Text mismatch"):
            load_and_validate_polarity_inputs(p_nhat, p_long, p_key)

    def test_stop_when_emotion_counts_not_50_each(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path, other_ok, other_ok, surp_ok, surp_ok
        )
        df_k = pd.read_csv(p_key)
        df_k.loc[60, "emotion_id"] = EMO_OTHER
        df_k.loc[60, "emotion_name"] = ID_TO_EMOTION7[EMO_OTHER]
        df_k.to_csv(p_key, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="Expected 50 Other and 50 Surprise"):
            load_and_validate_polarity_inputs(p_nhat, p_long, p_key)

    def test_stop_when_split_not_train(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path,
            other_ok,
            other_ok,
            surp_ok,
            surp_ok,
            include_split_col=True,
            split_val="dev",
        )
        with pytest.raises(ValueError, match="must come from 'train' split"):
            load_and_validate_polarity_inputs(p_nhat, p_long, p_key)

    def test_stop_when_fallback_emotion_columns_mismatch(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, _ = _write_synthetic_stage_a_files(
            tmp_path,
            other_ok,
            other_ok,
            surp_ok,
            surp_ok,
            include_key=False,
            include_fallback_cols_in_annotators=True,
        )
        df_l = pd.read_csv(p_long)
        df_l.loc[0, "emotion_id"] = EMO_SURPRISE
        df_l.loc[0, "emotion_name"] = ID_TO_EMOTION7[EMO_SURPRISE]
        df_l.to_csv(p_long, index=False, encoding="utf-8", lineterminator="\n")
        with pytest.raises(ValueError, match="Fallback emotion_id/emotion_name mismatch"):
            load_and_validate_polarity_inputs(p_nhat, p_long, tmp_path / "polarity_key.csv")

    def test_stop_when_missing_both_key_and_fallback_emotion_columns(self, tmp_path: Path) -> None:
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, _ = _write_synthetic_stage_a_files(
            tmp_path,
            other_ok,
            other_ok,
            surp_ok,
            surp_ok,
            include_key=False,
            include_fallback_cols_in_annotators=False,
        )
        with pytest.raises(ValueError, match="Missing polarity_key.csv"):
            load_and_validate_polarity_inputs(p_nhat, p_long, tmp_path / "polarity_key.csv")


# ---------------------------------------------------------------------------
# 8, 9, 10, 11, 12. Warnings, Disagreements, Determinism, and Row Shuffling
# ---------------------------------------------------------------------------
class TestStageAOutputsWarningsAndDeterminism:
    def test_missing_split_column_does_not_stop_and_emits_warning(self, tmp_path: Path) -> None:
        """8. Không có cột split: không dừng, nhưng có cảnh báo 'không kiểm chứng được nguồn câu'."""
        other_ok, surp_ok = _default_50_labels()
        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path / "in",
            other_ok,
            other_ok,
            surp_ok,
            surp_ok,
            include_split_col=False,
        )
        res = run_polarity_kappa_stage_a(
            p_nhat, p_long, p_key, tmp_path / "out", n_boot=100, seed=42
        )
        assert any("không kiểm chứng được nguồn câu" in w for w in res["warnings"])

    def test_disagreements_exact_ids_empty_final_label_and_sorted_by_id(
        self, tmp_path: Path
    ) -> None:
        """9. Danh sách bất đồng: đúng tập id hai file thực sự khác nhau, final_label để trống, sắp theo id."""
        other_nhat = [0] * 25 + [1] * 25
        other_long = [0] * 20 + [2] * 5 + [1] * 25  # 5 disagreements at i=20..24 (ids 30..34)
        surp_nhat = [1] * 25 + [2] * 25
        surp_long = [1] * 22 + [0] * 3 + [2] * 25  # 3 disagreements at i=22..24 (ids 522..524)

        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path / "in",
            other_nhat,
            other_long,
            surp_nhat,
            surp_long,
            shuffle_nhat=True,
            shuffle_long=True,
        )
        out_dir = tmp_path / "out"
        run_polarity_kappa_stage_a(p_nhat, p_long, p_key, out_dir, n_boot=100, seed=42)

        df_dis = pd.read_csv(out_dir / "polarity_disagreements.csv", dtype=str, keep_default_na=False)
        expected_ids = [30, 31, 32, 33, 34, 522, 523, 524]
        assert [int(x) for x in df_dis["id"].tolist()] == expected_ids
        assert all(val == "" for val in df_dis["final_label"].tolist())

    def test_warnings_wilson_060_050_opposite_sides_040_and_kappa_nan(
        self, tmp_path: Path
    ) -> None:
        """10. Cảnh báo: CI Wilson chứa 0,60; chứa 0,50; hai kappa riêng nằm hai phía 0,40; kappa NaN."""
        # Case 1: Other has low kappa (< 0.40), Surprise has high kappa (>= 0.40) and max pole = 30/50
        # At 30/50, Wilson 95% CI is [0.461845, 0.723880], which contains both 0.60 and 0.50.
        other_nhat = [0] * 15 + [1] * 30 + [2] * 5
        other_long = [1] * 15 + [2] * 15 + [0] * 15 + [1] * 5
        surp_nhat = [0] * 10 + [1] * 10 + [2] * 30
        surp_long = [0] * 10 + [1] * 10 + [2] * 30

        p_nhat1, p_long1, p_key1 = _write_synthetic_stage_a_files(
            tmp_path / "in_warn1", other_nhat, other_long, surp_nhat, surp_long
        )
        res1 = run_polarity_kappa_stage_a(
            p_nhat1, p_long1, p_key1, tmp_path / "out_warn1", n_boot=100, seed=42
        )
        w_text1 = "\n".join(res1["warnings"])
        assert "chứa 0.60" in w_text1
        assert "chứa 0.50" in w_text1
        assert "hai κ riêng nằm hai phía của 0.400000" in w_text1

        # Case 2: Other has single label (NEU=1 for all 50) -> kappa NaN on 'other'
        other_single = [1] * 50
        p_nhat2, p_long2, p_key2 = _write_synthetic_stage_a_files(
            tmp_path / "in_warn2", other_single, other_single, surp_nhat, surp_long
        )
        res2 = run_polarity_kappa_stage_a(
            p_nhat2, p_long2, p_key2, tmp_path / "out_warn2", n_boot=100, seed=42
        )
        w_text2 = "\n".join(res2["warnings"])
        assert "κ NaN" in w_text2

    def test_byte_determinism_formatting_and_no_timestamps_or_abs_paths(
        self, tmp_path: Path
    ) -> None:
        """11. Tính xác định: chạy hai lần giống hệt từng byte; không timestamp/path tuyệt đối; 6 chữ số; JSON chuẩn."""
        other_nhat = [0] * 15 + [1] * 25 + [2] * 10
        other_long = [0] * 12 + [1] * 28 + [2] * 10
        surp_nhat = [0] * 10 + [1] * 15 + [2] * 25
        surp_long = [0] * 10 + [1] * 12 + [2] * 28

        p_nhat, p_long, p_key = _write_synthetic_stage_a_files(
            tmp_path / "in", other_nhat, other_long, surp_nhat, surp_long
        )
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        run_polarity_kappa_stage_a(p_nhat, p_long, p_key, out1, n_boot=200, seed=42)
        run_polarity_kappa_stage_a(p_nhat, p_long, p_key, out2, n_boot=200, seed=42)

        csv_files = [
            "confusion_all.csv",
            "confusion_other.csv",
            "confusion_surprise.csv",
            "polarity_distribution.csv",
            "polarity_disagreements.csv",
        ]
        all_files = [
            "kappa_report.json",
            "kappa_report.md",
            *csv_files,
            "polarity_files_hash.json",
        ]
        for fname in all_files:
            b1 = (out1 / fname).read_bytes()
            b2 = (out2 / fname).read_bytes()
            assert b1 == b2
            assert not b1.startswith(b"\xef\xbb\xbf")  # UTF-8 without BOM
            text = (out1 / fname).read_text(encoding="utf-8")
            assert text.endswith("\n") and not text.endswith("\n\n")
            # No absolute path or ISO timestamp
            assert str(tmp_path) not in text
            assert str(_REPO_ROOT) not in text
            assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text)

        # CSV files specifically use LF (\n) line endings
        for cfile in csv_files:
            assert b"\r\n" not in (out1 / cfile).read_bytes()

        # Check JSON formatting: sort_keys=True, indent=2, single trailing newline
        for jname in ("kappa_report.json", "polarity_files_hash.json"):
            raw_json = (out1 / jname).read_text(encoding="utf-8")
            parsed = json.loads(raw_json)
            assert raw_json == json.dumps(parsed, indent=2, sort_keys=True) + "\n"

        # Check 6-decimal formatting in polarity_distribution.csv
        df_dist = pd.read_csv(out1 / "polarity_distribution.csv", dtype=str)
        for col in ("proportion", "wilson_ci_low", "wilson_ci_high"):
            for val in df_dist[col]:
                assert re.match(r"^-?\d+\.\d{6}$", val), f"Expected 6 decimals in {col}, got {val!r}"

    def test_shuffled_input_row_order_produces_identical_results(self, tmp_path: Path) -> None:
        """12. Xáo trộn thứ tự dòng đầu vào thì kết quả không đổi."""
        other_nhat = [0] * 15 + [1] * 25 + [2] * 10
        other_long = [0] * 12 + [1] * 28 + [2] * 10
        surp_nhat = [0] * 10 + [1] * 15 + [2] * 25
        surp_long = [0] * 10 + [1] * 12 + [2] * 28

        p_nhat1, p_long1, p_key1 = _write_synthetic_stage_a_files(
            tmp_path / "ordered",
            other_nhat,
            other_long,
            surp_nhat,
            surp_long,
            shuffle_nhat=False,
            shuffle_long=False,
            shuffle_key=False,
        )
        p_nhat2, p_long2, p_key2 = _write_synthetic_stage_a_files(
            tmp_path / "shuffled",
            other_nhat,
            other_long,
            surp_nhat,
            surp_long,
            shuffle_nhat=True,
            shuffle_long=True,
            shuffle_key=True,
            seed=123,
        )

        out1 = tmp_path / "out_ordered"
        out2 = tmp_path / "out_shuffled"
        run_polarity_kappa_stage_a(p_nhat1, p_long1, p_key1, out1, n_boot=200, seed=42)
        run_polarity_kappa_stage_a(p_nhat2, p_long2, p_key2, out2, n_boot=200, seed=42)

        content_files = [
            "kappa_report.json",
            "kappa_report.md",
            "confusion_all.csv",
            "confusion_other.csv",
            "confusion_surprise.csv",
            "polarity_distribution.csv",
            "polarity_disagreements.csv",
        ]
        for fname in content_files:
            assert (out1 / fname).read_bytes() == (out2 / fname).read_bytes()
