from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.check_step2_data import (
    aggregate_probs_7to3,
    check_group_a,
    check_group_b,
    check_group_c,
    check_group_d,
    check_group_e,
    check_group_f,
    check_group_g,
    check_group_h,
    check_group_i,
    check_group_j,
    compute_neuesc_data_hash,
    compute_vsmec_data_hash,
    load_step2_config,
    run_all_step2_checks,
    strong_normalize_text,
)
from src.common.constants import (
    EMOTIONS7,
    HARD_MERGE_GROUPS,
    HARD_MERGE_RULE,
    NEG,
    NEU,
    NEUESC_ORIG_TO_LABEL3,
    POS,
)
from src.common.hashing import sha256_file
from src.common.textnorm import normalize_text
from src.data.preprocess import preprocess_text


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_step2_config("configs/step2_data.yaml")


@pytest.fixture(scope="session")
def real_data(cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    clean_dir = Path(cfg["vsmec_clean_dir"])
    neuesc_file = Path(cfg["neuesc_file"])
    results, vsmec_dfs, neuesc_df = check_group_a(clean_dir, neuesc_file)
    assert all(r.status == "PASS" for r in results)
    assert neuesc_df is not None
    return vsmec_dfs, neuesc_df


# ==============================================================================
# Nhóm A — Tồn tại tệp, lược đồ cột, và kích thước chuẩn
# ==============================================================================


def test_A1_A2_A3_A4_real_data(cfg: dict) -> None:
    clean_dir = Path(cfg["vsmec_clean_dir"])
    neuesc_file = Path(cfg["neuesc_file"])
    results, vsmec_dfs, neu_df = check_group_a(clean_dir, neuesc_file)
    by_id = {r.id: r for r in results}

    assert by_id["A1"].level == "BLOCKER" and by_id["A1"].status == "PASS"
    assert by_id["A2"].level == "BLOCKER" and by_id["A2"].status == "PASS"
    assert by_id["A3"].level == "BLOCKER" and by_id["A3"].status == "PASS"
    assert by_id["A4"].level == "BLOCKER" and by_id["A4"].status == "PASS"

    assert len(vsmec_dfs["train"]) == 5542
    assert len(vsmec_dfs["dev"]) == 686
    assert len(vsmec_dfs["test"]) == 692

    assert neu_df is not None
    assert int((neu_df["subset"] == "train").sum()) == 23048
    assert int((neu_df["subset"] == "dev").sum()) == 3305
    assert int((neu_df["subset"] == "test").sum()) == 6613

    assert tuple(EMOTIONS7) == (
        "Enjoyment",
        "Sadness",
        "Fear",
        "Anger",
        "Disgust",
        "Surprise",
        "Other",
    )
    assert (NEG, NEU, POS) == (0, 1, 2)


# ==============================================================================
# Nhóm B — Tính nhất quán của không gian nhãn và ánh xạ 7 -> 3
# ==============================================================================


def test_B1_B2_B3_label_space_and_7to3(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame]
) -> None:
    _, neu_df = real_data
    results = check_group_b(neu_df)
    by_id = {r.id: r for r in results}

    assert by_id["B1"].status == "PASS"
    assert by_id["B2"].status == "PASS"
    assert by_id["B3"].status == "PASS"

    # Kiểm tra tường minh bảng B1
    assert dict(HARD_MERGE_RULE) == {0: 2, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1}
    assert {k: tuple(v) for k, v in HARD_MERGE_GROUPS.items()} == {
        2: (0,),
        0: (1, 2, 3, 4),
        1: (5, 6),
    }
    assert dict(NEUESC_ORIG_TO_LABEL3) == {0: 1, 1: 2, 2: 0, 3: 0}

    # Kiểm tra đại số hàm gộp xác suất 7 -> 3 (B2) trên 100 vector Dirichlet ngẫu nhiên
    rng = np.random.default_rng(123)
    p7 = rng.dirichlet(np.ones(7), size=100)
    p3 = aggregate_probs_7to3(p7)
    assert p3.shape == (100, 3)
    assert np.allclose(p3.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((p3 >= -1e-7) & (p3 <= 1.0 + 1e-7))

    # Kiểm tra 7 vector one-hot e_k khớp đúng cột HARD_MERGE_RULE[k]
    eye7 = np.eye(7)
    eye3 = aggregate_probs_7to3(eye7)
    for k in range(7):
        expected = np.zeros(3)
        expected[HARD_MERGE_RULE[k]] = 1.0
        assert np.allclose(eye3[k], expected, atol=1e-7)


# ==============================================================================
# Nhóm C — Trùng lặp nội bộ và rò rỉ giữa các split trong cùng một tập
# ==============================================================================


def test_C1_C2_C3_no_internal_duplicates_or_split_leakage(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
    tmp_path: Path,
) -> None:
    vsmec_splits, neu_df = real_data
    results = check_group_c(vsmec_splits, neu_df, reports_dir=tmp_path)
    by_id = {r.id: r for r in results}

    assert by_id["C1"].status == "PASS"
    assert by_id["C2"].status == "PASS"
    assert by_id["C3"].status == "PASS"

    # Đảm bảo khi có trùng lặp giả lập trên dữ liệu giả (tmp_path), hàm bắt được FAIL
    bad_vsmec = {k: v.copy() for k, v in vsmec_splits.items()}
    dup_row = bad_vsmec["train"].iloc[[0]].copy()
    dup_row["id"] = 999999
    dup_row["label"] = (int(dup_row["label"].iloc[0]) + 1) % 7
    bad_vsmec["train"] = pd.concat([bad_vsmec["train"], dup_row], ignore_index=True)
    bad_vsmec["test"] = pd.concat([bad_vsmec["test"], dup_row], ignore_index=True)

    bad_res = {r.id: r for r in check_group_c(bad_vsmec, neu_df, reports_dir=tmp_path)}
    assert bad_res["C1"].status == "FAIL"
    assert bad_res["C2"].status == "FAIL"
    assert bad_res["C3"].status == "FAIL"


# ==============================================================================
# Nhóm D — Rò rỉ chéo miền VSMEC <-> NEU-ESC (hai chiều)
# ==============================================================================


def test_D1_D2_D3_D4_cross_domain_overlap(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
    tmp_path: Path,
) -> None:
    vsmec_splits, neu_df = real_data
    results = check_group_d(vsmec_splits, neu_df, reports_dir=tmp_path)
    by_id = {r.id: r for r in results}

    # D1, D2 là BLOCKER và phải PASS (0 câu trùng chính xác sau normalize_text)
    assert by_id["D1"].level == "BLOCKER" and by_id["D1"].status == "PASS"
    assert by_id["D2"].level == "BLOCKER" and by_id["D2"].status == "PASS"
    assert by_id["D3"].level == "INFO" and by_id["D3"].status == "PASS"

    # Ma trận đếm chéo miền 3x3 trên exact hash đều bằng 0
    mat_3x3 = by_id["D3"].metrics["d3_matrix_3x3"]
    for vs in ("vsmec_train", "vsmec_dev", "vsmec_test"):
        for ns in ("neuesc_train", "neuesc_dev", "neuesc_test"):
            assert mat_3x3[vs][ns] == 0

    # D4 là WARN: phát hiện 2 cặp near-duplicate khi bỏ dấu câu/emoji và ghi ra CSV
    assert by_id["D4"].level == "WARN"
    assert by_id["D4"].status == "FAIL"
    d4_csv = tmp_path / "violations_D4_near_duplicates.csv"
    assert d4_csv.is_file()
    d4_df = pd.read_csv(d4_csv)
    assert len(d4_df) == 2
    assert set(d4_df["strong_norm"]) == {
        strong_normalize_text("chạm đáy nỗi đau 😞"),
        strong_normalize_text("vật chất quyết định ý thức ."),
    }


# ==============================================================================
# Nhóm E — Tập đánh giá phụ VSMEC-strict
# ==============================================================================


def test_E1_E2_E3_vsmec_strict_missing_and_synthetic_pass(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
    cfg: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vsmec_splits, _ = real_data
    # 1. Trên cấu hình thực tế (vsmec_strict_file: null) -> báo cáo MISSING rõ ràng
    res_real = {r.id: r for r in check_group_e(cfg, vsmec_splits)}
    assert res_real["E1"].status == "MISSING"
    assert res_real["E2"].status == "MISSING"
    assert res_real["E3"].status == "MISSING"

    # 2. Giả lập file vsmec_strict hợp lệ và docs/vsmec_strict_rule.md trong tmp_path
    vsmec_test = vsmec_splits["test"].copy()
    strict_df = vsmec_test[vsmec_test["label"].isin([0, 1, 2, 3, 4])][["id", "text", "label"]].copy()
    strict_df["annotator_1"] = strict_df["label"].map(HARD_MERGE_RULE)
    strict_df["annotator_2"] = strict_df["label"].map(HARD_MERGE_RULE)
    strict_path = tmp_path / "vsmec_strict.csv"
    strict_df.to_csv(strict_path, index=False, encoding="utf-8")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "vsmec_strict_rule.md").write_text("# Rule\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg_pass = dict(cfg)
    cfg_pass["vsmec_strict_file"] = str(strict_path)
    res_pass = {r.id: r for r in check_group_e(cfg_pass, vsmec_splits)}
    assert res_pass["E1"].status == "PASS"
    assert res_pass["E2"].status == "PASS"
    assert res_pass["E3"].status == "PASS"


# ==============================================================================
# Nhóm F — Mã băm dữ liệu và MANIFEST
# ==============================================================================


def test_F1_F2_F3_manifest_and_hashes(cfg: dict) -> None:
    clean_dir = Path(cfg["vsmec_clean_dir"])
    neuesc_file = Path(cfg["neuesc_file"])
    results_dir = Path(cfg.get("results_dir", "results"))
    res_real = check_group_f(clean_dir, neuesc_file, results_dir)
    by_id = {r.id: r for r in res_real}

    # Kiểm tra hash dữ liệu thực tế khớp công thức Bước 1.1 & 1.2
    assert by_id["F1"].metrics["current_vsmec_data_hash"] == compute_vsmec_data_hash(clean_dir)
    assert by_id["F1"].metrics["current_neuesc_data_hash"] == compute_neuesc_data_hash(neuesc_file)
    assert compute_neuesc_data_hash(neuesc_file) == sha256_file(neuesc_file)

    # Trên nhánh p1 chưa đồng bộ results/stage1.csv & results/neu3.csv -> F1, F3 là MISSING
    assert by_id["F1"].status == "MISSING"
    assert by_id["F3"].status == "MISSING"

    # F2 phát hiện đúng việc polarity_{long,nhat}.csv lệch MANIFEST và polarity_key.csv chưa liệt kê
    assert by_id["F2"].status == "FAIL"
    assert "polarity_long.csv" in by_id["F2"].detail
    assert "polarity_nhat.csv" in by_id["F2"].detail
    assert "polarity_key.csv" in by_id["F2"].detail


# ==============================================================================
# Nhóm G — Cổng chất lượng mô hình Bước 1 trước khi sang Bước 2
# ==============================================================================


def test_G1_G2_G3_step1_gate_missing_and_synthetic_pass(
    cfg: dict, tmp_path: Path
) -> None:
    results_dir = Path(cfg.get("results_dir", "results"))
    res_real = check_group_g(cfg, results_dir)
    by_id = {r.id: r for r in res_real}
    assert by_id["G1"].status == "MISSING"
    assert by_id["G2"].status == "MISSING"
    assert by_id["G3"].status == "MISSING"

    # Kiểm tra nhánh PASS khi có đủ kết quả trong stage1.csv và neu3.csv giả lập
    stage1_rows = []
    neu3_rows = []
    for bb in ("vinai/phobert-base-v2", "uitnlp/visobert"):
        for s in range(5):
            stage1_rows.append(
                {
                    "backbone": bb,
                    "seed": s,
                    "macro_f1": 0.61 + 0.002 * s,
                    "f1_enjoyment": 0.70,
                    "f1_sadness": 0.65,
                    "f1_fear": 0.55,
                    "f1_anger": 0.50,
                    "f1_disgust": 0.58,
                    "f1_surprise": 0.45,
                    "f1_other": 0.52,
                    "data_hash": "hash_vsmec",
                    "code_version": "abc1234",
                }
            )
            for sp in ("dev", "test"):
                neu3_rows.append(
                    {
                        "backbone": bb,
                        "seed": s,
                        "split": sp,
                        "run_kind": "base",
                        "macro_f1": 0.68 + 0.001 * s,
                        "f1_neg": 0.66,
                        "f1_neu": 0.78,
                        "f1_pos": 0.60,
                        "data_hash": "hash_neu",
                        "code_version": "abc1234",
                    }
                )
    pd.DataFrame(stage1_rows).to_csv(tmp_path / "stage1.csv", index=False)
    pd.DataFrame(neu3_rows).to_csv(tmp_path / "neu3.csv", index=False)

    cfg_syn = dict(cfg)
    cfg_syn["vsmec_published_macro_f1"] = 0.60
    res_syn = check_group_g(cfg_syn, tmp_path)
    by_syn = {r.id: r for r in res_syn}
    assert by_syn["G1"].status == "PASS"
    assert by_syn["G2"].status == "PASS"
    assert by_syn["G3"].status == "PASS"


# ==============================================================================
# Nhóm H — Phân bố lớp và độ lệch tiên nghiệm (label shift)
# ==============================================================================


def test_H1_H2_H3_distributions_and_js_divergence(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame]
) -> None:
    vsmec_splits, neu_df = real_data
    results = check_group_h(vsmec_splits, neu_df)
    by_id = {r.id: r for r in results}

    assert by_id["H1"].status == "PASS"
    assert by_id["H2"].status == "PASS"
    assert by_id["H3"].status == "PASS"

    # Kiểm tra lớp nhỏ nhất trên vsmec_test 7c là Surprise = 37 (>= 30)
    assert by_id["H2"].metrics["vsmec_test_7c_counts"]["Surprise"] == 37
    assert by_id["H2"].metrics["small_classes_lt_30"] == []
    # Kiểm tra tổng Surprise (37) + Other (129) = 166 (23.99%)
    assert by_id["H3"].metrics["surprise_count"] == 37
    assert by_id["H3"].metrics["other_count"] == 129
    assert by_id["H3"].metrics["neutral_b1_total"] == 166
    assert 0.0 < by_id["H2"].metrics["js_divergence_3class"]["train"]["js_bits"] < 1.0


# ==============================================================================
# Nhóm I — Tương thích đầu vào theo từng backbone trên tập đích
# ==============================================================================


def test_I1_I2_I3_I4_input_compatibility(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
    request: pytest.FixtureRequest,
) -> None:
    vsmec_splits, neu_df = real_data
    with_tok = bool(request.config.getoption("--with-tokenizer"))
    results = check_group_i(vsmec_splits, neu_df, with_tokenizer=with_tok)
    by_id = {r.id: r for r in results}

    assert by_id["I1"].status == "PASS"
    assert by_id["I3"].status == "PASS"
    assert by_id["I4"].status == "PASS"
    if with_tok:
        assert by_id["I2"].status in ("PASS", "FAIL")
    else:
        assert by_id["I2"].status == "SKIP"

    # Kiểm chứng trực tiếp tính bất biến luỹ đẳng của normalize_text và preprocess_text(..., emoji_mode="keep")
    for s in ["Chạm đáy nỗi đau 😞😞", "vật chất quyết định ý thức .", "Đẹp quáaaaa 😍"]:
        norm1 = normalize_text(s)
        assert normalize_text(norm1) == norm1
        viso1 = preprocess_text(s, emoji_mode="keep")
        assert preprocess_text(viso1, emoji_mode="keep") == viso1


# ==============================================================================
# Nhóm J — Tính sẵn sàng của checkpoint cho suy luận zero-shot
# ==============================================================================


def test_J1_J2_J3_checkpoint_readiness(
    cfg: dict, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    results_dir = Path(cfg.get("results_dir", "results"))
    check_ckpt = bool(request.config.getoption("--check-ckpt"))
    res_real = {r.id: r for r in check_group_j(cfg, results_dir, check_ckpt=check_ckpt)}
    if not check_ckpt:
        assert res_real["J1"].status == "SKIP"
        assert res_real["J2"].status == "SKIP"
        assert res_real["J3"].status == "SKIP"
    else:
        assert res_real["J1"].status in ("PASS", "FAIL", "MISSING")

    # Kiểm thử giả lập 20 thư mục checkpoint hợp lệ trong tmp_path khi check_ckpt=True
    ckpt_roots = {}
    hash_payload = {}
    group_specs = [
        ("vsmec_phobert", "PhoBERT", 7),
        ("vsmec_visobert", "ViSoBERT", 7),
        ("neu3_phobert", "PhoBERT", 3),
        ("neu3_visobert", "ViSoBERT", 3),
    ]
    for gname, alias, n_labels in group_specs:
        gdir = tmp_path / gname
        ckpt_roots[gname] = str(gdir)
        for s in range(5):
            sdir = gdir / f"seed_{s}" / alias
            (sdir / "encoder").mkdir(parents=True, exist_ok=True)
            head_pt = sdir / "head.pt"
            torch.save({"weight": torch.zeros((n_labels, 768))}, head_pt)
            hash_payload[f"{gname}_s{s}"] = {
                "path": str(sdir),
                "head_sha256": sha256_file(head_pt),
            }

    (tmp_path / "stage1_ckpt_hash.json").write_text(
        json.dumps(hash_payload), encoding="utf-8"
    )
    cfg_ckpt = {"ckpt_roots": ckpt_roots, "seeds": [0, 1, 2, 3, 4]}
    res_syn = {r.id: r for r in check_group_j(cfg_ckpt, tmp_path, check_ckpt=True)}
    assert res_syn["J1"].status == "PASS"
    assert res_syn["J2"].status == "PASS"
    assert res_syn["J3"].status == "PASS"


# ==============================================================================
# Kiểm thử tích hợp báo cáo tổng hợp
# ==============================================================================


def test_end_to_end_report_generation(tmp_path: Path) -> None:
    payload = run_all_step2_checks(
        config_path="configs/step2_data.yaml",
        reports_dir=tmp_path,
        with_tokenizer=False,
        check_ckpt=False,
    )
    assert not payload["ready"]
    assert "E1" in payload["blocking_unresolved"]
    assert "F2" in payload["blocking_unresolved"]
    assert (tmp_path / "readiness_report.md").is_file()
    assert (tmp_path / "readiness_report.json").is_file()
