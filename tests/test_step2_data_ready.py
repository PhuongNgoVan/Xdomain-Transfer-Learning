from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from scripts import check_step2_data as cs2
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
    load_step1_results,
    load_step2_config,
    run_all_step2_checks,
    strong_normalize_text,
    validate_vsmec_strict_file,
)
from scripts.make_strict_template import build_strict_template_df, generate_strict_template
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
from src.common.schema import PRED7_COLUMNS, PRED_COLUMNS, RESULT_COLUMNS, STAGE1_COLUMNS
from src.common.textnorm import normalize_text
from src.data.preprocess import preprocess_text

NEU3_COLUMNS: tuple[str, ...] = (
    "run_id",
    "backbone",
    "stage1",
    "stage2",
    "eval_domain",
    "label_view",
    "mode",
    "fraction",
    "seed",
    "split",
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "f1_neg",
    "f1_neu",
    "f1_pos",
    "f1_per_class",
    "ckpt_path",
    "run_kind",
    "variant",
    "lr",
    "best_step",
    "n_train",
    "precision",
    "gpu",
    "code_version",
    "data_hash",
    "config_path",
    "truncation_rate",
)


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


def _build_synthetic_full_results(
    clean_dir: Path,
    neuesc_file: Path,
    tmp_dir: Path,
    vsmec_hash_override: str | None = None,
    neuesc_hash_override: str | None = None,
    code_version: str = "abc1234",
    f1_surprise_val: float = 0.45,
) -> tuple[list[Path], dict[str, str]]:
    """Build synthetic Stage 1 + NEU-3 CSVs covering mixed schemas, sweep rows, and duplicates."""
    v_hash = vsmec_hash_override or compute_vsmec_data_hash(clean_dir)
    n_hash = neuesc_hash_override or compute_neuesc_data_hash(neuesc_file)

    # 1. VSMEC PhoBERT using RESULT_COLUMNS schema (has sweep row at seed 0 + duplicate row)
    res_rows = []
    for s in range(5):
        for sp in ("dev", "test"):
            row = {c: "" for c in RESULT_COLUMNS}
            row.update(
                {
                    "run_id": f"PhoBERT_Stage1-base_s{s}_lr2e-5",
                    "model": "vinai/phobert-base-v2",
                    "variant": "base",
                    "dataset": "vsmec",
                    "split": sp,
                    "inference_mode": "plain",
                    "fraction": 100,
                    "seed": s,
                    "lr": 2e-5,
                    "tau": 1.0,
                    "loss": "ce",
                    "precision": "fp16",
                    "max_length": 256,
                    "best_step": 100 + s * 10,
                    "n_train": 5542,
                    "prior_neg": 0.49,
                    "prior_neu": 0.23,
                    "prior_pos": 0.28,
                    "macro_f1": (0.61 if sp == "dev" else 0.62) + 0.003 * s,
                    "accuracy": 0.65,
                    "weighted_f1": 0.63,
                    "balanced_acc": 0.61,
                    "f1_neg": 0.66,
                    "f1_neu": 0.48,
                    "f1_pos": 0.72,
                    "ece": 0.03,
                    "gpu": "T4",
                    "code_version": code_version,
                    "data_hash": v_hash,
                    "config_path": "configs/exp/stage1_phobert.yaml",
                    "official": False,
                }
            )
            res_rows.append(row)
    # Seed 0 LR sweep row (dev only) -> MUST be filtered out
    sweep_row = dict(res_rows[0])
    sweep_row["run_id"] = "PhoBERT_Stage1-sweep1e-5_s0_lr1e-5"
    sweep_row["variant"] = "sweep_lr1e-5"
    sweep_row["split"] = "dev"
    res_rows.append(sweep_row)
    # Duplicate (run_id, split) row -> MUST be deduplicated
    res_rows.append(dict(res_rows[1]))
    f_v_pho = tmp_dir / "stage1_phobert_result_schema.csv"
    pd.DataFrame(res_rows).to_csv(f_v_pho, index=False)

    # 2. VSMEC ViSoBERT using STAGE1_COLUMNS + split column (and 7 per-emotion F1s)
    s1_rows = []
    for s in range(5):
        for sp in ("dev", "test"):
            row = {c: "" for c in STAGE1_COLUMNS}
            row.update(
                {
                    "run_id": f"ViSoBERT_Stage1-base_s{s}_lr2e-5",
                    "backbone": "uitnlp/visobert",
                    "seed": s,
                    "lr": 2e-5,
                    "best_step": 120 + s * 5,
                    "macro_f1": (0.63 if sp == "dev" else 0.64) + 0.002 * s,
                    "accuracy": 0.67,
                    "weighted_f1": 0.65,
                    "f1_enjoyment": 0.75,
                    "f1_sadness": 0.66,
                    "f1_fear": 0.58,
                    "f1_anger": 0.55,
                    "f1_disgust": 0.61,
                    "f1_surprise": f1_surprise_val,
                    "f1_other": 0.54,
                    "precision": "bf16",
                    "gpu": "T4",
                    "code_version": code_version,
                    "data_hash": v_hash,
                    "config_path": "configs/exp/stage1_visobert.yaml",
                    "split": sp,
                    "variant": "base",
                }
            )
            s1_rows.append(row)
    f_v_viso = tmp_dir / "stage1_visobert.csv"
    pd.DataFrame(s1_rows).to_csv(f_v_viso, index=False)

    # 3. NEU-3 PhoBERT + ViSoBERT using NEU3_COLUMNS schema (with sweep row at seed 0)
    neu_files = []
    for alias, bb, base_f1 in (
        ("PhoBERT", "vinai/phobert-base-v2", 0.68),
        ("ViSoBERT", "uitnlp/visobert", 0.70),
    ):
        n_rows = []
        for s in range(5):
            for sp in ("dev", "test"):
                row = {c: "" for c in NEU3_COLUMNS}
                row.update(
                    {
                        "run_id": f"{alias}_NEU3-base_s{s}_lr2e-5",
                        "backbone": bb,
                        "stage1": "NEU",
                        "stage2": "",
                        "eval_domain": "NEU",
                        "label_view": "V3-NEU",
                        "mode": "base",
                        "fraction": 100,
                        "seed": s,
                        "split": sp,
                        "macro_f1": base_f1 + 0.002 * s,
                        "weighted_f1": base_f1 + 0.01,
                        "accuracy": 0.75,
                        "f1_neg": 0.69,
                        "f1_neu": 0.80,
                        "f1_pos": 0.62,
                        "f1_per_class": json.dumps([0.69, 0.80, 0.62]),
                        "ckpt_path": f"checkpoints/neu3/{alias}",
                        "run_kind": "base",
                        "variant": "base-bf16",
                        "lr": 2e-5,
                        "best_step": 400 + s * 10,
                        "n_train": 23048,
                        "precision": "bf16",
                        "gpu": "T4",
                        "code_version": code_version,
                        "data_hash": n_hash,
                        "config_path": f"configs/exp/neu3_{alias.lower()}.yaml",
                        "truncation_rate": 0.01,
                    }
                )
                n_rows.append(row)
        sweep_n = dict(n_rows[0])
        sweep_n["run_id"] = f"{alias}_NEU3-sweep_s0_lr3e-5"
        sweep_n["run_kind"] = "sweep"
        sweep_n["variant"] = "sweep-lr3e-5"
        sweep_n["split"] = "dev"
        n_rows.append(sweep_n)
        fp = tmp_dir / f"neu3_{alias.lower()}.csv"
        pd.DataFrame(n_rows).to_csv(fp, index=False)
        neu_files.append(fp)

    return [f_v_pho, f_v_viso, *neu_files], {"vsmec": v_hash, "neuesc": n_hash}


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
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
) -> None:
    _, neu_df = real_data
    results = check_group_b(neu_df)
    by_id = {r.id: r for r in results}

    assert by_id["B1"].status == "PASS"
    assert by_id["B2"].status == "PASS"
    assert by_id["B3"].status == "PASS"

    assert dict(HARD_MERGE_RULE) == {0: 2, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1}
    assert {k: tuple(v) for k, v in HARD_MERGE_GROUPS.items()} == {
        2: (0,),
        0: (1, 2, 3, 4),
        1: (5, 6),
    }
    assert dict(NEUESC_ORIG_TO_LABEL3) == {0: 1, 1: 2, 2: 0, 3: 0}

    rng = np.random.default_rng(123)
    p7 = rng.dirichlet(np.ones(7), size=100)
    p3 = aggregate_probs_7to3(p7)
    assert p3.shape == (100, 3)
    assert np.allclose(p3.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((p3 >= -1e-7) & (p3 <= 1.0 + 1e-7))

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
# Nhóm D — Rò rỉ chéo miền & sensitivity_exclude_ids.csv (Việc 4)
# ==============================================================================


def test_D1_D2_D3_D4_cross_domain_overlap_and_sensitivity_csv(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
    tmp_path: Path,
) -> None:
    vsmec_splits, neu_df = real_data
    results = check_group_d(vsmec_splits, neu_df, reports_dir=tmp_path)
    by_id = {r.id: r for r in results}

    assert by_id["D1"].level == "BLOCKER" and by_id["D1"].status == "PASS"
    assert by_id["D2"].level == "BLOCKER" and by_id["D2"].status == "PASS"
    assert by_id["D3"].level == "INFO" and by_id["D3"].status == "PASS"

    mat_3x3 = by_id["D3"].metrics["d3_matrix_3x3"]
    for vs in ("vsmec_train", "vsmec_dev", "vsmec_test"):
        for ns in ("neuesc_train", "neuesc_dev", "neuesc_test"):
            assert mat_3x3[vs][ns] == 0

    assert by_id["D4"].level == "WARN"
    assert by_id["D4"].status == "FAIL"
    d4_csv = tmp_path / "violations_D4_near_duplicates.csv"
    assert d4_csv.is_file()
    d4_df = pd.read_csv(d4_csv)
    assert len(d4_df) == 2
    assert bool(d4_df["label_3class_match"].all())
    assert set(d4_df["strong_norm"]) == {
        strong_normalize_text("chạm đáy nỗi đau 😞"),
        strong_normalize_text("vật chất quyết định ý thức ."),
    }

    sens_csv = tmp_path / "sensitivity_exclude_ids.csv"
    assert sens_csv.is_file()
    sens_df = pd.read_csv(sens_csv)
    assert list(sens_df.columns) == ["id", "dataset", "reason"]
    pairs = set(zip(sens_df["dataset"].astype(str), sens_df["id"].astype(int)))
    assert pairs == {("vsmec_test", 6311), ("neuesc_test", 28482)}

    impact = by_id["D4"].metrics["impact"]
    assert impact["vsmec_test_affected"] == 1 and impact["vsmec_test_total"] == 692
    assert impact["neuesc_test_affected"] == 1 and impact["neuesc_test_total"] == 6613


# ==============================================================================
# Nhóm E — Tập đánh giá phụ VSMEC-strict & template script (Việc 3)
# ==============================================================================


def test_E1_E2_E3_vsmec_strict_missing_and_synthetic_cases(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
    cfg: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vsmec_splits, _ = real_data
    vsmec_test = vsmec_splits["test"].copy()

    # 1. Config thực tế null -> MISSING
    res_real = {r.id: r for r in check_group_e(cfg, vsmec_splits)}
    assert res_real["E1"].status == "MISSING"
    assert res_real["E2"].status == "MISSING"
    assert res_real["E3"].status == "MISSING"

    # 2. Hợp lệ với kappa >= 0.60 và có rule
    strict_df = vsmec_test[vsmec_test["label"].isin([5, 6])][["id", "text"]].copy()
    strict_df["annotator_1"] = [i % 3 for i in range(len(strict_df))]
    strict_df["annotator_2"] = list(strict_df["annotator_1"])
    strict_path = tmp_path / "strict_good.csv"
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
    assert res_pass["E2"].metrics["kappa"] == pytest.approx(1.0)
    assert res_pass["E2"].metrics["kappa_warn"] is False
    assert res_pass["E3"].status == "PASS"

    # 3. Kappa < 0.60 -> WARN trên E2
    low_k_df = strict_df.copy()
    low_k_df["annotator_2"] = [(i + 1) % 3 for i in range(len(low_k_df))]
    low_k_path = tmp_path / "strict_low_kappa.csv"
    low_k_df.to_csv(low_k_path, index=False, encoding="utf-8")
    res_lowk = {
        r.id: r for r in validate_vsmec_strict_file(low_k_path, vsmec_test, has_rule=True)
    }
    assert res_lowk["E2"].status == "PASS"
    assert res_lowk["E2"].metrics["kappa"] < 0.60
    assert res_lowk["E2"].metrics["kappa_warn"] is True
    assert "WARN" in res_lowk["E2"].detail

    # 4. ID ngoài vsmec_test & chỉ có 1 annotator -> FAIL
    bad_df = strict_df[["id", "text", "annotator_1"]].copy()
    bad_df.iloc[0, bad_df.columns.get_loc("id")] = 99999999
    bad_path = tmp_path / "strict_bad.csv"
    bad_df.to_csv(bad_path, index=False, encoding="utf-8")
    res_bad = {
        r.id: r for r in validate_vsmec_strict_file(bad_path, vsmec_test, has_rule=False)
    }
    assert res_bad["E1"].status == "FAIL"
    assert res_bad["E2"].status == "FAIL"
    assert res_bad["E3"].status == "MISSING"


def test_make_strict_template_dry_run_and_generate(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
    tmp_path: Path,
) -> None:
    vsmec_splits, _ = real_data
    vsmec_test = vsmec_splits["test"]

    out_csv = tmp_path / "strict_template.csv"
    dry_res = generate_strict_template(
        vsmec_test_path=Path("data/clean/vsmec_test.csv"),
        output_path=out_csv,
        n_samples=100,
        seed=42,
        generate=False,
    )
    assert dry_res is None
    assert not out_csv.exists()

    gen_res = generate_strict_template(
        vsmec_test_path=Path("data/clean/vsmec_test.csv"),
        output_path=out_csv,
        n_samples=100,
        seed=42,
        generate=True,
    )
    assert gen_res is not None
    assert out_csv.is_file()
    df_out = pd.read_csv(out_csv, keep_default_na=False)
    assert list(df_out.columns) == ["id", "text", "annotator_1", "annotator_2"]
    assert len(df_out) == 100
    assert (df_out["annotator_1"] == "").all()
    assert (df_out["annotator_2"] == "").all()

    so_ids = set(vsmec_test[vsmec_test["label"].isin([5, 6])]["id"].astype(int).tolist())
    assert set(df_out["id"].astype(int).tolist()).issubset(so_ids)

    df_out_2 = build_strict_template_df(vsmec_test, n_samples=100, seed=42)
    assert df_out_2["id"].tolist() == df_out["id"].tolist()


# ==============================================================================
# Nhóm F & G — step2_files, F2b, Loader 2 schema, F1/F3/G1/G2/G3 (Việc 1 & 2)
# ==============================================================================


def test_F2_step2_files_and_F2b_legacy_artifacts(cfg: dict) -> None:
    clean_dir = Path(cfg["vsmec_clean_dir"])
    neuesc_file = Path(cfg["neuesc_file"])
    empty_loaded = {"files": [], "df": pd.DataFrame(), "n_raw_rows": 0, "n_sweep_filtered": 0, "n_duplicates_dropped": 0}

    res_real = {
        r.id: r for r in check_group_f(cfg, clean_dir, neuesc_file, empty_loaded)
    }

    # Trên dữ liệu thật: F2 đọc từ config step2_files và PASS; F2b là WARN ghi artifact cũ
    assert res_real["F2"].level == "BLOCKER" and res_real["F2"].status == "PASS"
    assert res_real["F2b"].level == "WARN" and res_real["F2b"].status == "FAIL"

    art_files = {r["file"]: r for r in res_real["F2b"].metrics["artifacts"]}
    assert {"polarity_long.csv", "polarity_nhat.csv", "polarity_key.csv"}.issubset(
        set(art_files.keys())
    )
    assert art_files["polarity_key.csv"]["manifest_sha256"] == "không có"
    assert art_files["polarity_long.csv"]["actual_sha256"] != art_files["polarity_long.csv"]["manifest_sha256"]

    assert res_real["F1"].metrics["current_vsmec_data_hash"] == compute_vsmec_data_hash(clean_dir)
    assert res_real["F1"].metrics["current_neuesc_data_hash"] == sha256_file(neuesc_file)
    assert res_real["F1"].status == "MISSING"
    assert res_real["F3"].status == "MISSING"

    # Kiểm chứng F2 thực sự dùng khóa step2_files trong config (không viết cứng)
    cfg_with_polarity = dict(cfg)
    cfg_with_polarity["step2_files"] = [*cfg["step2_files"], "polarity_long.csv"]
    res_fail = {
        r.id: r for r in check_group_f(cfg_with_polarity, clean_dir, neuesc_file, empty_loaded)
    }
    assert res_fail["F2"].status == "FAIL"
    assert "polarity_long.csv" in res_fail["F2"].detail


def test_step1_loader_and_gates_full_pass_synthetic(cfg: dict, tmp_path: Path) -> None:
    clean_dir = Path(cfg["vsmec_clean_dir"])
    neuesc_file = Path(cfg["neuesc_file"])
    files, _ = _build_synthetic_full_results(clean_dir, neuesc_file, tmp_path)

    loaded = load_step1_results(results_files=files)
    assert loaded["n_sweep_filtered"] == 3  # 1 VSMEC PhoBERT + 2 NEU-3 sweep rows filtered
    assert loaded["n_duplicates_dropped"] == 1  # duplicate (run_id, split) removed
    assert len(loaded["df"]) == 40  # 4 groups * 5 seeds * 2 splits (dev & test)

    res_f = {r.id: r for r in check_group_f(cfg, clean_dir, neuesc_file, loaded)}
    assert res_f["F1"].status == "PASS"
    assert res_f["F2"].status == "PASS"
    assert res_f["F3"].status == "PASS"

    cfg_pub = dict(cfg)
    cfg_pub["vsmec_published_macro_f1"] = {"phobert": 0.62, "visobert": 0.63}
    res_g = {r.id: r for r in check_group_g(cfg_pub, loaded)}
    assert res_g["G1"].status == "PASS"
    assert "5/5" in res_g["G1"].detail
    assert res_g["G2"].status == "PASS"
    assert res_g["G2"].metrics["phobert"]["diff"] == pytest.approx(0.626 - 0.62, abs=1e-6)
    assert res_g["G3"].status == "PASS"


def test_step1_gates_failure_modes_synthetic(cfg: dict, tmp_path: Path) -> None:
    clean_dir = Path(cfg["vsmec_clean_dir"])
    neuesc_file = Path(cfg["neuesc_file"])

    # 1. Thiếu seed (NEU-3 ViSoBERT thiếu seed 3, 4; một seed khác chỉ có dev)
    dir_miss = tmp_path / "miss"
    dir_miss.mkdir()
    files_m, _ = _build_synthetic_full_results(clean_dir, neuesc_file, dir_miss)
    df_n_viso = pd.read_csv(files_m[3])
    # Remove seed 3, 4 completely and remove test row of seed 2 -> only seeds 0, 1 complete (2/5)
    df_n_viso = df_n_viso[
        ~df_n_viso["seed"].isin([3, 4])
        & ~((df_n_viso["seed"] == 2) & (df_n_viso["split"] == "test"))
    ]
    df_n_viso.to_csv(files_m[3], index=False)
    loaded_m = load_step1_results(results_files=files_m)
    res_gm = {r.id: r for r in check_group_g(cfg, loaded_m)}
    assert res_gm["G1"].status == "FAIL"
    assert "NEU-3 ViSoBERT: 2/5" in res_gm["G1"].detail
    assert "NEU-3 ViSoBERT" in res_gm["G1"].detail and "[2, 3, 4]" in res_gm["G1"].detail

    # 2. Lệch data_hash -> F1 FAIL
    dir_badh = tmp_path / "badh"
    dir_badh.mkdir()
    files_h, _ = _build_synthetic_full_results(
        clean_dir, neuesc_file, dir_badh, vsmec_hash_override="deadbeef" * 8
    )
    loaded_h = load_step1_results(results_files=files_h)
    res_fh = {r.id: r for r in check_group_f(cfg, clean_dir, neuesc_file, loaded_h)}
    assert res_fh["F1"].status == "FAIL"

    # 3. Nhiều code_version trong 1 nhóm -> F3 FAIL
    dir_multicv = tmp_path / "multicv"
    dir_multicv.mkdir()
    files_cv, _ = _build_synthetic_full_results(clean_dir, neuesc_file, dir_multicv)
    df_s1v = pd.read_csv(files_cv[1])
    df_s1v.loc[df_s1v["seed"] == 4, "code_version"] = "zzz9999"
    df_s1v.to_csv(files_cv[1], index=False)
    loaded_cv = load_step1_results(results_files=files_cv)
    res_fcv = {r.id: r for r in check_group_f(cfg, clean_dir, neuesc_file, loaded_cv)}
    assert res_fcv["F3"].status == "FAIL"

    # 4. Lớp có F1 TB < 0.30 -> G3 (WARN) FAIL
    dir_lowf1 = tmp_path / "lowf1"
    dir_lowf1.mkdir()
    files_lf, _ = _build_synthetic_full_results(
        clean_dir, neuesc_file, dir_lowf1, f1_surprise_val=0.22
    )
    loaded_lf = load_step1_results(results_files=files_lf)
    res_glf = {r.id: r for r in check_group_g(cfg, loaded_lf)}
    assert res_glf["G3"].level == "WARN" and res_glf["G3"].status == "FAIL"
    assert "f1_surprise" in res_glf["G3"].detail


# ==============================================================================
# Nhóm H — Phân bố lớp, JS divergence và Lệch bề mặt H4 (Việc 5)
# ==============================================================================


def test_H1_H2_H3_H4_distributions_and_surface_stats(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
) -> None:
    vsmec_splits, neu_df = real_data
    results = check_group_h(vsmec_splits, neu_df)
    by_id = {r.id: r for r in results}

    assert by_id["H1"].status == "PASS"
    assert by_id["H2"].status == "PASS"
    assert by_id["H3"].status == "PASS"
    assert by_id["H4"].level == "INFO" and by_id["H4"].status == "PASS"

    assert by_id["H2"].metrics["vsmec_test_7c_counts"]["Surprise"] == 37
    assert by_id["H2"].metrics["small_classes_lt_30"] == []
    assert by_id["H3"].metrics["surprise_count"] == 37
    assert by_id["H3"].metrics["other_count"] == 129
    assert by_id["H3"].metrics["neutral_b1_total"] == 166
    assert 0.0 < by_id["H2"].metrics["js_divergence_3class"]["train"]["js_bits"] < 1.0

    surf = by_id["H4"].metrics["surface_stats"]
    for dom in ("vsmec", "neuesc"):
        for sp in ("train", "dev", "test"):
            st = surf[dom][sp]
            for key in (
                "ends_with_space_dot_rate",
                "ends_with_dot_rate",
                "has_emoji_rate",
                "word_count_mean",
                "word_count_p95",
                "all_lowercase_rate",
            ):
                assert key in st
    # Kiểm chứng thực tế: NEU-ESC có ' .' gần như toàn bộ, VSMEC giữ emoji Unicode còn NEU-ESC = 0
    assert surf["neuesc"]["test"]["ends_with_space_dot_rate"] > 0.95
    assert surf["vsmec"]["test"]["has_emoji_rate"] > 0.15
    assert surf["neuesc"]["test"]["has_emoji_rate"] == 0.0


# ==============================================================================
# Nhóm I — Tương thích đầu vào theo từng backbone (Việc 6)
# ==============================================================================


def test_I1_I2_I3_I4_input_compatibility(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vsmec_splits, neu_df = real_data
    with_tok = bool(request.config.getoption("--with-tokenizer"))
    results = check_group_i(vsmec_splits, neu_df, with_tokenizer=with_tok)
    by_id = {r.id: r for r in results}

    assert by_id["I1"].status == "PASS"
    assert by_id["I3"].status == "PASS"
    assert by_id["I4"].status == "PASS"
    if with_tok:
        assert by_id["I2"].status in ("PASS", "SKIP")
    else:
        assert by_id["I2"].status == "SKIP"

    # Nếu không tải được tokenizer -> I2 phải trả về SKIP kèm lý do (không FAIL)
    def _fail_tok(_name: str):
        raise OSError("offline mock")

    monkeypatch.setattr(cs2, "_load_tokenizer_safe", _fail_tok)
    res_offline = {r.id: r for r in check_group_i(vsmec_splits, neu_df, with_tokenizer=True)}
    assert res_offline["I2"].status == "SKIP"
    assert "offline mock" in res_offline["I2"].detail

    for s in ["Chạm đáy nỗi đau 😞😞", "vật chất quyết định ý thức .", "Đẹp quáaaaa 😍"]:
        norm1 = normalize_text(s)
        assert normalize_text(norm1) == norm1
        viso1 = preprocess_text(s, emoji_mode="keep")
        assert preprocess_text(viso1, emoji_mode="keep") == viso1


# ==============================================================================
# Nhóm J — Checkpoint & --reload-check (Việc 6)
# ==============================================================================


def _create_tiny_hf_checkpoint(
    ckpt_dir: Path,
    vocab_words: list[str],
    num_labels: int,
) -> tuple[BertModel, BertTokenizerFast, torch.Tensor, torch.Tensor]:
    vocab_file = ckpt_dir / "encoder" / "vocab.txt"
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *vocab_words]
    seen = set()
    uniq_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq_tokens.append(t)
    vocab_file.write_text("\n".join(uniq_tokens) + "\n", encoding="utf-8")

    tok = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    tok.save_pretrained(ckpt_dir / "encoder")

    bcfg = BertConfig(
        vocab_size=len(uniq_tokens),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=512,
    )
    model = BertModel(bcfg)
    model.eval()
    model.save_pretrained(ckpt_dir / "encoder")

    torch.manual_seed(0)
    weight = torch.randn((num_labels, 16), dtype=torch.float32)
    bias = torch.randn((num_labels,), dtype=torch.float32)
    torch.save({"weight": weight, "bias": bias}, ckpt_dir / "head.pt")
    return model, tok, weight, bias


def test_J1_J2_J3_checkpoint_and_reload_check(
    real_data: tuple[dict[str, pd.DataFrame], pd.DataFrame],
    cfg: dict,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    vsmec_splits, neu_df = real_data
    results_dir = Path(cfg.get("results_dir", "results"))
    check_ckpt = bool(request.config.getoption("--check-ckpt"))
    res_real = {r.id: r for r in check_group_j(cfg, results_dir, check_ckpt=check_ckpt)}
    if not check_ckpt:
        assert res_real["J1"].status == "SKIP"
        assert res_real["J2"].status == "SKIP"
        assert res_real["J3"].status == "SKIP"

    # Tạo 4 nhóm × 5 seed với tiny HuggingFace encoder/head.pt & saved dev prediction CSVs
    ckpt_roots = {}
    pred7_dir = tmp_path / "predictions7"
    pred3_dir = tmp_path / "predictions3"
    pred7_dir.mkdir()
    pred3_dir.mkdir()
    hash_payload = {}

    v_dev16 = vsmec_splits["dev"].iloc[:16].copy()
    n_dev16 = neu_df[neu_df["subset"] == "dev"].iloc[:16].copy()

    specs = [
        ("vsmec_phobert", "vsmec", "PhoBERT", 7, v_dev16, pred7_dir),
        ("vsmec_visobert", "vsmec", "ViSoBERT", 7, v_dev16, pred7_dir),
        ("neu3_phobert", "neuesc", "PhoBERT", 3, n_dev16, pred3_dir),
        ("neu3_visobert", "neuesc", "ViSoBERT", 3, n_dev16, pred3_dir),
    ]
    for gname, dom, alias, n_labels, dev16, pdir in specs:
        gdir = tmp_path / "ckpts" / gname
        ckpt_roots[gname] = str(gdir)
        for s in range(5):
            sdir = gdir / f"seed_{s}" / alias
            if s == 0:
                model, tok, weight, bias = _create_tiny_hf_checkpoint(
                    sdir, ["vui", "buồn", "bình", "thường"], n_labels
                )
                texts = (
                    dev16["text_seg"].astype(str).tolist()
                    if alias == "PhoBERT"
                    else [preprocess_text(str(t), emoji_mode="keep") for t in dev16["text"].tolist()]
                )
                enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    logits = (model(**enc).last_hidden_state[:, 0, :] @ weight.T + bias).numpy()
                lcols = [f"logit_{i}" for i in range(7)] if dom == "vsmec" else ["logit_neg", "logit_neu", "logit_pos"]
                p_df = pd.DataFrame(logits, columns=lcols)
                p_df.insert(0, "id", dev16["id"].astype(int).tolist())
                split_tag = "vsmec_dev" if dom == "vsmec" else "neuesc_dev"
                p_df.to_csv(pdir / f"{alias}_s0_{split_tag}.csv", index=False)
            else:
                (sdir / "encoder").mkdir(parents=True, exist_ok=True)
                torch.save({"weight": torch.zeros((n_labels, 16)), "bias": torch.zeros(n_labels)}, sdir / "head.pt")
            hash_payload[f"{gname}_s{s}"] = {
                "path": str(sdir),
                "head_sha256": sha256_file(sdir / "head.pt"),
            }

    (tmp_path / "stage1_ckpt_hash.json").write_text(json.dumps(hash_payload), encoding="utf-8")
    cfg_ckpt = {"ckpt_roots": ckpt_roots, "seeds": [0, 1, 2, 3, 4], "results_inputs_dir": str(tmp_path)}
    res_syn = {
        r.id: r
        for r in check_group_j(
            cfg_ckpt,
            tmp_path,
            check_ckpt=True,
            reload_check=True,
            vsmec_dfs=vsmec_splits,
            neuesc_df=neu_df,
        )
    }
    assert res_syn["J1"].status == "PASS"
    assert res_syn["J2"].status == "PASS"
    assert res_syn["J3"].status == "PASS"

    # Làm lệch saved logit vượt quá 2e-3 -> J3 (reload-check) phải FAIL
    bad_pred = pred7_dir / "PhoBERT_s0_vsmec_dev.csv"
    bad_df = pd.read_csv(bad_pred)
    bad_df["logit_0"] = bad_df["logit_0"] + 0.05
    bad_df.to_csv(bad_pred, index=False)
    res_bad_reload = {
        r.id: r
        for r in check_group_j(
            cfg_ckpt,
            tmp_path,
            check_ckpt=True,
            reload_check=True,
            vsmec_dfs=vsmec_splits,
            neuesc_df=neu_df,
        )
    }
    assert res_bad_reload["J3"].status == "FAIL"


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
    # F2 đã PASS nhờ thu hẹp phạm vi step2_files trong config
    by_id = {c["id"]: c for c in payload["checks"]}
    assert by_id["F2"]["status"] == "PASS"
    assert by_id["F2b"]["level"] == "WARN"
    assert by_id["H4"]["level"] == "INFO" and by_id["H4"]["status"] == "PASS"
    for blk in ("E1", "E2", "E3", "F1", "F3", "G1", "G2"):
        assert blk in payload["blocking_unresolved"]
    assert "F2" not in payload["blocking_unresolved"]

    md_text = (tmp_path / "readiness_report.md").read_text(encoding="utf-8")
    assert 'Hiệu ứng này cần được xét ở phần "độ lệch nội tại" của Bước 2.' in md_text
    assert "--check-ckpt" in md_text
    assert (tmp_path / "readiness_report.json").is_file()
    assert (tmp_path / "sensitivity_exclude_ids.csv").is_file()
