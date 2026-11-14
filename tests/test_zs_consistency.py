import pandas as pd

from src.common.schema import STAGE1_COLUMNS
from src.zs.check_consistency import check_consistency


def test_vsmec_baseline_never_compares_against_neu_reference(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame([
        {"run_id": "refV", "backbone": "PhoBERT", "seed": 0, "stage1": "VSMEC", "mode": "ref", "map_method": "none", "label_view": "V7-VSMEC", "macro_f1": .7},
        {"run_id": "refN", "backbone": "PhoBERT", "seed": 0, "stage1": "NEU", "mode": "ref", "map_method": "none", "label_view": "V3-NEU", "macro_f1": .9},
    ]).to_csv(results / "zs.csv", index=False)
    row = {name: "" for name in STAGE1_COLUMNS}
    row.update(run_id="PhoBERT_Stage1-bf16_s0_lr2e-5", backbone="vinai/phobert-base-v2", seed=0, macro_f1=.7, split="test")
    baseline = tmp_path / "baseline.csv"
    pd.DataFrame([row]).to_csv(baseline, index=False)
    comparison = check_consistency(results, baseline)
    assert comparison["run_id"].tolist() == ["refV"]
    assert comparison["difference"].tolist() == [0]


def test_legacy_stage1_without_test_is_skipped(tmp_path, capsys):
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame({"run_id": ["reference"]}).to_csv(results / "zs.csv", index=False)
    baseline = tmp_path / "stage1.csv"
    pd.DataFrame({"run_id": ["baseline"], "macro_f1": [.8]}).to_csv(baseline, index=False)
    assert check_consistency(results, baseline).empty
    assert "không có dòng test, bỏ qua" in capsys.readouterr().out
