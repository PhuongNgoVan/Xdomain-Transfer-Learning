from pathlib import Path

import pandas as pd
import pytest

from scripts.check_step2_data import check_group_e, load_step2_config


def test_frozen_strict_rule_keeps_526_original_test_ids() -> None:
    cfg = load_step2_config()
    source = Path(cfg["vsmec_clean_dir"]) / "vsmec_test.csv"
    before = source.read_bytes()
    frame = pd.read_csv(source)
    checks = {r.id: r for r in check_group_e(cfg, {"test": frame})}
    assert {k: r.status for k, r in checks.items()} == {"E1": "PASS", "E2": "SKIP", "E3": "PASS"}
    assert checks["E1"].metrics["n"] == 526
    assert checks["E1"].metrics["n_excluded"] == 166
    assert checks["E1"].metrics["labels"] == [0, 2]
    assert source.read_bytes() == before


@pytest.mark.parametrize("case", ["invalid_label", "duplicate_id", "non_test", "empty_strict", "missing_columns"])
def test_rule_based_strict_rejects_bad_source(case: str) -> None:
    frame = pd.DataFrame({"id": [1, 2], "label": [0, 4], "subset": ["test", "test"]})
    if case == "invalid_label":
        frame.loc[0, "label"] = -1
    elif case == "duplicate_id":
        frame.loc[1, "id"] = 1
    elif case == "non_test":
        frame.loc[0, "subset"] = "dev"
    elif case == "empty_strict":
        frame["label"] = [5, 6]
    else:
        frame = frame.drop(columns="id")
    checks = {r.id: r for r in check_group_e({"vsmec_strict_rule": "label_le_4"}, {"test": frame})}
    assert checks["E1"].status == checks["E3"].status == "FAIL"


def test_rule_based_strict_requires_rule_document(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame({"id": [1], "label": [0], "subset": ["test"]})
    checks = {r.id: r for r in check_group_e({"vsmec_strict_rule": "label_le_4"}, {"test": frame})}
    assert checks["E1"].status == "PASS"
    assert checks["E3"].status == "MISSING"


def test_unknown_strict_rule_is_not_silently_accepted() -> None:
    assert all(r.status == "FAIL" for r in check_group_e({"vsmec_strict_rule": "typo"}, {}))
