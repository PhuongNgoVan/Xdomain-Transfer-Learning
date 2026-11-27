"""Task 11 LLM unit tests: offline CPU, synthetic splits, no network/GPU."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.common.constants import EMOTIONS7, LABELS3
from src.common.hashing import sha256_file
from src.common.schema import ZS_PRED_COLUMNS
from src.phase2.llm import (
    FewShotSelector,
    build_llm_messages,
    load_llm_splits,
    load_llm_task_config,
    parse_llm_label,
    run_llm_dev_sweep,
    run_llm_test_eval,
)
from src.phase2.subsets import generate_phase2_subsets
from test_phase2_task03 import _make_synthetic_clean_dir


def frozen_hashes():
    roots = [
        Path(p)
        for p in (
            "data/raw",
            "data/clean",
            "data/splits",
            "data/subsets_phase2",
            "configs/phase2.yaml",
            "src/common/constants.py",
            "src/seq",
            "src/zs",
            "src/stats",
        )
    ]
    files = [
        p
        for root in roots
        for p in (root.rglob("*") if root.is_dir() else [root])
        if p.is_file() and "__pycache__" not in p.parts
    ]
    return {str(p): sha256_file(p) for p in files}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase2-llm-corpus")
    clean = _make_synthetic_clean_dir(root)
    subsets = root / "subsets"
    generate_phase2_subsets(
        repo_root=Path.cwd(),
        config_path="configs/phase2_subsets.yaml",
        clean_dir_override=clean,
        subsets_dir_override=subsets,
        reports_dir_override=root / "reports",
    )
    return clean, subsets


def test_task11_llm_config_invariants():
    before = frozen_hashes()
    bundle = load_llm_task_config("configs/task11_llm.yaml")
    assert bundle["llm"]["temperature"] == 0.0
    assert bundle["llm"]["prompt_selection_split"] == "dev_only"
    assert bundle["llm"]["few_shot_pool"] == "train_10pct_only"
    assert len([s for s in bundle["llm"]["strategies"] if s != "zero_shot"]) == 3
    assert frozen_hashes() == before


def test_parse_llm_label_variants_and_invalid():
    assert parse_llm_label("Enjoyment", "vsmec") == (0, "Enjoyment", True)
    assert parse_llm_label("`sadness`.", "vsmec") == (1, "Sadness", True)
    assert parse_llm_label('{"label": "Anger"}', "vsmec") == (3, "Anger", True)
    assert parse_llm_label("Nhãn: positive", "neuesc") == (2, "positive", True)
    assert parse_llm_label("unknown gibberish", "neuesc") == (1, "neutral", False)
    assert parse_llm_label("", "vsmec") == (6, "Other", False)


def test_few_shot_selector_strategies_from_10pct_only(corpus):
    clean, subsets = corpus
    data = load_llm_splits("vsmec", fraction=10, seed=0, clean_dir=clean, subsets_dir=subsets, include_test=False)
    assert data["test_df"] is None
    train_set = set(data["train_texts"])
    assert train_set.isdisjoint(set(data["dev_texts"]))

    selector = FewShotSelector(data["train_texts"], data["train_labels"], EMOTIONS7, seed=0, shots_per_class=1)
    queries = data["dev_texts"][:4]
    zs = selector.select_batch(queries, "zero_shot")
    sf = selector.select_batch(queries, "stratified_fixed")
    knn = selector.select_batch(queries, "tfidf_knn")
    bknn = selector.select_batch(queries, "balanced_tfidf_knn")

    assert all(len(ex) == 0 for ex in zs)
    assert all(len(ex) == 7 for ex in sf)
    assert sf[0] == sf[1]
    assert all(len(ex) == 7 for ex in knn)
    assert all(len(ex) == 7 for ex in bknn)
    for batch in (sf, knn, bknn):
        for ex_list in batch:
            for txt, lbl in ex_list:
                assert txt in train_set
                assert lbl in EMOTIONS7
            if batch is bknn:
                assert {lbl for _, lbl in ex_list} == set(EMOTIONS7)


def test_dev_sweep_isolation_and_test_eval_contract(corpus, tmp_path):
    before = frozen_hashes()
    clean, subsets = corpus
    out_dir = tmp_path / "llm_out"

    def fake_predictor(prompts):
        # Return valid labels deterministically without touching test
        assert not any(" test " in p for p in prompts)
        return ["negative" if i % 3 == 0 else ("neutral" if i % 3 == 1 else "positive") for i in range(len(prompts))]

    sweep = run_llm_dev_sweep(
        "neuesc",
        fake_predictor,
        model_meta={"model_name": "fake-llm", "license": "apache-2.0"},
        strategies=("zero_shot", "stratified_fixed", "tfidf_knn", "balanced_tfidf_knn"),
        prompt_styles=("concise", "rubric"),
        shots_per_class=1,
        clean_dir=clean,
        subsets_dir=subsets,
        out_dir=out_dir,
        max_examples=12,
    )
    assert len(sweep["sweep_df"]) == 8
    sel_path = sweep["selection_path"]
    assert sel_path.is_file()
    assert not (out_dir / "neuesc" / "test_summary_neuesc.csv").exists()

    # Truncated dev selection must be rejected for full test eval
    with pytest.raises(ValueError, match="Truncated dev sweep"):
        run_llm_test_eval(
            "neuesc",
            sel_path,
            lambda prompts: ["neutral"] * len(prompts),
            clean_dir=clean,
            subsets_dir=subsets,
            out_dir=out_dir,
            max_examples=None,
        )

    # Full dev selection + test eval
    full_sweep = run_llm_dev_sweep(
        "neuesc",
        lambda prompts: ["positive" if "[ASSISTANT]" in p else "negative" for p in prompts],
        model_meta={"model_name": "fake-llm", "license": "apache-2.0"},
        strategies=("zero_shot", "balanced_tfidf_knn"),
        prompt_styles=("concise",),
        shots_per_class=1,
        clean_dir=clean,
        subsets_dir=subsets,
        out_dir=tmp_path / "llm_full",
    )
    test_res = run_llm_test_eval(
        "neuesc",
        full_sweep["selection_path"],
        lambda prompts: ["negative" if i % 2 == 0 else "positive" for i in range(len(prompts))],
        clean_dir=clean,
        subsets_dir=subsets,
        out_dir=tmp_path / "llm_full",
    )
    pred_csv = tmp_path / "llm_full" / "neuesc" / "test_zero_shot" / "predictions_test.csv"
    assert pred_csv.is_file()
    pred_df = pd.read_csv(pred_csv)
    assert tuple(pred_df.columns) == ZS_PRED_COLUMNS
    meta = json.loads((tmp_path / "llm_full" / "neuesc" / "test_zero_shot" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["pretraining_leakage_risk"] is True
    assert set(LABELS3) == {"negative", "neutral", "positive"}
    assert frozen_hashes() == before
