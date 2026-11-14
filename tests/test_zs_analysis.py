from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.common.constants import HARD_MERGE_GROUPS
from src.common.hashing import sha256_file
from src.common.schema import ZS_COLUMNS, ZS_PRED_COLUMNS
from src.zs.analysis import (baseline_floors, confusion_counts, contrast_stat,
                             js_divergence, merge_labels7, metrics_from_predictions, row_normalize)
from src.zs.analyze import (EXPECTED, JOBS, MAIN_COLUMNS, analyze_results, main)
from src.zs.infer import evaluation_data_hash


def make_analysis_fixture(root: Path):
    data = root / "data"
    results = root / "results"
    data.mkdir()
    predictions = results / "predictions_zs"
    predictions.mkdir(parents=True)
    v_true7 = np.tile(np.arange(7), 2)
    v_true3 = merge_labels7(v_true7)
    n_true3 = np.tile(np.arange(3), 4)
    pd.DataFrame({"label": np.tile(np.arange(7), 4)}).to_csv(data / "vsmec_train.csv", index=False)
    pd.DataFrame({"label": np.arange(7)}).to_csv(data / "vsmec_dev.csv", index=False)
    pd.DataFrame({"id": np.arange(100, 114), "label": v_true7, "subset": "test"}).to_csv(data / "vsmec_test.csv", index=False)
    pd.concat([pd.DataFrame({"id": np.arange(12), "label": n_true3, "subset": "train"}),
               pd.DataFrame({"id": np.arange(20, 32), "label": n_true3, "subset": "test"})]).to_csv(data / "neuesc_100.csv", index=False)
    manifest = {f"vsmec_{split}.csv": sha256_file(data / f"vsmec_{split}.csv") for split in ("train", "dev", "test")}
    (data / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    target_hashes = {"NEU": evaluation_data_hash(data, "neuesc"), "VSMEC": evaluation_data_hash(data, "vsmec")}
    rows = []
    for alias_index, alias in enumerate(("PhoBERT", "ViSoBERT")):
        for seed in range(5):
            for job_index, (job, (source, target)) in enumerate(JOBS.items()):
                run_id = f"{alias}_{job}_s{seed}"
                target_vsmec = target == "VSMEC"
                true3 = v_true3 if target_vsmec else n_true3
                true7 = v_true7 if target_vsmec else np.full(len(true3), -1)
                rng = np.random.default_rng(seed + alias_index * 100 + job_index * 10)
                p7 = rng.dirichlet(np.ones(7), size=len(true3)) if source == "VSMEC" else np.full((len(true3), 7), np.nan)
                if source == "VSMEC":
                    p3 = np.stack([p7[:, HARD_MERGE_GROUPS[i]].sum(axis=1) for i in range(3)], axis=1)
                    pred7 = p7.argmax(axis=1)
                    argmax3 = merge_labels7(pred7)
                else:
                    p3 = rng.dirichlet(np.ones(3), size=len(true3))
                    pred7 = np.full(len(true3), -1)
                    argmax3 = p3.argmax(axis=1)
                pred = pd.DataFrame({"id": np.arange(100, 114) if target_vsmec else np.arange(20, 32),
                                     "y_true3": true3, "y_true7": true7, "y_pred3_sum": p3.argmax(axis=1),
                                     "y_pred3_argmax": argmax3, "y_pred7": pred7,
                                     "p3_neg": p3[:, 0], "p3_neu": p3[:, 1], "p3_pos": p3[:, 2],
                                     **{f"p7_{i}": p7[:, i] for i in range(7)},
                                     "in_strict": true7 <= 4 if target_vsmec else np.nan}, columns=ZS_PRED_COLUMNS)
                filename = f"{run_id}_{'vsmec_test' if target_vsmec else 'neuesc_test'}.csv"
                pred.to_csv(predictions / filename, index=False)
                for view, method in sorted(EXPECTED[job]):
                    row = {c: "" for c in ZS_COLUMNS}
                    row.update(run_id=run_id, backbone=alias, seed=seed, stage1=source, eval_domain=target,
                               label_view=view, map_method=method, mode="zs" if source != target else "ref", fraction=100,
                               split="test", precision="fp32", n_eval=int((true7 <= 4).sum()) if view.endswith("strict") else len(true3),
                               macro_f1=999, accuracy=999, weighted_f1=999, data_hash=target_hashes[target], code_version="fixture")
                    rows.append(row)
    pd.DataFrame(rows, columns=ZS_COLUMNS).to_csv(results / "zs.csv", index=False)
    return data, results


def test_analysis_metrics_exact_and_neutral_strict_errors():
    true, pred = np.array([0, 0, 2, 2]), np.array([0, 1, 2, 1])
    m = metrics_from_predictions(true, pred, [0, 2])
    assert m["macro_f1"] == pytest.approx(f1_score(true, pred, labels=[0, 2], average="macro", zero_division=0))
    assert m["macro_f1"] == pytest.approx(2 / 3)
    assert float(contrast_stat({"zs": .2, "ref": .5}, "gap")) == pytest.approx(.3)
    assert float(contrast_stat({"zs": .2, "ref": .5}, "retention")) == pytest.approx(.4)
    assert np.isnan(contrast_stat({"zs": .2, "ref": 0}, "retention"))
    assert js_divergence([1, 0, 0], [1, 0, 0]) == 0
    assert js_divergence([1, 0, 0], [0, 1, 0]) == 1
    counts = confusion_counts([0, 6], [[1, 2], [0, 1]], 7, 3)
    assert counts.shape == (7, 3) and counts.sum() == 4
    np.testing.assert_array_equal(row_normalize(counts)[1], [0, 0, 0])
    floors = baseline_floors([0, 0, 2], [0, 0, 0, 2], [0, 2], n_sim=100)
    assert floors["majority"] == pytest.approx(.4)


def test_analyze_all_ten_pairs_reproducible_tables(tmp_path):
    data, results = make_analysis_fixture(tmp_path)
    first = analyze_results(results, data, tmp_path / "out1", boot=30, n_sim=40)
    second = analyze_results(results, data, tmp_path / "out2", boot=30, n_sim=40)
    for path in first["out_dir"].iterdir():
        assert path.read_bytes() == (second["out_dir"] / path.name).read_bytes()
    main_table = first["tables"]["zs_main.csv"]
    assert tuple(main_table.columns) == MAIN_COLUMNS
    assert len(main_table) == 6
    assert main_table.n_seeds.eq(5).all()
    assert main_table.zs_f1.between(0, 1).all()  # ignores intentionally incorrect metadata score999
    per_class = first["tables"]["zs_per_class.csv"]
    assert len(per_class) == 32
    assert set(per_class.loc[per_class.label_view == "V3-VSMEC-strict", "class"]) == {"neg", "pos"}
    for alias in ("PhoBERT", "ViSoBERT"):
        counts = pd.read_csv(first["out_dir"] / f"zs_matrix_7x3_{alias}_counts.csv", index_col=0)
        assert counts.shape == (7, 3) and counts.to_numpy().sum() == 14 * 5
    assert first["meta"]["n_rows"] == 100
    assert len(first["tables"]["zs_asymmetry.csv"]) == 14
    assert len(first["tables"]["zs_floors.csv"]) == 6
    assert first["tables"]["zs_js.csv"].kind.eq("prior").sum() == 1


def test_missing_seed_requires_flag_and_cli_reports(tmp_path, capsys):
    data, results = make_analysis_fixture(tmp_path)
    metadata = pd.read_csv(results / "zs.csv", keep_default_na=False)
    metadata = metadata[~((metadata.backbone == "ViSoBERT") & (metadata.seed == 4))]
    metadata.to_csv(results / "zs.csv", index=False)
    with pytest.raises(ValueError, match="missing seeds"):
        analyze_results(results, data, boot=10)
    main(["--results-dir", str(results), "--data-dir", str(data), "--out-dir", str(tmp_path / "partial"),
          "--boot", "10", "--allow-missing"])
    stdout = capsys.readouterr().out
    assert "ViSoBERT/zsA: missing seeds [4]" in stdout and "[ANALYZE DONE]" in stdout


def test_analyzer_rejects_misaligned_seed_ids(tmp_path):
    data, results = make_analysis_fixture(tmp_path)
    path = results / "predictions_zs" / "PhoBERT_zsA_s1_neuesc_test.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "id"] = 9999
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="identical test IDs"):
        analyze_results(results, data, boot=10, n_sim=10)


@pytest.mark.parametrize("target", ["NEU", "VSMEC"])
def test_analyzer_rejects_stale_evaluation_data_hash(tmp_path, target):
    data, results = make_analysis_fixture(tmp_path)
    metadata = pd.read_csv(results / "zs.csv", keep_default_na=False)
    metadata.loc[metadata.eval_domain == target, "data_hash"] = "0" * 64
    metadata.to_csv(results / "zs.csv", index=False)
    with pytest.raises(ValueError, match=f"data_hash does not match current {target}"):
        analyze_results(results, data, boot=10, n_sim=10)
    assert not (results / "tables").exists()


def test_analyzer_rejects_changed_floor_train_labels(tmp_path):
    data, results = make_analysis_fixture(tmp_path)
    frame = pd.read_csv(data / "vsmec_train.csv")
    frame.loc[0, "label"] = 6
    frame.to_csv(data / "vsmec_train.csv", index=False)
    with pytest.raises(ValueError, match="floor training data_hash"):
        analyze_results(results, data, boot=10, n_sim=10)
