import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.common.hashing import sha256_file
from src.common.schema import ZS_COLUMNS, ZS_PRED_COLUMNS
from src.zs.ckpt import locate_checkpoint
from src.zs.infer import evaluation_data_hash, run_job, score_prediction_view
from src.zs.mapping import project_probabilities
from src.zs.views import strict_mask


def checkpoint(root, alias="PhoBERT", domain="vsmec", seed=0, variant=None):
    variant = variant or ("Stage1-bf16" if domain == "vsmec" else "NEU3-base-bf16")
    run_name = f"{alias}_{variant}_s{seed}_lr2e-5"
    path = Path(root) / run_name
    (path / "encoder").mkdir(parents=True)
    (path / "encoder" / "config.json").write_text('{"hidden_size": 8}')
    (path / "encoder" / "model.safetensors").write_bytes(b"fake encoder bytes, never loaded")
    labels = 7 if domain == "vsmec" else 3
    torch.save({"weight": torch.zeros(labels, 8), "bias": torch.zeros(labels)}, path / "head.pt")
    return path


@pytest.fixture
def environment(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    vsmec = pd.DataFrame({"id": np.arange(7), "text": [f"v test {i}" for i in range(7)],
                          "text_seg": [f"v_test_{i}" for i in range(7)], "label": np.arange(7), "subset": "test"})
    vsmec.to_csv(data / "vsmec_test.csv", index=False)
    # Intentionally unusable train/dev content proves test filtering precedes validation.
    neu = pd.DataFrame({"id": [10, 11, 12, 13, 14], "text": ["FORBIDDEN_TRAIN", "FORBIDDEN_DEV", "n test 0", "n test 1", "n test 2"],
                        "text_seg": ["FORBIDDEN_TRAIN", "FORBIDDEN_DEV", "n_test_0", "n_test_1", "n_test_2"],
                        "label": [-99, -99, 0, 1, 2], "subset": ["train", "dev", "test", "test", "test"]})
    neu.to_csv(data / "neuesc_100.csv", index=False)
    # VSMEC train/dev do not exist. Their frozen digests are metadata, not inference inputs.
    manifest = {"vsmec_train.csv": "a" * 64, "vsmec_dev.csv": "b" * 64, "vsmec_test.csv": sha256_file(data / "vsmec_test.csv")}
    (data / "MANIFEST.json").write_text(json.dumps(manifest))
    roots = {"vsmec": tmp_path / "ckpt_v", "neu3": tmp_path / "ckpt_n"}
    for alias in ("PhoBERT", "ViSoBERT"):
        checkpoint(roots["vsmec"], alias=alias)
        checkpoint(roots["neu3"], alias=alias, domain="neu3")
    return data, roots


@pytest.mark.parametrize("variant", ["Stage1-bf16", "Stage1-base-bf16", "Stage1-emokeep-bf16", "Stage1-base-emokeep-bf16"])
def test_locator_accepts_real_vsmec_folder_variants(tmp_path, variant):
    expected = checkpoint(tmp_path, variant=variant)
    assert locate_checkpoint(tmp_path, "PhoBERT", "vsmec", 0) == expected.resolve()


def test_locator_excludes_sweeps_and_ambiguous_baselines(tmp_path):
    root = tmp_path / "baselines"
    expected = checkpoint(root)
    checkpoint(root / "sweep")
    checkpoint(root / "smoke")
    assert locate_checkpoint(root, "PhoBERT", "vsmec", 0) == expected.resolve()
    checkpoint(root, variant="Stage1-base-bf16")
    with pytest.raises(ValueError, match="exactly one"):
        locate_checkpoint(root, "PhoBERT", "vsmec", 0)


@pytest.mark.parametrize("failure", ["missing_config", "missing_weights", "wrong_head", "empty_manifest", "missing_entry", "encoder_tampered", "head_tampered"])
def test_locator_rejects_incomplete_or_corrupt_checkpoint(tmp_path, failure):
    path = checkpoint(tmp_path)
    if failure == "missing_config":
        (path / "encoder" / "config.json").unlink()
    elif failure == "missing_weights":
        (path / "encoder" / "model.safetensors").unlink()
    elif failure == "wrong_head":
        torch.save({"weight": torch.zeros(3, 8), "bias": torch.zeros(3)}, path / "head.pt")
    else:
        entry = {"path": str(path), "encoder_sha256": sha256_file(path / "encoder" / "model.safetensors"), "head_sha256": sha256_file(path / "head.pt")}
        manifest = {path.name: entry}
        if failure == "empty_manifest":
            manifest = {}
        elif failure == "missing_entry":
            manifest = {"unrelated_run": {**entry, "path": "/other/checkpoint"}}
        elif failure == "encoder_tampered":
            (path / "encoder" / "model.safetensors").write_bytes(b"changed")
        else:
            torch.save({"weight": torch.ones(7, 8), "bias": torch.zeros(7)}, path / "head.pt")
        (tmp_path / "stage1_ckpt_hash.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        locate_checkpoint(tmp_path, "PhoBERT", "vsmec", 0)


def test_mapping_and_strict_scoring_keep_neutral_errors():
    p3, summed, mapped = project_probabilities([[.30, .15, .10, .15, .10, .10, .10]])
    np.testing.assert_allclose(p3.sum(axis=1), 1)
    assert summed.tolist() == [0] and mapped.tolist() == [2]
    assert strict_mask(np.arange(7)).tolist() == [True] * 5 + [False] * 2
    frame = pd.DataFrame({"y_true3": [0, 2], "y_true7": [1, 0], "y_pred3_sum": [1, 2]})
    metrics = score_prediction_view(frame, "V3-VSMEC-strict", "none")
    assert metrics["macro_f1"] == metrics["accuracy"] == .5
    assert np.isnan(metrics["f1_neu"])


@pytest.mark.parametrize("run_scoped", [True, False])
def test_moved_alias_hash_requires_run_scoped_manifest(tmp_path, run_scoped):
    root = tmp_path / "baselines"
    path = checkpoint(root)
    manifest_directory = path / "results" if run_scoped else root
    manifest_directory.mkdir(exist_ok=True)
    entry = {"path": "/content/original/checkpoints/stage1/PhoBERT",
             "encoder_sha256": sha256_file(path / "encoder" / "model.safetensors"),
             "head_sha256": sha256_file(path / "head.pt")}
    (manifest_directory / "stage1_ckpt_hash.json").write_text(json.dumps({"PhoBERT": entry}))
    if run_scoped:
        assert locate_checkpoint(root, "PhoBERT", "vsmec", 0) == path.resolve()
    else:
        with pytest.raises(ValueError, match="no entry"):
            locate_checkpoint(root, "PhoBERT", "vsmec", 0)


@pytest.mark.parametrize("alias", ["PhoBERT", "ViSoBERT"])
def test_all_four_jobs_test_only_outputs_and_idempotence(environment, tmp_path, alias):
    data, roots = environment
    seen = []
    def predictor(texts, num_labels, seed):
        seen.extend(texts)
        return np.tile(np.arange(1, num_labels + 1) / np.arange(1, num_labels + 1).sum(), (len(texts), 1))
    arguments = dict(backbone=alias, data_dir=data, results_dir=tmp_path / "results", vsmec_ckpt_root=roots["vsmec"], neu_ckpt_root=roots["neu3"], predictor=predictor)
    for job, count in (("zsA", 2), ("zsB", 2), ("refV", 5), ("refN", 1)):
        result = run_job(job=job, **arguments)
        assert len(result["rows"]) == count
        prediction = pd.read_csv(result["prediction_path"])
        assert tuple(prediction.columns) == ZS_PRED_COLUMNS
        np.testing.assert_allclose(prediction[["p3_neg", "p3_neu", "p3_pos"]].sum(axis=1), 1)
        assert prediction["y_pred3_sum"].tolist() == prediction[["p3_neg", "p3_neu", "p3_pos"]].to_numpy().argmax(axis=1).tolist()
        assert run_job(job=job, **arguments)["skipped"]
    assert len(seen) == 20
    assert all("FORBIDDEN" not in text for text in seen)
    assert not any((data / f"vsmec_{sp}.csv").exists() for sp in ("train", "dev"))
    result_frame = pd.read_csv(arguments["results_dir"] / "zs.csv")
    assert tuple(result_frame.columns) == ZS_COLUMNS and len(result_frame) == 10
    assert result_frame.loc[result_frame["label_view"].eq("V3-VSMEC-strict"), "f1_neu"].isna().all()


def test_skip_rejects_changed_test_with_same_ids(environment, tmp_path):
    data, roots = environment
    arguments = dict(job="refN", data_dir=data, results_dir=tmp_path / "results", neu_ckpt_root=roots["neu3"],
                     predictor=lambda texts, num_labels, seed: np.full((len(texts), num_labels), 1 / num_labels))
    run_job(**arguments)
    frame = pd.read_csv(data / "neuesc_100.csv")
    frame.loc[frame["subset"].eq("test"), "text_seg"] = "CHANGED"
    frame.to_csv(data / "neuesc_100.csv", index=False)
    with pytest.raises(ValueError, match="different frozen data"):
        run_job(**arguments)


def test_vsmec_hash_matches_step1_formula_without_reading_train_dev(environment):
    data, _ = environment
    manifest = json.loads((data / "MANIFEST.json").read_text())
    expected = hashlib.sha256("".join(f"vsmec_{sp}.csv:{manifest[f'vsmec_{sp}.csv']}\n" for sp in ("train", "dev", "test")).encode()).hexdigest()
    assert evaluation_data_hash(data, "vsmec") == expected
