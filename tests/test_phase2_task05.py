"""Task 05 acceptance tests: local CPU models, synthetic data, no network/Drive."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from src.common.hashing import sha256_file
from src.phase2.data import load_phase2_prepared_data
from src.phase2.models import (build_phase2_model, count_model_parameters,
                               verify_c0_stage1_dft_checkpoint)
from src.phase2.protocol import load_phase2_config
from src.phase2.run import run_phase2_experiment
from src.phase2.select_lr import select_phase2_p1_lr
from src.phase2.subsets import generate_phase2_subsets, load_phase2_target_splits
from src.phase2.trainer import Phase2Trainer, build_default_adamw_optimizer
from src.plm.model_ckpt import Stage1Classifier
from test_phase2_task03 import _make_synthetic_clean_dir


def frozen_hashes():
    roots = [Path(p) for p in ("data/raw", "data/clean", "data/splits", "data/subsets_phase2",
        "configs/base.yaml", "configs/zs.yaml", "configs/seq.yaml", "configs/phase2.yaml",
        "src/common/constants.py", "src/seq", "src/zs", "src/stats")]
    files = [p for root in roots for p in (root.rglob("*") if root.is_dir() else [root])
             if p.is_file() and "__pycache__" not in p.parts]
    return {str(p): sha256_file(p) for p in files}


@pytest.fixture(scope="module", autouse=True)
def offline_cpu():
    before = frozen_hashes()
    patch = pytest.MonkeyPatch()
    patch.setenv("HF_HUB_OFFLINE", "1")
    patch.setenv("TRANSFORMERS_OFFLINE", "1")
    patch.setattr(torch.cuda, "is_available", lambda: False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield before
    torch.set_num_threads(threads)
    patch.undo()
    assert frozen_hashes() == before


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase2-corpus")
    clean = _make_synthetic_clean_dir(root)
    subsets = root / "subsets"
    generate_phase2_subsets(repo_root=Path.cwd(), config_path="configs/phase2_subsets.yaml",
        clean_dir_override=clean, subsets_dir_override=subsets, reports_dir_override=root / "reports")
    return clean, subsets


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase2-bert")
    (root / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\ngood\nbad\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(root / "vocab.txt"), do_lower_case=False).save_pretrained(root)
    torch.manual_seed(4)
    BertModel(BertConfig(vocab_size=9, hidden_size=12, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, hidden_dropout_prob=.1,
        attention_probs_dropout_prob=.1)).save_pretrained(root)
    return root


def make_model(tiny, direction="SA", seed=0, adapter=None):
    return build_phase2_model("dft", direction, "ViSoBERT", seed, 10, load_phase2_config(),
                             pretrained_model_path=tiny, local_files_only=True, model_adapter_fn=adapter)


class FakeTrainer:
    def __init__(self, tiny, direction="SA"):
        self.model, self.tokenizer, _ = make_model(tiny, direction)
        self.labels = self.model.num_labels
        self.seen = []
        self.selected = False

    def fit(self, train_texts, train_labels, dev_texts, dev_labels, work_dir):
        assert not any(" test " in t for t in train_texts + dev_texts)
        self.seen += train_texts + dev_texts
        self.selected = True
        self.cost = {**count_model_parameters(self.model), "train_seconds": .1,
            "gpu_hours": 0., "peak_vram_mb": 0., "gpu_name": "CPU", "n_gpus": 0,
            "actual_update_steps": 2, "trunc_rate": 0.}
        return dict(best_step=2, best_epoch=1., history=[{"dev_selection_score": .6}],
                    early_stopped=False, actual_update_steps=2, cost=self.cost)

    def predict(self, texts):
        assert self.selected
        self.seen += texts
        return np.tile(np.arange(1, self.labels + 1) / sum(range(1, self.labels + 1)), (len(texts), 1))

    def save(self, path):
        self.model.save_stage1_checkpoint(self.tokenizer, path)


def run_args(tmp_path, corpus, tiny):
    return dict(clean_dir=corpus[0], subsets_dir=corpus[1], results_dir=tmp_path / "results",
                ckpt_out_dir=tmp_path / "ckpt", pretrained_model_path=tiny)


def trainer_args(tiny, **settings):
    model, tokenizer, _ = make_model(tiny)
    defaults = {**load_phase2_config()["training"], "precision": "fp32", "lr": .002,
                "epochs": 3, "batch_size": 2, "gradient_accumulation_steps": 1, **settings}
    return dict(model=model, tokenizer=tokenizer, target_domain="neuesc", num_labels=3, seed=0, settings=defaults)


TRAIN = ["hello world", "good", "bad world", "hello", "good world", "bad"]
LABELS = [0, 1, 2, 1, 2, 0]


def test_data_loader_filters_exact_ids_and_rejects_underscore_leakage(corpus, monkeypatch):
    cfg = load_phase2_config()
    for direction, domain in (("SA", "neuesc"), ("SB", "vsmec")):
        tr, dv, te, meta = load_phase2_target_splits(domain, 10, 0, *corpus)
        data = load_phase2_prepared_data(direction, 10, 0, "ViSoBERT", cfg,
                                        clean_dir=corpus[0], subsets_dir=corpus[1])
        assert data["train_df"].id.tolist() == tr.id.tolist()
        assert len(data["dev_df"]) == len(dv) and len(data["test_df"]) == len(te)
        assert data["provenance"]["ids_sha256"] == meta["ids_sha256"]
        for mode in ("smoke", "sweep"):
            small = load_phase2_prepared_data(direction, 10, 0, "ViSoBERT", cfg,
                clean_dir=corpus[0], subsets_dir=corpus[1], run_mode=mode, max_examples=3)
            assert small["test_df"] is small["test_texts"] is small["test_labels"] is None
            assert small["provenance"]["ids_sha256"] == meta["ids_sha256"]
            assert len(small["train_df"]) == 3
    bad = copy.deepcopy(cfg)
    bad["backbones"]["ViSoBERT"]["text_col"] = "text_seg"
    with pytest.raises(ValueError, match="text_col"):
        load_phase2_prepared_data("SA", 10, 0, "ViSoBERT", bad)
    monkeypatch.setattr("src.phase2.data.preprocess_text", lambda t, **kw: t.replace(" ", "_"))
    with pytest.raises(ValueError, match="Underscore"):
        load_phase2_prepared_data("SA", 10, 0, "ViSoBERT", cfg, clean_dir=corpus[0], subsets_dir=corpus[1])


def test_model_init_dft_seq_c0_and_bitwise_head_match(tmp_path, corpus, tiny):
    cfg = load_phase2_config()
    for direction, width, source_domain in (("SA", 3, "vsmec"), ("SB", 7, "neu3")):
        model, tok, _ = make_model(tiny, direction)
        assert model.head.out_features == width
        source = tmp_path / direction / "source"
        name = "ViSoBERT_Stage1-base-bf16_s0_lr2e-05" if direction == "SA" else "ViSoBERT_NEU3-base-bf16_s0_lr2e-05"
        source_model = Stage1Classifier(copy.deepcopy(model.encoder), 7 if width == 3 else 3)
        with torch.no_grad():
            next(source_model.encoder.parameters()).add_(.25)
        source_model.save_stage1_checkpoint(tok, source / name)
        args = run_args(tmp_path / direction, corpus, tiny)
        dft = run_phase2_experiment(**args, direction=direction, trainer=FakeTrainer(tiny, direction))
        ids = json.loads((dft["run_dir"] / "metadata.json").read_text())["data_provenance"]["ids_sha256"]
        seq, _, seq_meta = build_phase2_model("seq", direction, "ViSoBERT", 0, 10, cfg, source_ckpt_root=source)
        c0, _, c0_meta = build_phase2_model("c0", direction, "ViSoBERT", 0, 10, cfg,
            c0_stage1_run_dir=dft["run_dir"], expected_ids_sha256=ids)
        assert torch.equal(seq.head.weight, c0.head.weight) and torch.equal(seq.head.bias, c0.head.bias)
        assert torch.equal(next(seq.encoder.parameters()), next(source_model.encoder.parameters()))
        assert torch.equal(next(c0.encoder.parameters()), next(model.encoder.parameters()))
        assert seq_meta["head_seed"] == c0_meta["head_seed"] == 0
        assert seq_meta["historical_run_provenance"] == "unverified"


def test_c0_rejects_100pct_target_checkpoint_and_mismatched_subset(tmp_path, corpus, tiny):
    args = run_args(tmp_path, corpus, tiny)
    result = run_phase2_experiment(**args, trainer=FakeTrainer(tiny))
    root = result["run_dir"]
    meta = json.loads((root / "metadata.json").read_text())
    ids = meta["data_provenance"]["ids_sha256"]
    for changes in ({"fraction": 100}, {"seed": 1}, {"run_mode": "smoke"}, {"max_examples": 2}):
        (root / "metadata.json").write_text(json.dumps({**meta, **changes}))
        with pytest.raises(ValueError, match="C0 low-resource violation"):
            verify_c0_stage1_dft_checkpoint(root, "SA", 10, 0, ids, load_phase2_config())
    (root / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="ids_sha256"):
        verify_c0_stage1_dft_checkpoint(root, "SA", 10, 0, "wrong", load_phase2_config())


def test_dev_selection_restores_best_epoch_using_macro_f1_v5(tmp_path, tiny, monkeypatch):
    model, tok, _ = make_model(tiny, "SB")
    trainer = Phase2Trainer(model, tok, "vsmec", 7, 0,
        {**load_phase2_config()["training"], "precision": "fp32", "lr": .001, "epochs": 10})
    expected = iter([.60, .85, .70, .68])
    states = []
    def score(truth, pred, domain):
        assert domain == "vsmec" and len(truth) == len(LABELS)
        states.append(copy.deepcopy(trainer.model.state_dict()))
        return next(expected)
    monkeypatch.setattr("src.phase2.trainer.compute_dev_selection_score", score)
    fit = trainer.fit(TRAIN, LABELS, TRAIN, LABELS, tmp_path)
    assert fit["best_epoch"] == 2 and fit["early_stopped"] and len(fit["history"]) == 4
    assert all(torch.equal(v, states[1][k]) for k, v in trainer.model.state_dict().items())
    assert not trainer.model.training


@pytest.mark.parametrize("mode", ["sweep", "smoke"])
@pytest.mark.parametrize("direction", ["SA", "SB"])
def test_sweep_and_smoke_never_open_or_score_test(tmp_path, corpus, tiny, monkeypatch, mode, direction):
    original = pd.read_csv
    def guarded(path, *args, **kw):
        assert not str(path).endswith("vsmec_test.csv")
        return original(path, *args, **kw)
    monkeypatch.setattr(pd, "read_csv", guarded)
    trainer = FakeTrainer(tiny, direction)
    result = run_phase2_experiment(**run_args(tmp_path, corpus, tiny), run_mode=mode,
                                   direction=direction, trainer=trainer)
    assert not any(" test " in t for t in trainer.seen)
    assert not (result["run_dir"] / "predictions_test.csv").exists()
    assert json.loads((result["run_dir"] / "metrics.json").read_text())["test_metrics"] is None
    assert all(row["split"] == "dev" for row in result["rows"])


@pytest.mark.parametrize("direction,width,test_views", [("SA", 3, 1), ("SB", 7, 6)])
def test_main_run_outputs_complete_artifacts_and_valid_probabilities(tmp_path, corpus, tiny, direction, width, test_views):
    trainer = FakeTrainer(tiny, direction)
    result = run_phase2_experiment(**run_args(tmp_path, corpus, tiny), direction=direction, trainer=trainer)
    root = result["run_dir"]
    assert {p.name for p in root.iterdir()} == {"config.yaml", "metadata.json", "metrics.json", "cost.json",
                                            "completion.json", "predictions_dev.csv", "predictions_test.csv"}
    prediction = pd.read_csv(root / "predictions_test.csv")
    columns = [f"p7_{i}" for i in range(7)] if width == 7 else ["p3_neg", "p3_neu", "p3_pos"]
    probs = prediction[columns].to_numpy()
    assert np.isfinite(probs).all() and (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(1), 1.)
    assert len([r for r in result["rows"] if r["split"] == "test"]) == test_views
    assert sum(" test " in t for t in trainer.seen) == len(prediction)
    if direction == "SB":
        strict = next(r for r in result["rows"] if r["label_view"] == "V3-VSMEC-strict")
        assert json.loads(strict["f1_per_class"])[1] is None


def test_resume_after_mid_training_interruption_matches_uninterrupted(tmp_path, tiny):
    baseline = Phase2Trainer(**trainer_args(tiny))
    fit = baseline.fit(TRAIN, LABELS, TRAIN, LABELS, tmp_path / "full")
    expected = baseline.predict(TRAIN)
    for interrupt_step in (1, 3):  # Between evaluations and at the end of epoch 1.
        interrupted = Phase2Trainer(**trainer_args(tiny))
        save = interrupted._save_resume
        def crash(state, path):
            save(state, path)
            if state["step"] == interrupt_step:
                raise RuntimeError("interrupted")
        interrupted._save_resume = crash
        work = tmp_path / f"resume-{interrupt_step}"
        with pytest.raises(RuntimeError, match="interrupted"):
            interrupted.fit(TRAIN, LABELS, TRAIN, LABELS, work)
        resumed = Phase2Trainer(**trainer_args(tiny))
        actual_fit = resumed.fit(TRAIN, LABELS, TRAIN, LABELS, work)
        np.testing.assert_allclose(resumed.predict(TRAIN), expected, atol=1e-6, rtol=0)
        assert actual_fit["history"] == fit["history"]
        assert actual_fit["best_step"] == fit["best_step"]
    fresh = Phase2Trainer(**trainer_args(tiny, epochs=1))
    fresh.fit(TRAIN, LABELS, TRAIN, LABELS, tmp_path / "c0-new")
    assert fresh.optimizer is not baseline.optimizer and fresh.scheduler is not baseline.scheduler
    assert fresh.cost["actual_update_steps"] == 3
    assert fresh.cost["gpu_name"] == "CPU" and fresh.cost["gpu_hours"] == 0


def test_resume_cadence_logs_and_replays_unsaved_update(tmp_path, tiny, capsys):
    saved_steps = []
    baseline = Phase2Trainer(**trainer_args(tiny, epochs=2, patience=10))
    save = baseline._save_resume
    def record(state, path):
        saved_steps.append(state["step"])
        save(state, path)
    baseline._save_resume = record
    fit = baseline.fit(TRAIN, LABELS, TRAIN, LABELS, tmp_path / "baseline")
    assert saved_steps == [1, 3, 6]
    logs = capsys.readouterr().out
    assert "[fit]" in logs and "[train]" in logs and "[eval]" in logs
    assert "train_loss=" in logs and "dev_score=" in logs and "[resume saved]" in logs

    interrupted = Phase2Trainer(**trainer_args(tiny, epochs=2, patience=10))
    batch = interrupted._batch
    calls = 0
    def crash_before_third_batch(encodings, labels=None):
        nonlocal calls
        if labels is not None:
            calls += 1
            if calls == 3:  # Update 2 has completed, but only update 1 was saved.
                raise RuntimeError("unsaved update")
        return batch(encodings, labels)
    interrupted._batch = crash_before_third_batch
    work = tmp_path / "replay"
    with pytest.raises(RuntimeError, match="unsaved update"):
        interrupted.fit(TRAIN, LABELS, TRAIN, LABELS, work)
    state = torch.load(work / "resume.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 1 and state["cursor"] == 1
    resumed = Phase2Trainer(**trainer_args(tiny, epochs=2, patience=10))
    actual = resumed.fit(TRAIN, LABELS, TRAIN, LABELS, work)
    assert actual["history"] == fit["history"]
    np.testing.assert_allclose(resumed.predict(TRAIN), baseline.predict(TRAIN), atol=1e-6, rtol=0)


def freeze_encoder(model, cfg):
    for p in model.encoder.parameters():
        p.requires_grad = False
    return model


def custom_optimizer(model, settings):
    return torch.optim.AdamW([{"params": [model.head.weight], "lr": .001},
                              {"params": [model.head.bias], "lr": .0001}])


def adapter_checkpoint_saver(model, tokenizer, path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.head.state_dict(), path / "adapter_head.pt")


def test_parameter_counting_and_task10_extension_hooks(tmp_path, corpus, tiny):
    model, tok, meta = make_model(tiny, adapter=freeze_encoder)
    assert 0 < meta["trainable_params"] < meta["total_params"]
    before = copy.deepcopy(model.encoder.state_dict())
    trainer = Phase2Trainer(model, tok, "neuesc", 3, 0, trainer_args(tiny, epochs=1)["settings"],
                            optimizer_builder=custom_optimizer)
    trainer.fit(TRAIN, LABELS, TRAIN, LABELS, tmp_path / "train")
    assert all(torch.equal(v, before[k]) for k, v in model.encoder.state_dict().items())
    assert [g["initial_lr"] for g in trainer.optimizer.param_groups] == [.001, .0001]
    assert count_model_parameters(model) == {k: trainer.cost[k] for k in ("total_params", "trainable_params")}
    calls = []
    def saver(model, tokenizer, path):
        calls.append(path)
        model.save_stage1_checkpoint(tokenizer, path)
    hooked = Phase2Trainer(model, tok, "neuesc", 3, 0, trainer_args(tiny)["settings"], checkpoint_saver=saver)
    hooked.save(tmp_path / "saved")
    assert calls == [tmp_path / "saved"]
    result = run_phase2_experiment(**run_args(tmp_path, corpus, tiny), run_mode="smoke", max_examples=3,
        strict_precision_override=False, epochs_override=1, model_adapter_fn=freeze_encoder,
        optimizer_builder=custom_optimizer)
    cost = json.loads((result["run_dir"] / "cost.json").read_text())
    assert 0 < cost["trainable_params"] < cost["total_params"]
    with pytest.raises(RuntimeError, match="strict bf16"):
        Phase2Trainer(model, tok, "neuesc", 3, 0, {"lr": .001})
    optimizer = build_default_adamw_optimizer(model, {"lr": .001, "weight_decay": .01})
    assert all(p.requires_grad for group in optimizer.param_groups for p in group["params"])
    custom_args = {**run_args(tmp_path / "adapter", corpus, tiny), "max_examples": 3,
        "strict_precision_override": False, "epochs_override": 1, "model_adapter_fn": freeze_encoder,
        "optimizer_builder": custom_optimizer, "checkpoint_saver": adapter_checkpoint_saver}
    custom = run_phase2_experiment(**custom_args)
    assert not custom["skipped"]
    assert run_phase2_experiment(**custom_args)["skipped"]


def test_skip_only_when_complete_and_reruns_if_artifact_deleted_or_tampered(tmp_path, corpus, tiny):
    args = run_args(tmp_path, corpus, tiny)
    def run(**kw):
        return run_phase2_experiment(**args, trainer=FakeTrainer(tiny), **kw)
    first = run()
    assert run()["skipped"]
    (first["run_dir"] / "predictions_test.csv").unlink()
    assert not run()["skipped"]
    (first["run_dir"] / "metrics.json").write_text("tampered")
    assert not run()["skipped"]
    completion = json.loads(first["manifest"].read_text())
    completion["complete"] = False
    first["manifest"].write_text(json.dumps(completion))
    assert not run()["skipped"]
    table = Path(args["results_dir"]) / "phase2_runs.csv"
    rows = pd.read_csv(table)
    pd.concat([rows, rows.iloc[:1]]).to_csv(table, index=False)
    assert not run()["skipped"]
    assert run()["skipped"]
    assert run(lr=3e-5)["run_id"] != first["run_id"]
    assert run(epochs_override=1)["run_id"] != first["run_id"]
    completion = json.loads(first["manifest"].read_text())
    completion["fingerprint"] = "wrong"
    first["manifest"].write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="fingerprint"):
        run()


def test_select_p1_lr_complete_grid_tie_break_and_edge_warning(tmp_path, corpus, tiny):
    args = run_args(tmp_path, corpus, tiny)
    cfg = load_phase2_config()
    # All scores tie; the smaller LR must win independently of row order.
    for direction in ("SA", "SB"):
        source = tmp_path / direction / "source"
        model, tok, _ = make_model(tiny)
        name = "ViSoBERT_Stage1-base-bf16_s0_lr2e-05" if direction == "SA" else "ViSoBERT_NEU3-base-bf16_s0_lr2e-05"
        Stage1Classifier(model.encoder, 7 if direction == "SA" else 3).save_stage1_checkpoint(tok, source / name)
        for method in ("dft", "seq"):
            for lr in cfg["sweeps"][f"{method}_lr_grid"]:
                run_phase2_experiment(**args, direction=direction, method=method, lr=lr,
                    run_mode="sweep", source_ckpt_root=source, trainer=FakeTrainer(tiny, direction))
    with pytest.warns(UserWarning, match="grid edge"):
        result = select_phase2_p1_lr(args["results_dir"])
    assert (args["results_dir"] / "selection_p1_lr.json").is_file()
    assert (args["results_dir"] / "p1_lr_selection.csv").is_file()
    for key, record in result["selected"].items():
        assert record["lr"] == min(cfg["sweeps"][f"{key.split('/')[-1]}_lr_grid"])
    # Main validates the lock; C0 reuses the selected Seq rate.
    main = run_phase2_experiment(**args, lr=2e-5, trainer=FakeTrainer(tiny),
                                 selection_path=args["results_dir"] / "selection_p1_lr.json")
    c0 = run_phase2_experiment(**args, method="c0", lr=1e-5, trainer=FakeTrainer(tiny),
        c0_stage1_run_dir=main["run_dir"], selection_path=args["results_dir"] / "selection_p1_lr.json")
    assert json.loads((c0["run_dir"] / "cost.json").read_text())["stage1_dft_cost_ref"]
    csv = args["results_dir"] / "phase2_runs.csv"
    original = pd.read_csv(csv, keep_default_na=False)
    sweep = original.loc[original.run_mode.eq("sweep")]
    for bad in (sweep.iloc[1:], sweep.assign(split="test"),
                sweep.assign(protocol_version="wrong")):
        bad.to_csv(csv, index=False)
        with pytest.raises(ValueError):
            select_phase2_p1_lr(args["results_dir"])
    bad = sweep.copy()
    bad.loc[bad.index[0], "data_hash"] = "wrong"
    bad.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="data_hash"):
        select_phase2_p1_lr(args["results_dir"])


def test_frozen_repo_invariants_preserved(offline_cpu):
    assert frozen_hashes() == offline_cpu
