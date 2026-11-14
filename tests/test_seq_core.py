import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from src.seq.protocol import atomic_json, digest, load_config
from src.seq.stage2 import run_stage2, build_seq_run_id
from src.seq.select_lr import select_lr
from src.seq.trainer import TorchTrainer, load_encoder_new_head
from src.seq.analysis import apply_primary_holm
from src.stats.bootstrap import bootstrap_contrast
from src.stats.holm import holm_adjust, bootstrap_pvalue
from src.common.hashing import sha256_file
from test_zs_analysis import make_analysis_fixture
from test_zs_infer import checkpoint


def test_paired_bootstrap_identity_seed_validation_and_legacy_exact():
    y = np.arange(24) % 3
    preds = np.array([y, np.roll(y, 1), np.roll(y, 2)])
    arm = dict(y_true=y, preds_by_seed=preds, labels=[0, 1, 2], seeds=[0, 1, 2])
    result = bootstrap_contrast({"a": arm, "b": arm}, lambda x: x["a"]-x["b"], True, boot=41, share_seeds=True)
    assert result["point"] == result["ci_lo"] == result["ci_hi"] == 0
    assert bootstrap_pvalue(result["draws"]) == 1
    with pytest.raises(ValueError, match="ordered seed"):
        bootstrap_contrast({"a": arm, "b": {**arm, "seeds": [2, 1, 0]}}, lambda x: 0, True, boot=10, share_seeds=True)
    # Recorded from pre-extension bootstrap at commit b226462. Stable after commits.
    arguments = ({"a": arm, "b": arm}, lambda x: x["a"]-x["b"], True)
    after = bootstrap_contrast(*arguments, boot=41)
    assert hashlib.sha256(after["draws"].tobytes()).hexdigest() == "e649aa1edac0c3a087f2f9aa87e10403493d0e1add69ac7b44a880a2d181d815"


def test_holm_fixed_family_and_pvalue_floor():
    np.testing.assert_allclose(holm_adjust([.01, .04, .03, .20]), [.04, .09, .09, .20])
    assert bootstrap_pvalue(np.ones(99)) == .02
    cfg = load_config()
    rows = [{**p, "p_boot": .001, "n_seeds": 5} for p in cfg["primary"]]
    assert all(r["significant"] for r in apply_primary_holm(rows, cfg))
    partial = apply_primary_holm(rows[:3], cfg)
    assert all(np.isnan(r["p_holm"]) and r["significant"] is None for r in partial)


def test_protocol_document_and_config_are_one_contract(tmp_path):
    import re
    import yaml
    cfg = load_config()
    doc = Path("docs/seq_primary_comparisons.md").read_text(encoding="utf-8")
    assert json.loads(re.search(r"```json\s*(.*?)\s*```", doc, re.S).group(1)) == cfg["primary"]
    changed = copy.deepcopy(cfg)
    changed["training"]["patience"] = 2
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen seq-v1"):
        load_config(p)


@pytest.fixture
def env(tmp_path):
    data, zs = make_analysis_fixture(tmp_path)
    # Add realistic target texts/splits to the frozen mock corpus.
    for filename in ("neuesc_100.csv", "vsmec_train.csv", "vsmec_dev.csv", "vsmec_test.csv"):
        frame = pd.read_csv(data / filename)
        split = "train" if "train" in filename else "dev" if "dev" in filename else "test"
        if "subset" not in frame:
            frame["subset"] = split
        if filename == "neuesc_100.csv":
            dev = frame.loc[frame.subset.eq("train")].copy()
            dev["subset"] = "dev"
            dev["id"] += 1000
            frame = pd.concat([frame, dev], ignore_index=True)
        if "id" not in frame:
            frame["id"] = np.arange(len(frame))
        frame["text"] = [f"{filename} {s} example {i} 😀" for i, s in enumerate(frame.subset)]
        frame["text_seg"] = frame.text
        frame.to_csv(data / filename, index=False)
    manifest = {f"vsmec_{s}.csv": sha256_file(data / f"vsmec_{s}.csv") for s in ("train", "dev", "test")}
    atomic_json(data / "MANIFEST.json", manifest)
    from src.zs.infer import evaluation_data_hash
    baseline = pd.read_csv(zs / "zs.csv")
    for domain, name in (("NEU", "neuesc"), ("VSMEC", "vsmec")):
        baseline.loc[baseline.eval_domain.eq(domain), "data_hash"] = evaluation_data_hash(data, name)
    baseline.to_csv(zs / "zs.csv", index=False)
    roots = {"vsmec": tmp_path / "vckpt", "neu3": tmp_path / "nckpt"}
    for alias in ("PhoBERT", "ViSoBERT"):
        for seed in range(5):
            for domain in roots:
                checkpoint(roots[domain], alias=alias, domain=domain, seed=seed)
    result = tmp_path / "seq"
    cfg = load_config()
    atomic_json(result / "selection_seq.json", {"protocol_hash": digest(cfg["training"]), "selected": {f"{a}/{d}": {"lr": 2e-5} for a in cfg["backbones"] for d in ("SA", "SB")}})
    return dict(data_dir=data, results_dir=result, ckpt_out_dir=tmp_path / "out", vsmec_ckpt_root=roots["vsmec"], neu_ckpt_root=roots["neu3"]), zs


class FakeTrainer:
    def __init__(self, labels):
        self.labels = labels
        self.selected = False
        self.seen = []

    def fit(self, train_texts, train_labels, dev_texts, dev_labels, work_dir):
        assert not any(" test " in t for t in train_texts + dev_texts)
        self.seen += train_texts + dev_texts
        self.selected = True
        return dict(best_step=2, best_epoch=1.0, history=[{"dev_macro_f1": .6}], trunc_rate=0.0)

    def predict(self, texts):
        assert self.selected
        self.seen += texts
        return np.tile(np.arange(1, self.labels+1) / sum(range(1, self.labels+1)), (len(texts), 1))

    def save(self, path):
        path = Path(path)
        (path / "encoder").mkdir(parents=True, exist_ok=True)
        (path / "encoder" / "config.json").write_text("{}")
        (path / "encoder" / "model.safetensors").write_bytes(b"saved")
        torch.save({"weight": torch.zeros(self.labels, 8), "bias": torch.zeros(self.labels)}, path / "head.pt")


@pytest.mark.parametrize("direction,labels,count", [("SA",3,2),("SB",7,6),("C0N",3,2),("C0V",7,6)])
def test_all_directions_and_complete_artifact_skip(env, direction, labels, count):
    args, _ = env
    model = FakeTrainer(labels)
    first = run_stage2(direction=direction, trainer=model, **args)
    assert len(first["rows"]) == count
    assert run_stage2(direction=direction, trainer=FakeTrainer(labels), **args)["skipped"]
    sidecar = json.loads(first["manifest"].read_text())
    assert sidecar["complete"] and len(sidecar["artifacts"]) >= 4
    pred_path = next(Path(p) for p in sidecar["artifacts"] if p.endswith("_test.csv"))
    pred_path.unlink()
    assert not run_stage2(direction=direction, trainer=FakeTrainer(labels), **args)["skipped"]
    assert all(("neuesc" in t) == (labels == 3) for t in model.seen)


def test_sweep_never_opens_vsmec_test_and_emojis_preserved(env, monkeypatch):
    args, _ = env
    original = pd.read_csv
    def guarded(path, *a, **kw):
        assert not str(path).endswith("vsmec_test.csv")
        return original(path, *a, **kw)
    monkeypatch.setattr(pd, "read_csv", guarded)
    model = FakeTrainer(7)
    result = run_stage2(direction="SB", backbone="ViSoBERT", variant="seqsweep-bf16", trainer=model, **args)
    assert len(result["rows"]) == 1 and result["rows"][0]["split"] == "dev"
    assert not result["rows"][0]["ckpt_path"]
    assert all("😀" in t for t in model.seen)


def test_resume_without_provenance_and_incomplete_result_keys(env):
    args, _ = env
    run_id = build_seq_run_id("PhoBERT", "SA", "seq-bf16", 0, 2e-5)
    work = args["results_dir"] / "work_seq" / run_id
    work.mkdir(parents=True)
    (work / "resume.pt").write_bytes(b"unidentified")
    with pytest.raises(ValueError, match="missing provenance"):
        run_stage2(trainer=FakeTrainer(3), **args)
    (work / "resume.pt").unlink()
    run_stage2(trainer=FakeTrainer(3), **args)
    path = args["results_dir"] / "seq.csv"
    rows = pd.read_csv(path)
    rows.loc[rows.split.eq("test"), "map_method"] = "wrong"
    rows.to_csv(path, index=False)
    assert not run_stage2(trainer=FakeTrainer(3), **args)["skipped"]


def test_lr_selection_complete_grid_tie_and_test_independence(env):
    args, _ = env
    for alias in ("PhoBERT", "ViSoBERT"):
        for direction, labels in (("SA", 3), ("SB", 7)):
            for lr in (1e-5, 2e-5, 3e-5):
                run_stage2(backbone=alias, direction=direction, lr=lr, variant="seqsweep-bf16", trainer=FakeTrainer(labels), **args)
    selected = select_lr(args["results_dir"])
    assert all(v["lr"] == 1e-5 for v in selected["selected"].values())
    path = args["results_dir"] / "seq.csv"
    frame = pd.read_csv(path)
    poisoned = frame.copy()
    poisoned["split"] = "test"
    poisoned["macro_f1"] = 1000
    pd.concat([frame, poisoned]).to_csv(path, index=False)
    assert selected == select_lr(args["results_dir"])
    frame.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="grid"):
        select_lr(args["results_dir"])


def test_head_replacement_seed_and_native_bf16_resume(tmp_path):
    torch.set_num_threads(1)
    p = tmp_path / "source"
    (p / "encoder").mkdir(parents=True)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(p / "encoder")
    encoder = BertModel(BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16))
    encoder.save_pretrained(p / "encoder")
    torch.save({"weight": torch.full((7,8), 999.), "bias": torch.zeros(7)}, p / "head.pt")
    one, _ = load_encoder_new_head(p, 3, 0)
    two, _ = load_encoder_new_head(p, 3, 0)
    three, _ = load_encoder_new_head(p, 3, 1)
    torch.testing.assert_close(one.head.weight, two.head.weight)
    assert not torch.equal(one.head.weight, three.head.weight)
    for k, v in encoder.state_dict().items():
        torch.testing.assert_close(one.encoder.state_dict()[k], v)
    settings = {**load_config()["training"], "strict_precision": False, "epochs": 2, "batch_size": 2, "lr": 2e-5}
    trainer = TorchTrainer(p, 3, 0, settings)
    observed = []
    trainer.model.head.register_forward_hook(lambda module, inputs, outputs: observed.append(outputs.dtype))
    texts, labels = ["hello world"]*5, [0,1,2,0,1]
    fit = trainer.fit(texts, labels, texts, labels, tmp_path / "work")
    assert torch.bfloat16 in observed
    assert fit["best_step"] > 0
    restored = TorchTrainer(p, 3, 0, settings)
    restored.fit(texts, labels, texts, labels, tmp_path / "work")
    np.testing.assert_allclose(restored.predict(texts), trainer.predict(texts), atol=1e-7)
    interrupted = TorchTrainer(p, 3, 0, settings)
    real_predict = interrupted.predict
    n_calls = 0
    def crash_after_first_safe_checkpoint(inputs):
        nonlocal n_calls
        n_calls += 1
        if n_calls == 2:
            raise RuntimeError("simulated interruption")
        return real_predict(inputs)
    interrupted.predict = crash_after_first_safe_checkpoint
    with pytest.raises(RuntimeError, match="simulated"):
        interrupted.fit(texts, labels, texts, labels, tmp_path / "interrupted")
    resumed = TorchTrainer(p, 3, 0, settings)
    resumed.fit(texts, labels, texts, labels, tmp_path / "interrupted")
    np.testing.assert_allclose(resumed.predict(texts), trainer.predict(texts), rtol=0, atol=1e-7)


def test_full_analysis_reproducible_and_holm_missing_family(env, tmp_path):
    from src.seq.analyze import analyze_results
    args, zs = env
    for alias in ("PhoBERT", "ViSoBERT"):
        for seed in range(5):
            for direction, labels in (("SA",3),("SB",7)):
                run_stage2(backbone=alias, seed=seed, direction=direction, trainer=FakeTrainer(labels), **args)
            if seed < 3:
                for direction, labels in (("C0N",3),("C0V",7)):
                    run_stage2(backbone=alias, seed=seed, direction=direction, trainer=FakeTrainer(labels), **args)
    first = analyze_results(args["results_dir"], zs, args["data_dir"], tmp_path / "tables1", boot=11)
    second = analyze_results(args["results_dir"], zs, args["data_dir"], tmp_path / "tables2", boot=11)
    for file in first["out_dir"].iterdir():
        assert file.read_bytes() == (second["out_dir"] / file.name).read_bytes()
    main = first["tables"]["seq_main.csv"]
    assert main.primary.sum() == 4 and main.p_holm.notna().sum() == 4
    assert len(first["tables"]["seq_c0.csv"]) == 8
    path = args["results_dir"] / "seq.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[~(frame.backbone.eq("PhoBERT") & frame.direction.eq("SA") & frame.seed.eq(4))]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing paired"):
        analyze_results(args["results_dir"], zs, args["data_dir"], boot=11)
    partial = analyze_results(args["results_dir"], zs, args["data_dir"], tmp_path / "partial", boot=11, allow_missing=True)
    assert partial["tables"]["seq_main.csv"].p_holm.isna().all()
