"""Task 11 XLM-R & synthesis unit tests: offline CPU, synthetic splits, no network/GPU."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from src.common.hashing import sha256_file
from src.common.schema import ZS_PRED_COLUMNS
from src.phase2.models import (
    _locate_phase2_xlmr_source_checkpoint,
    build_phase2_model,
    count_model_parameters,
    resolve_model_source,
)
from src.phase2.protocol import load_phase2_config
from src.phase2.run import run_phase2_experiment
from src.phase2.select_lr import select_phase2_p1_lr
from src.phase2.subsets import generate_phase2_subsets
from src.phase2.task11 import (
    benchmark_encoder_inference,
    build_task11_tables_and_report,
    run_xlmr_zs_eval,
)
from src.plm.model_ckpt import Stage1Classifier
from test_phase2_task03 import _make_synthetic_clean_dir


def _frozen_hashes():
    roots = [
        Path(p)
        for p in (
            "data/raw",
            "data/clean",
            "data/splits",
            "data/subsets_phase2",
            "configs/base.yaml",
            "configs/zs.yaml",
            "configs/seq.yaml",
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


@pytest.fixture(scope="module", autouse=True)
def offline_cpu():
    before = _frozen_hashes()
    patch = pytest.MonkeyPatch()
    patch.setenv("HF_HUB_OFFLINE", "1")
    patch.setenv("TRANSFORMERS_OFFLINE", "1")
    patch.setattr(torch.cuda, "is_available", lambda: False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield before
    torch.set_num_threads(threads)
    patch.undo()
    assert _frozen_hashes() == before


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase2-xlmr-corpus")
    clean = _make_synthetic_clean_dir(root)
    (clean / "MANIFEST.json").write_text(
        json.dumps(
            {
                name: sha256_file(clean / name)
                for name in ("vsmec_train.csv", "vsmec_dev.csv", "vsmec_test.csv", "neuesc_100.csv")
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    subsets = root / "subsets"
    generate_phase2_subsets(
        repo_root=Path.cwd(),
        config_path="configs/phase2_subsets.yaml",
        clean_dir_override=clean,
        subsets_dir_override=subsets,
        reports_dir_override=root / "reports",
    )
    return clean, subsets


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase2-xlmr-bert")
    (root / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\ngood\nbad\n",
        encoding="utf-8",
    )
    BertTokenizerFast(vocab_file=str(root / "vocab.txt"), do_lower_case=False).save_pretrained(root)
    torch.manual_seed(7)
    BertModel(
        BertConfig(
            vocab_size=9,
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
    ).save_pretrained(root)
    return root


class _FakeXLMRTrainer:
    def __init__(self, tiny_dir, direction="SA", seed=0):
        self.model, self.tokenizer, _ = build_phase2_model(
            "dft",
            direction,
            "XLMR",
            seed,
            10,
            load_phase2_config(),
            pretrained_model_path=tiny_dir,
            local_files_only=True,
        )
        self.labels = self.model.num_labels
        self.seed = int(seed)

    def fit(self, train_texts, train_labels, dev_texts, dev_labels, work_dir):
        self.cost = {
            **count_model_parameters(self.model),
            "train_seconds": 0.05,
            "gpu_hours": 0.0001,
            "peak_vram_mb": 128.0,
            "gpu_name": "CPU",
            "n_gpus": 0,
            "actual_update_steps": 2,
            "trunc_rate": 0.0,
            "effective_precision": "bf16",
        }
        return {
            "best_step": 2,
            "best_epoch": 1.0,
            "history": [{"dev_selection_score": 0.55}],
            "early_stopped": False,
            "actual_update_steps": 2,
            "cost": self.cost,
        }

    def predict(self, texts):
        base = np.arange(1, self.labels + 1, dtype=np.float64) + 0.1 * self.seed
        probs = base / base.sum()
        return np.tile(probs, (len(texts), 1))

    def save(self, path):
        self.model.save_stage1_checkpoint(self.tokenizer, path)


def test_xlmr_two_stage_lr_lock_source_verification_and_pipeline(tmp_path, corpus, tiny):
    clean, subsets = corpus
    cfg = load_phase2_config()
    results_dir = tmp_path / "results"
    ckpt_dir = tmp_path / "ckpt"

    def common(direction, **kw):
        return run_phase2_experiment(
            backbone="XLMR",
            direction=direction,
            clean_dir=clean,
            subsets_dir=subsets,
            results_dir=results_dir,
            ckpt_out_dir=ckpt_dir,
            pretrained_model_path=tiny,
            **kw,
        )

    # 1. DFT sweep (6 dev-only runs) -> lock DFT LRs only
    for direction in ("SA", "SB"):
        for lr in cfg["sweeps"]["dft_lr_grid"]:
            common(
                direction,
                method="dft",
                fraction=10,
                seed=0,
                lr=lr,
                run_mode="sweep",
                no_save_ckpt=True,
                trainer=_FakeXLMRTrainer(tiny, direction, 0),
            )
    dft_sel = select_phase2_p1_lr(results_dir, backbone="XLMR", methods="dft")
    dft_sel_path = results_dir / "selection_xlmr_dft_lr.json"
    assert dft_sel_path.is_file()
    assert (results_dir / "xlmr_dft_lr_selection.csv").is_file()
    assert set(dft_sel["selected"]) == {"XLMR/SA/dft", "XLMR/SB/dft"}

    # 2. DFT 100% (seed 0..2) validated with selection_xlmr_dft_lr.json
    for direction in ("SA", "SB"):
        lr = float(dft_sel["selected"][f"XLMR/{direction}/dft"]["lr"])
        for seed in (0, 1, 2):
            common(
                direction,
                method="dft",
                fraction=100,
                seed=seed,
                lr=lr,
                run_mode="main",
                selection_path=dft_sel_path,
                trainer=_FakeXLMRTrainer(tiny, direction, seed),
            )

    # Verify XLMR source checkpoint locator & provenance
    ckpt_s0, xlmr_meta = _locate_phase2_xlmr_source_checkpoint(
        ckpt_dir, "SA", 0, results_dir=results_dir, protocol_cfg=cfg, return_meta=True
    )
    assert xlmr_meta["verified_completion"] is True
    _, seq_prov = resolve_model_source(
        "seq", "SA", "XLMR", 0, 10, cfg, source_ckpt_root=ckpt_dir, results_dir=results_dir
    )
    assert seq_prov["source_run_origin"] == "phase2_xlmr_dft_100pct"
    assert seq_prov["source_dft_verified"] is True
    assert "historical_run_provenance" not in seq_prov

    # Reject truncated/incomplete XLMR DFT 100% checkpoint
    bad_run_dir = results_dir / "runs" / ckpt_s0.name
    orig_meta_txt = (bad_run_dir / "metadata.json").read_text(encoding="utf-8")
    orig_meta = json.loads(orig_meta_txt)
    (bad_run_dir / "metadata.json").write_text(
        json.dumps({**orig_meta, "max_examples": 50}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid XLMR Stage-1 source checkpoint"):
        _locate_phase2_xlmr_source_checkpoint(
            ckpt_dir, "SA", 0, results_dir=results_dir, protocol_cfg=cfg
        )
    (bad_run_dir / "metadata.json").write_text(orig_meta_txt, encoding="utf-8")

    # 3. Seq sweep -> lock full selection_xlmr_lr.json
    for direction in ("SA", "SB"):
        for lr in cfg["sweeps"]["seq_lr_grid"]:
            common(
                direction,
                method="seq",
                fraction=10,
                seed=0,
                lr=lr,
                run_mode="sweep",
                source_ckpt_root=ckpt_dir,
                no_save_ckpt=True,
                trainer=_FakeXLMRTrainer(tiny, direction, 0),
            )
    full_sel = select_phase2_p1_lr(results_dir, backbone="XLMR", methods=("dft", "seq"))
    full_sel_path = results_dir / "selection_xlmr_lr.json"
    assert full_sel_path.is_file()

    # Tampering selection_xlmr_dft_lr.json must be caught by full lock
    tampered_dft = copy.deepcopy(dft_sel)
    tampered_dft["selected"]["XLMR/SA/dft"]["lr"] = 9.9e-5
    dft_sel_path.write_text(json.dumps(tampered_dft), encoding="utf-8")
    with pytest.raises(ValueError, match="diverges"):
        select_phase2_p1_lr(results_dir, backbone="XLMR", methods=("dft", "seq"))
    dft_sel_path.write_text(json.dumps(dft_sel), encoding="utf-8")

    # 4. DFT 10% and Seq 10% (seed 0..2, --no-save-ckpt)
    for direction in ("SA", "SB"):
        for method in ("dft", "seq"):
            lr = float(full_sel["selected"][f"XLMR/{direction}/{method}"]["lr"])
            for seed in (0, 1, 2):
                common(
                    direction,
                    method=method,
                    fraction=10,
                    seed=seed,
                    lr=lr,
                    run_mode="main",
                    source_ckpt_root=ckpt_dir if method == "seq" else None,
                    selection_path=full_sel_path,
                    no_save_ckpt=True,
                    trainer=_FakeXLMRTrainer(tiny, direction, seed),
                )

    # 5. XLMR ZS evaluation + SKIP + tamper recovery
    zs_dir = results_dir / "xlmr_zs"

    def fake_zs_predict(*, ckpt, texts, num_labels, batch_size, max_length):
        vec = np.arange(1, num_labels + 1, dtype=np.float64)
        return np.tile(vec / vec.sum(), (len(texts), 1))

    zs1 = run_xlmr_zs_eval(
        results_dir,
        ckpt_dir,
        clean_dir=clean,
        out_dir=zs_dir,
        seeds=(0, 1, 2),
        predictor_fn=fake_zs_predict,
    )
    assert len(zs1["runs"]) == 6 and all(not r["skipped"] for r in zs1["runs"])
    pred_sample = pd.read_csv(zs1["runs"][0]["run_dir"] / "predictions_test.csv")
    assert tuple(pred_sample.columns) == ZS_PRED_COLUMNS

    zs2 = run_xlmr_zs_eval(
        results_dir,
        ckpt_dir,
        clean_dir=clean,
        out_dir=zs_dir,
        seeds=(0, 1, 2),
        predictor_fn=lambda **kw: (_ for _ in ()).throw(AssertionError("Must SKIP")),
    )
    assert len(zs2["runs"]) == 6 and all(r["skipped"] for r in zs2["runs"])

    # Delete one ZS prediction artifact -> only that run recomputes
    (zs1["runs"][0]["run_dir"] / "predictions_test.csv").unlink()
    zs3 = run_xlmr_zs_eval(
        results_dir,
        ckpt_dir,
        clean_dir=clean,
        out_dir=zs_dir,
        seeds=(0, 1, 2),
        predictor_fn=fake_zs_predict,
    )
    assert not zs3["runs"][0]["skipped"] and all(r["skipped"] for r in zs3["runs"][1:])

    # 6. Standardized CPU encoder benchmark + summary tables
    enc_cost_path = results_dir / "cost_encoder_inference.json"
    bench = benchmark_encoder_inference(
        backbones=("ViSoBERT", "XLMR"),
        domains=("neuesc", "vsmec"),
        clean_dir=clean,
        precision="fp32",
        n_warmup=1,
        n_b1=3,
        large_batch_size=4,
        n_large=8,
        pretrained_paths={"ViSoBERT": tiny, "XLMR": tiny},
        local_files_only=True,
        out_path=enc_cost_path,
    )
    assert len(bench["benchmarks"]) == 4
    assert all(b["total_params"] > 0 and b["infer_trainable_params"] == 0 for b in bench["benchmarks"])

    tables_dir = results_dir / "tables"
    report_path = tmp_path / "task11_xlmr_llm_10pct.md"
    synth = build_task11_tables_and_report(
        xlmr_results_dir=results_dir,
        xlmr_zs_dir=zs_dir,
        encoder_cost_path=enc_cost_path,
        out_dir=tables_dir,
        report_path=report_path,
    )
    assert (tables_dir / "task11_xlmr_manifest.csv").is_file()
    assert (tables_dir / "task11_metrics_summary.csv").is_file()
    assert (tables_dir / "task11_cost_performance_10pct.csv").is_file()
    assert report_path.is_file()

    strict_rows = synth["per_class_df"].loc[synth["per_class_df"]["label_view"].eq("V3-VSMEC-strict")]
    assert set(strict_rows["class_name"].tolist()) == {"negative", "positive"}
    assert not synth["contrasts_df"]["primary"].any()
    assert synth["contrasts_df"]["exploratory"].all()
    assert synth["contrasts_df"]["p_holm"].isna().all()
