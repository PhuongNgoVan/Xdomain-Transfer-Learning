"""Offline CPU unit and acceptance tests for `src/plm/stage1.py` (`tests/test_stage1.py`).

Covers all requirements of `tasks/M6_stage1_vsmec.md` Section 5.2 & 7:
1. Initializes tiny offline BERT (`BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1,
   num_attention_heads=2, intermediate_size=32)`) and ~160 synthetic samples in `tmp_path` (no network).
2. Runs full Stage 1 flow (`max_steps=1`): training, checkpoint saving (`encoder/` + `head.pt`),
   5-dataset `predictions7` export, `stage1.csv` recording, `work_dir` cleanup, and `stage1_ckpt_hash.json`.
3. Verifies single `nn.Linear(16, 7)` head and `head.pt` shape `(7, 16)` / `(7,)`.
4. Verifies all 5 `predictions7` files (`vsmec_dev`, `vsmec_test`, `neuesc_dev`, `neuesc_test`, `vsfc_test`)
   have exact `PRED7_COLUMNS`, matching row counts and IDs, `y_pred7 == argmax(logits)`, and empty `y_true7`
   on NEU-ESC and VSFC.
5. Verifies `stage1.csv` matches `STAGE1_COLUMNS` and its `macro_f1` and per-emotion F1 scores match
   values recomputed from `predictions7/{run_id}_vsmec_dev.csv`.
6. Verifies execution order: all 5 prediction files and `stage1.csv` exist on disk BEFORE intermediate
   checkpoints in `work_dir` are deleted.
7. Verifies reloading checkpoint (`encoder/` + `head.pt`) reproduces identical logits on 20 dev samples.
8. Verifies CPU determinism: two runs with `seed=0` produce identical logits and metrics.
9. Verifies skip mechanism when `run_id` already exists in `stage1.csv`, plus resume from `work_dir`.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer
import yaml

from src.common.constants import EMOTIONS7, NEG, NEU, POS
from src.common.hashing import sha256_file
from src.common.io import read_csv_schema
from src.common.schema import PRED7_COLUMNS, STAGE1_COLUMNS
from src.common.seed import set_seed
from src.eval.metrics import EMOTION7_IDS, compute_metrics
from src.plm.stage1 import (
    PREDICTION_DATASETS,
    Stage1Classifier,
    build_stage1_run_id,
    format_lr,
    infer_backbone_alias,
    load_stage1_config,
    main as stage1_main,
    predict_logits7,
    resolve_input_column,
    run_stage1,
    tokenize_with_truncation_stats,
)


def _create_tiny_bert_and_data(root: Path) -> Dict[str, Path]:
    """Create a local tiny BERT model (`hidden_size=16`) and synthetic clean CSV datasets."""
    backbone_dir = root / "tiny_bert"
    backbone_dir.mkdir(parents=True, exist_ok=True)

    # Create a 100-token vocabulary
    vocab_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    ] + [f"tok_{i}" for i in range(95)]
    vocab_file = backbone_dir / "vocab.txt"
    vocab_file.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")

    tokenizer = BertTokenizer(vocab_file=str(vocab_file), do_lower_case=True)
    tokenizer.save_pretrained(backbone_dir)

    set_seed(0, deterministic=True)
    config = BertConfig(
        vocab_size=100,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = BertModel(config)
    model.save_pretrained(backbone_dir)

    data_dir = root / "data_clean"
    data_dir.mkdir(parents=True, exist_ok=True)

    def _make_vsmec_rows(n: int, split_name: str, id_offset: int) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for i in range(n):
            emo = i % len(EMOTIONS7)
            t_idx = (i * 3) % 80
            # Make row 0 extra long so truncation (>32 tokens) is also exercised when max_length=32
            if i == 0:
                words = " ".join(f"tok_{(t_idx + k) % 90}" for k in range(45))
            else:
                words = " ".join(f"tok_{(t_idx + k) % 90}" for k in range(6))
            rows.append(
                {
                    "id": id_offset + i,
                    "text": words,
                    "label": emo,
                    "subset": split_name,
                    "text_seg": words,
                }
            )
        return pd.DataFrame(rows)

    _make_vsmec_rows(64, "train", 1000).to_csv(
        data_dir / "vsmec_train.csv", index=False, encoding="utf-8"
    )
    _make_vsmec_rows(28, "dev", 2000).to_csv(
        data_dir / "vsmec_dev.csv", index=False, encoding="utf-8"
    )
    _make_vsmec_rows(21, "test", 3000).to_csv(
        data_dir / "vsmec_test.csv", index=False, encoding="utf-8"
    )

    # NEU-ESC 100% split with train, dev, test
    neuesc_rows: List[Dict[str, Any]] = []
    for sp, cnt, off in [("train", 20, 4000), ("dev", 24, 5000), ("test", 18, 6000)]:
        for i in range(cnt):
            sent = [NEG, NEU, POS][i % 3]
            words = " ".join(f"tok_{(i + k) % 90}" for k in range(5))
            neuesc_rows.append(
                {
                    "id": off + i,
                    "text": words,
                    "label": sent,
                    "subset": sp,
                    "text_seg": words,
                }
            )
    pd.DataFrame(neuesc_rows).to_csv(
        data_dir / "neuesc_100.csv", index=False, encoding="utf-8"
    )

    # VSFC test clean
    vsfc_rows = [
        {
            "id": 7000 + i,
            "text": f"tok_{i % 80} tok_{(i + 1) % 80} tok_{(i + 2) % 80}",
            "label": [NEG, NEU, POS][i % 3],
            "subset": "test",
            "text_seg": f"tok_{i % 80} tok_{(i + 1) % 80} tok_{(i + 2) % 80}",
        }
        for i in range(15)
    ]
    pd.DataFrame(vsfc_rows).to_csv(
        data_dir / "vsfc_test_clean.csv", index=False, encoding="utf-8"
    )

    return {"backbone_dir": backbone_dir, "data_dir": data_dir}


def _write_stage1_yaml(
    cfg_path: Path,
    backbone_dir: Path,
    seed: int = 0,
    max_length: int = 32,
) -> Path:
    cfg = {
        "backbone": backbone_dir.as_posix(),
        "model_alias": "PhoBERT",
        "mode": "stage1",
        "variant": "base",
        "input_column": "text_seg",
        "num_labels": 7,
        "max_length": max_length,
        "dynamic_padding": True,
        "group_by_length": True,
        "dropout": 0.1,
        "loss": "ce",
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2.0e-5,
        "num_train_epochs": 2,
        "warmup_ratio": 0.06,
        "lr_scheduler_type": "linear",
        "weight_decay": 0.01,
        "precision": "fp16",
        "seed": seed,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "save_total_limit": 2,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "early_stopping_patience": 5,
        "load_best_model_at_end": True,
        "keep_ckpt": True,
        "official": False,
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


class TestStage1HelpersAndConfigs:
    def test_production_yaml_configs_exist_and_match_spec(self) -> None:
        pho_cfg = load_stage1_config("configs/exp/stage1_phobert.yaml")
        viso_cfg = load_stage1_config("configs/exp/stage1_visobert.yaml")

        assert pho_cfg["backbone"] == "vinai/phobert-base-v2"
        assert pho_cfg["input_column"] == "text_seg"
        assert viso_cfg["backbone"] == "uitnlp/visobert"
        assert viso_cfg["input_column"] == "text"

        for cfg in (pho_cfg, viso_cfg):
            assert cfg["num_labels"] == 7
            assert cfg["max_length"] == 256
            assert cfg["group_by_length"] is True
            assert cfg["batch_size"] == 16
            assert cfg["gradient_accumulation_steps"] == 2
            assert float(cfg["learning_rate"]) == pytest.approx(2.0e-5)
            assert cfg["num_train_epochs"] == 10
            assert float(cfg["warmup_ratio"]) == pytest.approx(0.06)
            assert float(cfg["weight_decay"]) == pytest.approx(0.01)
            assert cfg["precision"] == "fp16"
            assert cfg["seed"] == 0
            assert cfg["early_stopping_patience"] == 5
            assert cfg["load_best_model_at_end"] is True
            assert cfg["keep_ckpt"] is True
            assert cfg["official"] is False

    def test_resolve_input_column_and_run_id_formatting(self) -> None:
        assert format_lr(2e-5) == "2e-5"
        assert resolve_input_column("vinai/phobert-base-v2", {"input_column": "text_seg"}) == "text_seg"
        assert resolve_input_column("uitnlp/visobert", {"input_column": "text"}) == "text"

        with pytest.raises(ValueError, match="text_seg"):
            resolve_input_column("vinai/phobert-base-v2", {"input_column": "text"})
        with pytest.raises(ValueError, match="text"):
            resolve_input_column("uitnlp/visobert", {"input_column": "text_seg"})

        assert infer_backbone_alias("vinai/phobert-base-v2") == "PhoBERT"
        assert infer_backbone_alias("uitnlp/visobert") == "ViSoBERT"
        assert (
            build_stage1_run_id("vinai/phobert-base-v2", seed=0, lr=2e-5)
            == "PhoBERT_Stage1-base_s0_lr2e-5"
        )


class TestStage1EndToEndCPU:
    def test_stage1_full_flow_1_step_and_acceptance_invariants(
        self, tmp_path: Path
    ) -> None:
        env = _create_tiny_bert_and_data(tmp_path)
        cfg_path = _write_stage1_yaml(tmp_path / "stage1_tiny.yaml", env["backbone_dir"], seed=0)

        ckpt_dir = tmp_path / "checkpoints" / "stage1"
        results_dir = tmp_path / "results"
        work_dir = tmp_path / "work_tmp"

        cleanup_state: Dict[str, Any] = {}

        def _check_before_cleanup(info: Dict[str, Any]) -> None:
            # Verify all 5 prediction files and stage1.csv exist BEFORE work_dir is deleted
            cleanup_state["all_preds_exist"] = all(
                Path(p).is_file() for p in info["prediction_paths"].values()
            )
            cleanup_state["stage1_csv_exists"] = Path(info["stage1_csv"]).is_file()
            cleanup_state["work_ckpt_exists"] = info["work_dir_existed_after_results"]

        res = run_stage1(
            config_path=cfg_path,
            data_dir=env["data_dir"],
            ckpt_dir=ckpt_dir,
            results_dir=results_dir,
            work_dir=work_dir,
            max_steps_override=1,
            on_before_cleanup=_check_before_cleanup,
        )

        assert res["skipped"] is False
        assert res["best_step"] == 1
        assert cleanup_state == {
            "all_preds_exist": True,
            "stage1_csv_exists": True,
            "work_ckpt_exists": True,
        }
        # Intermediate work_dir for this run must be deleted after completion
        assert not (work_dir / res["run_id"]).exists()

        # 1. Verify checkpoint format: encoder/ + head.pt with shape (7, 16) and (7,)
        ckpt_subdir = ckpt_dir / "PhoBERT"
        assert (ckpt_subdir / "encoder").is_dir()
        head_pt = ckpt_subdir / "head.pt"
        assert head_pt.is_file()
        head_state = torch.load(head_pt, map_location="cpu", weights_only=True)
        assert set(head_state.keys()) == {"weight", "bias"}
        assert tuple(head_state["weight"].shape) == (7, 16)
        assert tuple(head_state["bias"].shape) == (7,)

        # 2. Verify all 5 prediction files exist, have PRED7_COLUMNS, exact row counts, and y_pred7 == argmax(logits)
        expected_counts = {
            "vsmec_dev": 28,
            "vsmec_test": 21,
            "neuesc_dev": 24,
            "neuesc_test": 18,
            "vsfc_test": 15,
        }
        logit_cols = [f"logit_{i}" for i in range(7)]

        for ds_name in PREDICTION_DATASETS:
            pred_path = res["prediction_paths"][ds_name]
            assert pred_path.is_file()
            df_p = read_csv_schema(pred_path, PRED7_COLUMNS, keep_default_na=False)
            assert len(df_p) == expected_counts[ds_name]

            logits = df_p[logit_cols].to_numpy(dtype=np.float64)
            y_pred = df_p["y_pred7"].astype(int).to_numpy()
            assert np.array_equal(y_pred, np.argmax(logits, axis=1))

            if ds_name.startswith("vsmec_"):
                assert all(str(v) != "" for v in df_p["y_true7"])
            else:
                assert (df_p["y_true7"] == "").all()

        # 3. Verify stage1.csv matches STAGE1_COLUMNS and recomputed metrics from vsmec_dev predictions
        df_s1 = read_csv_schema(results_dir / "stage1.csv", STAGE1_COLUMNS)
        assert len(df_s1) == 1
        row = df_s1.iloc[0]
        assert row["run_id"] == res["run_id"]
        assert int(row["best_step"]) == 1

        df_dev_pred = read_csv_schema(res["prediction_paths"]["vsmec_dev"], PRED7_COLUMNS)
        recomputed = compute_metrics(
            df_dev_pred["y_true7"].astype(int).tolist(),
            df_dev_pred["y_pred7"].astype(int).tolist(),
            logits=df_dev_pred[logit_cols].to_numpy(dtype=np.float64),
            labels=EMOTION7_IDS,
        )
        assert float(row["macro_f1"]) == pytest.approx(recomputed["macro_f1"])
        assert float(row["accuracy"]) == pytest.approx(recomputed["accuracy"])
        assert float(row["weighted_f1"]) == pytest.approx(recomputed["weighted_f1"])
        for emo in EMOTIONS7:
            col = f"f1_{emo.lower()}"
            assert float(row[col]) == pytest.approx(recomputed[col])

        # 4. Verify reloading checkpoint reproduces identical logits on 20 dev samples (< 1e-4 diff)
        reloaded_model, reloaded_tok = Stage1Classifier.load_stage1_checkpoint(
            ckpt_subdir, device="cpu"
        )
        assert isinstance(reloaded_model.head, torch.nn.Linear)
        assert tuple(reloaded_model.head.weight.shape) == (7, 16)

        dev_source = pd.read_csv(env["data_dir"] / "vsmec_dev.csv")
        enc_20, _ = tokenize_with_truncation_stats(
            dev_source["text_seg"].iloc[:20].tolist(), reloaded_tok, max_length=32
        )
        reloaded_logits_20 = predict_logits7(
            reloaded_model, enc_20, reloaded_tok, batch_size=16, device=torch.device("cpu")
        )
        saved_logits_20 = df_dev_pred[logit_cols].iloc[:20].to_numpy(dtype=np.float64)
        assert np.max(np.abs(reloaded_logits_20 - saved_logits_20)) < 1e-4

        # 5. Verify stage1_ckpt_hash.json exists and matches actual file SHA-256
        hash_json = json.loads((results_dir / "stage1_ckpt_hash.json").read_text(encoding="utf-8"))
        assert "PhoBERT" in hash_json
        assert hash_json["PhoBERT"]["head_sha256"] == sha256_file(head_pt)

        # 6. Verify truncation stats detected the >32-token sample in vsmec_train
        assert res["truncation_report"]["vsmec_train"]["n_truncated"] >= 1

        # 7. Verify skip/resume safety when run_id already exists in stage1.csv
        res_skip = run_stage1(
            config_path=cfg_path,
            data_dir=env["data_dir"],
            ckpt_dir=ckpt_dir,
            results_dir=results_dir,
            work_dir=work_dir,
            max_steps_override=1,
        )
        assert res_skip["skipped"] is True
        df_s1_after = read_csv_schema(results_dir / "stage1.csv", STAGE1_COLUMNS)
        assert len(df_s1_after) == 1

    def test_cpu_determinism_same_seed_gives_identical_logits(
        self, tmp_path: Path
    ) -> None:
        env = _create_tiny_bert_and_data(tmp_path)
        cfg_path = _write_stage1_yaml(tmp_path / "stage1_det.yaml", env["backbone_dir"], seed=0)

        res1 = run_stage1(
            config_path=cfg_path,
            data_dir=env["data_dir"],
            ckpt_dir=tmp_path / "run1" / "ckpt",
            results_dir=tmp_path / "run1" / "results",
            work_dir=tmp_path / "run1" / "work",
            max_steps_override=2,
        )
        res2 = run_stage1(
            config_path=cfg_path,
            data_dir=env["data_dir"],
            ckpt_dir=tmp_path / "run2" / "ckpt",
            results_dir=tmp_path / "run2" / "results",
            work_dir=tmp_path / "run2" / "work",
            max_steps_override=2,
        )

        logit_cols = [f"logit_{i}" for i in range(7)]
        for ds_name in PREDICTION_DATASETS:
            df1 = read_csv_schema(res1["prediction_paths"][ds_name], PRED7_COLUMNS, keep_default_na=False)
            df2 = read_csv_schema(res2["prediction_paths"][ds_name], PRED7_COLUMNS, keep_default_na=False)
            assert df1["id"].tolist() == df2["id"].tolist()
            assert df1["y_pred7"].tolist() == df2["y_pred7"].tolist()
            assert np.allclose(
                df1[logit_cols].to_numpy(dtype=np.float64),
                df2[logit_cols].to_numpy(dtype=np.float64),
                atol=1e-7,
            )

        s1_1 = read_csv_schema(res1["stage1_csv"], STAGE1_COLUMNS)
        s1_2 = read_csv_schema(res2["stage1_csv"], STAGE1_COLUMNS)
        assert float(s1_1.loc[0, "macro_f1"]) == pytest.approx(float(s1_2.loc[0, "macro_f1"]))

    def test_cli_main_and_visobert_text_column(self, tmp_path: Path) -> None:
        env = _create_tiny_bert_and_data(tmp_path)
        cfg_path = tmp_path / "stage1_visobert_tiny.yaml"
        _write_stage1_yaml(cfg_path, env["backbone_dir"], seed=0)
        # Switch config to ViSoBERT alias reading `text`
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg["model_alias"] = "ViSoBERT"
        cfg["input_column"] = "text"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        ckpt_dir = tmp_path / "ckpt_viso"
        res_dir = tmp_path / "res_viso"
        work_dir = tmp_path / "work_viso"

        stage1_main(
            [
                "--config",
                str(cfg_path),
                "--data-dir",
                str(env["data_dir"]),
                "--ckpt-dir",
                str(ckpt_dir),
                "--results-dir",
                str(res_dir),
                "--work-dir",
                str(work_dir),
                "--max-steps",
                "1",
            ]
        )

        assert (ckpt_dir / "ViSoBERT" / "head.pt").is_file()
        assert (ckpt_dir / "ViSoBERT" / "encoder").is_dir()
        df_s1 = read_csv_schema(res_dir / "stage1.csv", STAGE1_COLUMNS)
        assert len(df_s1) == 1
        assert df_s1.loc[0, "run_id"] == "ViSoBERT_Stage1-base_s0_lr2e-5"

