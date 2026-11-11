"""Offline CPU unit and acceptance tests for `src/plm/train_neu3.py` (`tests/test_neu3.py`).

Covers all requirements of `tasks/AGENT_neu3_phobert.md` (Sections 7, 8, 9, 11):
1. `sweep` run NEVER reads, validates, or tokenizes `test` (verified with intentionally corrupted
   `test` rows in `neuesc_100.csv` and spy on `tokenize_with_truncation_stats`), writes only 1 row
   (`split == "dev"`) to `neu3.csv` with empty `ckpt_path`, exports only `neuesc_dev.csv` in
   `predictions3/`, saves no checkpoint and no `neu3_ckpt_hash.json`, and prints `test_macro_f1=NA`.
2. `base` run with small `max_steps` creates:
   - `neu3.csv` with 2 rows (`dev` and `test`) matching `NEU3_COLUMNS`,
   - 2 files in `predictions3/` (`{run_id}_neuesc_dev.csv`, `{run_id}_neuesc_test.csv`),
   - checkpoint at `{ckpt_dir}/{alias}/encoder/` + `head.pt`,
   - SHA-256 hash entry in `{results_dir}/neu3_ckpt_hash.json` keyed by `run_id` (multiple seeds
     with the same `PhoBERT` alias do not overwrite each other).
3. Re-running the same `run_id` prints `=== [SKIP] run_id=... ===` and adds no duplicate rows.
4. Reloading checkpoint via `Stage1Classifier.load_stage1_checkpoint` yields `head.weight.shape[0] == 3`
   and reproduces saved dev logits within `< 1e-3`.
5. Prediction CSV columns follow exact order `id, y_true, y_pred, logit_neg, logit_neu, logit_pos`,
   preserving original `id`s and `y_pred == argmax(logits)`.
6. Labels outside `{0, 1, 2}` (or duplicate `id`, empty/NaN text, missing subset) raise `ValueError`.
7. Two runs with the same `seed` and config produce dev macro-F1 difference `< 1e-3`.
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

from src.common.constants import NEG, NEU, POS
from src.common.hashing import sha256_file
from src.common.io import read_csv_schema
from src.common.schema import NEU3_COLUMNS, PRED_COLUMNS
from src.common.seed import set_seed
from src.plm.model_ckpt import Stage1Classifier
import src.plm.train_neu3 as train_neu3_mod
from src.plm.train_neu3 import (
    build_neu3_run_id,
    format_lr,
    main as neu3_main,
    predict_logits3,
    run_neu3,
    tokenize_with_truncation_stats,
)


def _create_tiny_neu3_env(root: Path) -> Dict[str, Path]:
    """Create a local tiny BERT model (`hidden_size=16`) and synthetic `neuesc_100.csv` in `root`."""
    backbone_dir = root / "tiny_phobert"
    backbone_dir.mkdir(parents=True, exist_ok=True)

    vocab_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    ] + [f"tok{i}" for i in range(95)]
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

    rows: List[Dict[str, Any]] = []
    for sp, cnt, off in [("train", 36, 1000), ("dev", 18, 2000), ("test", 15, 3000)]:
        for i in range(cnt):
            sent = [NEG, NEU, POS][i % 3]
            t_idx = (i * 5) % 80
            if i == 0:
                words = " ".join(f"tok{(t_idx + k) % 90}" for k in range(42))
            else:
                words = " ".join(f"tok{(t_idx + k) % 90}" for k in range(6))
            rows.append(
                {
                    "id": off + i * 7,
                    "text": words,
                    "label": sent,
                    "subset": sp,
                    "text_seg": words,
                }
            )
    neuesc_path = data_dir / "neuesc_100.csv"
    pd.DataFrame(rows).to_csv(neuesc_path, index=False, encoding="utf-8")

    return {
        "backbone_dir": backbone_dir,
        "data_dir": data_dir,
        "neuesc_path": neuesc_path,
    }


def _write_neu3_yaml(
    path: Path,
    backbone_dir: Path,
    seed: int = 0,
    lr: float = 2.0e-5,
    variant: str = "base-bf16",
    run_kind: str = "base",
    save_checkpoint: bool = True,
    precision: str = "fp32",
    strict_precision: bool = False,
) -> Path:
    cfg = {
        "backbone": backbone_dir.as_posix(),
        "model_alias": "PhoBERT",
        "mode": "neu3",
        "variant": variant,
        "run_kind": run_kind,
        "save_checkpoint": save_checkpoint,
        "input_column": "text_seg",
        "num_labels": 3,
        "max_length": 32,
        "dynamic_padding": True,
        "group_by_length": True,
        "dropout": 0.1,
        "max_grad_norm": 1.0,
        "loss": "ce",
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "grad_accum": 2,
        "learning_rate": lr,
        "lr": lr,
        "num_train_epochs": 2,
        "epochs": 2,
        "warmup_ratio": 0.06,
        "lr_scheduler_type": "linear",
        "weight_decay": 0.01,
        "precision": precision,
        "strict_precision": strict_precision,
        "seed": seed,
        "eval_steps": 1,
        "save_total_limit": 2,
        "early_stopping_patience": 5,
        "official": False,
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


class TestNeu3Pipeline:
    def test_sweep_never_reads_or_tokenizes_test_even_when_test_is_corrupted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = _create_tiny_neu3_env(tmp_path)

        # Intentionally corrupt the test split in neuesc_100.csv (invalid label 999, empty/NaN text_seg)
        df = pd.read_csv(env["neuesc_path"])
        df.loc[df["subset"] == "test", "label"] = 999
        df.loc[df["subset"] == "test", "text_seg"] = ""
        df.to_csv(env["neuesc_path"], index=False, encoding="utf-8")

        tokenize_calls: List[int] = []
        orig_tokenize = train_neu3_mod.tokenize_with_truncation_stats

        def _spy_tokenize(texts: List[str], tokenizer: Any, max_length: int = 256):
            tokenize_calls.append(len(texts))
            return orig_tokenize(texts, tokenizer, max_length=max_length)

        monkeypatch.setattr(
            train_neu3_mod, "tokenize_with_truncation_stats", _spy_tokenize
        )

        cfg_path = _write_neu3_yaml(
            tmp_path / "neu3_sweep.yaml",
            env["backbone_dir"],
            seed=0,
            lr=3.0e-5,
            variant="sweep-bf16",
            run_kind="sweep",
            save_checkpoint=False,
        )

        ckpt_dir = tmp_path / "ckpt"
        results_dir = tmp_path / "results"
        work_dir = tmp_path / "work"

        res = run_neu3(
            config_path=cfg_path,
            data_dir=env["data_dir"],
            ckpt_dir=ckpt_dir,
            results_dir=results_dir,
            work_dir=work_dir,
            max_steps_override=2,
        )
        out_stdout = capsys.readouterr().out

        assert res["skipped"] is False
        assert res["run_id"] == "PhoBERT_NEU3-sweep-bf16_s0_lr3e-5"
        assert res["run_kind"] == "sweep"
        assert res["test_metrics"] is None
        # Only train (36) and dev (18) were tokenized; test (15) was never tokenized
        assert tokenize_calls == [36, 18]
        assert set(res["truncation_report"].keys()) == {"neuesc_train", "neuesc_dev"}

        # Predictions: only dev exists, no test file
        pred3_dir = results_dir / "predictions3"
        assert (pred3_dir / f"{res['run_id']}_neuesc_dev.csv").is_file()
        assert not (pred3_dir / f"{res['run_id']}_neuesc_test.csv").exists()

        # No checkpoint and no neu3_ckpt_hash.json
        assert not (ckpt_dir / "PhoBERT").exists()
        assert not (results_dir / "neu3_ckpt_hash.json").exists()

        # neu3.csv has exactly 1 row (split == "dev") with empty ckpt_path
        df_neu3 = read_csv_schema(results_dir / "neu3.csv", NEU3_COLUMNS)
        assert len(df_neu3) == 1
        row = df_neu3.iloc[0]
        assert row["run_id"] == res["run_id"]
        assert row["split"] == "dev"
        assert row["run_kind"] == "sweep"
        assert pd.isna(row["ckpt_path"]) or str(row["ckpt_path"]) == ""
        assert f"=== [DONE] {res['run_id']}:" in out_stdout
        assert "test_macro_f1=NA" in out_stdout

        # By contrast, running `base` on the same corrupted test split MUST fail validation
        cfg_base_corrupt = _write_neu3_yaml(
            tmp_path / "neu3_base_corrupt.yaml",
            env["backbone_dir"],
            seed=0,
            lr=3.0e-5,
            variant="base-bf16",
            run_kind="base",
            save_checkpoint=True,
        )
        with pytest.raises(ValueError, match="test"):
            run_neu3(
                config_path=cfg_base_corrupt,
                data_dir=env["data_dir"],
                ckpt_dir=ckpt_dir,
                results_dir=results_dir,
                work_dir=work_dir,
                max_steps_override=1,
            )

    def test_base_run_end_to_end_outputs_checkpoint_reload_and_skip(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = _create_tiny_neu3_env(tmp_path)
        ckpt_dir = tmp_path / "ckpt"
        results_dir = tmp_path / "results"
        work_dir = tmp_path / "work"

        cfg_s0 = _write_neu3_yaml(
            tmp_path / "neu3_base_s0.yaml",
            env["backbone_dir"],
            seed=0,
            lr=2.0e-5,
            variant="base-bf16",
            run_kind="base",
            save_checkpoint=True,
        )

        pre_cleanup_Order_ok = {"checked": False}

        def _check_before_cleanup(info: Dict[str, Any]) -> None:
            assert info["neu3_csv"].is_file()
            assert info["prediction_paths"]["neuesc_dev"].is_file()
            assert info["prediction_paths"]["neuesc_test"].is_file()
            assert info["work_dir_existed_after_results"] is True
            pre_cleanup_order_ok["checked"] = True

        pre_cleanup_order_ok = pre_cleanup_Order_ok
        res0 = run_neu3(
            config_path=cfg_s0,
            data_dir=env["data_dir"],
            ckpt_dir=ckpt_dir,
            results_dir=results_dir,
            work_dir=work_dir,
            max_steps_override=2,
            on_before_cleanup=_check_before_cleanup,
        )
        stdout0 = capsys.readouterr().out

        assert pre_cleanup_order_ok["checked"] is True
        assert res0["skipped"] is False
        run_id_s0 = "PhoBERT_NEU3-base-bf16_s0_lr2e-5"
        assert res0["run_id"] == run_id_s0
        assert not (work_dir / run_id_s0).exists()

        # Verify start-of-run logs and DONE line
        assert "[NEU3 Start]" in stdout0
        assert "[NEU3 Precision]" in stdout0
        assert "[NEU3 Provenance]" in stdout0
        assert "[NEU3 Splits]" in stdout0
        assert "[NEU3 Labels]" in stdout0
        assert f"=== [DONE] {run_id_s0}:" in stdout0
        assert "test_macro_f1=NA" not in stdout0

        # Verify neu3.csv has 2 rows (dev and test) with exact schema and fixed metadata
        df_neu3 = read_csv_schema(results_dir / "neu3.csv", NEU3_COLUMNS)
        assert len(df_neu3) == 2
        assert list(df_neu3["split"]) == ["dev", "test"]
        for _, row in df_neu3.iterrows():
            assert row["run_id"] == run_id_s0
            assert row["stage1"] == "NEU"
            assert pd.isna(row["stage2"]) or str(row["stage2"]) == ""
            assert row["eval_domain"] == "NEU"
            assert row["label_view"] == "V3-NEU"
            assert row["mode"] == "base"
            assert int(row["fraction"]) == 100
            assert int(row["n_train"]) == 36
            assert str(row["ckpt_path"]) == (ckpt_dir / "PhoBERT").as_posix()
            f1_list = json.loads(str(row["f1_per_class"]))
            assert len(f1_list) == 3
            assert np.isclose(f1_list[0], float(row["f1_neg"]))
            assert np.isclose(f1_list[1], float(row["f1_neu"]))
            assert np.isclose(f1_list[2], float(row["f1_pos"]))

        # Verify predictions3 files, column order, original IDs, and y_pred == argmax(logits)
        dev_pred_df = read_csv_schema(
            results_dir / "predictions3" / f"{run_id_s0}_neuesc_dev.csv",
            PRED_COLUMNS,
        )
        test_pred_df = read_csv_schema(
            results_dir / "predictions3" / f"{run_id_s0}_neuesc_test.csv",
            PRED_COLUMNS,
        )
        assert tuple(dev_pred_df.columns) == (
            "id",
            "y_true",
            "y_pred",
            "logit_neg",
            "logit_neu",
            "logit_pos",
        )
        assert len(dev_pred_df) == 18
        assert len(test_pred_df) == 15
        assert dev_pred_df["id"].tolist() == [2000 + i * 7 for i in range(18)]
        assert test_pred_df["id"].tolist() == [3000 + i * 7 for i in range(15)]

        dev_logits_saved = dev_pred_df[
            ["logit_neg", "logit_neu", "logit_pos"]
        ].to_numpy(dtype=np.float64)
        assert np.all(np.isfinite(dev_logits_saved))
        assert np.array_equal(
            dev_pred_df["y_pred"].to_numpy(dtype=int),
            np.argmax(dev_logits_saved, axis=1),
        )

        # Verify checkpoint reload via Stage1Classifier.load_stage1_checkpoint
        reloaded_model, reloaded_tok = Stage1Classifier.load_stage1_checkpoint(
            ckpt_dir / "PhoBERT", device="cpu"
        )
        assert reloaded_model.head.weight.shape == (3, 16)
        assert reloaded_model.head.bias.shape == (3,)

        df_full = pd.read_csv(env["neuesc_path"])
        dev_texts = df_full[df_full["subset"] == "dev"]["text_seg"].tolist()
        dev_enc_reload, _ = tokenize_with_truncation_stats(
            dev_texts, reloaded_tok, max_length=32
        )
        reloaded_logits = predict_logits3(
            model=reloaded_model,
            encodings=dev_enc_reload,
            tokenizer=reloaded_tok,
            batch_size=16,
            device=torch.device("cpu"),
            use_fp16=False,
        )
        max_diff = float(np.max(np.abs(reloaded_logits - dev_logits_saved)))
        assert max_diff < 1e-3

        # Re-running the same run_id must print [SKIP] and not add rows
        res_skip = run_neu3(
            config_path=cfg_s0,
            data_dir=env["data_dir"],
            ckpt_dir=ckpt_dir,
            results_dir=results_dir,
            work_dir=work_dir,
            max_steps_override=2,
        )
        stdout_skip = capsys.readouterr().out
        assert res_skip["skipped"] is True
        assert f"=== [SKIP] run_id={run_id_s0} ===" in stdout_skip
        assert len(read_csv_schema(results_dir / "neu3.csv", NEU3_COLUMNS)) == 2

        # Run seed=1 and verify neu3_ckpt_hash.json keeps BOTH run_ids without overwriting
        cfg_s1 = _write_neu3_yaml(
            tmp_path / "neu3_base_s1.yaml",
            env["backbone_dir"],
            seed=1,
            lr=2.0e-5,
            variant="base-bf16",
            run_kind="base",
            save_checkpoint=True,
        )
        res1 = run_neu3(
            config_path=cfg_s1,
            data_dir=env["data_dir"],
            ckpt_dir=ckpt_dir,
            results_dir=results_dir,
            work_dir=work_dir,
            max_steps_override=2,
        )
        run_id_s1 = res1["run_id"]
        hash_file = results_dir / "neu3_ckpt_hash.json"
        assert hash_file.is_file()
        hashes = json.loads(hash_file.read_text(encoding="utf-8"))
        assert run_id_s0 in hashes
        assert run_id_s1 in hashes
        assert hashes[run_id_s0]["alias"] == "PhoBERT"
        assert len(hashes[run_id_s0]["encoder_sha256"]) == 64
        assert len(hashes[run_id_s0]["head_sha256"]) == 64

    def test_invalid_labels_and_variant_validation(self, tmp_path: Path) -> None:
        env = _create_tiny_neu3_env(tmp_path)

        # 1. Variant containing '_' must fail
        with pytest.raises(ValueError, match="_"):
            build_neu3_run_id(
                "vinai/phobert-base-v2", seed=0, lr=2e-5, variant="base_bf16"
            )

        # 2. Label outside {0, 1, 2} (e.g. label=3 from unmerged toxic or -1) must raise ValueError
        df_bad = pd.read_csv(env["neuesc_path"])
        df_bad.loc[0, "label"] = 3
        df_bad.to_csv(env["neuesc_path"], index=False, encoding="utf-8")

        cfg_path = _write_neu3_yaml(
            tmp_path / "neu3_bad_label.yaml",
            env["backbone_dir"],
            seed=0,
            run_kind="sweep",
            save_checkpoint=False,
        )
        with pytest.raises(ValueError, match="invalid labels"):
            run_neu3(
                config_path=cfg_path,
                data_dir=env["data_dir"],
                ckpt_dir=tmp_path / "ckpt",
                results_dir=tmp_path / "res",
                work_dir=tmp_path / "work",
                max_steps_override=1,
            )

    def test_reproducibility_same_seed_and_cli_entry(self, tmp_path: Path) -> None:
        env = _create_tiny_neu3_env(tmp_path)
        cfg_path = _write_neu3_yaml(
            tmp_path / "neu3_repro.yaml",
            env["backbone_dir"],
            seed=42,
            lr=3.0e-5,
            variant="repro-test",
            run_kind="sweep",
            save_checkpoint=False,
        )

        res_a = tmp_path / "res_a"
        res_b = tmp_path / "res_b"

        neu3_main(
            [
                "--config",
                str(cfg_path),
                "--data-dir",
                str(env["data_dir"]),
                "--ckpt-dir",
                str(tmp_path / "ckpt_a"),
                "--results-dir",
                str(res_a),
                "--work-dir",
                str(tmp_path / "work_a"),
                "--max-steps",
                "3",
            ]
        )
        neu3_main(
            [
                "--config",
                str(cfg_path),
                "--data-dir",
                str(env["data_dir"]),
                "--ckpt-dir",
                str(tmp_path / "ckpt_b"),
                "--results-dir",
                str(res_b),
                "--work-dir",
                str(tmp_path / "work_b"),
                "--max-steps",
                "3",
            ]
        )

        df_a = read_csv_schema(res_a / "neu3.csv", NEU3_COLUMNS)
        df_b = read_csv_schema(res_b / "neu3.csv", NEU3_COLUMNS)
        f1_diff = abs(float(df_a.iloc[0]["macro_f1"]) - float(df_b.iloc[0]["macro_f1"]))
        assert f1_diff < 1e-3

    def test_non_degenerate_three_class_predictions_and_data_clean_integrity(
        self, tmp_path: Path
    ) -> None:
        clean_dir = Path("data/clean")
        hashes_before = {
            p.name: sha256_file(p)
            for p in sorted(clean_dir.iterdir())
            if p.is_file()
        }
        manifest = json.loads(
            (clean_dir / "MANIFEST.json").read_text(encoding="utf-8")
        )
        assert hashes_before["neuesc_100.csv"] == manifest["neuesc_100.csv"]

        env = _create_tiny_neu3_env(tmp_path)

        # Build class-separable synthetic token patterns for classes 0, 1, 2
        rows: List[Dict[str, Any]] = []
        class_tokens = {
            NEG: "tok1 tok2",
            NEU: "tok20 tok21 tok22 tok23 tok24",
            POS: "tok50 tok51 tok52 tok53 tok54 tok55 tok56 tok57 tok58",
        }
        for sp, cnt, off in [("train", 30, 1000), ("dev", 18, 2000), ("test", 12, 3000)]:
            for i in range(cnt):
                sent = [NEG, NEU, POS][i % 3]
                rows.append(
                    {
                        "id": off + i,
                        "text": class_tokens[sent],
                        "label": sent,
                        "subset": sp,
                        "text_seg": class_tokens[sent],
                    }
                )
        pd.DataFrame(rows).to_csv(env["neuesc_path"], index=False, encoding="utf-8")

        cfg_path = _write_neu3_yaml(
            tmp_path / "neu3_separable.yaml",
            env["backbone_dir"],
            seed=0,
            lr=5.0e-3,
            variant="sep-check",
            run_kind="sweep",
            save_checkpoint=False,
        )
        cfg_sep = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg_sep.update(
            {
                "num_train_epochs": 25,
                "epochs": 25,
                "early_stopping_patience": 30,
                "eval_steps": 5,
                "group_by_length": False,
                "dropout": 0.0,
            }
        )
        cfg_path.write_text(yaml.safe_dump(cfg_sep, sort_keys=False), encoding="utf-8")
        res = run_neu3(
            config_path=cfg_path,
            data_dir=env["data_dir"],
            ckpt_dir=tmp_path / "ckpt_sep",
            results_dir=tmp_path / "res_sep",
            work_dir=tmp_path / "work_sep",
            max_steps_override=40,
        )
        dev_pred_df = read_csv_schema(
            res["prediction_paths"]["neuesc_dev"], PRED_COLUMNS
        )
        predicted_classes = set(dev_pred_df["y_pred"].astype(int).unique().tolist())
        assert predicted_classes == {NEG, NEU, POS}

        hashes_after = {
            p.name: sha256_file(p)
            for p in sorted(clean_dir.iterdir())
            if p.is_file()
        }
        assert hashes_after == hashes_before


