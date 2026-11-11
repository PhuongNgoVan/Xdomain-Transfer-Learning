"""Offline CPU unit and acceptance tests for ViSoBERT Stage 1 sweep, precision, sidecar, checkpoint controls, and in-memory `emoji_mode='keep'` transformation (`tests/test_stage1_sweep.py`).

Covers all requirements of `tasks/AGENT_PLAN_visobert-vsmec-7c.md` (T1-T6):
1. Default configuration (without new keys) preserves `run_id == "ViSoBERT_Stage1-base_s0_lr2e-5"`
   and backward-compatible behavior.
2. Invalid `precision` raises `ValueError`; `strict_precision: true` + `bf16` on CPU raises
   `RuntimeError`; `strict_precision: false` + `bf16` on CPU runs and records
   `precision_effective == "fp32"`.
3. Two configurations differing only in `variant` (`emokeep-fp16` vs `emokeep-bf16`) produce
   distinct `run_id`s and do not skip each other. Also verifies `variant` containing `_` raises
   `ValueError`.
4. `sweep: true` does not create or overwrite `stage1_ckpt_hash.json`, and writes hashes into
   `ckpt_hash_sweep.json` keyed by `run_id`.
5. `save_checkpoint: false` skips creating the checkpoint directory and hash entry while still
   exporting all 5 `predictions7` CSVs, `stage1.csv` row, and `runs/<run_id>.json` sidecar from RAM.
6. Sidecar JSON (`<results_dir>/runs/<run_id>.json`) contains all required T2 keys, non-empty
   `eval_history`, integer `n_skipped_optimizer_steps` and `n_nonfinite_loss`, and `report_only`.
7. `build_input_texts`: `keep` preserves emojis and is idempotent; `raw` (or omitted) returns raw
   column unchanged; PhoBERT + `keep` (or invalid `emoji_mode` like `tag`/`remove`) raises `ValueError`.
8. End-to-end ViSoBERT run with `emoji_mode: keep` verifies via spy/mock tokenizer that all 6
   datasets (`vsmec_train`, `vsmec_dev`, `vsmec_test`, `neuesc_dev`, `neuesc_test`, `vsfc_test`)
   receive transformed texts.
9. `compute_emoji_and_normalization_report` runs offline with a dummy tokenizer and returns all
   required T5 fields.
10. Data integrity: SHA-256 of every file in `data/clean/` is identical before and after running
    the tests.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
from transformers import AutoTokenizer
import yaml

from src.common.hashing import sha256_file
from src.common.io import read_csv_schema
from src.common.schema import PRED7_COLUMNS, STAGE1_COLUMNS
from src.data.preprocess import preprocess_text
from src.plm.emoji_report import compute_emoji_and_normalization_report
from src.plm.input_transform import build_input_texts, compute_texts_sha256
from src.plm.stage1 import (
    PREDICTION_DATASETS,
    build_stage1_run_id,
    run_stage1,
)
from tests.test_stage1 import _create_tiny_bert_and_data, _write_stage1_yaml

SIDECAR_REQUIRED_KEYS = {
    "run_id",
    "alias",
    "backbone",
    "lr",
    "seed",
    "variant",
    "precision",
    "precision_effective",
    "emoji_mode",
    "git_commit",
    "data_hash",
    "torch_version",
    "transformers_version",
    "gpu",
    "best_step",
    "best_dev_macro_f1",
    "elapsed_sec",
    "truncation_report",
    "eval_history",
    "n_skipped_optimizer_steps",
    "n_nonfinite_loss",
    "report_only",
}


def _snapshot_data_clean_hashes() -> Dict[str, str]:
    clean_dir = Path("data/clean")
    assert clean_dir.is_dir(), "data/clean directory must exist"
    hashes: Dict[str, str] = {}
    for p in sorted(clean_dir.iterdir()):
        if p.is_file():
            hashes[p.name] = sha256_file(p)
    assert len(hashes) > 0
    return hashes


def _update_yaml(cfg_path: Path, updates: Dict[str, Any]) -> Path:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.update(updates)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def _write_visobert_tiny_yaml(
    cfg_path: Path,
    backbone_dir: Path,
    seed: int = 0,
    max_length: int = 32,
) -> Path:
    _write_stage1_yaml(cfg_path, backbone_dir, seed=seed, max_length=max_length)
    return _update_yaml(
        cfg_path,
        {
            "model_alias": "ViSoBERT",
            "input_column": "text",
            "variant": "base",
        },
    )


class TestStage1SweepAndPrecisionViSoBERT:
    def test_default_config_backward_compatibility_and_sidecar_schema(
        self, tmp_path: Path
    ) -> None:
        env = _create_tiny_bert_and_data(tmp_path)
        cfg_path = _write_visobert_tiny_yaml(
            tmp_path / "stage1_visobert_default.yaml", env["backbone_dir"], seed=0
        )

        ckpt_dir = tmp_path / "ckpt"
        results_dir = tmp_path / "results"
        work_dir = tmp_path / "work"

        res = run_stage1(
            config_path=cfg_path,
            data_dir=env["data_dir"],
            ckpt_dir=ckpt_dir,
            results_dir=results_dir,
            work_dir=work_dir,
            max_steps_override=1,
        )

        assert res["run_id"] == "ViSoBERT_Stage1-base_s0_lr2e-5"
        assert (results_dir / "stage1_ckpt_hash.json").is_file()
        assert not (results_dir / "ckpt_hash_sweep.json").exists()

        sidecar_path = results_dir / "runs" / f"{res['run_id']}.json"
        assert sidecar_path.is_file()
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        assert SIDECAR_REQUIRED_KEYS.issubset(set(sidecar.keys()))
        assert sidecar["run_id"] == "ViSoBERT_Stage1-base_s0_lr2e-5"
        assert sidecar["alias"] == "ViSoBERT"
        assert sidecar["emoji_mode"] == "raw"
        assert sidecar["precision"] == "fp16"
        assert sidecar["precision_effective"] == "fp32"
        assert isinstance(sidecar["eval_history"], list) and len(sidecar["eval_history"]) >= 1
        for ev in sidecar["eval_history"]:
            assert {"step", "epoch", "dev_macro_f1", "dev_loss", "dev_accuracy"}.issubset(
                set(ev.keys())
            )
        assert isinstance(sidecar["n_skipped_optimizer_steps"], int)
        assert isinstance(sidecar["n_nonfinite_loss"], int)
        assert set(sidecar["report_only"].keys()) == {"dev_macro_f1", "test_macro_f1"}

    def test_precision_validation_and_strict_precision_on_cpu(
        self, tmp_path: Path
    ) -> None:
        env = _create_tiny_bert_and_data(tmp_path)
        cfg_path = _write_visobert_tiny_yaml(
            tmp_path / "stage1_prec.yaml", env["backbone_dir"], seed=0
        )

        # 1. Invalid precision -> ValueError
        _update_yaml(cfg_path, {"precision": "fp8"})
        with pytest.raises(ValueError, match="Invalid precision"):
            run_stage1(
                config_path=cfg_path,
                data_dir=env["data_dir"],
                ckpt_dir=tmp_path / "ckpt_invalid",
                results_dir=tmp_path / "res_invalid",
                work_dir=tmp_path / "work_invalid",
                max_steps_override=1,
            )

        # 2. strict_precision: true + bf16 on CPU -> RuntimeError
        _update_yaml(
            cfg_path,
            {"precision": "bf16", "strict_precision": True, "variant": "emokeep-bf16"},
        )
        with pytest.raises(RuntimeError, match="strict_precision=True"):
            run_stage1(
                config_path=cfg_path,
                data_dir=env["data_dir"],
                ckpt_dir=tmp_path / "ckpt_strict",
                results_dir=tmp_path / "res_strict",
                work_dir=tmp_path / "work_strict",
                max_steps_override=1,
            )

        # 3. strict_precision: false + bf16 on CPU -> runs and records precision_effective == "fp32"
        _update_yaml(
            cfg_path,
            {"precision": "bf16", "strict_precision": False, "variant": "emokeep-bf16"},
        )
        res = run_stage1(
            config_path=cfg_path,
            data_dir=env["data_dir"],
            ckpt_dir=tmp_path / "ckpt_fallback",
            results_dir=tmp_path / "res_fallback",
            work_dir=tmp_path / "work_fallback",
            max_steps_override=1,
        )
        assert res["precision"] == "bf16"
        assert res["precision_effective"] == "fp32"
        sidecar = json.loads(res["sidecar_path"].read_text(encoding="utf-8"))
        assert sidecar["precision"] == "bf16"
        assert sidecar["precision_effective"] == "fp32"

        # 4. Variant containing '_' must raise ValueError
        with pytest.raises(ValueError, match="_"):
            build_stage1_run_id("uitnlp/visobert", seed=0, lr=2e-5, variant="emokeep_bf16")

    def test_emokeep_fp16_and_bf16_variants_have_distinct_run_ids_and_do_not_skip_each_other(
        self, tmp_path: Path
    ) -> None:
        env = _create_tiny_bert_and_data(tmp_path)
        results_dir = tmp_path / "shared_results"

        cfg_fp16 = _write_visobert_tiny_yaml(
            tmp_path / "cfg_fp16.yaml", env["backbone_dir"], seed=0
        )
        _update_yaml(
            cfg_fp16,
            {
                "precision": "fp16",
                "variant": "emokeep-fp16",
                "emoji_mode": "keep",
                "strict_precision": False,
                "sweep": True,
                "save_checkpoint": False,
            },
        )

        cfg_bf16 = _write_visobert_tiny_yaml(
            tmp_path / "cfg_bf16.yaml", env["backbone_dir"], seed=0
        )
        _update_yaml(
            cfg_bf16,
            {
                "precision": "bf16",
                "variant": "emokeep-bf16",
                "emoji_mode": "keep",
                "strict_precision": False,
                "sweep": True,
                "save_checkpoint": False,
            },
        )

        res_fp16 = run_stage1(
            config_path=cfg_fp16,
            data_dir=env["data_dir"],
            ckpt_dir=tmp_path / "ckpt_fp16",
            results_dir=results_dir,
            work_dir=tmp_path / "work",
            max_steps_override=20,
        )
        res_bf16 = run_stage1(
            config_path=cfg_bf16,
            data_dir=env["data_dir"],
            ckpt_dir=tmp_path / "ckpt_bf16",
            results_dir=results_dir,
            work_dir=tmp_path / "work",
            max_steps_override=20,
        )

        assert res_fp16["skipped"] is False
        assert res_bf16["skipped"] is False
        assert res_fp16["run_id"] == "ViSoBERT_Stage1-emokeep-fp16_s0_lr2e-5"
        assert res_bf16["run_id"] == "ViSoBERT_Stage1-emokeep-bf16_s0_lr2e-5"
        assert res_fp16["run_id"] != res_bf16["run_id"]

        df_s1 = read_csv_schema(results_dir / "stage1.csv", STAGE1_COLUMNS)
        assert len(df_s1) == 2
        assert set(df_s1["run_id"].tolist()) == {
            "ViSoBERT_Stage1-emokeep-fp16_s0_lr2e-5",
            "ViSoBERT_Stage1-emokeep-bf16_s0_lr2e-5",
        }

    def test_sweep_hash_isolation_and_save_checkpoint_false(
        self, tmp_path: Path
    ) -> None:
        env = _create_tiny_bert_and_data(tmp_path)
        results_dir = tmp_path / "results_sweep"

        # Case A: sweep=True, save_checkpoint=True -> writes ckpt_hash_sweep.json keyed by run_id
        cfg_s0 = _write_visobert_tiny_yaml(
            tmp_path / "cfg_s0.yaml", env["backbone_dir"], seed=0
        )
        _update_yaml(
            cfg_s0,
            {
                "precision": "bf16",
                "variant": "emokeep-bf16",
                "emoji_mode": "keep",
                "strict_precision": False,
                "sweep": True,
                "save_checkpoint": True,
                "learning_rate": 3.0e-5,
                "lr": 3.0e-5,
            },
        )
        ckpt_s0 = tmp_path / "ckpt_s0"
        res_s0 = run_stage1(
            config_path=cfg_s0,
            data_dir=env["data_dir"],
            ckpt_dir=ckpt_s0,
            results_dir=results_dir,
            work_dir=tmp_path / "work",
            max_steps_override=1,
        )

        assert res_s0["run_id"] == "ViSoBERT_Stage1-emokeep-bf16_s0_lr3e-5"
        assert (ckpt_s0 / "ViSoBERT" / "encoder").is_dir()
        assert (ckpt_s0 / "ViSoBERT" / "head.pt").is_file()
        assert not (results_dir / "stage1_ckpt_hash.json").exists()

        sweep_hash_file = results_dir / "ckpt_hash_sweep.json"
        assert sweep_hash_file.is_file()
        sweep_hashes = json.loads(sweep_hash_file.read_text(encoding="utf-8"))
        assert res_s0["run_id"] in sweep_hashes
        assert "ViSoBERT" not in sweep_hashes

        # Case B: sweep=True, save_checkpoint=False -> no checkpoint directory created,
        # but all 5 predictions7 files and sidecar exist
        cfg_s1 = _write_visobert_tiny_yaml(
            tmp_path / "cfg_s1.yaml", env["backbone_dir"], seed=1
        )
        _update_yaml(
            cfg_s1,
            {
                "precision": "bf16",
                "variant": "emokeep-bf16",
                "emoji_mode": "keep",
                "strict_precision": False,
                "sweep": True,
                "save_checkpoint": False,
                "learning_rate": 3.0e-5,
                "lr": 3.0e-5,
            },
        )
        ckpt_s1 = tmp_path / "ckpt_s1"
        res_s1 = run_stage1(
            config_path=cfg_s1,
            data_dir=env["data_dir"],
            ckpt_dir=ckpt_s1,
            results_dir=results_dir,
            work_dir=tmp_path / "work",
            max_steps_override=1,
        )

        assert res_s1["run_id"] == "ViSoBERT_Stage1-emokeep-bf16_s1_lr3e-5"
        assert not ckpt_s1.exists()
        for ds_name in PREDICTION_DATASETS:
            p = res_s1["prediction_paths"][ds_name]
            assert p.is_file()
            df_pred = read_csv_schema(p, PRED7_COLUMNS, keep_default_na=False)
            assert len(df_pred) > 0
        assert res_s1["sidecar_path"].is_file()

        sweep_hashes_after = json.loads(sweep_hash_file.read_text(encoding="utf-8"))
        assert set(sweep_hashes_after.keys()) == {res_s0["run_id"]}

    def test_input_transform_keep_idempotent_raw_and_phobert_guard(self) -> None:
        sample_df = pd.DataFrame(
            {
                "id": [1, 2],
                "text": [
                    "Đẹpppp quáaaa 😍 😭 :((",
                    "Buồnnnn thật sự 😡 !!!",
                ],
                "text_seg": [
                    "đẹp quá emo_love emo_sad emo_sad",
                    "buồn thật_sự emo_anger !",
                ],
            }
        )

        # 1. Omitted or 'raw' returns exact column without modification
        raw_viso = build_input_texts(sample_df, input_col="text", alias="ViSoBERT", emoji_mode=None)
        assert raw_viso == sample_df["text"].tolist()
        assert build_input_texts(sample_df, input_col="text", alias="ViSoBERT", emoji_mode="raw") == raw_viso

        raw_pho = build_input_texts(sample_df, input_col="text_seg", alias="PhoBERT", emoji_mode=None)
        assert raw_pho == sample_df["text_seg"].tolist()

        # 2. 'keep' on ViSoBERT preserves emojis, normalizes NFC/lowercase/repeats, and is idempotent
        keep_viso = build_input_texts(sample_df, input_col="text", alias="ViSoBERT", emoji_mode="keep")
        assert keep_viso == [
            "đẹp quá 😍 😭 :((",
            "buồn thật sự 😡 !",
        ]
        assert "😍" in keep_viso[0] and "😭" in keep_viso[0] and "😡" in keep_viso[1]
        keep_again = build_input_texts(
            pd.DataFrame({"text": keep_viso}),
            input_col="text",
            alias="ViSoBERT",
            emoji_mode="keep",
        )
        assert keep_again == keep_viso

        # 3. PhoBERT + 'keep' raises ValueError
        with pytest.raises(ValueError, match="ViSoBERT"):
            build_input_texts(sample_df, input_col="text_seg", alias="PhoBERT", emoji_mode="keep")

        # 4. Unsupported emoji_mode ('tag', 'remove', etc.) raises ValueError
        for bad_mode in ("tag", "remove", "invalid"):
            with pytest.raises(ValueError, match="Unsupported Stage 1 emoji_mode"):
                build_input_texts(sample_df, input_col="text", alias="ViSoBERT", emoji_mode=bad_mode)

    def test_visobert_keep_transforms_all_6_splits_before_tokenizer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = _create_tiny_bert_and_data(tmp_path)
        data_dir = env["data_dir"]

        # Inject unnormalized text (uppercase + repeated letters + emoji) into row 0 of each split
        marker_raw = "ĐẸPPPP QUÁAAA 😍"
        marker_keep = preprocess_text(marker_raw, emoji_mode="keep")
        assert marker_keep == "đẹp quá 😍" and marker_keep != marker_raw

        for csv_name in (
            "vsmec_train.csv",
            "vsmec_dev.csv",
            "vsmec_test.csv",
            "vsfc_test_clean.csv",
        ):
            p = data_dir / csv_name
            df = pd.read_csv(p)
            df.loc[0, "text"] = f"{marker_raw} {df.loc[0, 'text']}"
            df.to_csv(p, index=False, encoding="utf-8")

        neuesc_path = data_dir / "neuesc_100.csv"
        df_neu = pd.read_csv(neuesc_path)
        dev_first_idx = df_neu.index[df_neu["subset"] == "dev"][0]
        test_first_idx = df_neu.index[df_neu["subset"] == "test"][0]
        df_neu.loc[dev_first_idx, "text"] = f"{marker_raw} {df_neu.loc[dev_first_idx, 'text']}"
        df_neu.loc[test_first_idx, "text"] = f"{marker_raw} {df_neu.loc[test_first_idx, 'text']}"
        df_neu.to_csv(neuesc_path, index=False, encoding="utf-8")

        # Wrap AutoTokenizer.from_pretrained to spy on all batch text lists passed to tokenizer(...)
        real_from_pretrained = AutoTokenizer.from_pretrained
        captured_batches: List[List[str]] = []

        class SpyTokenizerWrapper:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __call__(self, texts: Any, *args: Any, **kwargs: Any) -> Any:
                if isinstance(texts, list) and kwargs.get("truncation") is True:
                    captured_batches.append(list(texts))
                return self._inner(texts, *args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        monkeypatch.setattr(
            AutoTokenizer,
            "from_pretrained",
            lambda *a, **kw: SpyTokenizerWrapper(real_from_pretrained(*a, **kw)),
        )

        cfg_path = _write_visobert_tiny_yaml(
            tmp_path / "visobert_keep.yaml", env["backbone_dir"], seed=0
        )
        _update_yaml(
            cfg_path,
            {
                "precision": "fp16",
                "variant": "emokeep-fp16",
                "emoji_mode": "keep",
                "strict_precision": False,
                "sweep": True,
                "save_checkpoint": False,
            },
        )

        res = run_stage1(
            config_path=cfg_path,
            data_dir=data_dir,
            ckpt_dir=tmp_path / "ckpt",
            results_dir=tmp_path / "results",
            work_dir=tmp_path / "work",
            max_steps_override=1,
        )

        # Exactly 6 splits tokenized (`vsmec_train`, `vsmec_dev`, `vsmec_test`, `neuesc_dev`, `neuesc_test`, `vsfc_test`)
        assert len(captured_batches) == 6
        for b in captured_batches:
            assert b[0].startswith("đẹp quá 😍")
            assert "ĐẸPPPP" not in b[0]

        sidecar = json.loads(res["sidecar_path"].read_text(encoding="utf-8"))
        assert sidecar["emoji_mode"] == "keep"
        expected_splits = {
            "vsmec_train",
            "vsmec_dev",
            "vsmec_test",
            "neuesc_dev",
            "neuesc_test",
            "vsfc_test",
        }
        assert set(sidecar["transformed_text_sha256"].keys()) == expected_splits
        for sp_name, b in zip(
            (
                "vsmec_train",
                "vsmec_dev",
                "vsmec_test",
                "neuesc_dev",
                "neuesc_test",
                "vsfc_test",
            ),
            captured_batches,
        ):
            assert sidecar["transformed_text_sha256"][sp_name] == compute_texts_sha256(b)
            assert sidecar["truncation_report"][sp_name]["text_sha256"] == compute_texts_sha256(b)

    def test_emoji_report_offline_with_tiny_tokenizer(self, tmp_path: Path) -> None:
        env = _create_tiny_bert_and_data(tmp_path)
        tok = AutoTokenizer.from_pretrained(str(env["backbone_dir"]))
        report = compute_emoji_and_normalization_report(
            data_dir=Path("data/clean"),
            tokenizer=tok,
            backbone="dummy-visobert",
        )
        assert report["n_vsmec_train_total"] == 5542
        assert report["n_vsmec_train_with_unicode_emoji"] > 0
        assert report["n_vsmec_train_lost_emoji_after_keep"] == 0
        assert report["n_vsmec_train_changed_by_keep"] > 0

    def test_data_clean_files_remain_bit_identical(self, tmp_path: Path) -> None:
        before_hashes = _snapshot_data_clean_hashes()

        env = _create_tiny_bert_and_data(tmp_path)
        cfg_path = _write_visobert_tiny_yaml(
            tmp_path / "stage1_imm.yaml", env["backbone_dir"], seed=0
        )
        _update_yaml(
            cfg_path,
            {
                "precision": "fp16",
                "variant": "emokeep-fp16",
                "emoji_mode": "keep",
                "strict_precision": False,
                "sweep": True,
                "save_checkpoint": False,
            },
        )
        run_stage1(
            config_path=cfg_path,
            data_dir=env["data_dir"],
            ckpt_dir=tmp_path / "ckpt",
            results_dir=tmp_path / "results",
            work_dir=tmp_path / "work",
            max_steps_override=1,
        )

        after_hashes = _snapshot_data_clean_hashes()
        assert before_hashes == after_hashes
