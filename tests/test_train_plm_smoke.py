"""End-to-end offline CPU smoke tests for `src/plm/train_plm.py` and `src/plm/predict.py` (`tests/test_train_plm_smoke.py`).

Covers all requirements of Task 4 in `tasks/CV_P8_stage1.md`:
- Initializes a tiny offline BERT model (`hidden_size=32, num_hidden_layers=2, num_attention_heads=2, vocab_size=256`),
  local vocabulary/tokenizer, 3x7 `wtrans.json`, and ~200 synthetic samples in `tmp_path` (100% offline).
- Tests all 4 training modes/heads end-to-end:
  1. `mode: stage1` (7 classes on VSMEC, verifies `STAGE1_COLUMNS`, `PRED7_COLUMNS`, and saved Stage 1 checkpoint)
  2. `mode: target` with `head: linear3`
  3. `mode: target` with `head: init` (`B4-init`, verifies `*_init_diag.json` and 2 rows `plain` + `prior` for `loss: la`)
  4. `mode: target` with `head: gesf` (`B5`, verifies `*_gesf_trace.csv` and `eval_test: true` generating `neuesc_test` & `vsfc_test`)
- Verifies:
  - `eval_test: false` never generates test prediction files (`neuesc_test`, `vsfc_test`).
  - `keep_ckpt: true` preserves the best checkpoint directory (and removes intermediate `checkpoint-*`);
    `keep_ckpt: false` removes the checkpoint directory completely.
  - Running twice with the same `seed` on CPU produces identical results and logits.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from src.common.constants import EMOTIONS7, HARD_MERGE_RULE, LABELS3, NEG, NEU, POS
from src.common.io import read_csv_schema
from src.common.schema import (
    PRED7_COLUMNS,
    PRED_COLUMNS,
    RESULT_COLUMNS,
    STAGE1_COLUMNS,
)
from src.common.seed import set_seed

pytest.importorskip("src.plm.callbacks")
pytest.importorskip("src.plm.predict")
pytest.importorskip("src.plm.train_plm")
from src.plm.callbacks import GESF_TRACE_COLUMNS
from src.plm.predict import export_stage1_logits_cli, resolve_text_column
from src.plm.train_plm import (
    enforce_git_discipline,
    infer_model_and_run_id,
    load_train_config,
    run_train_plm,
)


@pytest.fixture(scope="module")
def smoke_env(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """Create an isolated offline workspace with a tiny BERT backbone, tokenizer, wtrans, and ~200 samples."""
    root = tmp_path_factory.mktemp("plm_smoke")
    backbone_dir = root / "tiny_phobert"
    backbone_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create 256-token vocabulary file for BertTokenizer
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    domain_words = [
        "môn_học", "giảng_viên", "sinh_viên", "rất", "tốt", "hay", "tệ", "buồn",
        "vui", "sợ", "giận", "ghét", "bình_thường", "ổn", "thích", "kém",
        "emo_joy", "emo_sad", "emo_anger", "emo_fear", "emo_disgust", "emo_surprise",
        "nhiệt_tình", "chậm_trễ", "học_tập", "bài_giảng", "điểm", "thi", "lớp", "trường",
    ]
    filler_tokens = [f"tok_{i}" for i in range(256 - len(special_tokens) - len(domain_words))]
    vocab_list = special_tokens + domain_words + filler_tokens
    assert len(vocab_list) == 256

    vocab_file = backbone_dir / "vocab.txt"
    vocab_file.write_text("\n".join(vocab_list) + "\n", encoding="utf-8")

    tokenizer = BertTokenizer(
        vocab_file=str(vocab_file),
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(str(backbone_dir))

    set_seed(42)
    config = BertConfig(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        pad_token_id=0,
        num_labels=len(EMOTIONS7),
    )
    base_model = BertForSequenceClassification(config)
    base_model.save_pretrained(str(backbone_dir))

    # 2. Save 3x7 W_trans matrix
    wtrans_mat = np.zeros((len(LABELS3), len(EMOTIONS7)), dtype=np.float64)
    for emo_id, sent_id in HARD_MERGE_RULE.items():
        wtrans_mat[sent_id, emo_id] = 1.0
    wtrans_path = root / "wtrans.json"
    wtrans_path.write_text(json.dumps(wtrans_mat.tolist()), encoding="utf-8")

    # 3. Generate ~200 synthetic samples for VSMEC (7 classes), NEU-ESC (3 classes), and VSFC (3 classes)
    clean_dir = root / "data" / "clean"
    splits_dir = root / "data" / "splits"
    clean_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    def _make_vsmec_df(n_rows: int, subset_name: str, start_id: int) -> pd.DataFrame:
        rows = []
        for i in range(n_rows):
            lbl7 = i % len(EMOTIONS7)
            w1 = domain_words[(i + lbl7) % len(domain_words)]
            w2 = domain_words[(i * 3 + lbl7) % len(domain_words)]
            txt = f"{w1.replace('_', ' ')} {w2.replace('_', ' ')} tok_{i % 50}"
            txt_seg = f"{w1} {w2} tok_{i % 50}"
            rows.append(
                {
                    "id": start_id + i,
                    "text": txt,
                    "label": int(lbl7),
                    "subset": subset_name,
                    "text_seg": txt_seg,
                }
            )
        return pd.DataFrame(rows)

    _make_vsmec_df(120, "train", 0).to_csv(clean_dir / "vsmec_train.csv", index=False, lineterminator="\n")
    _make_vsmec_df(40, "dev", 1000).to_csv(clean_dir / "vsmec_dev.csv", index=False, lineterminator="\n")
    _make_vsmec_df(40, "test", 2000).to_csv(clean_dir / "vsmec_test.csv", index=False, lineterminator="\n")

    # NEU-ESC 200 samples (120 train, 40 dev, 40 test) with realistic class imbalance (NEG > NEU > POS)
    label3_cycle = [NEG, NEG, NEG, NEU, NEU, POS]
    neuesc_rows = []
    for i in range(200):
        if i < 120:
            sub = "train"
        elif i < 160:
            sub = "dev"
        else:
            sub = "test"
        lbl3 = label3_cycle[i % len(label3_cycle)]
        w1 = domain_words[(i + lbl3 * 2) % len(domain_words)]
        w2 = domain_words[(i * 2 + lbl3) % len(domain_words)]
        neuesc_rows.append(
            {
                "id": i,
                "text": f"{w1.replace('_', ' ')} {w2.replace('_', ' ')} tok_{i % 40}",
                "label": int(lbl3),
                "subset": sub,
                "text_seg": f"{w1} {w2} tok_{i % 40}",
            }
        )
    df_neuesc_clean = pd.DataFrame(neuesc_rows)
    df_neuesc_clean.to_csv(clean_dir / "neuesc_100.csv", index=False, lineterminator="\n")

    # Split files in `data/splits/` strictly have `id, text, label, subset`
    df_split = df_neuesc_clean.loc[:, ["id", "text", "label", "subset"]]
    df_split.to_csv(splits_dir / "neuesc_5_s0.csv", index=False, lineterminator="\n")
    df_split.to_csv(splits_dir / "neuesc_100.csv", index=False, lineterminator="\n")

    # VSFC OOD test (40 samples)
    vsfc_rows = []
    for i in range(40):
        lbl3 = label3_cycle[i % len(label3_cycle)]
        w1 = domain_words[(i + 5) % len(domain_words)]
        vsfc_rows.append(
            {
                "id": 5000 + i,
                "text": f"giảng viên {w1.replace('_', ' ')} tok_{i % 30}",
                "label": int(lbl3),
                "subset": "test",
                "text_seg": f"giảng_viên {w1} tok_{i % 30}",
            }
        )
    pd.DataFrame(vsfc_rows).to_csv(clean_dir / "vsfc_test_clean.csv", index=False, lineterminator="\n")

    return {
        "root": root,
        "backbone_dir": backbone_dir,
        "wtrans_path": wtrans_path,
        "clean_dir": clean_dir,
        "splits_dir": splits_dir,
    }


def test_smoke_1_stage1_and_predict_cli(smoke_env: Dict[str, Path], tmp_path: Path) -> None:
    """Case 1: `mode: stage1` (7-class VSMEC), keeps checkpoint, and exports 7-class logits via predict.py."""
    results_dir = tmp_path / "results"
    ckpt_dir = tmp_path / "checkpoints" / "stage1" / "PhoBERT_Stage1_VSMEC"

    cfg = {
        "run_id": "PhoBERT_Stage1_VSMEC",
        "backbone": str(smoke_env["backbone_dir"]),
        "text_column": "text_seg",
        "mode": "stage1",
        "loss": "ce",
        "lr": 1e-3,
        "seed": 42,
        "epochs": 2,
        "batch_size": 16,
        "grad_accum": 1,
        "max_length": 32,
        "precision": "fp16",
        "group_by_length": True,
        "early_stop": {"metric": "macro_f1_dev", "min_steps": 2, "patience": 2},
        "keep_ckpt": True,
        "official": False,
        "clean_dir": str(smoke_env["clean_dir"]),
        "splits_dir": str(smoke_env["splits_dir"]),
        "results_dir": str(results_dir),
        "predictions7_dir": str(results_dir / "predictions7"),
        "ckpt_dir": str(ckpt_dir),
    }

    res = run_train_plm(cfg, base_config_path=None)
    assert res["status"] == "completed"
    assert res["best_step"] >= 1

    # Verify STAGE1_COLUMNS in results/stage1.csv
    stage1_csv = results_dir / "stage1.csv"
    df_s1 = read_csv_schema(stage1_csv, STAGE1_COLUMNS, strict_order=True)
    assert len(df_s1) == 1
    assert df_s1.loc[0, "run_id"] == "PhoBERT_Stage1_VSMEC"

    # Verify PRED7_COLUMNS in predictions7/{run_id}_vsmec_{dev,test}.csv
    for ds_name in ("vsmec_dev", "vsmec_test"):
        pred7_file = results_dir / "predictions7" / f"PhoBERT_Stage1_VSMEC_{ds_name}.csv"
        df_p7 = read_csv_schema(pred7_file, PRED7_COLUMNS, strict_order=True)
        assert len(df_p7) == 40
        assert not df_p7["y_true7"].isna().any()

    # Verify keep_ckpt=True kept best checkpoint and cleaned intermediate checkpoint-* dirs
    assert ckpt_dir.is_dir()
    assert (ckpt_dir / "SHA256SUMS.json").is_file()
    assert not list(ckpt_dir.glob("checkpoint-*"))

    # Verify standalone predict.py CLI exporting 7-class logits on NEU-ESC test (y_true7 should be empty)
    exported_path = export_stage1_logits_cli(
        stage1_ckpt=ckpt_dir,
        input_csv=smoke_env["splits_dir"] / "neuesc_5_s0.csv",
        run_id="Hard_PhoBERT",
        dataset_name="neuesc_test",
        predictions7_dir=results_dir / "predictions7",
        subset="test",
        backbone=str(smoke_env["backbone_dir"]),
        text_column="text_seg",
        max_length=32,
        batch_size=16,
        device="cpu",
        clean_dir=smoke_env["clean_dir"],
    )
    df_exported = pd.read_csv(exported_path, keep_default_na=False)
    assert tuple(df_exported.columns) == PRED7_COLUMNS
    assert len(df_exported) == 40
    assert all(v == "" for v in df_exported["y_true7"])


def test_smoke_2_target_linear3_eval_test_false_and_determinism(
    smoke_env: Dict[str, Path], tmp_path: Path
) -> None:
    """Case 2: `mode: target`, `head: linear3`, `eval_test: false`, `keep_ckpt: false`, and CPU seed determinism."""
    for attempt in (1, 2):
        run_root = tmp_path / f"attempt_{attempt}"
        results_dir = run_root / "results"
        ckpt_dir = run_root / "checkpoints" / "PhoBERT_Tgt_run"

        cfg = {
            "run_id": "PhoBERT_Tgt-base_f5_s0_lr1e-3",
            "model": "PhoBERT_Tgt",
            "backbone": str(smoke_env["backbone_dir"]),
            "text_column": "text_seg",
            "mode": "target",
            "head": "linear3",
            "loss": "ce",
            "lr": 1e-3,
            "fraction": 5,
            "seed": 0,
            "epochs": 2,
            "batch_size": 16,
            "grad_accum": 2,
            "max_length": 32,
            "precision": "fp16",
            "group_by_length": True,
            "early_stop": {"metric": "val_loss", "min_steps": 2, "patience": 2},
            "eval_test": False,
            "keep_ckpt": False,
            "official": False,
            "clean_dir": str(smoke_env["clean_dir"]),
            "splits_dir": str(smoke_env["splits_dir"]),
            "results_dir": str(results_dir),
            "predictions_dir": str(results_dir / "predictions"),
            "ckpt_dir": str(ckpt_dir),
        }
        res = run_train_plm(cfg, base_config_path=None)
        assert res["status"] == "completed"

        # eval_test: false -> ONLY neuesc_dev prediction file exists; NO test files
        pred_dir = results_dir / "predictions"
        assert (pred_dir / "PhoBERT_Tgt-base_f5_s0_lr1e-3_neuesc_dev.csv").is_file()
        assert not (pred_dir / "PhoBERT_Tgt-base_f5_s0_lr1e-3_neuesc_test.csv").exists()
        assert not (pred_dir / "PhoBERT_Tgt-base_f5_s0_lr1e-3_vsfc_test.csv").exists()

        # keep_ckpt: false -> checkpoint directory is completely deleted
        assert not ckpt_dir.exists()

        df_res = read_csv_schema(results_dir / "phuong.csv", RESULT_COLUMNS, strict_order=True)
        assert len(df_res) == 1
        assert df_res.loc[0, "dataset"] == "neuesc_dev"
        assert df_res.loc[0, "inference_mode"] == "plain"

    # Verify exact CPU determinism across both runs with same seed
    pred_1 = pd.read_csv(
        tmp_path / "attempt_1" / "results" / "predictions" / "PhoBERT_Tgt-base_f5_s0_lr1e-3_neuesc_dev.csv"
    )
    pred_2 = pd.read_csv(
        tmp_path / "attempt_2" / "results" / "predictions" / "PhoBERT_Tgt-base_f5_s0_lr1e-3_neuesc_dev.csv"
    )
    pd.testing.assert_frame_equal(pred_1, pred_2, check_exact=False, atol=1e-6, rtol=1e-6)


def test_smoke_3_target_b4_init_and_4_b5_gesf(smoke_env: Dict[str, Path], tmp_path: Path) -> None:
    """Cases 3 & 4: `head: init` (B4-init with LA) and `head: gesf` (B5 with LA, trace CSV, and eval_test=True)."""
    results_dir = tmp_path / "results"
    stage1_ckpt_dir = tmp_path / "checkpoints" / "stage1" / "PhoBERT_Stage1_VSMEC"

    # First produce a quick Stage 1 checkpoint to initialize B4-init and B5 GESF
    run_train_plm(
        {
            "run_id": "PhoBERT_Stage1_VSMEC",
            "backbone": str(smoke_env["backbone_dir"]),
            "mode": "stage1",
            "loss": "ce",
            "lr": 1e-3,
            "seed": 0,
            "epochs": 1,
            "batch_size": 32,
            "grad_accum": 1,
            "max_length": 32,
            "keep_ckpt": True,
            "clean_dir": str(smoke_env["clean_dir"]),
            "splits_dir": str(smoke_env["splits_dir"]),
            "results_dir": str(results_dir),
            "predictions7_dir": str(results_dir / "predictions7"),
            "ckpt_dir": str(stage1_ckpt_dir),
        },
        base_config_path=None,
    )
    assert stage1_ckpt_dir.is_dir()

    # Case 3: B4-init (`head: init`, `loss: la`, `eval_test: false`)
    cfg_init = {
        "run_id": "PhoBERT_HR_Init",
        "model": "PhoBERT_HR_Init",
        "backbone": str(smoke_env["backbone_dir"]),
        "stage1_ckpt": str(stage1_ckpt_dir),
        "wtrans_path": str(smoke_env["wtrans_path"]),
        "wtrans_norm": "mean",
        "mode": "target",
        "head": "init",
        "loss": "la",
        "tau": 1.0,
        "lr": 1e-3,
        "fraction": 5,
        "seed": 0,
        "epochs": 2,
        "batch_size": 16,
        "grad_accum": 1,
        "max_length": 32,
        "eval_test": False,
        "keep_ckpt": False,
        "clean_dir": str(smoke_env["clean_dir"]),
        "splits_dir": str(smoke_env["splits_dir"]),
        "results_dir": str(results_dir),
        "predictions_dir": str(results_dir / "predictions"),
        "diagnostics_dir": str(results_dir / "diagnostics"),
        "ckpt_dir": str(tmp_path / "checkpoints" / "PhoBERT_HR_Init"),
    }
    res_init = run_train_plm(cfg_init, base_config_path=None)
    assert res_init["status"] == "completed"

    # Verify *_init_diag.json was generated before fine-tuning
    diag_json = results_dir / "diagnostics" / "PhoBERT_HR_Init_init_diag.json"
    assert diag_json.is_file()
    diag_data = json.loads(diag_json.read_text(encoding="utf-8"))
    assert "macro_f1_init" in diag_data
    assert "logit_var_init" in diag_data

    # Verify 2 rows (`plain` and `prior`) were written to results/phuong.csv for B4-init (`loss: la`)
    df_phuong = read_csv_schema(results_dir / "phuong.csv", RESULT_COLUMNS, strict_order=True)
    df_init_rows = df_phuong[df_phuong["run_id"] == "PhoBERT_HR_Init"]
    assert len(df_init_rows) == 2
    assert set(df_init_rows["inference_mode"]) == {"plain", "prior"}

    # Verify automatic skip when running the same run_id again
    res_skip = run_train_plm(cfg_init, base_config_path=None)
    assert res_skip["status"] == "skipped"

    # Case 4: B5 GESF (`head: gesf`, `loss: la`, `emo_input: p_emo`, `wtrans_norm: auto`, `alpha_init: auto`, `eval_test: true`)
    gesf_ckpt_dir = tmp_path / "checkpoints" / "PhoBERT_GESF_LA"
    cfg_gesf = {
        "run_id": "PhoBERT_GESF_LA",
        "model": "PhoBERT_GESF_LA",
        "backbone": str(smoke_env["backbone_dir"]),
        "stage1_ckpt": str(stage1_ckpt_dir),
        "wtrans_path": str(smoke_env["wtrans_path"]),
        "wtrans_norm": "auto",
        "emo_input": "p_emo",
        "alpha_init": "auto",
        "detach_emo": False,
        "mode": "target",
        "head": "gesf",
        "loss": "la",
        "tau": 1.0,
        "lr": 1e-3,
        "fraction": 5,
        "seed": 0,
        "epochs": 2,
        "batch_size": 16,
        "grad_accum": 1,
        "max_length": 32,
        "eval_test": True,
        "keep_ckpt": True,
        "clean_dir": str(smoke_env["clean_dir"]),
        "splits_dir": str(smoke_env["splits_dir"]),
        "results_dir": str(results_dir),
        "predictions_dir": str(results_dir / "predictions"),
        "diagnostics_dir": str(results_dir / "diagnostics"),
        "ckpt_dir": str(gesf_ckpt_dir),
    }
    res_gesf = run_train_plm(cfg_gesf, base_config_path=None)
    assert res_gesf["status"] == "completed"

    # Verify *_gesf_trace.csv was generated with valid columns and non-empty rows
    trace_csv = results_dir / "diagnostics" / "PhoBERT_GESF_LA_gesf_trace.csv"
    assert trace_csv.is_file()
    df_trace = read_csv_schema(trace_csv, GESF_TRACE_COLUMNS, strict_order=True)
    assert len(df_trace) >= 2
    assert (df_trace["p_emo_entropy"] > 0.0).all()
    assert (df_trace["w_emo_grad_norm"] > 0.0).any()

    # Verify eval_test=True produced predictions for neuesc_dev, neuesc_test, and vsfc_test
    for ds_name in ("neuesc_dev", "neuesc_test", "vsfc_test"):
        pred_file = results_dir / "predictions" / f"PhoBERT_GESF_LA_{ds_name}.csv"
        df_p = read_csv_schema(pred_file, PRED_COLUMNS, strict_order=True)
        assert len(df_p) == 40

    # Verify 6 rows (`plain` and `prior` across 3 datasets) added for PhoBERT_GESF_LA
    df_phuong_after = read_csv_schema(results_dir / "phuong.csv", RESULT_COLUMNS, strict_order=True)
    df_gesf_rows = df_phuong_after[df_phuong_after["run_id"] == "PhoBERT_GESF_LA"]
    assert len(df_gesf_rows) == 6
    assert set(df_gesf_rows["dataset"]) == {"neuesc_dev", "neuesc_test", "vsfc_test"}
    assert set(df_gesf_rows["inference_mode"]) == {"plain", "prior"}

    # Verify keep_ckpt=True preserved GESF best checkpoint and deleted intermediate checkpoint-*
    assert gesf_ckpt_dir.is_dir()
    assert (gesf_ckpt_dir / "SHA256SUMS.json").is_file()
    assert not list(gesf_ckpt_dir.glob("checkpoint-*"))


def test_smoke_5_exp_yaml_configs_and_git_discipline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the 4 YAML files in configs/exp/ and official=True git dirty guard."""
    c_s1_pho = load_train_config("configs/exp/stage1_phobert.yaml")
    assert c_s1_pho["run_id"] == "PhoBERT_Stage1_VSMEC"
    assert c_s1_pho["backbone"] == "vinai/phobert-base-v2"
    assert resolve_text_column(c_s1_pho["backbone"], c_s1_pho.get("text_column")) == "text_seg"
    assert c_s1_pho["mode"] == "stage1"
    assert c_s1_pho["loss"] == "ce"
    assert c_s1_pho["keep_ckpt"] is True
    assert c_s1_pho["official"] is False

    c_s1_viso = load_train_config("configs/exp/stage1_visobert.yaml")
    assert c_s1_viso["run_id"] == "ViSoBERT_Stage1_VSMEC"
    assert c_s1_viso["backbone"] == "uitnlp/visobert"
    assert resolve_text_column(c_s1_viso["backbone"], c_s1_viso.get("text_column")) == "text"
    assert c_s1_viso["mode"] == "stage1"
    assert c_s1_viso["loss"] == "ce"
    assert c_s1_viso["keep_ckpt"] is True
    assert c_s1_viso["official"] is False

    c_b4_init = load_train_config("configs/exp/b4_init_phobert.yaml")
    # run_id không cố định trong YAML: mỗi (mốc, seed, lr) phải có run_id riêng (hợp đồng 2.2)
    assert "run_id" not in c_b4_init
    assert infer_model_and_run_id(c_b4_init)[0] == "PhoBERT_HR_Init"
    assert infer_model_and_run_id({**c_b4_init, "seed": 3, "fraction": 5})[1] != infer_model_and_run_id(
        {**c_b4_init, "seed": 4, "fraction": 5}
    )[1]
    assert c_b4_init["mode"] == "target"
    assert c_b4_init["head"] == "init"
    assert c_b4_init["loss"] == "la"
    assert float(c_b4_init["tau"]) == 1.0
    assert c_b4_init["wtrans_norm"] == "mean"
    assert c_b4_init["official"] is False

    c_b5_gesf = load_train_config("configs/exp/b5_gesf_phobert.yaml")
    assert "run_id" not in c_b5_gesf
    assert infer_model_and_run_id(c_b5_gesf)[0] == "PhoBERT_GESF_LA"
    assert c_b5_gesf["mode"] == "target"

    c_b4_init_viso = load_train_config("configs/exp/b4_init_visobert.yaml")
    assert "run_id" not in c_b4_init_viso
    assert infer_model_and_run_id(c_b4_init_viso)[0] == "ViSoBERT_HR_Init"
    assert c_b4_init_viso["backbone"] == "uitnlp/visobert"
    assert resolve_text_column(c_b4_init_viso["backbone"], c_b4_init_viso.get("text_column")) == "text"
    assert c_b4_init_viso["head"] == "init"
    assert c_b4_init_viso["loss"] == "la"
    assert c_b4_init_viso["stage1_ckpt"] == "checkpoints/stage1/ViSoBERT_Stage1_VSMEC"
    assert c_b5_gesf["head"] == "gesf"
    assert c_b5_gesf["loss"] == "la"
    assert float(c_b5_gesf["tau"]) == 1.0
    assert c_b5_gesf["emo_input"] == "p_emo"
    assert c_b5_gesf["wtrans_norm"] == "auto"
    assert c_b5_gesf["alpha_init"] == "auto"
    assert c_b5_gesf["official"] is False

    # Verify official=True refuses to run when git working tree is dirty
    monkeypatch.setattr("src.plm.train_plm.get_git_info", lambda repo_dir=None: ("abc1234", True))
    with pytest.raises(RuntimeError, match="official=True"):
        enforce_git_discipline(official=True)
    assert enforce_git_discipline(official=False) == "abc1234-dirty"

