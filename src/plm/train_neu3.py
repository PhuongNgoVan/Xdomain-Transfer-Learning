"""Step 1.2 PLM fine-tuning (NEU-ESC 3-class, PhoBERT) viết lại bằng Hugging Face `Trainer`
(`src/plm/train_neu3_hf.py`).

Giữ nguyên contract của `train_neu3.py` (bản PhoBERT):
- `sweep`: không bao giờ đọc/validate/tokenize/evaluate `test`, không lưu checkpoint/hash.
- `base`: đánh giá dev + test FP32, lưu `{ckpt_dir}/{alias}/` (encoder/ + head.pt), ghi hash.
- Chọn model + early stopping chỉ dựa trên `neuesc_dev` (macro_f1).

Phần được thay bằng thư viện: vòng lặp train, AMP bf16/fp16, grad accumulation, clip grad,
AdamW (no-decay cho bias/LayerNorm), scheduler warmup tuyến tính, eval theo step,
early stopping, restore best model, resume, giới hạn checkpoint trung gian.
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from src.common.hashing import sha256_file
from src.common.io import append_row_schema, read_csv_schema
from src.common.schema import NEU3_COLUMNS, NEU3_KEY_COLUMNS, PRED_COLUMNS
from src.common.seed import set_seed
from src.eval.metrics import LABEL3_IDS, compute_metrics
from src.plm.model_ckpt import Stage1Classifier
from src.plm.tokenizer_dataset import (
    TokenizedTextDataset,
    resolve_input_column,
    tokenize_with_truncation_stats,
)
from src.plm.trainer_utils import (
    format_lr,
    get_git_commit,
    infer_backbone_alias,
    load_stage1_config,
    resolve_precision,
    write_stage1_sidecar,
)
from src.plm.train_neu3 import (
    NUM_LABELS3,
    _format_label_distribution,
    _resolve_run_kind,
    build_neu3_run_id,
    load_neuesc_splits,
    predict_logits3,
    update_neu3_ckpt_hash,
    write_prediction3_file,
)

__all__ = ["run_neu3_hf", "main"]


def _make_compute_metrics():
    def _fn(eval_pred: EvalPrediction) -> Dict[str, float]:
        preds_raw = eval_pred.predictions
        logits = np.asarray(preds_raw[0] if isinstance(preds_raw, (tuple, list)) else preds_raw)
        labels = np.asarray(eval_pred.label_ids).astype(int)
        m = compute_metrics(
            labels.tolist(),
            np.argmax(logits, axis=1).tolist(),
            logits=logits,
            labels=LABEL3_IDS,
        )
        return {
            "macro_f1": float(m["macro_f1"]),
            "weighted_f1": float(m["weighted_f1"]),
            "accuracy": float(m["accuracy"]),
        }

    return _fn


def run_neu3_hf(
    config_path: Union[str, Path],
    data_dir: Union[str, Path] = "data/clean",
    ckpt_dir: Union[str, Path] = "checkpoints/neu3",
    results_dir: Union[str, Path] = "results",
    work_dir: Union[str, Path] = "work_tmp/neu3",
    max_steps_override: Optional[int] = None,
) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    cfg = load_stage1_config(cfg_path)

    backbone = str(cfg["backbone"])
    alias = infer_backbone_alias(backbone, config=cfg)
    input_col = resolve_input_column(backbone, config=cfg)
    seed = int(cfg.get("seed", 0))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 2.0e-5)))
    variant = str(cfg.get("variant", "base"))
    run_kind = _resolve_run_kind(cfg)
    save_checkpoint = (
        False if run_kind == "sweep" else bool(cfg.get("save_checkpoint", True))
    )
    if int(cfg.get("num_labels", NUM_LABELS3)) != NUM_LABELS3:
        raise ValueError(f"NEU-3 requires num_labels={NUM_LABELS3}.")

    batch_size = int(cfg.get("batch_size", 16))
    eval_batch_size = int(cfg.get("eval_batch_size", max(batch_size * 4, 64)))
    grad_accum = int(cfg.get("gradient_accumulation_steps", cfg.get("grad_accum", 2)))
    epochs = int(cfg.get("num_train_epochs", cfg.get("epochs", 10)))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.06))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    dropout = float(cfg.get("dropout", 0.1))
    max_length = int(cfg.get("max_length", 256))
    precision = str(cfg.get("precision", "bf16")).strip()
    strict_precision = bool(cfg.get("strict_precision", False))
    group_by_length = bool(cfg.get("group_by_length", True))
    patience = int(cfg.get("early_stopping_patience", 5))
    save_total_limit = int(cfg.get("save_total_limit", 2))
    max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else (int(cfg["max_steps"]) if cfg.get("max_steps") is not None else None)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision_effective = resolve_precision(
        precision=precision, device=device, strict_precision=strict_precision
    )
    run_id = build_neu3_run_id(
        backbone=backbone, seed=seed, lr=lr, variant=variant, config=cfg
    )

    d_dir, c_dir, r_dir, w_dir = (
        Path(data_dir), Path(ckpt_dir), Path(results_dir), Path(work_dir)
    )
    neu3_csv = r_dir / "neu3.csv"
    ckpt_subdir = c_dir / alias

    # 1. SKIP nếu đã có kết quả
    required_splits = {"dev"} if run_kind == "sweep" else {"dev", "test"}
    if neu3_csv.is_file():
        df_existing = read_csv_schema(neu3_csv, NEU3_COLUMNS)
        df_run = df_existing[df_existing["run_id"].astype(str) == run_id]
        if required_splits.issubset(set(df_run["split"].astype(str).tolist())):
            print(f"=== [SKIP] run_id={run_id} ===")
            return {
                "skipped": True, "run_id": run_id, "run_kind": run_kind,
                "neu3_csv": neu3_csv, "ckpt_subdir": ckpt_subdir,
            }

    # 2. Seed
    set_seed(seed, deterministic=True)

    # 3. Load data (sweep: không đụng tới test)
    neuesc_path = d_dir / "neuesc_100.csv"
    data_hash = sha256_file(neuesc_path)
    code_version = get_git_commit()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    train_df, dev_df, test_df, split_sizes = load_neuesc_splits(
        neuesc_path, input_col=input_col, run_kind=run_kind
    )
    train_labels = train_df["label"].astype(int).tolist()
    dev_labels = dev_df["label"].astype(int).tolist()
    test_labels = test_df["label"].astype(int).tolist() if test_df is not None else None

    print(
        f"[NEU3-HF Start] run_id={run_id} | run_kind={run_kind} | backbone={backbone} ({alias}) "
        f"| variant={variant} | seed={seed} | lr={format_lr(lr)} "
        f"| precision={precision}->{precision_effective} | gpu={gpu_name}"
    )
    print(
        f"[NEU3-HF Splits] n_train={split_sizes['train']} | n_dev={split_sizes['dev']} | "
        f"n_test={split_sizes['test']} ({'loaded' if run_kind == 'base' else 'untouched in sweep'})"
    )
    print(f"[NEU3-HF Labels] train: {_format_label_distribution(train_labels)}")
    print(f"[NEU3-HF Labels] dev:   {_format_label_distribution(dev_labels)}")
    if test_labels is not None:
        print(f"[NEU3-HF Labels] test:  {_format_label_distribution(test_labels)}")

    # 4. Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = Stage1Classifier.from_pretrained_backbone(
        backbone_name_or_path=backbone, num_labels=NUM_LABELS3, dropout=dropout
    )

    # 5. Tokenize (test chỉ khi base)
    truncation_report: Dict[str, Dict[str, Union[int, float]]] = {}
    train_enc, truncation_report["neuesc_train"] = tokenize_with_truncation_stats(
        train_df[input_col].tolist(), tokenizer, max_length=max_length
    )
    dev_enc, truncation_report["neuesc_dev"] = tokenize_with_truncation_stats(
        dev_df[input_col].tolist(), tokenizer, max_length=max_length
    )
    test_enc: Optional[List[Dict[str, Any]]] = None
    if run_kind == "base" and test_df is not None:
        test_enc, truncation_report["neuesc_test"] = tokenize_with_truncation_stats(
            test_df[input_col].tolist(), tokenizer, max_length=max_length
        )
    for ds_name, st in truncation_report.items():
        print(
            f"[Tokenization] {ds_name} ({input_col}): "
            f"truncated={st['n_truncated']}/{st['n_total']} ({st['truncation_rate'] * 100:.2f}%)"
        )

    train_ds = TokenizedTextDataset(train_enc, labels=train_labels)
    dev_ds = TokenizedTextDataset(dev_enc, labels=dev_labels)

    # 6. TrainingArguments
    steps_per_epoch = max(
        1, math.ceil(max(1, math.ceil(len(train_enc) / batch_size)) / grad_accum)
    )
    eval_steps = int(cfg.get("eval_steps", max(1, steps_per_epoch // 2)))
    if max_steps is not None:
        eval_steps = min(eval_steps, max(1, max_steps))

    work_run_dir = w_dir / run_id
    work_run_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(work_run_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        max_grad_norm=max_grad_norm,
        bf16=(precision_effective == "bf16"),
        fp16=(precision_effective == "fp16"),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        group_by_length=group_by_length,
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,  # model là nn.Module tuỳ biến
        label_names=["labels"],
        report_to="none",
        dataloader_pin_memory=(device.type == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,  # CHỈ dev -> không thể chọn model theo test
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
        compute_metrics=_make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    # 7. Train (resume nếu có checkpoint trung gian)
    t0 = time.time()
    last_ckpt = get_last_checkpoint(str(work_run_dir))
    if last_ckpt is not None:
        print(f"=== Resuming from {last_ckpt} ===")
    trainer.train(resume_from_checkpoint=last_ckpt)
    elapsed_sec = time.time() - t0

    # 8. Trainer đã nạp best model (load_best_model_at_end)
    best_ckpt = trainer.state.best_model_checkpoint
    best_step = (
        int(Path(best_ckpt).name.split("-")[-1])
        if best_ckpt
        else int(trainer.state.global_step)
    )
    eval_history: List[Dict[str, Any]] = [
        {
            "step": int(h["step"]),
            "epoch": float(h["epoch"]),
            "dev_macro_f1": float(h["eval_macro_f1"]),
            "dev_loss": float(h["eval_loss"]) if "eval_loss" in h else None,
            "dev_accuracy": float(h["eval_accuracy"]),
        }
        for h in trainer.state.log_history
        if "eval_macro_f1" in h
    ]

    model = trainer.model
    model.eval()
    if save_checkpoint:
        model.save_stage1_checkpoint(tokenizer=tokenizer, ckpt_path=ckpt_subdir)

    # 9. Dự đoán FP32 + metrics từ CSV
    pred3_dir = r_dir / "predictions3"
    pred3_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths: Dict[str, Path] = {}
    logit_cols = ["logit_neg", "logit_neu", "logit_pos"]

    def _predict_and_score(name: str, df, enc, labels):
        logits = predict_logits3(
            model=model, encodings=enc, tokenizer=tokenizer,
            batch_size=eval_batch_size, device=trainer.args.device, use_fp16=False,
        )
        out_csv = pred3_dir / f"{run_id}_{name}.csv"
        write_prediction3_file(
            ids=df["id"].astype(int).tolist(), logits=logits,
            out_path=out_csv, y_true=labels,
        )
        prediction_paths[name] = out_csv
        pdf = read_csv_schema(out_csv, PRED_COLUMNS)
        return compute_metrics(
            pdf["y_true"].astype(int).tolist(),
            pdf["y_pred"].astype(int).tolist(),
            logits=pdf[logit_cols].to_numpy(dtype=np.float64),
            labels=LABEL3_IDS,
        )

    dev_metrics_final = _predict_and_score("neuesc_dev", dev_df, dev_enc, dev_labels)
    test_metrics_final: Optional[Dict[str, Any]] = None
    if run_kind == "base" and test_df is not None and test_enc is not None:
        test_metrics_final = _predict_and_score(
            "neuesc_test", test_df, test_enc, test_labels
        )

    # 10. Ghi neu3.csv
    ckpt_path_str = ckpt_subdir.as_posix() if save_checkpoint else ""

    def _row(split_name: str, m: Dict[str, Any], trunc_rate: float) -> Dict[str, Any]:
        f1 = [float(m["f1_neg"]), float(m["f1_neu"]), float(m["f1_pos"])]
        return {
            "run_id": run_id, "backbone": backbone, "stage1": "NEU", "stage2": "",
            "eval_domain": "NEU", "label_view": "V3-NEU", "mode": "base",
            "fraction": 100, "seed": seed, "split": split_name,
            "macro_f1": float(m["macro_f1"]), "weighted_f1": float(m["weighted_f1"]),
            "accuracy": float(m["accuracy"]),
            "f1_neg": f1[0], "f1_neu": f1[1], "f1_pos": f1[2],
            "f1_per_class": json.dumps(f1),
            "ckpt_path": ckpt_path_str, "run_kind": run_kind, "variant": variant,
            "lr": lr, "best_step": best_step, "n_train": int(len(train_df)),
            "precision": precision, "gpu": gpu_name, "code_version": code_version,
            "data_hash": data_hash, "config_path": cfg_path.as_posix(),
            "truncation_rate": float(trunc_rate),
        }

    append_row_schema(
        row=_row("dev", dev_metrics_final,
                 float(truncation_report["neuesc_dev"]["truncation_rate"])),
        path=neu3_csv, expected_columns=NEU3_COLUMNS, key_columns=NEU3_KEY_COLUMNS,
        overwrite_duplicate_key=True,
    )
    if run_kind == "base" and test_metrics_final is not None:
        append_row_schema(
            row=_row("test", test_metrics_final,
                     float(truncation_report["neuesc_test"]["truncation_rate"])),
            path=neu3_csv, expected_columns=NEU3_COLUMNS, key_columns=NEU3_KEY_COLUMNS,
            overwrite_duplicate_key=True,
        )

    sidecar_path = write_stage1_sidecar(
        results_dir=r_dir,
        run_id=run_id,
        payload={
            "run_id": run_id, "alias": alias, "backbone": backbone, "lr": lr,
            "seed": seed, "variant": variant, "run_kind": run_kind,
            "precision": precision, "precision_effective": precision_effective,
            "git_commit": code_version, "data_hash": data_hash,
            "torch_version": str(torch.__version__),
            "transformers_version": str(transformers.__version__),
            "gpu": gpu_name, "best_step": best_step,
            "best_dev_macro_f1": float(dev_metrics_final["macro_f1"]),
            "elapsed_sec": float(elapsed_sec),
            "truncation_report": truncation_report, "eval_history": eval_history,
            "report_only": {
                "dev_macro_f1": float(dev_metrics_final["macro_f1"]),
                "test_macro_f1": (
                    float(test_metrics_final["macro_f1"])
                    if test_metrics_final is not None else None
                ),
            },
        },
    )

    # 11. Dọn checkpoint trung gian
    shutil.rmtree(work_run_dir, ignore_errors=True)

    # 12. Hash checkpoint
    ckpt_hashes = (
        update_neu3_ckpt_hash(
            run_id=run_id, alias=alias, ckpt_subdir=ckpt_subdir, results_dir=r_dir
        )
        if save_checkpoint else {}
    )

    test_f1_str = (
        f"{test_metrics_final['macro_f1']:.4f}" if test_metrics_final else "NA"
    )
    print(
        f"=== [DONE] {run_id}: best_step={best_step}, "
        f"dev_macro_f1={dev_metrics_final['macro_f1']:.4f}, "
        f"test_macro_f1={test_f1_str}, elapsed={elapsed_sec:.1f}s ==="
    )
    return {
        "skipped": False, "run_id": run_id, "run_kind": run_kind, "alias": alias,
        "best_step": best_step, "precision": precision,
        "precision_effective": precision_effective,
        "dev_metrics": dev_metrics_final, "test_metrics": test_metrics_final,
        "truncation_report": truncation_report, "eval_history": eval_history,
        "prediction_paths": prediction_paths, "neu3_csv": neu3_csv,
        "sidecar_path": sidecar_path, "ckpt_subdir": ckpt_subdir,
        "ckpt_hashes": ckpt_hashes, "elapsed_sec": elapsed_sec,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Step 1.2 NEU-3 fine-tuning (Hugging Face Trainer, PhoBERT)."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/clean"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/neu3"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--work-dir", type=Path, default=Path("work_tmp/neu3"))
    p.add_argument("--max-steps", type=int, default=None)
    a = p.parse_args(argv)
    run_neu3_hf(
        config_path=a.config, data_dir=a.data_dir, ckpt_dir=a.ckpt_dir,
        results_dir=a.results_dir, work_dir=a.work_dir, max_steps_override=a.max_steps,
    )


if __name__ == "__main__":
    main()
