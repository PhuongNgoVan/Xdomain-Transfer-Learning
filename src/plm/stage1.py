"""Stage 1 PLM fine-tuning (7-class emotion on UIT-VSMEC) (`src/plm/stage1.py`).

Facade and entry-point per `stage1_refactor_spec.md` & `tasks/M6_stage1_vsmec.md`:
- Re-exports all public symbols from:
  - `src.plm.tokenizer_dataset`
  - `src.plm.model_ckpt`
  - `src.plm.trainer_utils`
- Implements `run_stage1` and CLI `main()` on top of the Hugging Face stack:
  1. `datasets.Dataset` tokenization and `transformers.Trainer` for the training loop (AdamW,
     linear warmup schedule, fp16 `GradScaler` / bf16 via `accelerate`, gradient clipping and
     accumulation, `group_by_length`, `EarlyStoppingCallback`, `load_best_model_at_end`).
  2. FP32 post-training inference (`Trainer.predict` with `fp16=False` on a freshly rebuilt model
     loaded with the best weights) so saved CSV logits match checkpoint reload within `< 1e-4`.
  3. Intermediate `checkpoint-{step}` dirs live under `work_dir` (resume via
     `resume_from_checkpoint`); the final `encoder/` + `head.pt` checkpoint is written once at the
     end of training to avoid repeated Google Drive FUSE I/O stalls.
"""

import argparse
import gc
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from src.common.constants import EMOTIONS7, ID_TO_EMOTION7
from src.common.io import append_row_schema, read_csv_schema
from src.common.schema import PRED7_COLUMNS, SPLIT_COLUMNS, STAGE1_COLUMNS
from src.common.seed import set_seed
from src.eval.metrics import EMOTION7_IDS, compute_metrics
from src.plm.model_ckpt import (
    Stage1Classifier,
    _find_encoder_weight_file,
    update_stage1_ckpt_hash,
)
from src.plm.tokenizer_dataset import (
    TokenizedTextDataset,
    resolve_input_column,
    tokenize_with_truncation_stats,
)
from src.plm.trainer_utils import (
    NUM_EMOTIONS,
    PREDICTION_DATASETS,
    VALID_PRECISIONS,
    Stage1Trainer,
    _find_latest_work_checkpoint,
    _load_split_dataframe,
    build_stage1_run_id,
    build_stage1_trainer,
    build_training_arguments,
    compute_cross_entropy_from_logits,
    compute_stage1_metrics,
    compute_vsmec_data_hash,
    extract_eval_history,
    format_lr,
    get_git_commit,
    infer_backbone_alias,
    load_stage1_config,
    predict_logits7,
    resolve_best_step,
    resolve_precision,
    write_prediction7_file,
    write_stage1_sidecar,
)

__all__ = [
    "EMOTIONS7",
    "ID_TO_EMOTION7",
    "NUM_EMOTIONS",
    "PREDICTION_DATASETS",
    "Stage1Classifier",
    "Stage1Trainer",
    "TokenizedTextDataset",
    "VALID_PRECISIONS",
    "_find_encoder_weight_file",
    "_find_latest_work_checkpoint",
    "_load_split_dataframe",
    "build_stage1_run_id",
    "build_stage1_trainer",
    "build_training_arguments",
    "compute_cross_entropy_from_logits",
    "compute_stage1_metrics",
    "compute_vsmec_data_hash",
    "extract_eval_history",
    "format_lr",
    "get_git_commit",
    "infer_backbone_alias",
    "load_stage1_config",
    "main",
    "predict_logits7",
    "resolve_best_step",
    "resolve_input_column",
    "resolve_precision",
    "run_stage1",
    "tokenize_with_truncation_stats",
    "update_stage1_ckpt_hash",
    "write_prediction7_file",
    "write_stage1_sidecar",
]


def run_stage1(
    config_path: Union[str, Path],
    data_dir: Union[str, Path] = "data/clean",
    ckpt_dir: Union[str, Path] = "checkpoints/stage1",
    results_dir: Union[str, Path] = "results",
    work_dir: Union[str, Path] = "work_tmp/stage1",
    max_steps_override: Optional[int] = None,
    on_before_cleanup: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Execute the full Stage 1 training, prediction export, evaluation, and cleanup pipeline."""
    cfg_path = Path(config_path)
    cfg = load_stage1_config(cfg_path)

    backbone = str(cfg["backbone"])
    alias = infer_backbone_alias(backbone, config=cfg)
    input_col = resolve_input_column(backbone, config=cfg)
    seed = int(cfg.get("seed", 0))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 2.0e-5)))
    variant = str(cfg.get("variant", "base"))
    batch_size = int(cfg.get("batch_size", 16))
    eval_batch_size = int(cfg.get("eval_batch_size", max(batch_size * 4, 64)))
    grad_accum = int(cfg.get("gradient_accumulation_steps", cfg.get("grad_accum", 2)))
    epochs = int(cfg.get("num_train_epochs", cfg.get("epochs", 10)))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.06))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    dropout = float(cfg.get("dropout", 0.1))
    max_length = int(cfg.get("max_length", 256))
    precision = str(cfg.get("precision", "fp16")).strip()
    strict_precision = bool(cfg.get("strict_precision", False))
    emoji_mode = str(cfg.get("emoji_mode", "tag"))
    sweep = bool(cfg.get("sweep", False))
    save_checkpoint = bool(cfg.get("save_checkpoint", True))
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

    run_id = build_stage1_run_id(
        backbone=backbone, seed=seed, lr=lr, variant=variant, config=cfg
    )

    d_dir = Path(data_dir)
    c_dir = Path(ckpt_dir)
    r_dir = Path(results_dir)
    w_dir = Path(work_dir)
    stage1_csv = r_dir / "stage1.csv"

    # 1. Skip if `run_id` already exists in `stage1.csv`
    if stage1_csv.is_file():
        df_existing = read_csv_schema(stage1_csv, STAGE1_COLUMNS)
        if run_id in set(df_existing["run_id"].astype(str)):
            print(f"=== [SKIP] run_id={run_id} already exists in {stage1_csv} ===")
            return {
                "skipped": True,
                "run_id": run_id,
                "stage1_csv": stage1_csv,
                "ckpt_subdir": c_dir / alias,
            }

    # 2. Set deterministic seed
    set_seed(seed, deterministic=True)

    # 3. Verify and load all source datasets
    vsmec_train_df = _load_split_dataframe(
        d_dir / "vsmec_train.csv", (*SPLIT_COLUMNS, input_col)
    )
    vsmec_dev_df = _load_split_dataframe(
        d_dir / "vsmec_dev.csv", (*SPLIT_COLUMNS, input_col)
    )
    vsmec_test_df = _load_split_dataframe(
        d_dir / "vsmec_test.csv", (*SPLIT_COLUMNS, input_col)
    )
    neuesc_full_df = _load_split_dataframe(
        d_dir / "neuesc_100.csv", (*SPLIT_COLUMNS, input_col)
    )
    neuesc_dev_df = neuesc_full_df[neuesc_full_df["subset"] == "dev"].reset_index(
        drop=True
    )
    neuesc_test_df = neuesc_full_df[neuesc_full_df["subset"] == "test"].reset_index(
        drop=True
    )
    vsfc_test_df = _load_split_dataframe(
        d_dir / "vsfc_test_clean.csv", ("id", input_col)
    )

    # 4. Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = Stage1Classifier.from_pretrained_backbone(
        backbone_name_or_path=backbone,
        num_labels=NUM_EMOTIONS,
        dropout=dropout,
    )

    train_labels = vsmec_train_df["label"].astype(int).tolist()
    dev_labels = vsmec_dev_df["label"].astype(int).tolist()
    test_labels = vsmec_test_df["label"].astype(int).tolist()

    # 5. Tokenize into HF `datasets.Dataset`s and record truncation statistics across all datasets
    truncation_report: Dict[str, Dict[str, Union[int, float]]] = {}
    train_ds, truncation_report["vsmec_train"] = tokenize_with_truncation_stats(
        vsmec_train_df[input_col].tolist(),
        tokenizer,
        max_length=max_length,
        labels=train_labels,
    )
    dev_ds, truncation_report["vsmec_dev"] = tokenize_with_truncation_stats(
        vsmec_dev_df[input_col].tolist(),
        tokenizer,
        max_length=max_length,
        labels=dev_labels,
    )
    test_ds, truncation_report["vsmec_test"] = tokenize_with_truncation_stats(
        vsmec_test_df[input_col].tolist(),
        tokenizer,
        max_length=max_length,
        labels=test_labels,
    )
    neuesc_dev_ds, truncation_report["neuesc_dev"] = tokenize_with_truncation_stats(
        neuesc_dev_df[input_col].tolist(), tokenizer, max_length=max_length
    )
    neuesc_test_ds, truncation_report["neuesc_test"] = tokenize_with_truncation_stats(
        neuesc_test_df[input_col].tolist(), tokenizer, max_length=max_length
    )
    vsfc_test_ds, truncation_report["vsfc_test"] = tokenize_with_truncation_stats(
        vsfc_test_df[input_col].tolist(), tokenizer, max_length=max_length
    )

    for ds_name, st in truncation_report.items():
        print(
            f"[Tokenization] {ds_name} ({input_col}): "
            f"truncated={st['n_truncated']}/{st['n_total']} ({st['truncation_rate'] * 100:.2f}%)"
        )

    # 6. Step budget (the schedule length is fixed explicitly via `max_steps`)
    num_batches_per_epoch = max(1, math.ceil(len(train_ds) / batch_size))
    steps_per_epoch = max(1, math.ceil(num_batches_per_epoch / grad_accum))
    total_optim_steps = (
        int(max_steps) if max_steps is not None else steps_per_epoch * epochs
    )
    warmup_steps = int(math.ceil(total_optim_steps * warmup_ratio))

    eval_steps = int(cfg.get("eval_steps", max(1, steps_per_epoch // 2)))
    if max_steps is not None:
        eval_steps = min(eval_steps, max(1, max_steps))

    ckpt_subdir = c_dir / alias
    work_run_dir = w_dir / run_id
    work_run_dir.mkdir(parents=True, exist_ok=True)

    # 7. HF Trainer: AdamW + linear warmup, fp16/bf16, clipping, early stopping, best-model selection
    training_args = build_training_arguments(
        work_run_dir,
        learning_rate=lr,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        grad_accum=grad_accum,
        num_train_epochs=epochs,
        max_steps=total_optim_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        group_by_length=group_by_length,
        precision_effective=precision_effective,
        device=device,
    )
    trainer = build_stage1_trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        early_stopping_patience=patience,
    )

    # Optional resume from the latest intermediate checkpoint in work_run_dir
    latest_resume = _find_latest_work_checkpoint(work_run_dir)
    if latest_resume is not None:
        print(f"=== Resuming from {latest_resume} ===")

    t0 = time.time()
    trainer.train(
        resume_from_checkpoint=str(latest_resume) if latest_resume is not None else None
    )

    # `load_best_model_at_end=True`: `trainer.model` now holds the best-dev-macro-F1 weights.
    train_state = trainer.state
    best_macro_f1 = float(train_state.best_metric) if train_state.best_metric is not None else -1.0
    best_step = resolve_best_step(train_state)
    eval_history: List[Dict[str, Any]] = extract_eval_history(train_state.log_history)
    n_skipped_optimizer_steps = int(trainer.n_skipped_optimizer_steps)
    n_nonfinite_loss = int(trainer.n_nonfinite_loss)
    best_state_dict = {
        k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()
    }

    # Rebuild a pristine FP32 model (no accelerate/autocast wrappers left on `forward`) holding the best
    # weights; it is used for checkpoint export and all prediction files.
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    eval_model = Stage1Classifier.from_pretrained_backbone(
        backbone_name_or_path=backbone,
        num_labels=NUM_EMOTIONS,
        dropout=dropout,
        load_pretrained_weights=False,
    )
    eval_model.load_state_dict(best_state_dict)
    eval_model.to(device)
    eval_model.eval()
    del best_state_dict

    if save_checkpoint:
        eval_model.save_stage1_checkpoint(tokenizer=tokenizer, ckpt_path=ckpt_subdir)

    elapsed_sec = time.time() - t0

    # 8. Post-training Step (a): Export 5 raw-logit prediction files (`PRED7_COLUMNS`) in FP32
    pred7_dir = r_dir / "predictions7"
    pred7_dir.mkdir(parents=True, exist_ok=True)

    eval_targets: List[
        Tuple[str, Sequence[int], Any, Optional[Sequence[int]]]
    ] = [
        (
            "vsmec_dev",
            vsmec_dev_df["id"].astype(int).tolist(),
            dev_ds,
            dev_labels,
        ),
        (
            "vsmec_test",
            vsmec_test_df["id"].astype(int).tolist(),
            test_ds,
            test_labels,
        ),
        (
            "neuesc_dev",
            neuesc_dev_df["id"].astype(int).tolist(),
            neuesc_dev_ds,
            None,
        ),
        (
            "neuesc_test",
            neuesc_test_df["id"].astype(int).tolist(),
            neuesc_test_ds,
            None,
        ),
        (
            "vsfc_test",
            vsfc_test_df["id"].astype(int).tolist(),
            vsfc_test_ds,
            None,
        ),
    ]

    prediction_paths: Dict[str, Path] = {}
    for ds_name, ids_list, ds_obj, y_true_opt in eval_targets:
        logits_arr = predict_logits7(
            model=eval_model,
            encodings=ds_obj,
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
            device=device,
            use_fp16=False,
        )
        out_csv = pred7_dir / f"{run_id}_{ds_name}.csv"
        write_prediction7_file(
            ids=ids_list,
            logits=logits_arr,
            out_path=out_csv,
            y_true7=y_true_opt,
        )
        prediction_paths[ds_name] = out_csv

    # 9. Post-training Step (b): Compute dev VSMEC metrics from the written `predictions7` file
    #    and append 1 row to `{results_dir}/stage1.csv` (`STAGE1_COLUMNS`)
    dev_pred_df = read_csv_schema(prediction_paths["vsmec_dev"], PRED7_COLUMNS)
    y_true_dev = dev_pred_df["y_true7"].astype(int).tolist()
    y_pred_dev = dev_pred_df["y_pred7"].astype(int).tolist()
    logit_cols = [f"logit_{i}" for i in range(NUM_EMOTIONS)]
    logits_dev = dev_pred_df[logit_cols].to_numpy(dtype=np.float64)

    dev_metrics_final = compute_metrics(
        y_true_dev,
        y_pred_dev,
        logits=logits_dev,
        labels=EMOTION7_IDS,
    )

    # Also compute test VSMEC metrics once for reporting (never used for model selection)
    test_pred_df = read_csv_schema(prediction_paths["vsmec_test"], PRED7_COLUMNS)
    test_metrics_final = compute_metrics(
        test_pred_df["y_true7"].astype(int).tolist(),
        test_pred_df["y_pred7"].astype(int).tolist(),
        logits=test_pred_df[logit_cols].to_numpy(dtype=np.float64),
        labels=EMOTION7_IDS,
    )

    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    code_version = get_git_commit()
    data_hash = compute_vsmec_data_hash(d_dir)

    stage1_row: Dict[str, Any] = {
        "run_id": run_id,
        "backbone": backbone,
        "seed": seed,
        "lr": lr,
        "best_step": best_step,
        "macro_f1": float(dev_metrics_final["macro_f1"]),
        "accuracy": float(dev_metrics_final["accuracy"]),
        "weighted_f1": float(dev_metrics_final["weighted_f1"]),
        "precision": precision,
        "gpu": gpu_name,
        "code_version": code_version,
        "data_hash": data_hash,
        "config_path": cfg_path.as_posix(),
    }
    for emo_name in EMOTIONS7:
        col = f"f1_{emo_name.lower()}"
        stage1_row[col] = float(dev_metrics_final[col])

    append_row_schema(
        row=stage1_row,
        path=stage1_csv,
        expected_columns=STAGE1_COLUMNS,
        key_columns=("run_id",),
        overwrite_duplicate_key=False,
    )

    # Write diagnostics sidecar (`<results_dir>/runs/<run_id>.json`) after predictions & stage1.csv
    sidecar_payload: Dict[str, Any] = {
        "run_id": run_id,
        "alias": alias,
        "backbone": backbone,
        "lr": lr,
        "seed": seed,
        "variant": variant,
        "precision": precision,
        "precision_effective": precision_effective,
        "emoji_mode": emoji_mode,
        "git_commit": code_version,
        "data_hash": data_hash,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "gpu": gpu_name,
        "best_step": int(best_step),
        "best_dev_macro_f1": float(dev_metrics_final["macro_f1"]),
        "elapsed_sec": float(elapsed_sec),
        "truncation_report": truncation_report,
        "eval_history": eval_history,
        "n_skipped_optimizer_steps": int(n_skipped_optimizer_steps),
        "n_nonfinite_loss": int(n_nonfinite_loss),
        "report_only": {
            "dev_macro_f1": float(dev_metrics_final["macro_f1"]),
            "test_macro_f1": float(test_metrics_final["macro_f1"]),
        },
    }
    sidecar_path = write_stage1_sidecar(
        results_dir=r_dir,
        run_id=run_id,
        payload=sidecar_payload,
    )

    # Verify ordering invariant before deleting intermediate checkpoints
    work_dir_existed_after_results = bool(
        work_run_dir.is_dir() and any(work_run_dir.glob("checkpoint-*"))
    )
    pre_cleanup_info = {
        "run_id": run_id,
        "prediction_paths": prediction_paths,
        "stage1_csv": stage1_csv,
        "sidecar_path": sidecar_path,
        "work_run_dir": work_run_dir,
        "work_dir_existed_after_results": work_dir_existed_after_results,
    }
    if on_before_cleanup is not None:
        on_before_cleanup(pre_cleanup_info)

    # 10. Post-training Step (c): Remove intermediate checkpoints in `work_dir`, keep best checkpoint
    if work_run_dir.is_dir():
        shutil.rmtree(work_run_dir, ignore_errors=True)

    # 11. Post-training Step (d): Hash encoder weights & `head.pt` when `save_checkpoint=True`
    if save_checkpoint:
        ckpt_hashes = update_stage1_ckpt_hash(
            alias=alias,
            ckpt_subdir=ckpt_subdir,
            results_dir=r_dir,
            sweep=sweep,
            run_id=run_id,
        )
    else:
        ckpt_hashes = {}

    print(
        f"=== [DONE] {run_id}: best_step={best_step}, "
        f"dev_macro_f1={dev_metrics_final['macro_f1']:.4f}, "
        f"test_macro_f1={test_metrics_final['macro_f1']:.4f}, "
        f"elapsed={elapsed_sec:.1f}s ==="
    )

    return {
        "skipped": False,
        "run_id": run_id,
        "alias": alias,
        "best_step": best_step,
        "precision": precision,
        "precision_effective": precision_effective,
        "dev_metrics": dev_metrics_final,
        "test_metrics": test_metrics_final,
        "truncation_report": truncation_report,
        "eval_history": eval_history,
        "n_skipped_optimizer_steps": n_skipped_optimizer_steps,
        "n_nonfinite_loss": n_nonfinite_loss,
        "prediction_paths": prediction_paths,
        "stage1_csv": stage1_csv,
        "sidecar_path": sidecar_path,
        "ckpt_subdir": ckpt_subdir,
        "ckpt_hashes": ckpt_hashes,
        "elapsed_sec": elapsed_sec,
        "work_dir_existed_after_results": work_dir_existed_after_results,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 PLM 7-class fine-tuning on UIT-VSMEC (CV-P8)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Stage 1 YAML config (e.g. configs/exp/stage1_phobert.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/clean"),
        help="Directory containing cleaned CSV datasets.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("checkpoints/stage1"),
        help="Directory to store the best Stage 1 checkpoints (encoder/ + head.pt).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write stage1.csv, predictions7/, and stage1_ckpt_hash.json.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work_tmp/stage1"),
        help="Local temporary directory for intermediate training checkpoints.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional override for maximum optimization steps (for smoke testing).",
    )
    args = parser.parse_args(argv)

    run_stage1(
        config_path=args.config,
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        results_dir=args.results_dir,
        work_dir=args.work_dir,
        max_steps_override=args.max_steps,
    )


if __name__ == "__main__":
    main()
