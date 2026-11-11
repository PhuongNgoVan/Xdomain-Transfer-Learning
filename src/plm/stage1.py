"""Stage 1 PLM fine-tuning (7-class emotion on UIT-VSMEC) (`src/plm/stage1.py`).

Facade and entry-point per `stage1_refactor_spec.md` & `tasks/M6_stage1_vsmec.md`:
- Re-exports all public symbols from:
  - `src.plm.tokenizer_dataset`
  - `src.plm.model_ckpt`
  - `src.plm.trainer_utils`
- Implements `run_stage1` and CLI `main()` with:
  1. Proper FP16 `GradScaler.unscale_` + `clip_grad_norm_` + scale-checked `scheduler.step()`.
  2. FP32 post-training & validation inference (`use_fp16=False`) so saved CSV logits match
     CPU/GPU checkpoint reload within `< 1e-4`.
  3. In-RAM `best_state_dict` tracking during training and single `save_stage1_checkpoint` write
     at the end of training to eliminate repeated Google Drive FUSE I/O stalls.
  4. `eval_batch_size`, `pin_memory=(device.type == "cuda")`, and `non_blocking=True` tensor transfers.
"""

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

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
    build_length_grouped_indices,
    resolve_input_column,
    tokenize_with_truncation_stats,
)
from src.plm.trainer_utils import (
    NUM_EMOTIONS,
    PREDICTION_DATASETS,
    VALID_PRECISIONS,
    _find_latest_work_checkpoint,
    _load_split_dataframe,
    _save_temp_work_checkpoint,
    build_stage1_run_id,
    compute_cross_entropy_from_logits,
    compute_vsmec_data_hash,
    format_lr,
    get_git_commit,
    infer_backbone_alias,
    load_stage1_config,
    predict_logits7,
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
    "TokenizedTextDataset",
    "VALID_PRECISIONS",
    "_find_encoder_weight_file",
    "_find_latest_work_checkpoint",
    "_load_split_dataframe",
    "_save_temp_work_checkpoint",
    "build_length_grouped_indices",
    "build_stage1_run_id",
    "compute_cross_entropy_from_logits",
    "compute_vsmec_data_hash",
    "format_lr",
    "get_git_commit",
    "infer_backbone_alias",
    "load_stage1_config",
    "main",
    "predict_logits7",
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
    # Dev evaluation and prediction export always run in FP32 so training precision is the sole difference
    eval_fp16 = False
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
    use_fp16 = precision_effective == "fp16"
    use_bf16 = precision_effective == "bf16"
    amp_enabled = use_fp16 or use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

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
    model.to(device)

    # 5. Tokenize and record truncation statistics across all datasets
    truncation_report: Dict[str, Dict[str, Union[int, float]]] = {}
    train_enc, truncation_report["vsmec_train"] = tokenize_with_truncation_stats(
        vsmec_train_df[input_col].tolist(), tokenizer, max_length=max_length
    )
    dev_enc, truncation_report["vsmec_dev"] = tokenize_with_truncation_stats(
        vsmec_dev_df[input_col].tolist(), tokenizer, max_length=max_length
    )
    test_enc, truncation_report["vsmec_test"] = tokenize_with_truncation_stats(
        vsmec_test_df[input_col].tolist(), tokenizer, max_length=max_length
    )
    neuesc_dev_enc, truncation_report["neuesc_dev"] = tokenize_with_truncation_stats(
        neuesc_dev_df[input_col].tolist(), tokenizer, max_length=max_length
    )
    neuesc_test_enc, truncation_report["neuesc_test"] = tokenize_with_truncation_stats(
        neuesc_test_df[input_col].tolist(), tokenizer, max_length=max_length
    )
    vsfc_test_enc, truncation_report["vsfc_test"] = tokenize_with_truncation_stats(
        vsfc_test_df[input_col].tolist(), tokenizer, max_length=max_length
    )

    for ds_name, st in truncation_report.items():
        print(
            f"[Tokenization] {ds_name} ({input_col}): "
            f"truncated={st['n_truncated']}/{st['n_total']} ({st['truncation_rate'] * 100:.2f}%)"
        )

    train_labels = vsmec_train_df["label"].astype(int).tolist()
    dev_labels = vsmec_dev_df["label"].astype(int).tolist()

    # 6. Optimizer & linear warmup scheduler
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    num_batches_per_epoch = max(1, math.ceil(len(train_enc) / batch_size))
    steps_per_epoch = max(1, math.ceil(num_batches_per_epoch / grad_accum))
    total_optim_steps = (
        int(max_steps) if max_steps is not None else steps_per_epoch * epochs
    )
    warmup_steps = int(math.ceil(total_optim_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps,
    )

    eval_steps = int(cfg.get("eval_steps", max(1, steps_per_epoch // 2)))
    if max_steps is not None:
        eval_steps = min(eval_steps, max(1, max_steps))

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    ckpt_subdir = c_dir / alias
    work_run_dir = w_dir / run_id
    work_run_dir.mkdir(parents=True, exist_ok=True)

    # Optional resume from latest intermediate checkpoint in work_run_dir
    global_step = 0
    start_epoch = 0
    best_macro_f1 = -1.0
    best_step = 0
    patience_counter = 0
    best_state_dict: Optional[Dict[str, Tensor]] = None
    eval_history: List[Dict[str, Any]] = []
    n_skipped_optimizer_steps: int = 0
    n_nonfinite_loss: int = 0

    latest_resume = _find_latest_work_checkpoint(work_run_dir)
    if latest_resume is not None:
        resume_data = torch.load(latest_resume, map_location="cpu", weights_only=False)
        global_step = int(resume_data["step"])
        start_epoch = int(resume_data["epoch"])
        best_macro_f1 = float(resume_data["best_macro_f1"])
        best_step = int(resume_data["best_step"])
        patience_counter = int(resume_data["patience_counter"])
        eval_history = list(resume_data.get("eval_history", []))
        n_skipped_optimizer_steps = int(resume_data.get("n_skipped_optimizer_steps", 0))
        n_nonfinite_loss = int(resume_data.get("n_nonfinite_loss", 0))
        model.load_state_dict(resume_data["model_state_dict"])
        model.to(device)
        optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        scheduler.load_state_dict(resume_data["scheduler_state_dict"])
        best_state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        print(
            f"=== Resumed from {latest_resume} at step={global_step}, best_macro_f1={best_macro_f1:.4f} ==="
        )

    def _evaluate_and_checkpoint(curr_step: int, curr_epoch: int) -> bool:
        nonlocal best_macro_f1, best_step, patience_counter, best_state_dict
        dev_logits = predict_logits7(
            model=model,
            encodings=dev_enc,
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
            device=device,
            use_fp16=eval_fp16,
        )
        dev_preds = np.argmax(dev_logits, axis=1).tolist()
        dev_metrics = compute_metrics(
            dev_labels, dev_preds, logits=dev_logits, labels=EMOTION7_IDS
        )
        curr_f1 = float(dev_metrics["macro_f1"])
        curr_acc = float(dev_metrics["accuracy"])
        curr_loss = compute_cross_entropy_from_logits(dev_logits, dev_labels)

        eval_history.append(
            {
                "step": int(curr_step),
                "epoch": int(curr_epoch),
                "dev_macro_f1": curr_f1,
                "dev_loss": curr_loss,
                "dev_accuracy": curr_acc,
            }
        )

        if curr_f1 > best_macro_f1:
            best_macro_f1 = curr_f1
            best_step = int(curr_step)
            patience_counter = 0
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

        _save_temp_work_checkpoint(
            work_run_dir=work_run_dir,
            step=curr_step,
            epoch=curr_epoch,
            best_macro_f1=best_macro_f1,
            best_step=best_step,
            patience_counter=patience_counter,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_total_limit=save_total_limit,
            eval_history=eval_history,
            n_skipped_optimizer_steps=n_skipped_optimizer_steps,
            n_nonfinite_loss=n_nonfinite_loss,
        )
        model.train()
        return patience_counter >= patience

    # 7. Training loop
    t0 = time.time()
    stop_early = False
    last_evaluated_step = -1

    for epoch in range(start_epoch, epochs):
        if stop_early or (max_steps is not None and global_step >= max_steps):
            break

        ordered_indices = build_length_grouped_indices(
            train_enc,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            group_by_length=group_by_length,
        )
        epoch_enc = [train_enc[i] for i in ordered_indices]
        epoch_labels = [train_labels[i] for i in ordered_indices]
        train_ds = TokenizedTextDataset(epoch_enc, labels=epoch_labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=(device.type == "cuda"),
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for b_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device, non_blocking=True)
            labels_t = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels_t,
                )
                if not bool(torch.isfinite(out.loss).all().item()):
                    n_nonfinite_loss += 1
                loss = out.loss / float(grad_accum)

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accum_boundary = ((b_idx + 1) % grad_accum == 0) or (
                (b_idx + 1) == len(train_loader)
            )
            if is_accum_boundary:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    if max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= scale_before:
                        scheduler.step()
                    else:
                        n_skipped_optimizer_steps += 1
                else:
                    if max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % eval_steps == 0 or (
                    max_steps is not None and global_step >= max_steps
                ):
                    last_evaluated_step = global_step
                    should_stop = _evaluate_and_checkpoint(global_step, epoch)
                    if should_stop or (
                        max_steps is not None and global_step >= max_steps
                    ):
                        stop_early = True
                        break

    if last_evaluated_step != global_step:
        _evaluate_and_checkpoint(global_step, epochs)

    # Load best model at end (`load_best_model_at_end: true`) and optionally write checkpoint once
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
    model.eval()
    if save_checkpoint:
        model.save_stage1_checkpoint(tokenizer=tokenizer, ckpt_path=ckpt_subdir)

    elapsed_sec = time.time() - t0

    # 8. Post-training Step (a): Export 5 raw-logit prediction files (`PRED7_COLUMNS`) in FP32
    pred7_dir = r_dir / "predictions7"
    pred7_dir.mkdir(parents=True, exist_ok=True)

    eval_targets: List[
        Tuple[str, Sequence[int], List[Dict[str, Any]], Optional[Sequence[int]]]
    ] = [
        (
            "vsmec_dev",
            vsmec_dev_df["id"].astype(int).tolist(),
            dev_enc,
            dev_labels,
        ),
        (
            "vsmec_test",
            vsmec_test_df["id"].astype(int).tolist(),
            test_enc,
            vsmec_test_df["label"].astype(int).tolist(),
        ),
        (
            "neuesc_dev",
            neuesc_dev_df["id"].astype(int).tolist(),
            neuesc_dev_enc,
            None,
        ),
        (
            "neuesc_test",
            neuesc_test_df["id"].astype(int).tolist(),
            neuesc_test_enc,
            None,
        ),
        (
            "vsfc_test",
            vsfc_test_df["id"].astype(int).tolist(),
            vsfc_test_enc,
            None,
        ),
    ]

    prediction_paths: Dict[str, Path] = {}
    for ds_name, ids_list, enc_list, y_true_opt in eval_targets:
        logits_arr = predict_logits7(
            model=model,
            encodings=enc_list,
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
