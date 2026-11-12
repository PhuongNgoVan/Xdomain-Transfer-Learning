"""Stage 1 training utilities built on HF `Trainer`, logit prediction, CSV export (`src/plm/trainer_utils.py`)."""

import hashlib
import inspect
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
import yaml

from src.common.constants import EMOTIONS7
from src.common.hashing import sha256_file
from src.common.io import read_csv_schema, write_csv_schema
from src.common.schema import PRED7_COLUMNS
from src.plm.model_ckpt import Stage1Classifier
from src.plm.tokenizer_dataset import TokenizedTextDataset

NUM_EMOTIONS: int = len(EMOTIONS7)
PREDICTION_DATASETS: Tuple[str, ...] = (
    "vsmec_dev",
    "vsmec_test",
    "neuesc_dev",
    "neuesc_test",
    "vsfc_test",
)


VALID_PRECISIONS: Tuple[str, ...] = ("fp16", "bf16", "fp32")


def resolve_precision(
    precision: str,
    device: torch.device,
    strict_precision: bool = False,
) -> str:
    """Validate requested `precision` (`fp16`, `bf16`, `fp32`) and resolve effective precision on `device`.

    When `strict_precision` is True, raises `RuntimeError` if the hardware/device cannot satisfy
    the requested precision (e.g., `fp16`/`bf16` without CUDA, or `bf16` when
    `torch.cuda.is_bf16_supported()` is False). When `strict_precision` is False, falls back to `'fp32'`.
    """
    p = str(precision).strip()
    if p not in VALID_PRECISIONS:
        raise ValueError(
            f"Invalid precision {precision!r}. Must be one of {VALID_PRECISIONS}."
        )
    if p == "fp32":
        return "fp32"

    has_cuda = device.type == "cuda" and torch.cuda.is_available()
    if p == "fp16":
        if has_cuda:
            return "fp16"
        if strict_precision:
            raise RuntimeError(
                "strict_precision=True: precision='fp16' requires an active CUDA device, "
                f"but running on device={device.type!r}."
            )
        return "fp32"

    # p == "bf16"
    bf16_ok = bool(
        has_cuda
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if bf16_ok:
        return "bf16"
    if strict_precision:
        raise RuntimeError(
            "strict_precision=True: precision='bf16' requires CUDA with bfloat16 support "
            f"(device={device.type!r}, cuda_available={has_cuda})."
        )
    return "fp32"


def format_lr(lr: float) -> str:
    """Format learning rate compactly for `run_id` (e.g. `2e-05` -> `2e-5`)."""
    s = f"{float(lr):g}"
    if "e" in s:
        mantissa, exp = s.split("e", 1)
        sign = "-" if exp.startswith("-") else ""
        digits = exp.lstrip("+-").lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}"
    return s


def infer_backbone_alias(backbone: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Determine canonical short alias (`PhoBERT` or `ViSoBERT`) from config or backbone."""
    if config and config.get("model_alias"):
        return str(config["model_alias"])
    lower = str(backbone).lower()
    if "phobert" in lower:
        return "PhoBERT"
    if "visobert" in lower:
        return "ViSoBERT"
    return Path(str(backbone)).name


def build_stage1_run_id(
    backbone: str,
    seed: int,
    lr: float,
    variant: str = "base",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Construct Stage 1 `run_id` per Section 6 (`{alias}_Stage1-{variant}_s{seed}_lr{lr}`)."""
    v_str = str(variant).strip()
    if not v_str or "_" in v_str:
        raise ValueError(
            f"Stage 1 variant must be non-empty and must not contain '_' (got {variant!r})."
        )
    alias = infer_backbone_alias(backbone, config=config)
    return f"{alias}_Stage1-{v_str}_s{int(seed)}_lr{format_lr(lr)}"


def get_git_commit() -> str:
    """Return full `git rev-parse HEAD` commit hash (or `'unknown'` if unavailable)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def compute_vsmec_data_hash(data_dir: Union[str, Path]) -> str:
    """Compute combined SHA-256 digest of `vsmec_train.csv`, `vsmec_dev.csv`, `vsmec_test.csv`."""
    d_path = Path(data_dir)
    combined = hashlib.sha256()
    for sp in ("train", "dev", "test"):
        fp = d_path / f"vsmec_{sp}.csv"
        file_digest = sha256_file(fp)
        combined.update(f"vsmec_{sp}.csv:{file_digest}\n".encode("utf-8"))
    return combined.hexdigest()


# ---------------------------------------------------------------------------
# HF Trainer integration
# ---------------------------------------------------------------------------


def _init_params(cls: Any) -> Sequence[str]:
    """Names of `cls.__init__` parameters (used for transformers cross-version compatibility)."""
    return tuple(inspect.signature(cls.__init__).parameters)


def _tokenizer_kwarg(tokenizer: Any) -> Dict[str, Any]:
    """`Trainer(processing_class=...)` on new transformers, `Trainer(tokenizer=...)` on old ones."""
    key = "processing_class" if "processing_class" in _init_params(Trainer) else "tokenizer"
    return {key: tokenizer}


def build_training_arguments(
    output_dir: Union[str, Path],
    *,
    learning_rate: float,
    batch_size: int,
    eval_batch_size: int,
    grad_accum: int,
    num_train_epochs: int,
    max_steps: int,
    warmup_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    eval_steps: int,
    save_total_limit: int,
    seed: int,
    group_by_length: bool,
    precision_effective: str,
    device: torch.device,
) -> TrainingArguments:
    """Create `TrainingArguments` equivalent to the former hand-written Stage 1 loop.

    - AdamW (`adamw_torch`, no weight decay on bias/LayerNorm) + linear warmup/decay schedule.
    - fp16 (GradScaler + unscale + clip) / bf16 handled by `Trainer`/`accelerate`.
    - Evaluate + save every `eval_steps`; best checkpoint chosen by dev `macro_f1`
      (`load_best_model_at_end=True`).
    """
    params = _init_params(TrainingArguments)
    kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(eval_batch_size),
        gradient_accumulation_steps=int(grad_accum),
        num_train_epochs=float(num_train_epochs),
        max_steps=int(max_steps),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        warmup_steps=int(warmup_steps),
        lr_scheduler_type="linear",
        optim="adamw_torch",
        max_grad_norm=float(max_grad_norm),
        fp16=(precision_effective == "fp16"),
        bf16=(precision_effective == "bf16"),
        seed=int(seed),
        data_seed=int(seed),
        group_by_length=bool(group_by_length),
        logging_strategy="steps",
        logging_steps=int(eval_steps),
        save_strategy="steps",
        save_steps=int(eval_steps),
        eval_steps=int(eval_steps),
        save_total_limit=int(save_total_limit) if int(save_total_limit) > 0 else None,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=True,
    )
    # `evaluation_strategy` was renamed to `eval_strategy` in transformers 4.46.
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    if device.type != "cuda":
        kwargs["use_cpu"] = True
    return TrainingArguments(**kwargs)


def compute_stage1_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """`Trainer.compute_metrics` callback: dev macro-F1 / accuracy / weighted-F1 via `src.eval.metrics`."""
    from src.eval.metrics import EMOTION7_IDS, compute_metrics

    logits = eval_pred.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(eval_pred.label_ids).astype(int).tolist()
    y_pred = np.argmax(logits, axis=1).tolist()
    m = compute_metrics(y_true, y_pred, logits=logits, labels=EMOTION7_IDS)
    return {
        "macro_f1": float(m["macro_f1"]),
        "accuracy": float(m["accuracy"]),
        "weighted_f1": float(m["weighted_f1"]),
    }


class Stage1Trainer(Trainer):
    """`Trainer` that additionally counts non-finite training losses and skipped optimizer steps.

    Counters are kept for the sidecar diagnostics (`n_nonfinite_loss`, `n_skipped_optimizer_steps`);
    they restart from 0 when training is resumed from a checkpoint.
    """

    n_nonfinite_loss: int = 0
    n_skipped_optimizer_steps: int = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )
        if model.training:
            loss = result[0] if return_outputs else result
            if not bool(torch.isfinite(loss).all().item()):
                self.n_nonfinite_loss += 1
        return result


class _EvalAtEndCallback(TrainerCallback):
    """Force one evaluation + checkpoint on the final optimizer step (even if off the `eval_steps` grid)."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps > 0 and state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


class _SkippedStepCallback(TrainerCallback):
    """Count optimizer steps skipped by the fp16 `GradScaler` (inf/NaN gradients)."""

    def __init__(self, trainer: Stage1Trainer) -> None:
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        accelerator = getattr(self._trainer, "accelerator", None)
        if accelerator is not None and bool(
            getattr(accelerator, "optimizer_step_was_skipped", False)
        ):
            self._trainer.n_skipped_optimizer_steps += 1
        return control


def build_stage1_trainer(
    model: Stage1Classifier,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: Any,
    early_stopping_patience: int = 0,
) -> Stage1Trainer:
    """Assemble `Stage1Trainer` with dynamic padding, metrics, final-step eval and early stopping."""
    trainer = Stage1Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
        compute_metrics=compute_stage1_metrics,
        **_tokenizer_kwarg(tokenizer),
    )
    trainer.add_callback(_EvalAtEndCallback())
    trainer.add_callback(_SkippedStepCallback(trainer))
    if int(early_stopping_patience) > 0:
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=int(early_stopping_patience))
        )
    return trainer


def extract_eval_history(log_history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert `Trainer.state.log_history` eval records into the sidecar `eval_history` format."""
    history: List[Dict[str, Any]] = []
    for rec in log_history:
        if "eval_macro_f1" not in rec:
            continue
        history.append(
            {
                "step": int(rec.get("step", 0)),
                "epoch": float(rec.get("epoch", 0.0)),
                "dev_macro_f1": float(rec["eval_macro_f1"]),
                "dev_loss": float(rec.get("eval_loss", float("nan"))),
                "dev_accuracy": float(rec.get("eval_accuracy", float("nan"))),
            }
        )
    return history


def resolve_best_step(state: Any) -> int:
    """Global step of the best checkpoint from `TrainerState` (falls back to the checkpoint dir name)."""
    step = getattr(state, "best_global_step", None)
    if step:
        return int(step)
    ckpt = getattr(state, "best_model_checkpoint", None)
    if ckpt:
        tail = Path(str(ckpt)).name.split("-")[-1]
        if tail.isdigit():
            return int(tail)
    return 0


def _find_latest_work_checkpoint(work_run_dir: Path) -> Optional[Path]:
    """Return the latest HF `checkpoint-{step}` directory in `work_run_dir` (or `None`)."""
    if not work_run_dir.is_dir():
        return None
    latest = get_last_checkpoint(str(work_run_dir))
    return Path(latest) if latest else None


@torch.no_grad()
def predict_logits7(
    model: Stage1Classifier,
    encodings: Union[Dataset, Sequence[Dict[str, Any]]],
    tokenizer: Any,
    batch_size: int,
    device: torch.device,
    use_fp16: bool = False,
) -> np.ndarray:
    """Run `Trainer.predict` on `encodings` and return raw 7-class logits of shape `(N, 7)`.

    `encodings` is a tokenized `datasets.Dataset` (a list of encodings is also accepted). Any `labels`
    column is dropped. Inference is FP32 unless `use_fp16=True` and `device` is CUDA.
    """
    dataset = (
        encodings
        if isinstance(encodings, Dataset)
        else TokenizedTextDataset(list(encodings))
    )
    if "labels" in dataset.column_names:
        dataset = dataset.remove_columns("labels")
    if len(dataset) == 0:
        return np.zeros((0, NUM_EMOTIONS), dtype=np.float64)

    on_cuda = device.type == "cuda"
    with tempfile.TemporaryDirectory() as tmp_dir:
        pred_kwargs: Dict[str, Any] = dict(
            output_dir=tmp_dir,
            per_device_eval_batch_size=int(batch_size),
            fp16=bool(use_fp16 and on_cuda),
            report_to="none",
            disable_tqdm=True,
            remove_unused_columns=True,
        )
        if not on_cuda:
            pred_kwargs["use_cpu"] = True
        predictor = Trainer(
            model=model,
            args=TrainingArguments(**pred_kwargs),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True),
            **_tokenizer_kwarg(tokenizer),
        )
        output = predictor.predict(dataset)

    logits = output.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return np.asarray(logits, dtype=np.float32)


def write_prediction7_file(
    ids: Sequence[int],
    logits: np.ndarray,
    out_path: Path,
    y_true7: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Build and atomically write a `PRED7_COLUMNS` CSV file."""
    if logits.ndim != 2 or logits.shape[1] != NUM_EMOTIONS:
        raise ValueError(
            f"Expected logits of shape (N, {NUM_EMOTIONS}), got {logits.shape}."
        )
    y_pred7 = np.argmax(logits, axis=1).astype(int)
    if y_true7 is not None:
        y_true_col: List[Union[int, str]] = [int(x) for x in y_true7]
    else:
        y_true_col = ["" for _ in range(len(ids))]

    data: Dict[str, Any] = {
        "id": [int(x) for x in ids],
        "y_true7": y_true_col,
        "y_pred7": y_pred7.tolist(),
    }
    for c_idx in range(NUM_EMOTIONS):
        data[f"logit_{c_idx}"] = logits[:, c_idx].astype(np.float64).tolist()

    df_pred = pd.DataFrame(data, columns=list(PRED7_COLUMNS))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".csv.tmp")
    write_csv_schema(df_pred, tmp_path, PRED7_COLUMNS)
    tmp_path.replace(out_path)
    return df_pred


def compute_cross_entropy_from_logits(
    logits: np.ndarray,
    labels: Sequence[int],
) -> float:
    """Compute mean cross-entropy loss in FP32 from raw logits `(N, C)` and integer labels `(N,)`."""
    if len(labels) == 0 or logits.shape[0] == 0:
        return 0.0
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(list(labels), dtype=torch.long)
    loss = F.cross_entropy(logits_t, labels_t)
    return float(loss.item())


def write_stage1_sidecar(
    results_dir: Union[str, Path],
    run_id: str,
    payload: Dict[str, Any],
) -> Path:
    """Atomically write Stage 1 diagnostics sidecar JSON to `<results_dir>/runs/<run_id>.json`."""
    runs_dir = Path(results_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = runs_dir / f"{run_id}.json"
    tmp_path = sidecar_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp_path.replace(sidecar_path)
    return sidecar_path


def load_stage1_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Read and normalize Stage 1 YAML configuration."""
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(raw).__name__}")
    return raw


def _load_split_dataframe(csv_path: Path, required_cols: Sequence[str]) -> pd.DataFrame:
    """Read a dataset CSV and verify required columns exist (`strict_order=False`)."""
    return read_csv_schema(
        csv_path, required_cols, strict_order=False, keep_default_na=False
    )
