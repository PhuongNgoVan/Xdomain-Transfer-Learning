"""Stage 1 training utilities, logit prediction, CSV export, and Hugging Face Trainer helpers (`src/plm/trainer_utils.py`).

Hugging Face integration notes:
- Training/evaluation/checkpoint-resume are delegated to `transformers.Trainer`
  (`build_training_arguments`, `Stage1Trainer`, `build_compute_metrics`, `extract_eval_history`,
  `best_step_from_trainer`).
- `predict_logits7` runs inference through `Trainer.predict` (FP32 unless `use_fp16=True`).
- The hand-written temporary checkpoint saver (`_save_temp_work_checkpoint`) was removed; the Trainer
  writes `checkpoint-{step}/` itself (honouring `save_total_limit`). `_find_latest_work_checkpoint`
  is kept as a thin wrapper over `transformers.trainer_utils.get_last_checkpoint`.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
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
from src.eval.metrics import compute_metrics
from src.plm.model_ckpt import Stage1Classifier
from src.plm.tokenizer_dataset import TokenizedTextDataset, build_data_collator

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
# Inference through `Trainer.predict`
# ---------------------------------------------------------------------------
def predict_logits_hf(
    model: Stage1Classifier,
    encodings: Sequence[Dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
    device: torch.device,
    num_labels: int,
    use_fp16: bool = False,
) -> np.ndarray:
    """Run `Trainer.predict` on `encodings` and return raw logits `(N, num_labels)` as float32.

    Inference is FP32 by default (`Stage1Classifier.force_fp32_eval`); with `use_fp16=True` it runs
    under fp16 autocast (CUDA only).
    """
    if len(encodings) == 0:
        return np.zeros((0, num_labels), dtype=np.float64)

    has_cuda = device.type == "cuda" and torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(Path(tempfile.gettempdir()) / "stage1_predict_unused"),
        per_device_eval_batch_size=int(batch_size),
        fp16=bool(use_fp16 and has_cuda),
        use_cpu=(device.type != "cuda"),
        remove_unused_columns=False,
        dataloader_pin_memory=has_cuda,
        report_to="none",
        save_strategy="no",
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=build_data_collator(tokenizer),
    )

    prev_flag = getattr(model, "force_fp32_eval", True)
    model.force_fp32_eval = not use_fp16
    try:
        out = trainer.predict(TokenizedTextDataset(list(encodings), labels=None))
    finally:
        model.force_fp32_eval = prev_flag

    preds = out.predictions
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    return np.asarray(preds, dtype=np.float32)


def predict_logits7(
    model: Stage1Classifier,
    encodings: Sequence[Dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
    device: torch.device,
    use_fp16: bool = False,
) -> np.ndarray:
    """Run inference on `encodings` and return raw 7-class logits of shape `(N, 7)`."""
    return predict_logits_hf(
        model=model,
        encodings=encodings,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        num_labels=NUM_EMOTIONS,
        use_fp16=use_fp16,
    )


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


# ---------------------------------------------------------------------------
# Hugging Face Trainer helpers
# ---------------------------------------------------------------------------
def build_training_arguments(
    output_dir: Union[str, Path],
    *,
    batch_size: int,
    eval_batch_size: int,
    grad_accum: int,
    epochs: int,
    max_steps: Optional[int],
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    max_grad_norm: float,
    precision_effective: str,
    eval_steps: int,
    save_total_limit: int,
    group_by_length: bool,
    seed: int,
    pin_memory: bool,
) -> TrainingArguments:
    """Build `TrainingArguments` equivalent to the former hand-written training loop.

    - Linear warmup/decay scheduler with `warmup_steps` (computed as `ceil(total_steps * warmup_ratio)`).
    - AdamW (Trainer default) with no weight decay on bias / LayerNorm parameters.
    - Eval + save every `eval_steps` optimizer steps; best model = highest `macro_f1` on dev,
      restored at the end (`load_best_model_at_end=True`).
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(eval_batch_size),
        gradient_accumulation_steps=int(grad_accum),
        num_train_epochs=float(epochs),
        max_steps=int(max_steps) if max_steps is not None else -1,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        warmup_steps=int(warmup_steps),
        lr_scheduler_type="linear",
        max_grad_norm=float(max_grad_norm),
        bf16=(precision_effective == "bf16"),
        fp16=(precision_effective == "fp16"),
        eval_strategy="steps",
        eval_steps=int(eval_steps),
        save_strategy="steps",
        save_steps=int(eval_steps),
        save_total_limit=int(save_total_limit),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        group_by_length=bool(group_by_length),
        logging_strategy="steps",
        logging_steps=int(eval_steps),
        seed=int(seed),
        data_seed=int(seed),
        remove_unused_columns=False,  # custom nn.Module (not a PreTrainedModel)
        label_names=["labels"],
        report_to="none",
        dataloader_pin_memory=bool(pin_memory),
    )


def build_compute_metrics(
    label_ids: Sequence[int],
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """`compute_metrics` for `Trainer`; `macro_f1` drives best-model selection + early stopping."""
    ids = list(label_ids)

    def _fn(eval_pred: EvalPrediction) -> Dict[str, float]:
        preds = eval_pred.predictions
        logits = np.asarray(preds[0] if isinstance(preds, (tuple, list)) else preds)
        labels = np.asarray(eval_pred.label_ids).astype(int)
        m = compute_metrics(
            labels.tolist(),
            np.argmax(logits, axis=1).tolist(),
            logits=logits,
            labels=ids,
        )
        return {
            "macro_f1": float(m["macro_f1"]),
            "weighted_f1": float(m["weighted_f1"]),
            "accuracy": float(m["accuracy"]),
        }

    return _fn


class _SkippedStepCallback(TrainerCallback):
    """Counts optimizer steps skipped by the fp16 GradScaler (inf/NaN grads)."""

    def __init__(self, trainer: "Stage1Trainer") -> None:
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        acc = getattr(self._trainer, "accelerator", None)
        if acc is not None and bool(getattr(acc, "optimizer_step_was_skipped", False)):
            self._trainer.n_skipped_optimizer_steps += 1
        return control


class _EvaluateAtEndCallback(TrainerCallback):
    """Always evaluate + save at the last optimizer step (as the former loop did)."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps and state.max_steps > 0 and state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


class Stage1Trainer(Trainer):
    """`Trainer` that also tracks diagnostics for the sidecar JSON.

    Attributes: `n_nonfinite_loss` (micro-batches with non-finite loss) and
    `n_skipped_optimizer_steps` (fp16 GradScaler skipped steps).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.n_nonfinite_loss: int = 0
        self.n_skipped_optimizer_steps: int = 0
        self.add_callback(_SkippedStepCallback(self))
        self.add_callback(_EvaluateAtEndCallback())

    def training_step(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        loss = super().training_step(*args, **kwargs)
        if not bool(torch.isfinite(loss).all().item()):
            self.n_nonfinite_loss += 1
        return loss


def best_step_from_trainer(trainer: Trainer) -> int:
    """Optimizer step of the best checkpoint (falls back to the final global step)."""
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        tail = Path(best_ckpt).name.split("-")[-1]
        if tail.isdigit():
            return int(tail)
    return int(trainer.state.global_step)


def extract_eval_history(trainer: Trainer) -> List[Dict[str, Any]]:
    """Convert `trainer.state.log_history` eval entries into the sidecar `eval_history` format."""
    history: List[Dict[str, Any]] = []
    for h in trainer.state.log_history:
        if "eval_macro_f1" not in h:
            continue
        history.append(
            {
                "step": int(h["step"]),
                "epoch": float(h["epoch"]),
                "dev_macro_f1": float(h["eval_macro_f1"]),
                "dev_loss": float(h["eval_loss"]) if "eval_loss" in h else None,
                "dev_accuracy": float(h["eval_accuracy"]),
            }
        )
    return history


def _find_latest_work_checkpoint(work_run_dir: Path) -> Optional[Path]:
    """Return the latest Trainer `checkpoint-{step}/` directory in `work_run_dir` if present."""
    if not Path(work_run_dir).is_dir():
        return None
    last = get_last_checkpoint(str(work_run_dir))
    return Path(last) if last else None


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
