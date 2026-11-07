"""Stage 1 training utilities, logit prediction, CSV export, and temporary checkpoint management (`src/plm/trainer_utils.py`)."""

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
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
    alias = infer_backbone_alias(backbone, config=config)
    return f"{alias}_Stage1-{variant}_s{int(seed)}_lr{format_lr(lr)}"


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


@torch.no_grad()
def predict_logits7(
    model: Stage1Classifier,
    encodings: Sequence[Dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
    device: torch.device,
    use_fp16: bool = False,
) -> np.ndarray:
    """Run inference on `encodings` and return raw 7-class logits of shape `(N, 7)`."""
    model.eval()
    dataset = TokenizedTextDataset(list(encodings), labels=None)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    all_logits: List[np.ndarray] = []
    amp_enabled = bool(use_fp16 and device.type == "cuda")
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        logits_np = out.logits.detach().float().cpu().numpy()
        all_logits.append(logits_np)

    if not all_logits:
        return np.zeros((0, NUM_EMOTIONS), dtype=np.float64)
    return np.vstack(all_logits)


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


def _save_temp_work_checkpoint(
    work_run_dir: Path,
    step: int,
    epoch: int,
    best_macro_f1: float,
    best_step: int,
    patience_counter: int,
    model: Stage1Classifier,
    optimizer: AdamW,
    scheduler: Any,
    save_total_limit: int = 2,
) -> Path:
    """Save an intermediate training state in `work_run_dir/checkpoint-{step}` and prune old ones."""
    ckpt_dir = work_run_dir / f"checkpoint-{int(step)}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / "trainer_state.pt"
    torch.save(
        {
            "step": int(step),
            "epoch": int(epoch),
            "best_macro_f1": float(best_macro_f1),
            "best_step": int(best_step),
            "patience_counter": int(patience_counter),
            "model_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        state_path,
    )

    # Enforce save_total_limit on intermediate checkpoints
    all_ckpts = sorted(
        work_run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if save_total_limit > 0 and len(all_ckpts) > save_total_limit:
        for old_ckpt in all_ckpts[:-save_total_limit]:
            shutil.rmtree(old_ckpt, ignore_errors=True)
    return ckpt_dir


def _find_latest_work_checkpoint(work_run_dir: Path) -> Optional[Path]:
    """Return the latest `checkpoint-{step}/trainer_state.pt` in `work_run_dir` if present."""
    if not work_run_dir.is_dir():
        return None
    all_ckpts = sorted(
        work_run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    for ckpt in reversed(all_ckpts):
        st = ckpt / "trainer_state.pt"
        if st.is_file():
            return st
    return None


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
