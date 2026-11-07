"""Stage 1 PLM fine-tuning (7-class emotion on UIT-VSMEC) (`src/plm/stage1.py`).

Independent implementation per `tasks/M6_stage1_vsmec.md` (CV-P8):
- Architecture: `AutoModel` + `nn.Dropout(0.1)` + single `nn.Linear(hidden_size, 7)`
  applied to the `<s>` token representation (`last_hidden_state[:, 0, :]`).
  Does NOT use `AutoModelForSequenceClassification` (whose RoBERTa head has 2 layers).
- Data discipline:
  - PhoBERT reads `text_seg`; ViSoBERT reads `text`.
  - Trains on `vsmec_train.csv`, selects best checkpoint by `macro_f1` on `vsmec_dev.csv`.
  - Logs truncation rate at `max_length = 256`.
- Fixed checkpoint format in `{ckpt_dir}/{PhoBERT|ViSoBERT}/`:
  - `encoder/`: `encoder.save_pretrained(...)` + `tokenizer.save_pretrained(...)`
  - `head.pt`: `{"weight": Tensor(7, hidden_size), "bias": Tensor(7)}`
- Strict post-training order:
  1. Write 5 raw-logit prediction files in `{results_dir}/predictions7/{run_id}_{dataset}.csv`
     for `vsmec_dev`, `vsmec_test`, `neuesc_dev`, `neuesc_test`, `vsfc_test` (`PRED7_COLUMNS`).
  2. Compute `vsmec_dev` metrics from the written prediction file via `compute_metrics`
     and append 1 row to `{results_dir}/stage1.csv` (`STAGE1_COLUMNS`).
  3. Only after steps (1) and (2) succeed, delete intermediate checkpoints in `work_dir`
     while keeping the best checkpoint in `ckpt_dir`.
  4. Compute SHA-256 of encoder weights and `head.pt` and update
     `{results_dir}/stage1_ckpt_hash.json`.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import yaml

from src.common.constants import EMOTIONS7, ID_TO_EMOTION7
from src.common.hashing import sha256_file
from src.common.io import append_row_schema, read_csv_schema, write_csv_schema
from src.common.schema import PRED7_COLUMNS, SPLIT_COLUMNS, STAGE1_COLUMNS
from src.common.seed import set_seed
from src.eval.metrics import EMOTION7_IDS, compute_metrics

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


def resolve_input_column(backbone: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Resolve and validate the text column (`text_seg` for PhoBERT, `text` for ViSoBERT)."""
    lower = str(backbone).lower()
    cfg_col = config.get("input_column") if config else None

    if "phobert" in lower:
        if cfg_col is not None and cfg_col != "text_seg":
            raise ValueError(
                f"PhoBERT backbone ({backbone}) must use input_column='text_seg', got {cfg_col!r}."
            )
        return "text_seg"
    if "visobert" in lower:
        if cfg_col is not None and cfg_col != "text":
            raise ValueError(
                f"ViSoBERT backbone ({backbone}) must use input_column='text', got {cfg_col!r}."
            )
        return "text"
    return str(cfg_col) if cfg_col else "text"


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


class Stage1Classifier(nn.Module):
    """Single-linear-layer 7-class emotion classifier on top of `AutoModel` token `<s>` (index 0).

    Per `tasks/M6_stage1_vsmec.md` Section 4.2:
    - Does NOT use `AutoModelForSequenceClassification` (which has a 2-layer head in RoBERTa).
    - Applies `nn.Dropout(dropout)` + `nn.Linear(hidden_size, 7)` directly on
      `last_hidden_state[:, 0, :]`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int = NUM_EMOTIONS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = int(getattr(encoder.config, "hidden_size"))
        self.hidden_size = hidden_size
        self.num_labels = int(num_labels)
        self.dropout = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(hidden_size, self.num_labels)

    @classmethod
    def from_pretrained_backbone(
        cls,
        backbone_name_or_path: Union[str, Path],
        num_labels: int = NUM_EMOTIONS,
        dropout: float = 0.1,
    ) -> "Stage1Classifier":
        """Load `AutoModel` from `backbone_name_or_path` and attach fresh 7-class linear head."""
        encoder = AutoModel.from_pretrained(str(backbone_name_or_path))
        return cls(encoder=encoder, num_labels=num_labels, dropout=dropout)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput:
        encoder_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.head(self.dropout(cls_repr))

        loss: Optional[Tensor] = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels.long())
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def save_stage1_checkpoint(
        self,
        tokenizer: Any,
        ckpt_path: Union[str, Path],
    ) -> Dict[str, Path]:
        """Save `encoder/` (`save_pretrained` + tokenizer) and `head.pt` (`weight`, `bias`)."""
        root = Path(ckpt_path)
        encoder_dir = root / "encoder"
        encoder_dir.mkdir(parents=True, exist_ok=True)

        self.encoder.save_pretrained(encoder_dir)
        tokenizer.save_pretrained(encoder_dir)

        head_path = root / "head.pt"
        head_state = {
            "weight": self.head.weight.detach().cpu().clone(),
            "bias": self.head.bias.detach().cpu().clone(),
        }
        torch.save(head_state, head_path)
        return {"encoder_dir": encoder_dir, "head_path": head_path}

    @classmethod
    def load_stage1_checkpoint(
        cls,
        ckpt_path: Union[str, Path],
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple["Stage1Classifier", Any]:
        """Reload `Stage1Classifier` and tokenizer from `{ckpt_path}/encoder/` and `head.pt`."""
        root = Path(ckpt_path)
        encoder_dir = root / "encoder"
        head_path = root / "head.pt"
        if not encoder_dir.is_dir():
            raise FileNotFoundError(f"Encoder directory not found: {encoder_dir}")
        if not head_path.is_file():
            raise FileNotFoundError(f"Head file not found: {head_path}")

        tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
        encoder = AutoModel.from_pretrained(str(encoder_dir))
        head_state = torch.load(head_path, map_location="cpu", weights_only=True)
        weight: Tensor = head_state["weight"]
        bias: Tensor = head_state["bias"]

        model = cls(encoder=encoder, num_labels=int(weight.shape[0]), dropout=dropout)
        with torch.no_grad():
            model.head.weight.copy_(weight)
            model.head.bias.copy_(bias)

        if device is not None:
            model.to(device)
        model.eval()
        return model, tokenizer


class TokenizedTextDataset(Dataset):
    """Dataset of pre-tokenized inputs (without padding; padded dynamically per batch)."""

    def __init__(
        self,
        encodings: List[Dict[str, Any]],
        labels: Optional[Sequence[int]] = None,
    ) -> None:
        self.encodings = encodings
        self.labels = [int(x) for x in labels] if labels is not None else None

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.encodings[idx])
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def tokenize_with_truncation_stats(
    texts: Sequence[str],
    tokenizer: Any,
    max_length: int = 256,
) -> Tuple[List[Dict[str, Any]], Dict[str, Union[int, float]]]:
    """Tokenize `texts` with `max_length` truncation and compute exact truncation statistics."""
    str_texts = [str(t) for t in texts]
    n_total = len(str_texts)
    if n_total == 0:
        return [], {"n_total": 0, "n_truncated": 0, "truncation_rate": 0.0}

    # Measure full untruncated token lengths (including special tokens)
    untruncated = tokenizer(
        str_texts,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    n_truncated = sum(
        1 for ids in untruncated["input_ids"] if len(ids) > int(max_length)
    )
    trunc_rate = float(n_truncated) / float(n_total)

    truncated_batch = tokenizer(
        str_texts,
        truncation=True,
        max_length=int(max_length),
        padding=False,
    )
    keys = list(truncated_batch.keys())
    encodings: List[Dict[str, Any]] = []
    for i in range(n_total):
        encodings.append({k: truncated_batch[k][i] for k in keys})

    stats: Dict[str, Union[int, float]] = {
        "n_total": int(n_total),
        "n_truncated": int(n_truncated),
        "truncation_rate": trunc_rate,
    }
    return encodings, stats


def build_length_grouped_indices(
    encodings: Sequence[Dict[str, Any]],
    batch_size: int,
    seed: int,
    epoch: int = 0,
    group_by_length: bool = True,
) -> List[int]:
    """Produce deterministic training sample order (length-grouped mega-batches or shuffled)."""
    n = len(encodings)
    rng = np.random.default_rng(int(seed) + int(epoch) * 1009)
    perm = rng.permutation(n).tolist()
    if not group_by_length or n <= batch_size:
        return perm

    # Group into mega-batches of 50 * batch_size, sort by descending token length within mega-batch,
    # then shuffle batch chunks deterministically.
    mega_size = max(batch_size * 50, batch_size)
    ordered_batches: List[List[int]] = []
    for start in range(0, n, mega_size):
        mega = perm[start : start + mega_size]
        mega.sort(key=lambda idx: len(encodings[idx]["input_ids"]), reverse=True)
        for b_start in range(0, len(mega), batch_size):
            ordered_batches.append(mega[b_start : b_start + batch_size])

    batch_perm = rng.permutation(len(ordered_batches)).tolist()
    flat: List[int] = []
    for b_idx in batch_perm:
        flat.extend(ordered_batches[b_idx])
    return flat


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
    )

    all_logits: List[np.ndarray] = []
    amp_enabled = bool(use_fp16 and device.type == "cuda")
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
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


def _find_encoder_weight_file(encoder_dir: Path) -> Path:
    """Locate the primary weight file saved by `AutoModel.save_pretrained`."""
    for candidate in ("model.safetensors", "pytorch_model.bin"):
        p = encoder_dir / candidate
        if p.is_file():
            return p
    files = sorted(encoder_dir.glob("*.safetensors")) + sorted(encoder_dir.glob("*.bin"))
    if files:
        return files[0]
    raise FileNotFoundError(f"No encoder weight file found in {encoder_dir}")


def update_stage1_ckpt_hash(
    alias: str,
    ckpt_subdir: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    """Compute SHA-256 of encoder weights and `head.pt` and update `stage1_ckpt_hash.json`."""
    encoder_dir = ckpt_subdir / "encoder"
    head_path = ckpt_subdir / "head.pt"
    enc_weight_file = _find_encoder_weight_file(encoder_dir)

    enc_sha = sha256_file(enc_weight_file)
    head_sha = sha256_file(head_path)

    hash_file = results_dir / "stage1_ckpt_hash.json"
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if hash_file.is_file():
        try:
            existing = json.loads(hash_file.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    existing[alias] = {
        "path": ckpt_subdir.as_posix(),
        "encoder_sha256": enc_sha,
        "head_sha256": head_sha,
    }
    hash_file.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return existing


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
    grad_accum = int(cfg.get("gradient_accumulation_steps", cfg.get("grad_accum", 2)))
    epochs = int(cfg.get("num_train_epochs", cfg.get("epochs", 10)))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.06))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    dropout = float(cfg.get("dropout", 0.1))
    max_length = int(cfg.get("max_length", 256))
    precision = str(cfg.get("precision", "fp16"))
    group_by_length = bool(cfg.get("group_by_length", True))
    patience = int(cfg.get("early_stopping_patience", 5))
    save_total_limit = int(cfg.get("save_total_limit", 2))
    max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else (int(cfg["max_steps"]) if cfg.get("max_steps") is not None else None)
    )

    run_id = build_stage1_run_id(backbone=backbone, seed=seed, lr=lr, variant=variant, config=cfg)

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
    neuesc_dev_df = neuesc_full_df[neuesc_full_df["subset"] == "dev"].reset_index(drop=True)
    neuesc_test_df = neuesc_full_df[neuesc_full_df["subset"] == "test"].reset_index(drop=True)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(precision == "fp16" and device.type == "cuda")
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

    latest_resume = _find_latest_work_checkpoint(work_run_dir)
    if latest_resume is not None:
        resume_data = torch.load(latest_resume, map_location="cpu", weights_only=False)
        global_step = int(resume_data["step"])
        start_epoch = int(resume_data["epoch"])
        best_macro_f1 = float(resume_data["best_macro_f1"])
        best_step = int(resume_data["best_step"])
        patience_counter = int(resume_data["patience_counter"])
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
            batch_size=batch_size,
            device=device,
            use_fp16=use_fp16,
        )
        dev_preds = np.argmax(dev_logits, axis=1).tolist()
        dev_metrics = compute_metrics(
            dev_labels, dev_preds, logits=dev_logits, labels=EMOTION7_IDS
        )
        curr_f1 = float(dev_metrics["macro_f1"])

        if curr_f1 > best_macro_f1:
            best_macro_f1 = curr_f1
            best_step = int(curr_step)
            patience_counter = 0
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            model.save_stage1_checkpoint(tokenizer=tokenizer, ckpt_path=ckpt_subdir)
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
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for b_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels_t = batch["labels"].to(device)

            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_fp16
            ):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels_t,
                )
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
                    scaler.step(optimizer)
                    scaler.update()
                else:
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

    # Load best model at end (`load_best_model_at_end: true`)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
    model.eval()
    # Ensure best checkpoint is saved cleanly in `ckpt_subdir`
    model.save_stage1_checkpoint(tokenizer=tokenizer, ckpt_path=ckpt_subdir)

    elapsed_sec = time.time() - t0

    # 8. Post-training Step (a): Export 5 raw-logit prediction files (`PRED7_COLUMNS`)
    pred7_dir = r_dir / "predictions7"
    pred7_dir.mkdir(parents=True, exist_ok=True)

    eval_targets: List[Tuple[str, Sequence[int], List[Dict[str, Any]], Optional[Sequence[int]]]] = [
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
            batch_size=batch_size,
            device=device,
            use_fp16=use_fp16,
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

    # Verify ordering invariant before deleting intermediate checkpoints
    work_dir_existed_after_results = bool(
        work_run_dir.is_dir() and any(work_run_dir.glob("checkpoint-*"))
    )
    pre_cleanup_info = {
        "run_id": run_id,
        "prediction_paths": prediction_paths,
        "stage1_csv": stage1_csv,
        "work_run_dir": work_run_dir,
        "work_dir_existed_after_results": work_dir_existed_after_results,
    }
    if on_before_cleanup is not None:
        on_before_cleanup(pre_cleanup_info)

    # 10. Post-training Step (c): Remove intermediate checkpoints in `work_dir`, keep best checkpoint
    if work_run_dir.is_dir():
        shutil.rmtree(work_run_dir, ignore_errors=True)

    # 11. Post-training Step (d): Hash encoder weights & `head.pt` into `stage1_ckpt_hash.json`
    ckpt_hashes = update_stage1_ckpt_hash(
        alias=alias,
        ckpt_subdir=ckpt_subdir,
        results_dir=r_dir,
    )

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
        "dev_metrics": dev_metrics_final,
        "test_metrics": test_metrics_final,
        "truncation_report": truncation_report,
        "prediction_paths": prediction_paths,
        "stage1_csv": stage1_csv,
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
