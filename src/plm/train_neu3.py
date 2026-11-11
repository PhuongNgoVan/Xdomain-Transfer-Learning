"""Step 1.2 PLM fine-tuning (3-class sentiment polarity on NEU-ESC) (`src/plm/train_neu3.py`).

Implements `run_neu3` and CLI `main()` per `tasks/AGENT_neu3_visobert.md`:
- Uses `Stage1Classifier(num_labels=3)` (`nn.Dropout` + single `nn.Linear(hidden_size, 3)` on `<s>`).
- Applies in-memory text transformation via `build_input_texts` (`emoji_mode: keep` for ViSoBERT on
  column `text`) without modifying any files in `data/clean/`.
- Enforces strict dev-only model selection and early stopping on `neuesc_dev`.
- When `run_kind == "sweep"`, `neuesc_test` is never validated, transformed, tokenized, or evaluated,
  and no model checkpoint or checkpoint hash is written.
- When `run_kind == "base"`, evaluates both `neuesc_dev` and `neuesc_test` in FP32,
  saves the best checkpoint to `{ckpt_dir}/{alias}/` (`encoder/` + `head.pt`), and records
  SHA-256 hashes in `{results_dir}/neu3_ckpt_hash.json` keyed by `run_id`.
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
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

from src.common.constants import ID_TO_LABEL3, LABELS3, NEG, NEU, POS
from src.common.hashing import sha256_file
from src.common.io import append_row_schema, read_csv_schema, write_csv_schema
from src.common.schema import NEU3_COLUMNS, NEU3_KEY_COLUMNS, PRED_COLUMNS
from src.common.seed import set_seed
from src.eval.metrics import LABEL3_IDS, compute_metrics
from src.plm.input_transform import (
    VALID_STAGE1_EMOJI_MODES,
    build_input_texts,
    compute_texts_sha256,
    normalize_emoji_mode,
)
from src.plm.model_ckpt import (
    Stage1Classifier,
    _find_encoder_weight_file,
)
from src.plm.tokenizer_dataset import (
    TokenizedTextDataset,
    build_length_grouped_indices,
    resolve_input_column,
    tokenize_with_truncation_stats,
)
from src.plm.trainer_utils import (
    VALID_PRECISIONS,
    _find_latest_work_checkpoint,
    _save_temp_work_checkpoint,
    compute_cross_entropy_from_logits,
    format_lr,
    get_git_commit,
    infer_backbone_alias,
    load_stage1_config,
    resolve_precision,
    write_stage1_sidecar,
)

NUM_LABELS3: int = len(LABELS3)
VALID_RUN_KINDS: Tuple[str, ...] = ("sweep", "base")
VALID_LABEL3_SET = {NEG, NEU, POS}

__all__ = [
    "LABELS3",
    "LABEL3_IDS",
    "NUM_LABELS3",
    "Stage1Classifier",
    "TokenizedTextDataset",
    "VALID_PRECISIONS",
    "VALID_RUN_KINDS",
    "VALID_STAGE1_EMOJI_MODES",
    "build_input_texts",
    "build_length_grouped_indices",
    "build_neu3_run_id",
    "compute_cross_entropy_from_logits",
    "compute_texts_sha256",
    "format_lr",
    "get_git_commit",
    "infer_backbone_alias",
    "load_neuesc_splits",
    "load_stage1_config",
    "main",
    "normalize_emoji_mode",
    "predict_logits3",
    "resolve_input_column",
    "resolve_precision",
    "run_neu3",
    "tokenize_with_truncation_stats",
    "update_neu3_ckpt_hash",
    "write_prediction3_file",
]


def build_neu3_run_id(
    backbone: str,
    seed: int,
    lr: float,
    variant: str = "base",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Construct NEU-3 `run_id` per Section 7 (`{alias}_NEU3-{variant}_s{seed}_lr{format_lr(lr)}`)."""
    v_str = str(variant).strip()
    if not v_str or "_" in v_str:
        raise ValueError(
            f"NEU-3 variant must be non-empty and must not contain '_' (got {variant!r})."
        )
    alias = infer_backbone_alias(backbone, config=config)
    return f"{alias}_NEU3-{v_str}_s{int(seed)}_lr{format_lr(lr)}"


def _resolve_run_kind(cfg: Dict[str, Any]) -> str:
    """Resolve `run_kind` (`'sweep'` or `'base'`) from config."""
    if "run_kind" in cfg and cfg["run_kind"] is not None:
        rk = str(cfg["run_kind"]).strip()
        if rk not in VALID_RUN_KINDS:
            raise ValueError(
                f"Invalid run_kind {rk!r}. Must be one of {VALID_RUN_KINDS}."
            )
        return rk
    if bool(cfg.get("sweep", False)):
        return "sweep"
    return "base"


def _validate_split_df(
    df: pd.DataFrame,
    split_name: str,
    input_col: str,
    label_col: str,
) -> None:
    """Validate a single NEU-ESC split DataFrame (`train`, `dev`, or `test`)."""
    if len(df) == 0:
        raise ValueError(f"NEU-ESC subset={split_name!r} is empty.")

    if df["id"].isna().any():
        raise ValueError(f"NEU-ESC subset={split_name!r} contains NaN in 'id'.")
    if df["id"].duplicated().any():
        dup_ids = df.loc[df["id"].duplicated(), "id"].tolist()[:5]
        raise ValueError(
            f"NEU-ESC subset={split_name!r} contains duplicate IDs: {dup_ids}"
        )

    if df[input_col].isna().any():
        raise ValueError(
            f"NEU-ESC subset={split_name!r} contains NaN in input column {input_col!r}."
        )
    empty_mask = df[input_col].astype(str).str.strip() == ""
    if bool(empty_mask.any()):
        raise ValueError(
            f"NEU-ESC subset={split_name!r} contains empty/whitespace text in column {input_col!r}."
        )

    if df[label_col].isna().any():
        raise ValueError(
            f"NEU-ESC subset={split_name!r} contains NaN in label column {label_col!r}."
        )
    try:
        labels_int = df[label_col].astype(int)
        if not np.all(labels_int == df[label_col].to_numpy()):
            raise ValueError("Non-integer label values detected.")
    except Exception as exc:
        raise ValueError(
            f"NEU-ESC subset={split_name!r} has non-integer labels in {label_col!r}: {exc}"
        ) from exc

    unique_labels = set(labels_int.unique().tolist())
    if not unique_labels.issubset(VALID_LABEL3_SET):
        invalid = sorted(unique_labels - VALID_LABEL3_SET)
        raise ValueError(
            f"NEU-ESC subset={split_name!r} contains invalid labels {invalid}; "
            f"expected only {sorted(VALID_LABEL3_SET)}."
        )


def load_neuesc_splits(
    csv_path: Union[str, Path],
    input_col: str,
    run_kind: str = "base",
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, int]]:
    """Load `neuesc_100.csv` and validate active splits.

    Important contract guarantee (Section 2 & Section 9):
    - When `run_kind == "sweep"`, `test` rows are NEVER inspected, validated, or returned
      (`test_df` is `None`), preventing any accidental use of `test` during LR sweeps.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"NEU-ESC file not found: {path}")

    df = pd.read_csv(path)
    label_col = "label" if "label" in df.columns else (
        "sentiment_id" if "sentiment_id" in df.columns else "label"
    )
    required_cols = ("id", "subset", input_col, label_col)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {path}. Found columns: {list(df.columns)}"
        )

    if df["subset"].isna().any():
        raise ValueError(f"Column 'subset' in {path} contains NaN values.")

    subset_counts_raw = df["subset"].astype(str).value_counts().to_dict()
    for req_sub in ("train", "dev", "test"):
        if subset_counts_raw.get(req_sub, 0) <= 0:
            raise ValueError(
                f"Required subset {req_sub!r} is missing or empty in {path} (counts={subset_counts_raw})."
            )
    unexpected_subsets = set(subset_counts_raw.keys()) - {"train", "dev", "test"}
    if unexpected_subsets:
        raise ValueError(
            f"Unexpected subset values {sorted(unexpected_subsets)} in {path}."
        )

    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    train_df = df[df["subset"] == "train"].reset_index(drop=True)
    dev_df = df[df["subset"] == "dev"].reset_index(drop=True)

    _validate_split_df(train_df, "train", input_col=input_col, label_col="label")
    _validate_split_df(dev_df, "dev", input_col=input_col, label_col="label")

    # Check cross-split duplicate IDs across active splits
    train_dev_ids = pd.concat([train_df["id"], dev_df["id"]], ignore_index=True)
    if train_dev_ids.duplicated().any():
        raise ValueError("Duplicate 'id' detected across NEU-ESC train and dev splits.")

    test_df: Optional[pd.DataFrame] = None
    if run_kind == "base":
        test_df = df[df["subset"] == "test"].reset_index(drop=True)
        _validate_split_df(test_df, "test", input_col=input_col, label_col="label")
        all_ids = pd.concat([train_dev_ids, test_df["id"]], ignore_index=True)
        if all_ids.duplicated().any():
            raise ValueError(
                "Duplicate 'id' detected across NEU-ESC train/dev/test splits."
            )

    split_sizes = {
        "train": int(len(train_df)),
        "dev": int(len(dev_df)),
        "test": int(subset_counts_raw["test"]),
    }
    return train_df, dev_df, test_df, split_sizes


@torch.no_grad()
def predict_logits3(
    model: Stage1Classifier,
    encodings: Sequence[Dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
    device: torch.device,
    use_fp16: bool = False,
) -> np.ndarray:
    """Run inference on `encodings` and return raw 3-class logits of shape `(N, 3)`."""
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
        return np.zeros((0, NUM_LABELS3), dtype=np.float64)
    return np.vstack(all_logits)


def write_prediction3_file(
    ids: Sequence[int],
    logits: np.ndarray,
    out_path: Path,
    y_true: Sequence[int],
) -> pd.DataFrame:
    """Build and atomically write a 3-class `PRED_COLUMNS` CSV file (`id, y_true, y_pred, logit_neg, logit_neu, logit_pos`)."""
    if logits.ndim != 2 or logits.shape[1] != NUM_LABELS3:
        raise ValueError(
            f"Expected logits of shape (N, {NUM_LABELS3}), got {logits.shape}."
        )
    y_pred = np.argmax(logits, axis=1).astype(int)
    data: Dict[str, Any] = {
        "id": [int(x) for x in ids],
        "y_true": [int(x) for x in y_true],
        "y_pred": y_pred.tolist(),
        "logit_neg": logits[:, NEG].astype(np.float64).tolist(),
        "logit_neu": logits[:, NEU].astype(np.float64).tolist(),
        "logit_pos": logits[:, POS].astype(np.float64).tolist(),
    }

    df_pred = pd.DataFrame(data, columns=list(PRED_COLUMNS))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".csv.tmp")
    write_csv_schema(df_pred, tmp_path, PRED_COLUMNS)
    tmp_path.replace(out_path)
    return df_pred


def update_neu3_ckpt_hash(
    run_id: str,
    alias: str,
    ckpt_subdir: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    """Compute SHA-256 of encoder weights and `head.pt` and update `neu3_ckpt_hash.json` keyed by `run_id`."""
    if not run_id:
        raise ValueError("run_id is required when updating neu3_ckpt_hash.json.")

    encoder_dir = ckpt_subdir / "encoder"
    head_path = ckpt_subdir / "head.pt"
    enc_weight_file = _find_encoder_weight_file(encoder_dir)

    enc_sha = sha256_file(enc_weight_file)
    head_sha = sha256_file(head_path)

    hash_file = Path(results_dir) / "neu3_ckpt_hash.json"
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if hash_file.is_file():
        try:
            existing = json.loads(hash_file.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    existing[str(run_id)] = {
        "alias": str(alias),
        "path": ckpt_subdir.as_posix(),
        "encoder_sha256": enc_sha,
        "head_sha256": head_sha,
    }
    tmp_path = hash_file.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp_path.replace(hash_file)
    return existing


def _format_label_distribution(labels: Sequence[int]) -> str:
    """Format count and percentage of `{0: negative, 1: neutral, 2: positive}` for logging."""
    total = max(1, len(labels))
    arr = np.asarray(labels, dtype=int)
    parts: List[str] = []
    for cid in LABEL3_IDS:
        cnt = int(np.sum(arr == cid))
        pct = (cnt / total) * 100.0
        parts.append(f"{cid}:{ID_TO_LABEL3[cid]}={cnt} ({pct:.2f}%)")
    return ", ".join(parts)


def run_neu3(
    config_path: Union[str, Path],
    data_dir: Union[str, Path] = "data/clean",
    ckpt_dir: Union[str, Path] = "checkpoints/neu3",
    results_dir: Union[str, Path] = "results",
    work_dir: Union[str, Path] = "work_tmp/neu3",
    max_steps_override: Optional[int] = None,
    on_before_cleanup: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Execute the full NEU-3 training, prediction export, evaluation, and cleanup pipeline."""
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
        False
        if run_kind == "sweep"
        else bool(cfg.get("save_checkpoint", True))
    )
    raw_emoji_mode = cfg.get("emoji_mode")
    emoji_mode = normalize_emoji_mode(raw_emoji_mode)
    num_labels = int(cfg.get("num_labels", NUM_LABELS3))
    if num_labels != NUM_LABELS3:
        raise ValueError(
            f"NEU-3 requires num_labels={NUM_LABELS3}, got {num_labels}."
        )

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
    autocast_dtype_str = (
        "torch.bfloat16"
        if use_bf16
        else ("torch.float16" if use_fp16 else "torch.float32 (autocast disabled)")
    )

    run_id = build_neu3_run_id(
        backbone=backbone, seed=seed, lr=lr, variant=variant, config=cfg
    )

    d_dir = Path(data_dir)
    c_dir = Path(ckpt_dir)
    r_dir = Path(results_dir)
    w_dir = Path(work_dir)
    neu3_csv = r_dir / "neu3.csv"
    ckpt_subdir = c_dir / alias

    # 1. Check SKIP condition in `neu3.csv` (`sweep` requires `dev`; `base` requires `dev` and `test`)
    required_splits = {"dev"} if run_kind == "sweep" else {"dev", "test"}
    if neu3_csv.is_file():
        df_existing = read_csv_schema(neu3_csv, NEU3_COLUMNS)
        df_run = df_existing[df_existing["run_id"].astype(str) == run_id]
        existing_splits = set(df_run["split"].astype(str).tolist())
        if required_splits.issubset(existing_splits):
            print(f"=== [SKIP] run_id={run_id} ===")
            return {
                "skipped": True,
                "run_id": run_id,
                "run_kind": run_kind,
                "neu3_csv": neu3_csv,
                "ckpt_subdir": ckpt_subdir,
            }

    # 2. Set deterministic seed
    set_seed(seed, deterministic=True)

    # 3. Load and validate `neuesc_100.csv` (never touches test when `run_kind == "sweep"`)
    neuesc_path = d_dir / "neuesc_100.csv"
    data_hash = sha256_file(neuesc_path)
    code_version = get_git_commit()
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )

    train_df, dev_df, test_df, split_sizes = load_neuesc_splits(
        neuesc_path, input_col=input_col, run_kind=run_kind
    )
    train_labels = train_df["label"].astype(int).tolist()
    dev_labels = dev_df["label"].astype(int).tolist()
    test_labels: Optional[List[int]] = (
        test_df["label"].astype(int).tolist() if test_df is not None else None
    )

    print(
        f"[NEU3 Start] run_id={run_id} | run_kind={run_kind} | backbone={backbone} ({alias}) | "
        f"variant={variant} | input_col={input_col} | emoji_mode={emoji_mode} | seed={seed} | lr={format_lr(lr)}"
    )
    print(
        f"[NEU3 Precision] requested={precision} | effective={precision_effective} | "
        f"autocast_dtype={autocast_dtype_str} | strict_precision={strict_precision} | gpu={gpu_name}"
    )
    print(
        f"[NEU3 Provenance] git_commit={code_version} | data_hash={data_hash} | "
        f"config_path={cfg_path.as_posix()}"
    )
    print(
        f"[NEU3 Splits] n_train={split_sizes['train']} | n_dev={split_sizes['dev']} | "
        f"n_test={split_sizes['test']} ({'loaded' if run_kind == 'base' else 'untouched in sweep'})"
    )
    print(f"[NEU3 Labels] train: {_format_label_distribution(train_labels)}")
    print(f"[NEU3 Labels] dev:   {_format_label_distribution(dev_labels)}")
    if test_labels is not None:
        print(f"[NEU3 Labels] test:  {_format_label_distribution(test_labels)}")

    # 4. Initialize tokenizer and 3-class Stage1Classifier
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = Stage1Classifier.from_pretrained_backbone(
        backbone_name_or_path=backbone,
        num_labels=NUM_LABELS3,
        dropout=dropout,
    )
    model.to(device)

    # 5. Transform input texts in-memory (`emoji_mode`) and tokenize train & dev (and test ONLY when `run_kind == "base"`)
    active_split_dfs: List[Tuple[str, pd.DataFrame]] = [
        ("neuesc_train", train_df),
        ("neuesc_dev", dev_df),
    ]
    if run_kind == "base" and test_df is not None:
        active_split_dfs.append(("neuesc_test", test_df))

    transformed_texts: Dict[str, List[str]] = {}
    transformed_text_hashes: Dict[str, str] = {}
    truncation_report: Dict[str, Dict[str, Union[int, float, str]]] = {}
    encoded_splits: Dict[str, List[Dict[str, Any]]] = {}

    for ds_name, df_split in active_split_dfs:
        texts_list = build_input_texts(
            df=df_split,
            input_col=input_col,
            alias=alias,
            emoji_mode=raw_emoji_mode,
        )
        transformed_texts[ds_name] = texts_list
        text_sha = compute_texts_sha256(texts_list)
        transformed_text_hashes[ds_name] = text_sha

        enc_list, trunc_st = tokenize_with_truncation_stats(
            texts_list, tokenizer, max_length=max_length
        )
        trunc_st_with_hash: Dict[str, Union[int, float, str]] = dict(trunc_st)
        trunc_st_with_hash["text_sha256"] = text_sha
        truncation_report[ds_name] = trunc_st_with_hash
        encoded_splits[ds_name] = enc_list

    train_enc = encoded_splits["neuesc_train"]
    dev_enc = encoded_splits["neuesc_dev"]
    test_enc: Optional[List[Dict[str, Any]]] = encoded_splits.get("neuesc_test")

    for ds_name, st in truncation_report.items():
        print(
            f"[Tokenization] {ds_name} ({input_col}, emoji_mode={emoji_mode}): "
            f"truncated={st['n_truncated']}/{st['n_total']} ({float(st['truncation_rate']) * 100:.2f}%)"
        )

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
        dev_logits = predict_logits3(
            model=model,
            encodings=dev_enc,
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
            device=device,
            use_fp16=eval_fp16,
        )
        dev_preds = np.argmax(dev_logits, axis=1).tolist()
        dev_metrics = compute_metrics(
            dev_labels, dev_preds, logits=dev_logits, labels=LABEL3_IDS
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

    # 8. Restore best state dict and save checkpoint once (only when `run_kind == "base"` and `save_checkpoint`)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
    model.eval()
    if save_checkpoint:
        model.save_stage1_checkpoint(tokenizer=tokenizer, ckpt_path=ckpt_subdir)

    elapsed_sec = time.time() - t0

    # 9. Export FP32 predictions to `predictions3/` and compute metrics from the written CSVs
    pred3_dir = r_dir / "predictions3"
    pred3_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths: Dict[str, Path] = {}

    dev_logits_arr = predict_logits3(
        model=model,
        encodings=dev_enc,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        device=device,
        use_fp16=False,
    )
    dev_pred_csv = pred3_dir / f"{run_id}_neuesc_dev.csv"
    write_prediction3_file(
        ids=dev_df["id"].astype(int).tolist(),
        logits=dev_logits_arr,
        out_path=dev_pred_csv,
        y_true=dev_labels,
    )
    prediction_paths["neuesc_dev"] = dev_pred_csv

    logit3_cols = ["logit_neg", "logit_neu", "logit_pos"]
    dev_pred_df = read_csv_schema(dev_pred_csv, PRED_COLUMNS)
    dev_metrics_final = compute_metrics(
        dev_pred_df["y_true"].astype(int).tolist(),
        dev_pred_df["y_pred"].astype(int).tolist(),
        logits=dev_pred_df[logit3_cols].to_numpy(dtype=np.float64),
        labels=LABEL3_IDS,
    )

    test_metrics_final: Optional[Dict[str, Any]] = None
    if run_kind == "base" and test_df is not None and test_enc is not None and test_labels is not None:
        test_logits_arr = predict_logits3(
            model=model,
            encodings=test_enc,
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
            device=device,
            use_fp16=False,
        )
        test_pred_csv = pred3_dir / f"{run_id}_neuesc_test.csv"
        write_prediction3_file(
            ids=test_df["id"].astype(int).tolist(),
            logits=test_logits_arr,
            out_path=test_pred_csv,
            y_true=test_labels,
        )
        prediction_paths["neuesc_test"] = test_pred_csv

        test_pred_df = read_csv_schema(test_pred_csv, PRED_COLUMNS)
        test_metrics_final = compute_metrics(
            test_pred_df["y_true"].astype(int).tolist(),
            test_pred_df["y_pred"].astype(int).tolist(),
            logits=test_pred_df[logit3_cols].to_numpy(dtype=np.float64),
            labels=LABEL3_IDS,
        )

    # 10. Append row(s) to `{results_dir}/neu3.csv` (`NEU3_COLUMNS`)
    ckpt_path_str = ckpt_subdir.as_posix() if save_checkpoint else ""
    n_train = int(len(train_df))

    def _build_neu3_row(
        split_name: str,
        metrics_dict: Dict[str, Any],
        trunc_rate: float,
    ) -> Dict[str, Any]:
        f1_neg = float(metrics_dict["f1_neg"])
        f1_neu = float(metrics_dict["f1_neu"])
        f1_pos = float(metrics_dict["f1_pos"])
        return {
            "run_id": run_id,
            "backbone": backbone,
            "stage1": "NEU",
            "stage2": "",
            "eval_domain": "NEU",
            "label_view": "V3-NEU",
            "mode": "base",
            "fraction": 100,
            "seed": seed,
            "split": split_name,
            "macro_f1": float(metrics_dict["macro_f1"]),
            "weighted_f1": float(metrics_dict["weighted_f1"]),
            "accuracy": float(metrics_dict["accuracy"]),
            "f1_neg": f1_neg,
            "f1_neu": f1_neu,
            "f1_pos": f1_pos,
            "f1_per_class": json.dumps([f1_neg, f1_neu, f1_pos]),
            "ckpt_path": ckpt_path_str,
            "run_kind": run_kind,
            "variant": variant,
            "lr": lr,
            "best_step": best_step,
            "n_train": n_train,
            "precision": precision,
            "gpu": gpu_name,
            "code_version": code_version,
            "data_hash": data_hash,
            "config_path": cfg_path.as_posix(),
            "truncation_rate": float(trunc_rate),
        }

    dev_row = _build_neu3_row(
        split_name="dev",
        metrics_dict=dev_metrics_final,
        trunc_rate=float(truncation_report["neuesc_dev"]["truncation_rate"]),
    )
    append_row_schema(
        row=dev_row,
        path=neu3_csv,
        expected_columns=NEU3_COLUMNS,
        key_columns=NEU3_KEY_COLUMNS,
        overwrite_duplicate_key=True,
    )

    if run_kind == "base" and test_metrics_final is not None:
        test_row = _build_neu3_row(
            split_name="test",
            metrics_dict=test_metrics_final,
            trunc_rate=float(truncation_report["neuesc_test"]["truncation_rate"]),
        )
        append_row_schema(
            row=test_row,
            path=neu3_csv,
            expected_columns=NEU3_COLUMNS,
            key_columns=NEU3_KEY_COLUMNS,
            overwrite_duplicate_key=True,
        )

    # Write diagnostics sidecar (`<results_dir>/runs/<run_id>.json`)
    sidecar_payload: Dict[str, Any] = {
        "run_id": run_id,
        "alias": alias,
        "backbone": backbone,
        "lr": lr,
        "seed": seed,
        "variant": variant,
        "run_kind": run_kind,
        "precision": precision,
        "precision_effective": precision_effective,
        "emoji_mode": emoji_mode,
        "transformed_text_sha256": transformed_text_hashes,
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
            "test_macro_f1": (
                float(test_metrics_final["macro_f1"])
                if test_metrics_final is not None
                else None
            ),
        },
    }
    sidecar_path = write_stage1_sidecar(
        results_dir=r_dir,
        run_id=run_id,
        payload=sidecar_payload,
    )

    work_dir_existed_after_results = bool(
        work_run_dir.is_dir() and any(work_run_dir.glob("checkpoint-*"))
    )
    pre_cleanup_info = {
        "run_id": run_id,
        "prediction_paths": prediction_paths,
        "neu3_csv": neu3_csv,
        "sidecar_path": sidecar_path,
        "work_run_dir": work_run_dir,
        "work_dir_existed_after_results": work_dir_existed_after_results,
    }
    if on_before_cleanup is not None:
        on_before_cleanup(pre_cleanup_info)

    # 11. Remove intermediate checkpoints in `work_dir`
    if work_run_dir.is_dir():
        shutil.rmtree(work_run_dir, ignore_errors=True)

    # 12. Update `neu3_ckpt_hash.json` keyed by `run_id` when `save_checkpoint=True`
    if save_checkpoint:
        ckpt_hashes = update_neu3_ckpt_hash(
            run_id=run_id,
            alias=alias,
            ckpt_subdir=ckpt_subdir,
            results_dir=r_dir,
        )
    else:
        ckpt_hashes = {}

    test_f1_str = (
        f"{test_metrics_final['macro_f1']:.4f}"
        if test_metrics_final is not None
        else "NA"
    )
    print(
        f"=== [DONE] {run_id}: best_step={best_step}, "
        f"dev_macro_f1={dev_metrics_final['macro_f1']:.4f}, "
        f"test_macro_f1={test_f1_str}, "
        f"elapsed={elapsed_sec:.1f}s ==="
    )

    return {
        "skipped": False,
        "run_id": run_id,
        "run_kind": run_kind,
        "alias": alias,
        "best_step": best_step,
        "precision": precision,
        "precision_effective": precision_effective,
        "emoji_mode": emoji_mode,
        "transformed_texts": transformed_texts,
        "transformed_text_sha256": transformed_text_hashes,
        "dev_metrics": dev_metrics_final,
        "test_metrics": test_metrics_final,
        "truncation_report": truncation_report,
        "eval_history": eval_history,
        "n_skipped_optimizer_steps": n_skipped_optimizer_steps,
        "n_nonfinite_loss": n_nonfinite_loss,
        "prediction_paths": prediction_paths,
        "neu3_csv": neu3_csv,
        "sidecar_path": sidecar_path,
        "ckpt_subdir": ckpt_subdir,
        "ckpt_hashes": ckpt_hashes,
        "elapsed_sec": elapsed_sec,
        "work_dir_existed_after_results": work_dir_existed_after_results,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Step 1.2 PLM 3-class sentiment fine-tuning on NEU-ESC."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to NEU-3 YAML config (e.g. configs/exp/neu3_visobert.yaml).",
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
        default=Path("checkpoints/neu3"),
        help="Directory to store the best NEU-3 checkpoints (encoder/ + head.pt).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write neu3.csv, predictions3/, and neu3_ckpt_hash.json.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work_tmp/neu3"),
        help="Local temporary directory for intermediate training checkpoints.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional override for maximum optimization steps (for smoke testing).",
    )
    args = parser.parse_args(argv)

    run_neu3(
        config_path=args.config,
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        results_dir=args.results_dir,
        work_dir=args.work_dir,
        max_steps_override=args.max_steps,
    )


if __name__ == "__main__":
    main()

