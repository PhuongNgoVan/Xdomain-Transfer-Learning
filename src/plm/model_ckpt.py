"""Stage 1 classifier architecture, checkpoint persistence, and SHA-256 hashing (`src/plm/model_ckpt.py`).

Hugging Face integration notes:
- `Stage1Classifier` stays a plain `nn.Module` (custom single-linear head; checkpoint contract is
  `encoder/` via `save_pretrained` + `head.pt`), but exposes the attributes `transformers.Trainer`
  looks for (`config`, `main_input_name`, `accepts_loss_kwargs`, `_keys_to_ignore_on_save`,
  `gradient_checkpointing_enable`).
- `accepts_loss_kwargs = False` is important: the model returns a *mean* cross-entropy, so the
  Trainer must divide the loss by `gradient_accumulation_steps` itself (otherwise newer
  `transformers` versions assume a sum-normalized loss and the gradient is scaled by `grad_accum`).
- In `eval()` mode the forward pass is forced to FP32 (autocast disabled) regardless of the
  Trainer's AMP setting, so dev selection metrics are computed in FP32 as in the original design.
"""

import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from src.common.constants import EMOTIONS7
from src.common.hashing import sha256_file

NUM_EMOTIONS: int = len(EMOTIONS7)


class Stage1Classifier(nn.Module):
    """Single-linear-layer classifier on top of `AutoModel` token `<s>` (index 0).

    Per `tasks/M6_stage1_vsmec.md` Section 4.2:
    - Does NOT use `AutoModelForSequenceClassification` (which has a 2-layer head in RoBERTa).
    - Applies `nn.Dropout(dropout)` + `nn.Linear(hidden_size, num_labels)` directly on
      `last_hidden_state[:, 0, :]`.
    """

    # --- attributes read by `transformers.Trainer` -------------------------------------------
    main_input_name: str = "input_ids"
    supports_gradient_checkpointing: bool = True
    # Mean-reduced loss: let Trainer divide by grad_accum (do NOT pass `num_items_in_batch`).
    accepts_loss_kwargs: bool = False
    _keys_to_ignore_on_save: Optional[list] = None

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int = NUM_EMOTIONS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        hidden_size = int(getattr(encoder.config, "hidden_size"))
        self.hidden_size = hidden_size
        self.num_labels = int(num_labels)
        self.dropout = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(hidden_size, self.num_labels)
        # When True, `eval()` forward runs with autocast disabled (pure FP32 inference).
        self.force_fp32_eval: bool = True

    @classmethod
    def from_pretrained_backbone(
        cls,
        backbone_name_or_path: Union[str, Path],
        num_labels: int = NUM_EMOTIONS,
        dropout: float = 0.1,
    ) -> "Stage1Classifier":
        """Load `AutoModel` from `backbone_name_or_path` and attach fresh linear head."""
        encoder = AutoModel.from_pretrained(str(backbone_name_or_path))
        return cls(encoder=encoder, num_labels=num_labels, dropout=dropout)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """Delegate to the encoder so `TrainingArguments(gradient_checkpointing=True)` works."""
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.encoder.gradient_checkpointing_disable()

    def _precision_guard(self, device_type: str):
        """Disable autocast in eval mode (FP32 dev evaluation) when `force_fp32_eval` is set."""
        if (not self.training) and self.force_fp32_eval and device_type in ("cuda", "cpu"):
            return torch.autocast(device_type=device_type, enabled=False)
        return contextlib.nullcontext()

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

        with self._precision_guard(input_ids.device.type):
            outputs = self.encoder(**encoder_kwargs)
            cls_repr = outputs.last_hidden_state[:, 0, :]
            logits = self.head(self.dropout(cls_repr))

        loss: Optional[Tensor] = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.float(), labels.long())
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
    sweep: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute SHA-256 of encoder weights and `head.pt` and update `stage1_ckpt_hash.json` (or `ckpt_hash_sweep.json` when `sweep=True`)."""
    encoder_dir = ckpt_subdir / "encoder"
    head_path = ckpt_subdir / "head.pt"
    enc_weight_file = _find_encoder_weight_file(encoder_dir)

    enc_sha = sha256_file(enc_weight_file)
    head_sha = sha256_file(head_path)

    if sweep:
        if not run_id:
            raise ValueError("run_id is required when updating sweep checkpoint hash.")
        hash_file = results_dir / "ckpt_hash_sweep.json"
        key = str(run_id)
    else:
        hash_file = results_dir / "stage1_ckpt_hash.json"
        key = str(alias)

    hash_file.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if hash_file.is_file():
        try:
            existing = json.loads(hash_file.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    existing[key] = {
        "path": ckpt_subdir.as_posix(),
        "encoder_sha256": enc_sha,
        "head_sha256": head_sha,
    }
    hash_file.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return existing
