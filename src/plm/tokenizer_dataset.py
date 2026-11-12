"""Tokenization, truncation statistics, dataset, and collation (`src/plm/tokenizer_dataset.py`).

Hugging Face integration notes:
- Tokenization is done with the model's own `AutoTokenizer` (fast batch call).
- `TokenizedTextDataset` yields `input_ids` / `attention_mask` (+ `labels`), so `transformers.Trainer`
  can consume it directly, including its `LengthGroupedSampler` (`group_by_length=True`), which reads
  `input_ids` lengths straight from the items.
- `build_data_collator` returns the standard `DataCollatorWithPadding` (dynamic per-batch padding).
- `build_length_grouped_indices` is the legacy hand-written sampler; the Trainer-based training
  path no longer uses it (kept for backward compatibility with older imports/tests).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding


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


def build_data_collator(
    tokenizer: Any,
    pad_to_multiple_of: Optional[int] = None,
) -> DataCollatorWithPadding:
    """Standard HF dynamic-padding collator (`labels` are stacked into a LongTensor)."""
    return DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=pad_to_multiple_of
    )


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
    """Legacy: deterministic training sample order (length-grouped mega-batches or shuffled).

    Not used by the Trainer-based pipeline (`TrainingArguments(group_by_length=...)` is used instead).
    """
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
