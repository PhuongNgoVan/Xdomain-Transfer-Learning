"""Tokenization, truncation statistics, and HF `datasets` construction (`src/plm/tokenizer_dataset.py`).

Length-grouped sampling is no longer implemented here: it is delegated to
`transformers.TrainingArguments(group_by_length=True)` (`LengthGroupedSampler`).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from datasets import Dataset


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


def TokenizedTextDataset(  # noqa: N802 - kept as a factory for backward-compatible naming
    encodings: Sequence[Dict[str, Any]],
    labels: Optional[Sequence[int]] = None,
) -> Dataset:
    """Build a `datasets.Dataset` from pre-tokenized `encodings` (no padding; padded per batch).

    `labels`, when given, are stored in a `labels` column (the name `Trainer` expects).
    """
    enc_list = list(encodings)
    if not enc_list:
        ds = Dataset.from_dict({"input_ids": [], "attention_mask": []})
    else:
        ds = Dataset.from_list(enc_list)
    if labels is not None:
        ds = ds.add_column("labels", [int(x) for x in labels])
    return ds


def tokenize_with_truncation_stats(
    texts: Sequence[str],
    tokenizer: Any,
    max_length: int = 256,
    labels: Optional[Sequence[int]] = None,
) -> Tuple[Dataset, Dict[str, Union[int, float]]]:
    """Tokenize `texts` into a `datasets.Dataset` (truncated to `max_length`) with exact truncation stats.

    The returned dataset has the tokenizer's native columns (`input_ids`, `attention_mask`,
    and `token_type_ids` if the tokenizer emits it) plus `labels` when `labels` is provided.
    """
    str_texts = [str(t) for t in texts]
    n_total = len(str_texts)
    max_len = int(max_length)
    if n_total == 0:
        return TokenizedTextDataset([], labels), {
            "n_total": 0,
            "n_truncated": 0,
            "truncation_rate": 0.0,
        }

    len_col = "__n_tokens_untruncated__"

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        # Full untruncated length (including special tokens) -> exact truncation statistics.
        untruncated = tokenizer(
            batch["text"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        truncated = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        out: Dict[str, Any] = {k: truncated[k] for k in truncated.keys()}
        out[len_col] = [len(ids) for ids in untruncated["input_ids"]]
        return out

    ds = Dataset.from_dict({"text": str_texts}).map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    n_truncated = sum(1 for n in ds[len_col] if n > max_len)
    ds = ds.remove_columns(len_col)
    if labels is not None:
        ds = ds.add_column("labels", [int(x) for x in labels])

    stats: Dict[str, Union[int, float]] = {
        "n_total": int(n_total),
        "n_truncated": int(n_truncated),
        "truncation_rate": float(n_truncated) / float(n_total),
    }
    return ds, stats
