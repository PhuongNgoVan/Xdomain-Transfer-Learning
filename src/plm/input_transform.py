"""In-memory text transformation for Stage 1 ViSoBERT emoji experiments (`src/plm/input_transform.py`).

Per `tasks/AGENT_PLAN_visobert-vsmec-7c.md` Section 4 (T4):
- Provides `build_input_texts(df, input_col, alias, emoji_mode) -> List[str]`:
  - When `emoji_mode` is `None`, `""`, or `"raw"`: returns `[str(t) for t in df[input_col]]`
    unchanged (backward-compatible default).
  - When `emoji_mode == "keep"` and `alias == "ViSoBERT"` (or `"visobert"` in `alias.lower()`):
    returns `[preprocess_text(str(t), emoji_mode="keep") for t in df["text"]]` (applies NFC,
    lowercase, repeated-character collapsing, and preserves Unicode emojis and ASCII emoticons).
  - Any other `emoji_mode` (`"tag"`, `"remove"`, etc.) or `emoji_mode == "keep"` on PhoBERT
    raises `ValueError`.
- Provides `compute_texts_sha256(texts)` to record the deterministic SHA-256 digest of transformed
  texts per split in the Stage 1 sidecar file without touching `data/`.
"""

import hashlib
from typing import List, Optional, Sequence

import pandas as pd

from src.data.preprocess import preprocess_text

VALID_STAGE1_EMOJI_MODES = ("raw", "keep")


def normalize_emoji_mode(emoji_mode: Optional[str]) -> str:
    """Normalize optional `emoji_mode` config value (`None` or `'raw'` -> `'raw'`)."""
    if emoji_mode is None:
        return "raw"
    mode = str(emoji_mode).strip()
    return mode if mode else "raw"


def build_input_texts(
    df: pd.DataFrame,
    input_col: str,
    alias: str,
    emoji_mode: Optional[str] = None,
) -> List[str]:
    """Build in-memory list of input texts for tokenization according to `emoji_mode` and `alias`.

    Args:
        df: Split DataFrame loaded from `data/clean/*.csv`.
        input_col: Resolved input column name (`"text"` for ViSoBERT, `"text_seg"` for PhoBERT).
        alias: Backbone alias (`"ViSoBERT"`, `"PhoBERT"`, etc.).
        emoji_mode: Optional mode from config (`None`/`"raw"` or `"keep"`).

    Returns:
        List of transformed strings (same length and order as `df`).

    Raises:
        ValueError: If `emoji_mode` is unsupported or `"keep"` is used with a non-ViSoBERT backbone.
    """
    mode = normalize_emoji_mode(emoji_mode)
    if mode not in VALID_STAGE1_EMOJI_MODES:
        raise ValueError(
            f"Unsupported Stage 1 emoji_mode={emoji_mode!r}. "
            f"Allowed values are None, 'raw', or 'keep' (for ViSoBERT)."
        )

    if mode == "raw":
        if input_col not in df.columns:
            raise ValueError(
                f"Column {input_col!r} not found in DataFrame columns {list(df.columns)}."
            )
        return [str(t) for t in df[input_col].tolist()]

    # mode == "keep"
    alias_lower = str(alias).strip().lower()
    if "phobert" in alias_lower or "visobert" not in alias_lower:
        raise ValueError(
            f"emoji_mode='keep' is only supported for ViSoBERT backbone, got alias={alias!r}."
        )
    if "text" not in df.columns:
        raise ValueError(
            f"Column 'text' is required for emoji_mode='keep', got columns {list(df.columns)}."
        )

    return [preprocess_text(str(t), emoji_mode="keep") for t in df["text"].tolist()]


def compute_texts_sha256(texts: Sequence[str]) -> str:
    """Compute deterministic SHA-256 hex digest over a sequence of in-memory strings."""
    hasher = hashlib.sha256()
    for idx, t in enumerate(texts):
        hasher.update(f"{idx}:{t}\n".encode("utf-8"))
    return hasher.hexdigest()

