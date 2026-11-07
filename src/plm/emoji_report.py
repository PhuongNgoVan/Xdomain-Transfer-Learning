"""Emoji & text normalization inspection report for ViSoBERT Stage 1 (`src/plm/emoji_report.py`).

Per `tasks/AGENT_PLAN_visobert-vsmec-7c.md` Section 4 (T5) and Section 5:
- Purely read-only diagnostic module (never modifies `data/` or `preprocess_text`).
- Computes and prints:
  1. Total sentences in VSMEC train (`vsmec_train.csv`), and number of sentences containing
     Unicode emojis (and combined Unicode emojis + ASCII emoticons).
  2. Number of sentences that lose emojis after `preprocess_text(text, emoji_mode="keep")`
     (expected: 0).
  3. Number of sentences changed by normalization (`preprocess_text(text, emoji_mode="keep") != text`),
     plus breakdown of raw `text` vs normalized text across `vsmec_train`, `vsmec_dev`, `vsmec_test`
     (confirming Section 0.5 observation: uppercase characters, repeated characters, etc.).
  4. Number of unique emojis in VSMEC train (and in `UNICODE_EMOJI_MAP`) that tokenize into `<unk>`
     when tokenized standalone by the ViSoBERT tokenizer (`uitnlp/visobert`).
  5. Token `<unk>` rate on sentences containing emojis in raw format vs `keep` format.
"""

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
from transformers import AutoTokenizer

from src.data.preprocess import (
    EMOTICON_PATTERNS,
    UNICODE_EMOJI_MAP,
    _GENERIC_EMOJI_RE,
    preprocess_text,
)

_REPEAT_3PLUS_RE = re.compile(r"([^\s\d])\1{2,}", flags=re.UNICODE)


def extract_unicode_emojis(text: str) -> List[str]:
    """Return all Unicode emoji characters found in `text` (preserving order of appearance)."""
    s = str(text)
    found: List[str] = []
    for ch in s:
        if ch in UNICODE_EMOJI_MAP or _GENERIC_EMOJI_RE.match(ch):
            found.append(ch)
    return found


def has_ascii_emoticon(text: str) -> bool:
    """Return True if `text` matches any canonical ASCII emoticon pattern."""
    s = str(text)
    return any(pat.search(s) is not None for pat, _ in EMOTICON_PATTERNS)


def _tokenize_content_ids(tokenizer: Any, text: str) -> Tuple[List[int], int]:
    """Return `(content_token_ids, unk_token_id)` excluding special tokens (`<s>`, `</s>`, `<pad>`)."""
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is None:
        unk_id = 3
    ids: List[int] = list(
        tokenizer.encode(str(text), add_special_tokens=False, truncation=False)
    )
    return ids, int(unk_id)


def compute_emoji_and_normalization_report(
    data_dir: Union[str, Path] = "data/clean",
    tokenizer: Optional[Any] = None,
    backbone: str = "uitnlp/visobert",
) -> Dict[str, Any]:
    """Run read-only emoji and text normalization diagnostics on `data/clean/vsmec_*.csv`."""
    d_dir = Path(data_dir)
    train_df = pd.read_csv(d_dir / "vsmec_train.csv")
    raw_texts = [str(t) for t in train_df["text"].tolist()]
    keep_texts = [preprocess_text(t, emoji_mode="keep") for t in raw_texts]

    n_total_train = len(raw_texts)
    n_with_unicode_emoji = 0
    n_with_any_emoji_or_emoticon = 0
    n_lost_unicode_emoji_after_keep = 0
    n_changed_by_keep = 0
    n_with_uppercase = 0
    n_with_repeated_chars = 0

    unique_emojis_in_train: Set[str] = set()
    emoji_sentence_indices: List[int] = []

    for idx, (raw_t, keep_t) in enumerate(zip(raw_texts, keep_texts)):
        raw_emos = extract_unicode_emojis(raw_t)
        keep_emos = extract_unicode_emojis(keep_t)
        has_uni = len(raw_emos) > 0
        has_emo_any = has_uni or has_ascii_emoticon(raw_t)

        if has_uni:
            n_with_unicode_emoji += 1
            unique_emojis_in_train.update(raw_emos)
            emoji_sentence_indices.append(idx)
            if set(raw_emos) - set(keep_emos):
                n_lost_unicode_emoji_after_keep += 1
        if has_emo_any:
            n_with_any_emoji_or_emoticon += 1
        if raw_t != keep_t:
            n_changed_by_keep += 1
        if any(ch.isupper() for ch in raw_t):
            n_with_uppercase += 1
        if _REPEAT_3PLUS_RE.search(raw_t):
            n_with_repeated_chars += 1

    # Check all 3 VSMEC splits for Section 0.5 observation confirmation
    splits_normalization_summary: Dict[str, Dict[str, int]] = {}
    for sp in ("train", "dev", "test"):
        sp_path = d_dir / f"vsmec_{sp}.csv"
        if sp_path.is_file():
            sp_df = pd.read_csv(sp_path)
            sp_raw = [str(t) for t in sp_df["text"].tolist()]
            sp_keep = [preprocess_text(t, emoji_mode="keep") for t in sp_raw]
            splits_normalization_summary[f"vsmec_{sp}"] = {
                "n_total": len(sp_raw),
                "n_changed_by_keep": sum(1 for r, k in zip(sp_raw, sp_keep) if r != k),
                "n_uppercase": sum(1 for r in sp_raw if any(c.isupper() for c in r)),
                "n_repeated_3plus": sum(
                    1 for r in sp_raw if _REPEAT_3PLUS_RE.search(r) is not None
                ),
                "n_with_unicode_emoji": sum(
                    1 for r in sp_raw if len(extract_unicode_emojis(r)) > 0
                ),
            }

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(backbone, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(backbone)

    # Standalone emoji tokenization check
    sorted_train_emojis = sorted(unique_emojis_in_train)
    n_standalone_emoji_unk = 0
    standalone_unk_emojis: List[str] = []
    standalone_known_emojis: List[str] = []

    for emo in sorted_train_emojis:
        ids, unk_id = _tokenize_content_ids(tokenizer, emo)
        if not ids or any(i == unk_id for i in ids):
            n_standalone_emoji_unk += 1
            standalone_unk_emojis.append(emo)
        else:
            standalone_known_emojis.append(emo)

    # `<unk>` token rate on emoji-containing sentences in raw vs keep
    raw_total_tokens = 0
    raw_unk_tokens = 0
    keep_total_tokens = 0
    keep_unk_tokens = 0

    for idx in emoji_sentence_indices:
        r_ids, unk_id = _tokenize_content_ids(tokenizer, raw_texts[idx])
        k_ids, _ = _tokenize_content_ids(tokenizer, keep_texts[idx])
        raw_total_tokens += len(r_ids)
        raw_unk_tokens += sum(1 for i in r_ids if i == unk_id)
        keep_total_tokens += len(k_ids)
        keep_unk_tokens += sum(1 for i in k_ids if i == unk_id)

    raw_unk_rate = (
        float(raw_unk_tokens) / float(raw_total_tokens) if raw_total_tokens > 0 else 0.0
    )
    keep_unk_rate = (
        float(keep_unk_tokens) / float(keep_total_tokens)
        if keep_total_tokens > 0
        else 0.0
    )

    return {
        "backbone": backbone,
        "n_vsmec_train_total": n_total_train,
        "n_vsmec_train_with_unicode_emoji": n_with_unicode_emoji,
        "n_vsmec_train_with_any_emoji_or_emoticon": n_with_any_emoji_or_emoticon,
        "n_vsmec_train_lost_emoji_after_keep": n_lost_unicode_emoji_after_keep,
        "n_vsmec_train_changed_by_keep": n_changed_by_keep,
        "n_vsmec_train_with_uppercase": n_with_uppercase,
        "n_vsmec_train_with_repeated_chars": n_with_repeated_chars,
        "n_unique_unicode_emojis_in_train": len(sorted_train_emojis),
        "n_standalone_emojis_mapped_to_unk": n_standalone_emoji_unk,
        "standalone_unk_emojis": standalone_unk_emojis,
        "standalone_known_emojis": standalone_known_emojis,
        "emoji_sentences_raw_total_tokens": raw_total_tokens,
        "emoji_sentences_raw_unk_tokens": raw_unk_tokens,
        "emoji_sentences_raw_unk_rate": round(raw_unk_rate, 6),
        "emoji_sentences_keep_total_tokens": keep_total_tokens,
        "emoji_sentences_keep_unk_tokens": keep_unk_tokens,
        "emoji_sentences_keep_unk_rate": round(keep_unk_rate, 6),
        "splits_normalization_summary": splits_normalization_summary,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="T5: Read-only emoji & normalization report for ViSoBERT on UIT-VSMEC."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--backbone", type=str, default="uitnlp/visobert")
    args = parser.parse_args(argv)

    report = compute_emoji_and_normalization_report(
        data_dir=args.data_dir,
        backbone=args.backbone,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

