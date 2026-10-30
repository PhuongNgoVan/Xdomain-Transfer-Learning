"""Text normalization utilities shared by preprocessing and duplicate checking.

Per `tasks/M0_scaffold.md`:
- Unicode NFC normalization
- Lowercase conversion
- Collapse repeated characters and whitespace
- Strictly idempotent: `normalize_text(normalize_text(x)) == normalize_text(x)`
"""

import re
import unicodedata

# Match 3 or more consecutive identical non-digit, non-whitespace characters
_REPEAT_CHAR_RE = re.compile(r"([^\s\d])\1{2,}", flags=re.UNICODE)
# Match 2 or more consecutive identical punctuation characters
_REPEAT_PUNCT_RE = re.compile(r"([!?.~,;:\-_+=*/\\|^&%$#@])\1+", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)

# Map Vietnamese precomposed accented vowels to their unaccented base vowel
# so trailing elongations like "quáaaa" or "trờiiii" collapse into "quá" / "trời".
_VIET_VOWEL_GROUPS = {
    "a": "aàáảãạ",
    "ă": "ăằắẳẵặ",
    "â": "âầấẩẫậ",
    "e": "eèéẻẽẹ",
    "ê": "êềếểễệ",
    "i": "iìíỉĩị",
    "o": "oòóỏõọ",
    "ô": "ôồốổỗộ",
    "ơ": "ơờớởỡợ",
    "u": "uùúủũụ",
    "ư": "ưừứửữự",
    "y": "yỳýỷỹỵ",
}

_ACCENTED_FOLLOWED_BY_BASE_PATTERNS = [
    re.compile(f"([{group}]){base}+", flags=re.UNICODE)
    for base, group in _VIET_VOWEL_GROUPS.items()
]


def collapse_repeated_chars(text: str) -> str:
    """Collapse elongated characters and repeated punctuation.

    1. Collapses 3+ identical non-digit, non-space characters to 1 (`đẹpppp` -> `đẹp`).
    2. Collapses Vietnamese accented vowel followed by repeated unaccented base vowel
       (`quáaaaa` -> `quá`, `trờiiii` -> `trời`).
    3. Collapses 2+ identical punctuation marks to 1 (`!!!` -> `!`).
    """
    text = _REPEAT_CHAR_RE.sub(r"\1", text)
    for pattern in _ACCENTED_FOLLOWED_BY_BASE_PATTERNS:
        text = pattern.sub(r"\1", text)
    text = _REPEAT_CHAR_RE.sub(r"\1", text)
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    return text


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection and preprocessing.

    Steps:
    1. Unicode NFC normalization (composes decomposed Vietnamese diacritics).
    2. Lowercase conversion.
    3. Collapse repeated characters and punctuation, followed by NFC re-normalization.
    4. Collapse all whitespace sequences (spaces, tabs, newlines, NBSP) to a single
       ASCII space and strip leading/trailing spaces.

    Args:
        text: Input string.

    Returns:
        Normalized string (guaranteed idempotent).

    Raises:
        TypeError: If `text` is not a `str`.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    out = unicodedata.normalize("NFC", text)
    out = out.lower()
    out = collapse_repeated_chars(out)
    out = unicodedata.normalize("NFC", out)
    out = collapse_repeated_chars(out)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out
