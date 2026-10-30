"""CV-N2: Unified text preprocessing and one-time Vietnamese word segmentation.

Per `docs/contract.md` and `tasks/M2_preprocess.md`:
- Steps:
  1. Emoji / emoticon mapping to canonical tokens (`emoji_mode = "tag"` by default,
     documented in `docs/preprocess.md` and configured in `configs/base.yaml`).
  2. Unicode NFC, lowercase, collapse repeated characters and whitespace
     (via `src.common.textnorm.normalize_text`).
  3. Vietnamese word segmentation via VnCoreNLP wrapper for PhoBERT (`text_seg` column);
     ViSoBERT uses the unsegmented normalized text.
- Runs word segmentation ONCE and stores `text_seg` into `data/clean/*.csv` (and
  optionally `data/splits/*.csv`), updating SHA-256 hashes in `data/clean/MANIFEST.json`.
- VSMEC, NEU-ESC, and UIT-VSFC share the exact same preprocessing function.
- Short-sentence filter (`<= 2` words) is a configurable flag (`filter_short: bool = False`),
  OFF by default.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import unicodedata

import pandas as pd
import yaml

from src.common.hashing import build_manifest
from src.common.io import read_csv_schema, validate_columns, write_csv_schema
from src.common.schema import SPLIT_COLUMNS
from src.common.textnorm import normalize_text

CLEAN_SEG_COLUMNS: Tuple[str, ...] = (*SPLIT_COLUMNS, "text_seg")

# ---------------------------------------------------------------------------
# 1. Canonical Emoji & Emoticon Mapping (`emoji_mode = "tag"`)
# ---------------------------------------------------------------------------
EMOTICON_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<!\w)(?:=+\)+|:+\){2,}|:-\)+|:+\)+)(?!\w)"), " emo_joy "),
    (re.compile(r"(?<!\w)(?::+v+|:+d+|x+d+)(?!\w)", flags=re.IGNORECASE), " emo_joy "),
    (re.compile(r"(?<!\w)(?:<+3+)(?!\w)"), " emo_love "),
    (re.compile(r"(?<!\w)(?::+\(+|:-\(+|T_T|;_\s*;)(?!\w)"), " emo_sad "),
    (re.compile(r"(?<!\w)(?::-?/|:-?\\|-_-)(?!\w)"), " emo_neutral "),
    (re.compile(r"(?<!\w)(?::+o+|:+0+)(?!\w)", flags=re.IGNORECASE), " emo_surprise "),
)

UNICODE_EMOJI_MAP: Dict[str, str] = {
    # Joy / Enjoyment
    "😀": "emo_joy",
    "😁": "emo_joy",
    "😂": "emo_joy",
    "🤣": "emo_joy",
    "😃": "emo_joy",
    "😄": "emo_joy",
    "😅": "emo_joy",
    "😆": "emo_joy",
    "😉": "emo_joy",
    "😊": "emo_joy",
    "😋": "emo_joy",
    "😎": "emo_joy",
    "🙂": "emo_joy",
    "🤗": "emo_joy",
    "🥳": "emo_joy",
    "😸": "emo_joy",
    "😹": "emo_joy",
    # Love / Positive
    "😍": "emo_love",
    "🥰": "emo_love",
    "😘": "emo_love",
    "😗": "emo_love",
    "😙": "emo_love",
    "😚": "emo_love",
    "❤️": "emo_love",
    "❤": "emo_love",
    "💕": "emo_love",
    "💖": "emo_love",
    "💗": "emo_love",
    "💓": "emo_love",
    "💞": "emo_love",
    "💘": "emo_love",
    "👍": "emo_like",
    "👏": "emo_clap",
    # Sadness
    "😢": "emo_sad",
    "😭": "emo_sad",
    "😞": "emo_sad",
    "😔": "emo_sad",
    "😟": "emo_sad",
    "😕": "emo_sad",
    "🙁": "emo_sad",
    "☹️": "emo_sad",
    "☹": "emo_sad",
    "😣": "emo_sad",
    "😖": "emo_sad",
    "😫": "emo_sad",
    "😩": "emo_sad",
    "🥺": "emo_sad",
    "💔": "emo_sad",
    # Anger
    "😠": "emo_anger",
    "😡": "emo_anger",
    "🤬": "emo_anger",
    "😤": "emo_anger",
    "💢": "emo_anger",
    # Disgust / Dislike
    "🤢": "emo_disgust",
    "🤮": "emo_disgust",
    "💩": "emo_disgust",
    "👎": "emo_dislike",
    "🙄": "emo_disgust",
    # Fear
    "😨": "emo_fear",
    "😰": "emo_fear",
    "😱": "emo_fear",
    "😳": "emo_fear",
    # Surprise
    "😮": "emo_surprise",
    "😯": "emo_surprise",
    "😲": "emo_surprise",
    "🤯": "emo_surprise",
    # Neutral / Thinking
    "🤔": "emo_thinking",
    "😐": "emo_neutral",
    "😑": "emo_neutral",
    "😶": "emo_neutral",
}

# Match general Unicode emoji code points (Emoticons, Symbols & Pictographs, Supplemental)
_GENERIC_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000027BF"
    "]",
    flags=re.UNICODE,
)

# Variation selectors (e.g. U+FE0F) to strip around emojis
_VARIATION_SELECTOR_RE = re.compile(r"[\uFE0E\uFE0F]")


def map_emojis(text: str, emoji_mode: str = "tag") -> str:
    """Map Unicode emojis and common ASCII emoticons to canonical tokens.

    Args:
        text: Input string.
        emoji_mode:
            - `"tag"` (default): Replace emojis/emoticons with semantic tokens
              like `emo_joy`, `emo_sad`, or `emo_uXXXX`.
            - `"remove"`: Strip emojis.
            - `"keep"`: Keep original emojis untouched.
    """
    if emoji_mode == "keep":
        return text
    if emoji_mode not in ("tag", "remove"):
        raise ValueError(f"Unsupported emoji_mode: {emoji_mode!r}")

    out = _VARIATION_SELECTOR_RE.sub("", text)

    if emoji_mode == "remove":
        for pat, _ in EMOTICON_PATTERNS:
            out = pat.sub(" ", out)
        out = _GENERIC_EMOJI_RE.sub(" ", out)
        return out

    # Replace ASCII emoticons first
    for pat, replacement in EMOTICON_PATTERNS:
        out = pat.sub(replacement, out)

    # Replace known Unicode emojis
    chars: List[str] = []
    for ch in out:
        if ch in UNICODE_EMOJI_MAP:
            chars.append(f" {UNICODE_EMOJI_MAP[ch]} ")
        elif _GENERIC_EMOJI_RE.match(ch):
            chars.append(f" emo_u{ord(ch):x} ")
        else:
            chars.append(ch)
    return "".join(chars)


def preprocess_text(text: str, emoji_mode: str = "tag") -> str:
    """Unified text normalization + emoji mapping for VSMEC, NEU-ESC, and UIT-VSFC.

    Idempotent: `preprocess_text(preprocess_text(x)) == preprocess_text(x)`.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    # Step 1: NFC normalize first so decomposed characters are canonical
    s = unicodedata.normalize("NFC", text)
    # Step 2: Map emojis and emoticons to canonical tokens
    s = map_emojis(s, emoji_mode=emoji_mode)
    # Step 3: Apply `normalize_text` (NFC, lowercase, collapse repeated chars & whitespace)
    s = normalize_text(s)
    return s


# ---------------------------------------------------------------------------
# 2. Vietnamese Word Segmentation (`py_vncorenlp`, strict — no fallback)
# ---------------------------------------------------------------------------
_VNCORENLP_CACHE: Dict[str, Any] = {}


def _ensure_java_home() -> None:
    """Populate `JAVA_HOME` in `os.environ` from Windows Registry if not inherited by the shell."""
    if os.environ.get("JAVA_HOME"):
        return
    if sys.platform == "win32":
        try:
            import winreg

            for root, subkey in (
                (winreg.HKEY_CURRENT_USER, r"Environment"),
                (
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                ),
            ):
                try:
                    with winreg.OpenKey(root, subkey) as key:
                        val, _ = winreg.QueryValueEx(key, "JAVA_HOME")
                        if val and Path(str(val)).is_dir():
                            os.environ["JAVA_HOME"] = str(val)
                            return
                except OSError:
                    continue
        except Exception:
            pass


class VietnameseWordSegmenter:
    """Strict wrapper around `py_vncorenlp` (`annotators=['wseg']`) producing `_`-joined tokens.

    Does NOT fall back to `underthesea` or any other segmenter: if VnCoreNLP cannot be
    initialized from `vncorenlp_dir`, raises a clear `RuntimeError` stating the path tried.
    For unit tests in `pytest`, pass `custom_fn` (mock segmenter) to avoid JVM/network dependency.
    """

    def __init__(
        self,
        backend: str = "vncorenlp",
        vncorenlp_dir: Optional[Union[str, Path]] = None,
        custom_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.backend = backend
        self.vncorenlp_dir = Path(vncorenlp_dir) if vncorenlp_dir is not None else None
        self._custom_fn = custom_fn
        self._annotator: Any = None
        self._active_backend: str = "uninitialized"
        self._init_backend()

    def _init_backend(self) -> None:
        if self._custom_fn is not None:
            self._active_backend = "custom"
            return

        tried_path = str(self.vncorenlp_dir) if self.vncorenlp_dir is not None else "<None>"
        if self.backend != "vncorenlp":
            raise RuntimeError(
                f"Unsupported segmenter backend {self.backend!r} (tried path: {tried_path}). "
                "Only 'vncorenlp' is allowed; automatic fallback is disabled."
            )

        if self.vncorenlp_dir is None:
            raise RuntimeError(
                f"Failed to initialize VnCoreNLP: 'vncorenlp_dir' is not set "
                f"(tried path: {tried_path}). Configure 'preprocess.vncorenlp_dir' in configs/base.yaml."
            )

        resolved_dir = self.vncorenlp_dir.resolve()
        if not resolved_dir.is_dir():
            raise RuntimeError(
                f"Failed to initialize VnCoreNLP: directory does not exist "
                f"(tried path: {resolved_dir})."
            )

        jar_files = list(resolved_dir.glob("VnCoreNLP*.jar"))
        models_dir = resolved_dir / "models"
        if not jar_files or not models_dir.is_dir():
            raise RuntimeError(
                f"Failed to initialize VnCoreNLP: missing VnCoreNLP*.jar or models/ directory "
                f"(tried path: {resolved_dir})."
            )

        cache_key = str(resolved_dir)
        if cache_key in _VNCORENLP_CACHE:
            self._annotator = _VNCORENLP_CACHE[cache_key]
            self._active_backend = "vncorenlp"
            return

        _ensure_java_home()
        prev_cwd = os.getcwd()
        try:
            import py_vncorenlp  # type: ignore[import-untyped]

            self._annotator = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir=str(resolved_dir)
            )
            _VNCORENLP_CACHE[cache_key] = self._annotator
            self._active_backend = "vncorenlp"
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize VnCoreNLP from tried path '{resolved_dir}': {exc}"
            ) from exc
        finally:
            # py_vncorenlp.VnCoreNLP.__init__ changes working directory via os.chdir(save_dir);
            # always restore the caller's working directory.
            try:
                os.chdir(prev_cwd)
            except OSError:
                pass

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def segment(self, normalized_text: str) -> str:
        """Segment a single normalized sentence into PhoBERT format (`word_one word_two`).

        Protects `emo_*` tags so they remain intact single tokens.
        """
        if not normalized_text:
            return ""

        if self._custom_fn is not None:
            seg = self._custom_fn(normalized_text)
        elif self._active_backend == "vncorenlp":
            sents = self._annotator.word_segment(normalized_text)
            seg = " ".join(sents)
        else:
            raise RuntimeError(f"Segmenter backend not initialized: {self._active_backend!r}")

        # Ensure any `emo_ xxx` or `emo _ xxx` split by VnCoreNLP is rejoined to `emo_xxx`
        seg = re.sub(r"\bemo\s*_\s*([a-z0-9]+)\b", r"emo_\1", seg)
        seg = re.sub(r"\s+", " ", seg).strip()
        return seg


def count_words(text_seg: str) -> int:
    """Count word tokens (excluding pure punctuation tokens) in segmented or normalized text."""
    if not text_seg or not text_seg.strip():
        return 0
    tokens = [tok for tok in text_seg.strip().split(" ") if any(ch.isalnum() for ch in tok)]
    return len(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    segmenter: Optional[VietnameseWordSegmenter] = None,
    emoji_mode: str = "tag",
    filter_short: bool = False,
    min_words: int = 3,
) -> pd.DataFrame:
    """Preprocess `df` (with `text` column) and populate `text_seg`.

    Uses the exact same logic for VSMEC, NEU-ESC, and UIT-VSFC.
    When `filter_short=True`, filters out sentences with `<= 2` words (`< min_words`).
    By default (`filter_short=False`), keeps all rows.
    """
    if "text" not in df.columns:
        raise ValueError(f"DataFrame must contain 'text' column, got {list(df.columns)}")

    if segmenter is not None:
        seg_engine = segmenter
    else:
        cfg = load_preprocess_config()
        seg_engine = VietnameseWordSegmenter(
            backend=str(cfg.get("segmenter", "vncorenlp")),
            vncorenlp_dir=cfg.get("vncorenlp_dir"),
        )

    out = df.copy()
    norm_texts = [preprocess_text(str(t), emoji_mode=emoji_mode) for t in out["text"]]
    seg_texts = [seg_engine.segment(nt) for nt in norm_texts]
    out["text_seg"] = seg_texts

    if filter_short:
        keep_mask = [count_words(st) >= min_words for st in out["text_seg"]]
        out = out.loc[keep_mask].reset_index(drop=True)

    return out


def load_preprocess_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load `preprocess` section from YAML config (`configs/base.yaml` by default)."""
    defaults: Dict[str, Any] = {
        "emoji_mode": "tag",
        "filter_short": False,
        "min_words": 3,
        "segmenter": "vncorenlp",
        "vncorenlp_dir": None,
    }
    if config_path is None:
        default_yaml = Path("configs/base.yaml")
        if default_yaml.is_file():
            config_path = default_yaml
    if config_path is not None and Path(config_path).is_file():
        raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        if isinstance(raw_cfg.get("preprocess"), dict):
            defaults.update(raw_cfg["preprocess"])
    return defaults


def run_preprocess_all(
    clean_dir: Union[str, Path] = "data/clean",
    splits_dir: Optional[Union[str, Path]] = "data/splits",
    config_path: Optional[Union[str, Path]] = None,
    segmenter: Optional[VietnameseWordSegmenter] = None,
    filter_short: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run one-time preprocessing and word segmentation on all clean & split CSV files.

    Saves `text_seg` into:
    - `data/clean/vsmec_{train,dev,test}.csv`
    - `data/clean/vsfc_test_clean.csv`
    - `data/clean/neuesc_100.csv` (copied from `splits/neuesc_100.csv` if present so
      all 3 datasets have `text_seg` in `data/clean/`)
    - Also updates `data/clean/MANIFEST.json`.
    """
    cfg = load_preprocess_config(config_path)
    emoji_mode = str(cfg.get("emoji_mode", "tag"))
    do_filter_short = (
        bool(filter_short)
        if filter_short is not None
        else bool(cfg.get("filter_short", False))
    )
    min_words = int(cfg.get("min_words", 3))

    seg_engine = (
        segmenter
        if segmenter is not None
        else VietnameseWordSegmenter(
            backend=str(cfg.get("segmenter", "vncorenlp")),
            vncorenlp_dir=cfg.get("vncorenlp_dir"),
        )
    )

    c_dir = Path(clean_dir)
    c_dir.mkdir(parents=True, exist_ok=True)

    # If `splits_dir/neuesc_100.csv` exists and `clean_dir/neuesc_100.csv` does not yet exist,
    # copy it into `clean_dir` so VSMEC, NEU-ESC, and VSFC all reside in `data/clean/` with `text_seg`.
    if splits_dir is not None:
        s_dir = Path(splits_dir)
        sp_100 = s_dir / "neuesc_100.csv"
        clean_100 = c_dir / "neuesc_100.csv"
        if sp_100.is_file() and not clean_100.is_file():
            df_sp100 = pd.read_csv(sp_100)
            df_sp100.to_csv(clean_100, index=False, encoding="utf-8", lineterminator="\n")

    target_files = [
        "vsmec_train.csv",
        "vsmec_dev.csv",
        "vsmec_test.csv",
        "vsfc_test_clean.csv",
        "neuesc_100.csv",
    ]

    processed_counts: Dict[str, int] = {}
    for fname in target_files:
        fpath = c_dir / fname
        if not fpath.is_file():
            continue
        df_in = pd.read_csv(fpath)
        validate_columns(df_in, SPLIT_COLUMNS, strict_order=False)
        df_base = df_in.loc[:, list(SPLIT_COLUMNS)].copy()
        df_out = preprocess_dataframe(
            df_base,
            segmenter=seg_engine,
            emoji_mode=emoji_mode,
            filter_short=do_filter_short,
            min_words=min_words,
        )
        write_csv_schema(df_out, fpath, CLEAN_SEG_COLUMNS)
        processed_counts[fname] = len(df_out)

    # Update MANIFEST.json in clean_dir
    existing_csvs = sorted(c_dir.glob("*.csv"))
    manifest = build_manifest(existing_csvs, base_dir=c_dir)
    manifest_path = c_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "segmenter_backend": seg_engine.active_backend,
        "emoji_mode": emoji_mode,
        "filter_short": do_filter_short,
        "processed_counts": processed_counts,
        "manifest": manifest,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="CV-N2: Preprocess and word-segment clean datasets."
    )
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--filter-short",
        action="store_true",
        default=None,
        help="Filter out sentences with <= 2 words (disabled by default).",
    )
    args = parser.parse_args(argv)

    res = run_preprocess_all(
        clean_dir=args.clean_dir,
        splits_dir=args.splits_dir,
        config_path=args.config,
        filter_short=args.filter_short,
    )
    print(
        f"=== Preprocessing finished (backend={res['segmenter_backend']}, "
        f"emoji_mode={res['emoji_mode']}, filter_short={res['filter_short']}) ==="
    )
    for k, cnt in res["processed_counts"].items():
        print(f"  {k}: {cnt} rows")


if __name__ == "__main__":
    main()
