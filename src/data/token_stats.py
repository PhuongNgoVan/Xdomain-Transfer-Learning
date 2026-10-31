"""CV-N3: Token length statistics (`p95`, `p99`, truncation rate at 256 tokens).

Per `tasks/M2_preprocess.md`:
- Computes token length distribution (`p95`, `p99`) and truncation rate at
  `max_length = 256` on NEU-ESC for both PhoBERT and ViSoBERT tokenizers read from `configs/base.yaml`.
- PhoBERT (`backbone: vinai/phobert-base-v2`) reads column `text_seg`.
- ViSoBERT (`visobert_backbone: uitnlp/visobert`) reads column `text`.
- Exports CSV table to `results/token_stats.csv`.
- Unit tests use a mock / local small tokenizer without any network access.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from src.common.io import write_csv_schema

TOKEN_STATS_COLUMNS: Tuple[str, ...] = (
    "tokenizer",
    "subset",
    "n_samples",
    "max_length",
    "min_tokens",
    "mean_tokens",
    "median_tokens",
    "p95_tokens",
    "p99_tokens",
    "max_tokens",
    "n_truncated",
    "truncation_rate",
)


def count_tokens_with_tokenizer(
    texts: Iterable[str],
    tokenizer: Any,
) -> List[int]:
    """Count token lengths (including `[CLS]`/`<s>` and `[SEP]`/`</s>` special tokens)
    using a HuggingFace tokenizer, mock tokenizer with `.encode()`, or callable.
    """
    lengths: List[int] = []
    for t in texts:
        s = "" if t is None else str(t)
        if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
            ids = tokenizer.encode(s, add_special_tokens=True, truncation=False)
            lengths.append(len(ids))
        elif callable(tokenizer):
            out = tokenizer(s)
            if isinstance(out, dict) and "input_ids" in out:
                lengths.append(len(out["input_ids"]))
            elif isinstance(out, (list, tuple)):
                lengths.append(len(out))
            elif isinstance(out, int):
                lengths.append(out)
            else:
                raise TypeError(f"Unsupported callable tokenizer output: {type(out)!r}")
        else:
            raise TypeError(
                f"Tokenizer must provide `.encode(text)` or be callable, got {type(tokenizer)!r}"
            )
    return lengths


def compute_length_distribution_row(
    lengths: Sequence[int],
    tokenizer_name: str,
    subset: str,
    max_length: int = 256,
) -> Dict[str, Any]:
    """Compute summary statistics (`p95`, `p99`, `truncation_rate` at `max_length`) for token lengths."""
    arr = np.asarray(lengths, dtype=np.float64)
    n_samples = int(arr.size)
    if n_samples == 0:
        return {
            "tokenizer": tokenizer_name,
            "subset": subset,
            "n_samples": 0,
            "max_length": int(max_length),
            "min_tokens": 0,
            "mean_tokens": 0.0,
            "median_tokens": 0.0,
            "p95_tokens": 0.0,
            "p99_tokens": 0.0,
            "max_tokens": 0,
            "n_truncated": 0,
            "truncation_rate": 0.0,
        }

    n_trunc = int(np.sum(arr > max_length))
    trunc_rate = float(n_trunc / n_samples)

    return {
        "tokenizer": tokenizer_name,
        "subset": subset,
        "n_samples": n_samples,
        "max_length": int(max_length),
        "min_tokens": int(np.min(arr)),
        "mean_tokens": round(float(np.mean(arr)), 4),
        "median_tokens": round(float(np.percentile(arr, 50)), 4),
        "p95_tokens": round(float(np.percentile(arr, 95)), 4),
        "p99_tokens": round(float(np.percentile(arr, 99)), 4),
        "max_tokens": int(np.max(arr)),
        "n_truncated": n_trunc,
        "truncation_rate": round(trunc_rate, 6),
    }


def load_hf_tokenizer(model_name: str) -> Any:
    """Load a real HuggingFace `AutoTokenizer` strictly (no local approximator fallback)."""
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    return AutoTokenizer.from_pretrained(model_name)


def load_tokenizers_from_config(
    config_path: Union[str, Path] = "configs/base.yaml",
) -> Dict[str, str]:
    """Read both PhoBERT (`backbone`) and ViSoBERT (`visobert_backbone`) checkpoints from `configs/base.yaml`."""
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    raw_cfg: Dict[str, Any] = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}

    resolved: Dict[str, str] = {}
    pho_ckpt = raw_cfg.get("backbone")
    if pho_ckpt:
        resolved[f"PhoBERT ({pho_ckpt})"] = str(pho_ckpt)

    viso_ckpt = raw_cfg.get("visobert_backbone")
    if viso_ckpt:
        resolved[f"ViSoBERT ({viso_ckpt})"] = str(viso_ckpt)

    return resolved


def compute_token_stats(
    df: pd.DataFrame,
    tokenizers: Dict[str, Any],
    max_length: int = 256,
    out_csv: Optional[Union[str, Path]] = "results/token_stats.csv",
    include_by_subset: bool = True,
) -> pd.DataFrame:
    """Compute token length statistics (`p95`, `p99`, truncation rate at `max_length`) on `df`.

    - PhoBERT (`"phobert"` in tokenizer key/name) strictly reads column `text_seg`.
    - ViSoBERT (`"visobert"` in tokenizer key/name) strictly reads column `text`.
    """
    missing_cols = [c for c in ("text", "text_seg") if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"DataFrame must contain both 'text' and 'text_seg' columns; missing {missing_cols}"
        )

    subsets_to_eval: List[Tuple[str, pd.DataFrame]] = [("all", df)]
    if include_by_subset and "subset" in df.columns:
        for sp in ("train", "dev", "test"):
            sub_df = df[df["subset"] == sp]
            if not sub_df.empty:
                subsets_to_eval.append((sp, sub_df))

    rows: List[Dict[str, Any]] = []
    for tok_name, tok_obj in tokenizers.items():
        use_segmented = "phobert" in tok_name.lower()
        col_name = "text_seg" if use_segmented else "text"

        for sp_name, sub_df in subsets_to_eval:
            lengths = count_tokens_with_tokenizer(sub_df[col_name], tok_obj)
            row = compute_length_distribution_row(
                lengths=lengths,
                tokenizer_name=tok_name,
                subset=sp_name,
                max_length=max_length,
            )
            rows.append(row)

    result_df = pd.DataFrame(rows, columns=list(TOKEN_STATS_COLUMNS))
    if out_csv is not None:
        write_csv_schema(result_df, out_csv, TOKEN_STATS_COLUMNS)
    return result_df


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="CV-N3: Compute token length statistics (p95, p99, truncation rate at 256)."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/clean/neuesc_100.csv"),
        help="Path to NEU-ESC CSV (with text_seg and text columns).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/token_stats.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to YAML config containing official PhoBERT and ViSoBERT checkpoint names.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args(argv)

    input_path = args.input_csv
    if not input_path.is_file():
        raise FileNotFoundError(f"NEU-ESC file not found: {args.input_csv}")

    ckpt_map = load_tokenizers_from_config(config_path=args.config)
    if len(ckpt_map) < 2:
        raise RuntimeError(
            f"Expected both 'backbone' (PhoBERT) and 'visobert_backbone' (ViSoBERT) in {args.config}, "
            f"got {ckpt_map}"
        )

    df = pd.read_csv(input_path)
    tokenizers = {
        label: load_hf_tokenizer(ckpt_id) for label, ckpt_id in ckpt_map.items()
    }

    stats_df = compute_token_stats(
        df=df,
        tokenizers=tokenizers,
        max_length=args.max_length,
        out_csv=args.out_csv,
        include_by_subset=True,
    )
    print("=== NEU-ESC Token Length Statistics ===")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
