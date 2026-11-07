"""Unit tests for `src/data/token_stats.py` (M2 / CV-N3).

Uses mock/local tokenizers only — zero network access.
"""

from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.common.constants import NEG, NEU, POS
from src.common.io import read_csv_schema
from src.data.token_stats import (
    TOKEN_STATS_COLUMNS,
    compute_length_distribution_row,
    compute_token_stats,
    count_tokens_with_tokenizer,
    load_tokenizers_from_config,
)


class DummyWhitespaceTokenizer:
    """Offline mock tokenizer that splits on whitespace and adds 2 special tokens (`<s>`, `</s>`)."""

    def __init__(self, split_underscores: bool = False) -> None:
        self.split_underscores = split_underscores

    def encode(
        self, text: str, add_special_tokens: bool = True, truncation: bool = False
    ) -> List[int]:
        if not text or not text.strip():
            return [0, 2] if add_special_tokens else []
        s = text.replace("_", " ") if self.split_underscores else text
        words = s.strip().split()
        n = len(words) + (2 if add_special_tokens else 0)
        return list(range(n))


class TestTokenStats:
    def test_count_tokens_and_distribution_row_exact(self) -> None:
        tok = DummyWhitespaceTokenizer(split_underscores=False)
        texts = [
            "một hai",  # 2 + 2 = 4 tokens
            "sinh_viên đại_học kinh_tế",  # 3 + 2 = 5 tokens
            " ".join(["từ"] * 260),  # 260 + 2 = 262 tokens (> 256 -> truncated)
        ]
        lengths = count_tokens_with_tokenizer(texts, tok)
        assert lengths == [4, 5, 262]

        row = compute_length_distribution_row(
            lengths, tokenizer_name="MockPhoBERT", subset="all", max_length=256
        )
        assert row["n_samples"] == 3
        assert row["min_tokens"] == 4
        assert row["max_tokens"] == 262
        assert row["n_truncated"] == 1
        assert row["truncation_rate"] == pytest.approx(1.0 / 3.0, abs=1e-5)

    def test_compute_token_stats_phobert_vs_visobert_offline(
        self, tmp_path: Path
    ) -> None:
        # PhoBERT reads `text_seg` (where compound words are 1 token `sinh_viên`),
        # whereas ViSoBERT reads unsegmented `text` (where `sinh viên` is 2 tokens).
        rows = []
        for i in range(95):
            rows.append(
                {
                    "id": i,
                    "text": "sinh viên trường đại học kinh tế quốc dân",  # 9 syllables -> 9+2=11 ViSoBERT tokens
                    "text_seg": "sinh_viên trường đại_học kinh_tế quốc_dân",  # 5 words -> 5+2=7 PhoBERT tokens
                    "label": NEU,
                    "subset": "train" if i < 80 else "dev",
                }
            )
        # Add 5 long samples (> 256 tokens for both tokenizers)
        long_seg = " ".join(["sinh_viên"] * 260)  # 260 + 2 = 262 PhoBERT tokens
        long_raw = " ".join(["sinh viên"] * 260)  # 520 + 2 = 522 ViSoBERT tokens
        for i in range(95, 100):
            rows.append(
                {
                    "id": i,
                    "text": long_raw,
                    "text_seg": long_seg,
                    "label": NEG if i % 2 == 0 else POS,
                    "subset": "test",
                }
            )

        df = pd.DataFrame(rows)
        out_csv = tmp_path / "results" / "token_stats.csv"

        tokenizers = {
            "PhoBERT": DummyWhitespaceTokenizer(split_underscores=False),
            "ViSoBERT": DummyWhitespaceTokenizer(split_underscores=True),
        }

        stats_df = compute_token_stats(
            df=df,
            tokenizers=tokenizers,
            max_length=256,
            out_csv=out_csv,
            include_by_subset=True,
        )

        assert out_csv.is_file()
        loaded = read_csv_schema(out_csv, TOKEN_STATS_COLUMNS)
        assert len(loaded) == len(stats_df)

        pho_all = stats_df[
            (stats_df["tokenizer"] == "PhoBERT") & (stats_df["subset"] == "all")
        ].iloc[0]
        viso_all = stats_df[
            (stats_df["tokenizer"] == "ViSoBERT") & (stats_df["subset"] == "all")
        ].iloc[0]

        assert pho_all["n_samples"] == 100
        assert pho_all["n_truncated"] == 5
        assert pho_all["truncation_rate"] == pytest.approx(0.05)
        assert pho_all["median_tokens"] == 7.0
        assert pho_all["max_tokens"] == 262

        assert viso_all["n_samples"] == 100
        assert viso_all["n_truncated"] == 5
        assert viso_all["truncation_rate"] == pytest.approx(0.05)
        assert viso_all["median_tokens"] == 11.0
        assert viso_all["max_tokens"] == 522
        assert viso_all["p99_tokens"] > pho_all["p99_tokens"]

    def test_load_tokenizers_from_config_no_guessing(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "base.yaml"
        cfg_file.write_text("backbone: vinai/phobert-base-v2\n", encoding="utf-8")

        ckpt_map = load_tokenizers_from_config(config_path=cfg_file)
        assert ckpt_map == {"PhoBERT (vinai/phobert-base-v2)": "vinai/phobert-base-v2"}
        assert not any("visobert" in k.lower() for k in ckpt_map)

        # Official configs/base.yaml must declare both PhoBERT and ViSoBERT
        official_map = load_tokenizers_from_config(config_path=Path("configs/base.yaml"))
        assert official_map == {
            "PhoBERT (vinai/phobert-base-v2)": "vinai/phobert-base-v2",
            "ViSoBERT (uitnlp/visobert)": "uitnlp/visobert",
        }

        # Missing text or text_seg column raises ValueError
        with pytest.raises(ValueError, match="text_seg"):
            compute_token_stats(
                df=pd.DataFrame({"text": ["xin chào"]}),
                tokenizers={"PhoBERT": DummyWhitespaceTokenizer()},
                out_csv=None,
            )
