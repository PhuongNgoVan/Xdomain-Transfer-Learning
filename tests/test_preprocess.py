"""Unit and acceptance tests for `src/data/preprocess.py` (M2 / CV-N2)."""

from pathlib import Path
import unicodedata

import pandas as pd
import pytest

from src.common.constants import NEG, NEU, POS
from src.common.hashing import sha256_file
from src.common.io import read_csv_schema
from src.data.preprocess import (
    CLEAN_SEG_COLUMNS,
    VietnameseWordSegmenter,
    count_words,
    load_preprocess_config,
    map_emojis,
    preprocess_dataframe,
    preprocess_text,
    run_preprocess_all,
)


class TestPreprocessText:
    def test_emoji_mapping_and_repeated_chars_and_nfc(self) -> None:
        # Decomposed Unicode (NFD) + elongated characters + emojis/emoticons
        raw_nfd = unicodedata.normalize(
            "NFD", "Đẹpppp   quáaaaa 😍 :))  Trường đại học tuyệt vờiiii 😭"
        )
        raw_nfc = unicodedata.normalize(
            "NFC", "đẹp quá 😍 :)) trường đại học tuyệt vời 😭"
        )

        out_nfd = preprocess_text(raw_nfd, emoji_mode="tag")
        out_nfc = preprocess_text(raw_nfc, emoji_mode="tag")

        assert out_nfd == out_nfc
        assert out_nfd == "đẹp quá emo_love emo_joy trường đại học tuyệt vời emo_sad"
        assert unicodedata.is_normalized("NFC", out_nfd)

    @pytest.mark.parametrize(
        "sample",
        [
            "Đẹpppp quáaaa 😍",
            "Buồnnnn quá đi mất 😭 :((",
            "Giận dữ thật sự 😡 !!!",
            unicodedata.normalize("NFD", "Tiếng Việt chuẩn hóa đồng nhất ❤️ <3"),
            "Bình thường không có gì đặc biệt -_-",
        ],
    )
    def test_preprocess_text_idempotent_and_stable(self, sample: str) -> None:
        once = preprocess_text(sample, emoji_mode="tag")
        twice = preprocess_text(once, emoji_mode="tag")
        assert twice == once

    def test_emoji_modes_remove_and_keep(self) -> None:
        s = "Hay quá 😍 :))"
        assert preprocess_text(s, emoji_mode="remove") == "hay quá"
        assert "😍" in map_emojis(s, emoji_mode="keep")
        with pytest.raises(ValueError):
            map_emojis(s, emoji_mode="invalid_mode")


class TestWordSegmentationAndDataFrame:
    def test_custom_mock_segmenter_and_emo_tag_protection(self) -> None:
        # Simulate a segmenter that joins "sinh viên" -> "sinh_viên" and "đại học" -> "đại_học"
        # and splits "emo_love" into "emo _ love" (just like VnCoreNLP does)
        def dummy_seg(text: str) -> str:
            return (
                text.replace("sinh viên", "sinh_viên")
                .replace("đại học", "đại_học")
                .replace("emo_love", "emo _ love")  # simulate accidental split by VnCoreNLP
            )

        seg_engine = VietnameseWordSegmenter(custom_fn=dummy_seg)
        assert seg_engine.active_backend == "custom"

        norm = preprocess_text("Sinh viên đại học 😍", emoji_mode="tag")
        segmented = seg_engine.segment(norm)
        assert segmented == "sinh_viên đại_học emo_love"

    def test_missing_or_invalid_vncorenlp_raises_error_no_fallback(
        self, tmp_path: Path
    ) -> None:
        # 1. Missing vncorenlp_dir (None) -> must raise RuntimeError, never fall back
        with pytest.raises(RuntimeError, match="tried path"):
            VietnameseWordSegmenter(backend="vncorenlp", vncorenlp_dir=None)

        # 2. Non-existent directory -> must raise RuntimeError mentioning the tried path
        missing_dir = tmp_path / "nonexistent_vncorenlp"
        with pytest.raises(RuntimeError, match="nonexistent_vncorenlp"):
            VietnameseWordSegmenter(backend="vncorenlp", vncorenlp_dir=missing_dir)

        # 3. Existing directory without VnCoreNLP jar/models -> must raise RuntimeError
        empty_dir = tmp_path / "empty_vncorenlp"
        empty_dir.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="empty_vncorenlp"):
            VietnameseWordSegmenter(backend="vncorenlp", vncorenlp_dir=empty_dir)

        # 4. Non-vncorenlp backend name -> must raise RuntimeError (no underthesea/whitespace fallback)
        with pytest.raises(RuntimeError, match="Unsupported segmenter backend"):
            VietnameseWordSegmenter(backend="underthesea", vncorenlp_dir=empty_dir)

    def test_short_sentence_filter_flag_off_by_default(self) -> None:
        df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "text": [
                    "ok",  # 1 word
                    "đẹpppp quáaaa",  # 2 words
                    "bài giảng rất hay và bổ ích",  # 6 words
                    "hay 😍 :))",  # 3 tokens ("hay", "emo_love", "emo_joy")
                ],
                "label": [NEU, POS, POS, POS],
                "subset": ["train", "train", "dev", "test"],
            }
        )
        seg_engine = VietnameseWordSegmenter(custom_fn=lambda s: s)

        # Default: filter_short=False -> keeps all 4 rows
        out_default = preprocess_dataframe(df, segmenter=seg_engine, filter_short=False)
        assert len(out_default) == 4
        assert "text_seg" in out_default.columns

        # When filter_short=True -> filters out sentences with <= 2 words (rows 0 and 1)
        out_filtered = preprocess_dataframe(
            df, segmenter=seg_engine, filter_short=True, min_words=3
        )
        assert len(out_filtered) == 2
        assert out_filtered["id"].tolist() == [2, 3]

    def test_config_default_has_filter_short_false(self) -> None:
        cfg = load_preprocess_config()
        assert cfg["filter_short"] is False
        assert cfg["emoji_mode"] == "tag"
        assert count_words("xin_chào các bạn !") == 3

    def test_run_preprocess_all_twice_gives_identical_manifest_hashes(
        self, tmp_path: Path
    ) -> None:
        clean_dir = tmp_path / "clean"
        splits_dir = tmp_path / "splits"
        clean_dir.mkdir(parents=True)
        splits_dir.mkdir(parents=True)

        sample_rows = pd.DataFrame(
            {
                "id": [0, 1],
                "text": ["Đẹpppp quáaaa 😍", "Giảng viên nhiệt tình :))"],
                "label": [POS, NEG],
                "subset": ["train", "test"],
            }
        )
        for fname in ("vsmec_train.csv", "vsmec_dev.csv", "vsmec_test.csv", "vsfc_test_clean.csv"):
            sample_rows.to_csv(clean_dir / fname, index=False, encoding="utf-8")
        sample_rows.to_csv(splits_dir / "neuesc_100.csv", index=False, encoding="utf-8")

        mock_seg = VietnameseWordSegmenter(
            custom_fn=lambda s: s.replace("giảng viên", "giảng_viên").replace(
                "nhiệt tình", "nhiệt_tình"
            )
        )

        res1 = run_preprocess_all(
            clean_dir=clean_dir,
            splits_dir=splits_dir,
            segmenter=mock_seg,
            filter_short=False,
        )
        hashes_run1 = dict(res1["manifest"])

        # Re-run a second time on the same directory
        res2 = run_preprocess_all(
            clean_dir=clean_dir,
            splits_dir=splits_dir,
            segmenter=mock_seg,
            filter_short=False,
        )
        hashes_run2 = dict(res2["manifest"])

        assert hashes_run1 == hashes_run2
        for fname in (
            "vsmec_train.csv",
            "vsmec_dev.csv",
            "vsmec_test.csv",
            "vsfc_test_clean.csv",
            "neuesc_100.csv",
        ):
            df_out = read_csv_schema(clean_dir / fname, CLEAN_SEG_COLUMNS)
            assert df_out["text_seg"].tolist() == [
                "đẹp quá emo_love",
                "giảng_viên nhiệt_tình emo_joy",
            ]
            assert sha256_file(clean_dir / fname) == hashes_run2[fname]
