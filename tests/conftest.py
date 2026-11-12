from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--with-tokenizer",
        action="store_true",
        default=False,
        help="Chạy cả bài kiểm tra I2 (nạp tokenizer thật của PhoBERT và ViSoBERT).",
    )
    parser.addoption(
        "--check-ckpt",
        action="store_true",
        default=False,
        help="Chạy cả nhóm J (kiểm tra sự tồn tại của 20 checkpoint trên đĩa/Drive).",
    )
