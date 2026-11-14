from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from src.plm.model_ckpt import Stage1Classifier


@pytest.mark.parametrize("alias", ["PhoBERT", "ViSoBERT"])
@pytest.mark.parametrize("num_labels", [3, 7])
def test_checkpoint_reload_infers_head_width_offline(tmp_path: Path, alias: str, num_labels: int) -> None:
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nxin\nchao\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    config = BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    model = Stage1Classifier(BertModel(config), num_labels=num_labels).eval()
    inputs = tokenizer("xin chao", return_tensors="pt")
    with torch.no_grad():
        expected = model(**inputs).logits
    checkpoint = tmp_path / alias
    model.save_stage1_checkpoint(tokenizer, checkpoint)
    reloaded, reloaded_tokenizer = Stage1Classifier.load_stage1_checkpoint(checkpoint, device="cpu")
    assert reloaded.num_labels == num_labels
    assert reloaded.head.weight.shape == (num_labels, 8)
    assert reloaded.accepts_loss_kwargs is False
    with torch.no_grad():
        actual = reloaded(**reloaded_tokenizer("xin chao", return_tensors="pt")).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
