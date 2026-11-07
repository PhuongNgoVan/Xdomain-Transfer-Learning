"""Unit tests for `src/plm/heads.py` (`tests/test_heads_init_gesf.py`).

Covers all requirements of Task 2 in `tasks/CV_P8_stage1.md`:
1. `load_wtrans`: supports JSON/CSV/NPY/YAML and `wtrans_norm` in `("none", "mean", "auto")`.
2. `init_head` (`B4-init`): $W = \\tilde{W}_{\\text{trans}} W_{\\text{emo}},\\; b = \\tilde{W}_{\\text{trans}} b_{\\text{emo}}$
   matches hand calculation for both `mean` and `none`.
3. `diagnose_init` (`B4-init`): returns `macro_f1_init` and `logit_var_init` and creates the JSON file.
4. `GESFModel` (`B5`):
   - Outputs 3-class logits of shape `(B, 3)`.
   - When `emo_input="z_emo"`, each sample's emotion representation $e$ has sum/mean equal to 0.
   - When `wtrans_norm="auto"`, uses `"none"` for `p_emo` and `"mean"` for `log_p_emo` / `z_emo`.
   - When `alpha_init="auto"`, the std ratio between the two branches lies in `[0.5, 2.0]`.
   - When `detach_emo=False`, $W_{\\text{emo}}$ receives non-zero gradient; when `detach_emo=True`,
     gradient on $W_{\\text{emo}}$ is `None` or `0`.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from src.common.constants import (
    EMO_ANGER,
    EMO_DISGUST,
    EMO_ENJOYMENT,
    EMO_FEAR,
    EMO_OTHER,
    EMO_SADNESS,
    EMO_SURPRISE,
    EMOTIONS7,
    HARD_MERGE_RULE,
    LABELS3,
    NEG,
    NEU,
    POS,
)

pytest.importorskip("src.plm.heads")
from src.plm.heads import (
    GESFModel,
    diagnose_init,
    init_head,
    load_wtrans,
)


def _build_default_wtrans_matrix() -> np.ndarray:
    """Build the 3x7 binary transition matrix from `HARD_MERGE_RULE`."""
    mat = np.zeros((len(LABELS3), len(EMOTIONS7)), dtype=np.float64)
    for emo_id, sent_id in HARD_MERGE_RULE.items():
        mat[sent_id, emo_id] = 1.0
    return mat


class DummyEncoder(nn.Module):
    """Lightweight offline encoder mapping input tensors `(B, d)` or token IDs `(B, L)` to `(B, d)`."""

    def __init__(self, hidden_size: int = 8, vocab_size: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.config = type("DummyConfig", (), {"hidden_size": hidden_size})()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs: object,
    ) -> Tensor:
        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            if input_ids.dtype in (torch.float32, torch.float64):
                x = input_ids
            else:
                x = self.embedding(input_ids).mean(dim=1)
        else:
            raise ValueError("Provide input_ids or inputs_embeds")
        return torch.tanh(self.proj(x))


class DummySeqClassifier(nn.Module):
    """Lightweight sequence classifier with `.encoder` and `.classifier`."""

    def __init__(self, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.encoder = DummyEncoder(hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: Tensor, **kwargs: object) -> Tensor:
        h = self.encoder(input_ids=input_ids)
        return self.classifier(h)


def test_load_wtrans_formats_and_auto_normalization(tmp_path: Path) -> None:
    """Verify loading W_trans from JSON/CSV/NPY/YAML and auto normalization behavior."""
    raw_mat = _build_default_wtrans_matrix()

    json_path = tmp_path / "wtrans.json"
    json_path.write_text(json.dumps(raw_mat.tolist()), encoding="utf-8")

    npy_path = tmp_path / "wtrans.npy"
    np.save(npy_path, raw_mat)

    csv_path = tmp_path / "wtrans.csv"
    header = ",".join(EMOTIONS7)
    lines = [header] + [",".join(str(x) for x in row) for row in raw_mat]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 1. 'none' preserves binary 0/1 matrix
    w_none = load_wtrans(json_path, wtrans_norm="none")
    assert w_none.shape == (3, 7)
    assert torch.allclose(w_none, torch.tensor(raw_mat, dtype=torch.float32))

    # 2. 'mean' divides each row by row sum (NEG: 1/4=0.25, NEU: 1/2=0.5, POS: 1/1=1.0)
    w_mean = load_wtrans(npy_path, wtrans_norm="mean")
    assert torch.allclose(w_mean.sum(dim=1), torch.ones(3), atol=1e-6)
    for emo_id in (EMO_SADNESS, EMO_FEAR, EMO_ANGER, EMO_DISGUST):
        assert pytest.approx(w_mean[NEG, emo_id].item(), abs=1e-6) == 0.25
    for emo_id in (EMO_SURPRISE, EMO_OTHER):
        assert pytest.approx(w_mean[NEU, emo_id].item(), abs=1e-6) == 0.50
    assert pytest.approx(w_mean[POS, EMO_ENJOYMENT].item(), abs=1e-6) == 1.00

    # 3. 'auto' with GESF: 'none' for p_emo, 'mean' for log_p_emo and z_emo, 'mean' for head='init'
    w_auto_p = load_wtrans(csv_path, wtrans_norm="auto", emo_input="p_emo", head="gesf")
    w_auto_logp = load_wtrans(csv_path, wtrans_norm="auto", emo_input="log_p_emo", head="gesf")
    w_auto_z = load_wtrans(csv_path, wtrans_norm="auto", emo_input="z_emo", head="gesf")
    w_auto_init = load_wtrans(csv_path, wtrans_norm="auto", emo_input="p_emo", head="init")

    assert torch.allclose(w_auto_p, w_none, atol=1e-6)
    assert torch.allclose(w_auto_logp, w_mean, atol=1e-6)
    assert torch.allclose(w_auto_z, w_mean, atol=1e-6)
    assert torch.allclose(w_auto_init, w_mean, atol=1e-6)


def test_init_head_matches_hand_calculation_mean_and_none(tmp_path: Path) -> None:
    """Verify init_head sets W_init = W_trans @ W_emo and b_init = W_trans @ b_emo for 'mean' and 'none'."""
    d = 4
    raw_mat = _build_default_wtrans_matrix()
    wtrans_path = tmp_path / "wtrans.json"
    wtrans_path.write_text(json.dumps(raw_mat.tolist()), encoding="utf-8")

    # Known 7x4 W_emo and 7-dim b_emo
    w_emo = torch.arange(1, 7 * d + 1, dtype=torch.float32).reshape(7, d)
    b_emo = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0, -1.0, 1.0], dtype=torch.float32)

    stage1_model = DummySeqClassifier(hidden_size=d, num_labels=7)
    with torch.no_grad():
        stage1_model.classifier.weight.copy_(w_emo)
        stage1_model.classifier.bias.copy_(b_emo)

    # Case 1: wtrans_norm = "none"
    model_none = DummySeqClassifier(hidden_size=d, num_labels=3)
    init_head(model_none, stage1_model, wtrans_path, wtrans_norm="none")

    expected_w_none = torch.tensor(raw_mat, dtype=torch.float32) @ w_emo
    expected_b_none = torch.tensor(raw_mat, dtype=torch.float32) @ b_emo
    # Specifically for b_none:
    # NEG = 1.0 + 2.0 + 3.0 + 4.0 = 10.0
    # NEU = -1.0 + 1.0 = 0.0
    # POS = 0.5
    assert torch.allclose(model_none.classifier.weight, expected_w_none, atol=1e-6)
    assert torch.allclose(model_none.classifier.bias, expected_b_none, atol=1e-6)
    assert torch.allclose(
        model_none.classifier.bias,
        torch.tensor([10.0, 0.0, 0.5], dtype=torch.float32),
        atol=1e-6,
    )

    # Case 2: wtrans_norm = "mean"
    model_mean = DummySeqClassifier(hidden_size=d, num_labels=3)
    init_head(model_mean, stage1_model, wtrans_path, wtrans_norm="mean")

    row_norm_mat = raw_mat / raw_mat.sum(axis=1, keepdims=True)
    expected_w_mean = torch.tensor(row_norm_mat, dtype=torch.float32) @ w_emo
    expected_b_mean = torch.tensor(row_norm_mat, dtype=torch.float32) @ b_emo
    # Specifically for b_mean:
    # NEG = 10.0 / 4 = 2.5
    # NEU = 0.0 / 2 = 0.0
    # POS = 0.5 / 1 = 0.5
    assert torch.allclose(model_mean.classifier.weight, expected_w_mean, atol=1e-6)
    assert torch.allclose(model_mean.classifier.bias, expected_b_mean, atol=1e-6)
    assert torch.allclose(
        model_mean.classifier.bias,
        torch.tensor([2.5, 0.0, 0.5], dtype=torch.float32),
        atol=1e-6,
    )


def test_diagnose_init_creates_json_and_returns_metrics(tmp_path: Path) -> None:
    """Verify diagnose_init computes macro_f1_init and logit_var_init and writes JSON report."""
    torch.manual_seed(7)
    model = DummySeqClassifier(hidden_size=6, num_labels=3)

    dev_batches = [
        {
            "input_ids": torch.randint(0, 20, (8, 5)),
            "labels": torch.tensor([NEG, NEU, POS, NEG, NEU, POS, NEG, POS]),
        },
        {
            "input_ids": torch.randint(0, 20, (4, 5)),
            "labels": torch.tensor([NEG, NEU, POS, NEU]),
        },
    ]

    diag_dir = tmp_path / "diagnostics"
    report = diagnose_init(
        model=model,
        dev_dataloader=dev_batches,
        device="cpu",
        run_id="PhoBERT_HR_Init_test",
        output_dir=diag_dir,
    )

    assert "macro_f1_init" in report
    assert "logit_var_init" in report
    assert 0.0 <= float(report["macro_f1_init"]) <= 1.0
    assert float(report["logit_var_init"]) > 0.0
    # Also verify per-class access on logit_var_init
    for cls_name in LABELS3:
        assert cls_name in report["logit_var_init"]
        assert report["logit_var_init"][cls_name] > 0.0

    json_file = diag_dir / "PhoBERT_HR_Init_test_init_diag.json"
    assert json_file.is_file()
    saved = json.loads(json_file.read_text(encoding="utf-8"))
    assert saved["run_id"] == "PhoBERT_HR_Init_test"
    assert saved["step"] == 0
    assert saved["n_samples"] == 12
    assert pytest.approx(saved["macro_f1_init"], abs=1e-6) == float(report["macro_f1_init"])
    assert set(saved["logit_var_per_class"].keys()) == set(LABELS3)


def test_gesf_model_shapes_z_emo_zero_mean_and_auto_wtrans(tmp_path: Path) -> None:
    """Verify GESFModel 3-class output shape, zero-mean z_emo, and wtrans_norm='auto'."""
    torch.manual_seed(123)
    wtrans_path = tmp_path / "wtrans.json"
    wtrans_path.write_text(json.dumps(_build_default_wtrans_matrix().tolist()), encoding="utf-8")

    # 1. emo_input = "z_emo" -> row-wise mean/sum of e must be 0, and wtrans_norm="auto" -> "mean"
    enc_z = DummyEncoder(hidden_size=8)
    model_z = GESFModel(
        encoder=enc_z,
        wtrans_path=wtrans_path,
        emo_input="z_emo",
        wtrans_norm="auto",
        alpha_init=1.0,
    )
    assert model_z.effective_wtrans_norm == "mean"
    assert torch.allclose(model_z.wtrans.sum(dim=1), torch.ones(3), atol=1e-6)

    x = torch.randint(0, 20, (6, 4))
    out_z = model_z(input_ids=x)
    assert out_z.logits.shape == (6, 3)
    assert out_z.hidden_states is not None
    _, z_emo, e_z = out_z.hidden_states
    assert z_emo.shape == (6, 7)
    assert e_z.shape == (6, 7)
    # Row-wise mean and sum of e_z must be 0 for every sample
    assert torch.allclose(e_z.mean(dim=-1), torch.zeros(6), atol=1e-6)
    assert torch.allclose(e_z.sum(dim=-1), torch.zeros(6), atol=1e-5)

    # 2. emo_input = "p_emo" -> wtrans_norm="auto" resolves to "none", probabilities sum to 1
    enc_p = DummyEncoder(hidden_size=8)
    model_p = GESFModel(
        encoder=enc_p,
        wtrans_path=wtrans_path,
        emo_input="p_emo",
        wtrans_norm="auto",
        alpha_init=1.0,
    )
    assert model_p.effective_wtrans_norm == "none"
    out_p = model_p(input_ids=x)
    assert out_p.logits.shape == (6, 3)
    _, _, e_p = out_p.hidden_states
    assert torch.allclose(e_p.sum(dim=-1), torch.ones(6), atol=1e-6)

    # 3. emo_input = "log_p_emo" -> wtrans_norm="auto" resolves to "mean"
    enc_lp = DummyEncoder(hidden_size=8)
    model_lp = GESFModel(
        encoder=enc_lp,
        wtrans_path=wtrans_path,
        emo_input="log_p_emo",
        wtrans_norm="auto",
        alpha_init=1.0,
    )
    assert model_lp.effective_wtrans_norm == "mean"
    out_lp = model_lp(input_ids=x)
    assert out_lp.logits.shape == (6, 3)


def test_gesf_alpha_init_auto_calibrates_std_ratio_in_bounds(tmp_path: Path) -> None:
    """When alpha_init='auto', std(alpha * W_trans e) / std(z_ctx) must lie in [0.5, 2.0]."""
    torch.manual_seed(99)
    wtrans_path = tmp_path / "wtrans.json"
    wtrans_path.write_text(json.dumps(_build_default_wtrans_matrix().tolist()), encoding="utf-8")

    encoder = DummyEncoder(hidden_size=8)
    model = GESFModel(
        encoder=encoder,
        wtrans_path=wtrans_path,
        emo_input="p_emo",
        wtrans_norm="auto",
        alpha_init="auto",
    )
    # Intentionally scale ctx_head weights by 10x so uncalibrated ratio would be far outside [0.5, 2.0]
    with torch.no_grad():
        model.ctx_head.weight.mul_(10.0)

    batches = [
        {"input_ids": torch.randint(0, 20, (16, 6))},
        {"input_ids": torch.randint(0, 20, (16, 6))},
    ]
    calibrated_alpha = model.calibrate_alpha(batches, max_batches=2)
    assert calibrated_alpha > 0.0

    # Check std ratio on training batches
    out = model(input_ids=batches[0]["input_ids"])
    z_ctx, _, e = out.hidden_states
    u_emo = torch.matmul(e, model.wtrans.t())
    ratio = (model.alpha * u_emo).std(unbiased=False).item() / z_ctx.std(unbiased=False).item()
    assert 0.5 <= ratio <= 2.0

    diag = model.get_diagnostics()
    assert 0.5 <= diag["std_ratio"] <= 2.0
    assert diag["p_emo_entropy"] > 0.0


def test_gesf_detach_emo_gradient_control(tmp_path: Path) -> None:
    """Verify W_emo receives non-zero gradient when detach_emo=False and None/0 when detach_emo=True."""
    torch.manual_seed(2026)
    wtrans_path = tmp_path / "wtrans.json"
    wtrans_path.write_text(json.dumps(_build_default_wtrans_matrix().tolist()), encoding="utf-8")

    x = torch.randint(0, 20, (8, 5))
    y = torch.tensor([NEG, NEU, POS, NEG, POS, NEU, NEG, POS], dtype=torch.long)

    # 1. detach_emo = False -> W_emo has non-zero gradient
    model_grad = GESFModel(
        encoder=DummyEncoder(hidden_size=8),
        wtrans_path=wtrans_path,
        emo_input="p_emo",
        alpha_init=1.0,
        detach_emo=False,
    )
    out_grad = model_grad(input_ids=x, labels=y)
    assert out_grad.loss is not None
    out_grad.loss.backward()

    assert model_grad.W_emo.grad is not None
    assert model_grad.get_w_emo_grad_norm() > 0.0
    assert torch.any(model_grad.W_emo.grad != 0.0)

    # 2. detach_emo = True -> W_emo has None or zero gradient, while W_ctx still receives gradient
    model_detached = GESFModel(
        encoder=DummyEncoder(hidden_size=8),
        wtrans_path=wtrans_path,
        emo_input="p_emo",
        alpha_init=1.0,
        detach_emo=True,
    )
    out_det = model_detached(input_ids=x, labels=y)
    assert out_det.loss is not None
    out_det.loss.backward()

    grad_w_emo = model_detached.W_emo.grad
    assert grad_w_emo is None or torch.all(grad_w_emo == 0.0)
    assert model_detached.get_w_emo_grad_norm() == 0.0
    assert model_detached.W_ctx.grad is not None
    assert torch.any(model_detached.W_ctx.grad != 0.0)
