"""Unit tests for `src/plm/losses.py` (`tests/test_losses.py`).

Verifies mathematical properties required by `tasks/CV_P8_stage1.md`:
1. `LogitAdjustmentLoss` with `tau = 0.0` matches standard `CrossEntropyLoss` (`torch.allclose`).
2. `LogitAdjustmentLoss` with `tau = 1.0` matches hand-calculated values and matches
   `BalancedSoftmaxLoss` (`torch.allclose`).
3. `build_loss` factory and `compute_class_priors` validation.
"""

import math

import pytest
import torch

from src.common.constants import NEG, NEU, POS

pytest.importorskip("src.plm.losses")
from src.plm.losses import (
    BalancedSoftmaxLoss,
    CrossEntropyLoss,
    LogitAdjustmentLoss,
    build_loss,
    compute_class_priors,
)


def test_la_tau_zero_equals_cross_entropy() -> None:
    """LA with tau = 0.0 must equal standard CrossEntropyLoss across reductions."""
    torch.manual_seed(42)
    logits = torch.randn(16, 3, dtype=torch.float64, requires_grad=True)
    logits_ce = logits.detach().clone().requires_grad_(True)
    targets = torch.tensor([NEG, NEU, POS, NEG, NEU, POS, NEG, NEG] * 2, dtype=torch.long)
    priors = [0.6, 0.25, 0.15]

    ce_loss_fn = CrossEntropyLoss(reduction="mean")
    la_zero_fn = LogitAdjustmentLoss(tau=0.0, priors=priors, reduction="mean")

    loss_ce = ce_loss_fn(logits_ce, targets)
    loss_la0 = la_zero_fn(logits, targets)

    assert torch.allclose(loss_la0, loss_ce, atol=1e-12, rtol=1e-12)

    loss_ce.backward()
    loss_la0.backward()
    assert logits.grad is not None and logits_ce.grad is not None
    assert torch.allclose(logits.grad, logits_ce.grad, atol=1e-12, rtol=1e-12)


def test_la_tau_one_matches_hand_calculation_and_balanced_softmax() -> None:
    """LA with tau = 1.0 must match hand-computed value and equal BalancedSoftmaxLoss."""
    # 2 samples, 3 classes (NEG=0, NEU=1, POS=2)
    logits = torch.tensor(
        [
            [1.5, -0.5, 0.5],
            [-0.2, 0.8, 1.2],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    logits_bs = logits.detach().clone().requires_grad_(True)
    targets = torch.tensor([NEG, POS], dtype=torch.long)
    priors = [0.5, 0.3, 0.2]
    eps = 1e-12

    # Hand calculation for each sample:
    # z'_k = z_k + 1.0 * log(pi_k + eps)
    # L_i = -z'_{y_i} + log(sum_k exp(z'_k))
    log_pi = [math.log(p + eps) for p in priors]

    # Sample 0 (target = NEG = 0)
    adj_0 = [logits[0, k].item() + log_pi[k] for k in range(3)]
    lse_0 = math.log(sum(math.exp(v) for v in adj_0))
    loss_0 = -adj_0[NEG] + lse_0

    # Sample 1 (target = POS = 2)
    adj_1 = [logits[1, k].item() + log_pi[k] for k in range(3)]
    lse_1 = math.log(sum(math.exp(v) for v in adj_1))
    loss_1 = -adj_1[POS] + lse_1

    expected_mean_loss = (loss_0 + loss_1) / 2.0

    la_fn = LogitAdjustmentLoss(tau=1.0, priors=priors, eps=eps, reduction="mean")
    bs_fn = BalancedSoftmaxLoss(priors=priors, eps=eps, reduction="mean")

    out_la = la_fn(logits, targets)
    out_bs = bs_fn(logits_bs, targets)

    assert math.isclose(out_la.item(), expected_mean_loss, rel_tol=1e-6, abs_tol=1e-6)
    assert torch.allclose(out_la, out_bs, atol=1e-12, rtol=1e-12)

    out_la.backward()
    out_bs.backward()
    assert logits.grad is not None and logits_bs.grad is not None
    assert torch.allclose(logits.grad, logits_bs.grad, atol=1e-12, rtol=1e-12)


def test_la_custom_tau_and_positional_arg_flexibility() -> None:
    """Verify non-unit tau and both (tau, priors) and (priors, tau) constructor signatures."""
    logits = torch.tensor([[0.2, 0.4, -0.1]], dtype=torch.float32)
    targets = torch.tensor([NEU], dtype=torch.long)
    priors = [0.7, 0.2, 0.1]

    la_a = LogitAdjustmentLoss(tau=1.5, priors=priors)
    la_b = LogitAdjustmentLoss(priors, 1.5)
    assert torch.allclose(la_a(logits, targets), la_b(logits, targets), atol=1e-7)


def test_build_loss_and_compute_class_priors() -> None:
    """Verify build_loss factory and compute_class_priors helper."""
    labels = [NEG, NEG, NEG, NEU, NEU, POS]
    priors = compute_class_priors(labels, num_classes=3)
    expected = torch.tensor([3 / 6, 2 / 6, 1 / 6], dtype=torch.float32)
    assert torch.allclose(priors, expected, atol=1e-6)

    logits = torch.randn(4, 3)
    targets = torch.tensor([NEG, NEU, POS, NEG])

    ce = build_loss("ce")
    la = build_loss("la", priors=priors, tau=1.0)
    bs = build_loss("balanced_softmax", priors=priors)

    assert isinstance(ce, CrossEntropyLoss)
    assert isinstance(la, LogitAdjustmentLoss)
    assert isinstance(bs, BalancedSoftmaxLoss)
    assert torch.allclose(la(logits, targets), bs(logits, targets), atol=1e-7)

    with pytest.raises(ValueError, match="priors"):
        build_loss("la", priors=None)

    with pytest.raises(ValueError, match="Unsupported loss_name"):
        build_loss("unknown_loss")
