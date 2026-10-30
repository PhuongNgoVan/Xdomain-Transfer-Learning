"""Deterministic seed utilities for Python `random`, `numpy`, and `torch`."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix random seeds across `random`, `numpy`, and `torch` for reproducibility.

    Args:
        seed: Non-negative integer seed value.
        deterministic: Whether to configure PyTorch cuDNN for deterministic execution.

    Raises:
        ValueError: If `seed` is not an integer or is negative.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
