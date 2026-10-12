#  Copyright (c) Prior Labs GmbH 2026.

"""Check for the enable_gqa parameter of PyTorch scaled_dot_product_attention."""

from __future__ import annotations

import functools

import torch
from torch.torch_version import TorchVersion


@functools.cache
def gqa_is_supported() -> bool:
    """Check if PyTorch's scaled_dot_product_attention supports enable_gqa parameter.

    This checks whether torch.nn.functional.scaled_dot_product_attention has a
    kwarg enable_gqa
    For CUDA, checks for sufficient NVIDIA compute capability and PyTorch 2.5+.
    For XPU, checks for PyTorch 2.8+.
    """
    if torch.cuda.is_available():
        # Check compute capability only if CUDA is available
        # We need compute capability >= 8.0 for efficient GQA
        has_enable_gqa = torch.__version__ >= TorchVersion("2.5")
        if not has_enable_gqa:
            return False
        device = torch.cuda.current_device()
        nvidia_compute_capability = torch.cuda.get_device_capability(device)
        return nvidia_compute_capability[0] >= 8
    if torch.xpu.is_available():
        return torch.__version__ >= TorchVersion("2.8")
    return False
