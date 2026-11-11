from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .extension import load_extension


_EXT = None


def _custom(q, k, v, kernel, br):
    global _EXT
    if _EXT is None:
        _EXT = load_extension()
    return _EXT.forward(q, k, v, kernel, br)


def custom_naive(q, k, v, br=64):
    return _custom(q, k, v, 0, br)


def fa1(q, k, v, br=64):
    return _custom(q, k, v, 1, br)


def fa2(q, k, v, br=64):
    return _custom(q, k, v, 2, br)


def torch_sdpa(q, k, v, backend=None):
    if backend is None:
        context = nullcontext()
    else:
        context = torch.nn.attention.sdpa_kernel(backend)
    with context:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def torch_math(q, k, v):
    return torch_sdpa(q, k, v, torch.nn.attention.SDPBackend.MATH)


def torch_flash(q, k, v):
    return torch_sdpa(q, k, v, torch.nn.attention.SDPBackend.FLASH_ATTENTION)


OPERATORS = {
    "naive": custom_naive,
    "fa1": fa1,
    "fa2": fa2,
    "torch": torch_sdpa,
    "torch_math": torch_math,
    "torch_flash": torch_flash,
}
