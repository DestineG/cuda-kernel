from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parents[2]
_NATIVE = _ROOT / "native"
_CACHE = _ROOT / "build"


def load_extension():
    _CACHE.mkdir(parents=True, exist_ok=True)
    return load(
        name="flashattn_torch_ext",
        sources=[str(_NATIVE / "torch_ext.cpp"), str(_NATIVE / "torch_ext.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        build_directory=str(_CACHE),
        verbose=False,
    )
