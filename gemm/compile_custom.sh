#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
NVCC_BIN="${NVCC_BIN:-/usr/local/cuda/bin/nvcc}"
CUDA_ARCH="${CUDA_ARCH:-sm_89}"

mkdir -p "$BUILD_DIR"
"$NVCC_BIN" -O3 -std=c++17 -arch="$CUDA_ARCH" -lineinfo -Xptxas=-v "$SCRIPT_DIR/gemm_custom.cu" -o "$BUILD_DIR/gemm_custom" -lcublas
echo "Output: $BUILD_DIR/gemm_custom"
