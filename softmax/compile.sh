#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
NVCC_BIN="${NVCC_BIN:-/usr/local/cuda/bin/nvcc}"
CUDA_ARCH="${CUDA_ARCH:-sm_89}"
SOURCE_FILE="$SCRIPT_DIR/softmax.cu"
OUTPUT_FILE="$BUILD_DIR/softmax"

if [[ ! -x "$NVCC_BIN" ]]; then
    echo "Error: nvcc not found or not executable: $NVCC_BIN" >&2
    exit 1
fi

if [[ ! -f "$SOURCE_FILE" ]]; then
    echo "Error: source file not found: $SOURCE_FILE" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "Compiling $SOURCE_FILE"
echo "Architecture: $CUDA_ARCH"

"$NVCC_BIN" -O3 -std=c++17 -arch="$CUDA_ARCH" -lineinfo -Xptxas=-v "$SOURCE_FILE" -o "$OUTPUT_FILE" -lcudnn

echo "Output: $OUTPUT_FILE"
