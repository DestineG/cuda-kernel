#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WARMUP="${WARMUP:-100}"
ITERATIONS="${ITERATIONS:-1000}"

LARGE_ITERATIONS="${LARGE_ITERATIONS:-100}"

printf 'custom benchmark\n'
printf 'warmup=%s iterations=%s large_iterations=%s\n' "$WARMUP" "$ITERATIONS" "$LARGE_ITERATIONS"
printf '\n[1024x1024x1024]\n'
"$SCRIPT_DIR/build/gemm_custom" benchmark cublas 1024 1024 1024 "$WARMUP" "$ITERATIONS"
"$SCRIPT_DIR/build/gemm_custom" benchmark naive 1024 1024 1024 "$WARMUP" "$ITERATIONS"
"$SCRIPT_DIR/build/gemm_custom" benchmark custom 1024 1024 1024 "$WARMUP" "$ITERATIONS"
printf '\n[2048x2048x2048]\n'
"$SCRIPT_DIR/build/gemm_custom" benchmark cublas 2048 2048 2048 "$WARMUP" "$ITERATIONS"
"$SCRIPT_DIR/build/gemm_custom" benchmark naive 2048 2048 2048 "$WARMUP" "$ITERATIONS"
"$SCRIPT_DIR/build/gemm_custom" benchmark custom 2048 2048 2048 "$WARMUP" "$ITERATIONS"
printf '\n[4096x4096x4096]\n'
"$SCRIPT_DIR/build/gemm_custom" benchmark cublas 4096 4096 4096 "$WARMUP" "$LARGE_ITERATIONS"
"$SCRIPT_DIR/build/gemm_custom" benchmark custom 4096 4096 4096 "$WARMUP" "$LARGE_ITERATIONS"
printf '\n[2048x1024x2048]\n'
"$SCRIPT_DIR/build/gemm_custom" benchmark cublas 2048 1024 2048 "$WARMUP" "$ITERATIONS"
"$SCRIPT_DIR/build/gemm_custom" benchmark custom 2048 1024 2048 "$WARMUP" "$ITERATIONS"
printf '\n[1024x2048x2048]\n'
"$SCRIPT_DIR/build/gemm_custom" benchmark cublas 1024 2048 2048 "$WARMUP" "$ITERATIONS"
"$SCRIPT_DIR/build/gemm_custom" benchmark custom 1024 2048 2048 "$WARMUP" "$ITERATIONS"
