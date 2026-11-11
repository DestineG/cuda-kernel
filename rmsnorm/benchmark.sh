#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
N_VALUES=(512 1024 2048 4096 8192 16384)
M_VALUES=(128 256 512 1024 2048 4096)
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-100}"
SAMPLES="${SAMPLES:-10}"

for N in "${N_VALUES[@]}"; do
    for M in "${M_VALUES[@]}"; do
        echo "Benchmarking N=$N, M=$M"
        "$SCRIPT_DIR/build/rmsnorm" benchmark all "$N" "$M" "$WARMUP" "$ITERATIONS" "$SAMPLES"
    done
done
