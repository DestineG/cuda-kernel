#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
N_VALUES=(1 31 512)
M_VALUES=(1 31 32 33 127 128 129 255 256 257 511 512 513 1023 1024 1025 2047 2048 2049 4095 4096)

for N in "${N_VALUES[@]}"; do
    for M in "${M_VALUES[@]}"; do
        echo "Verifying N=$N, M=$M"
        "$SCRIPT_DIR/build/layernorm" verify all "$N" "$M"
    done
done
