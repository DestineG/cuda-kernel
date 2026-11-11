#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/build/gemm_custom" verify custom 128 128 64
"$SCRIPT_DIR/build/gemm_custom" verify custom 1024 1024 1024
"$SCRIPT_DIR/build/gemm_custom" verify custom 2048 2048 2048
