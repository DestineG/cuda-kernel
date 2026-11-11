#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
REPORT_DIR=""
NVCC_BIN="/usr/local/cuda/bin/nvcc"
NCU_BIN="/usr/local/cuda/bin/ncu"
CUDA_ARCH="sm_89"
WARMUP=10
ITERATIONS=100
PROFILE_WARMUP=0
PROFILE_ITERATIONS=1
NCU_SET="full"
KERNEL="all"
USE_SUDO=1
SIZES=()
SUMMARY_ROWS=()
PROFILE_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  benchmark.sh [options] [N:HEAD_DIM:Br ...]

Options:
  --warmup N              Benchmark warmup launches (default: 10)
  --iterations N          Benchmark measured launches (default: 100)
  --profile-warmup N      NCU profile warmup launches (default: 0)
  --profile-iterations N  NCU profile measured launches (default: 1)
  --kernel all|naive|fa1|fa2
                          Kernels to verify, benchmark, and profile (default: all)
  --ncu-set SET           Nsight Compute set (default: full)
  --build-dir DIR         Launcher output directory (default: flashattn/build)
  --report-dir DIR        NCU report directory (default: BUILD_DIR/ncu_reports)
  --arch ARCH             CUDA architecture (default: sm_89)
  --nvcc PATH             nvcc executable
  --ncu PATH              ncu executable
  --no-sudo               Do not use sudo for ncu
  -h, --help              Show this help

Each positional size is N:HEAD_DIM:Br, for example 256:64:32. Bc is fixed at 64.
If no sizes are given,
the default matrix is used.
EOF
}

positive() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
nonnegative() { [[ "$1" =~ ^[0-9]+$ ]]; }

need_value() {
    if (($# < 2)); then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
    fi
}

while (($# > 0)); do
    case "$1" in
        --warmup) need_value "$@"; WARMUP="$2"; shift 2 ;;
        --iterations) need_value "$@"; ITERATIONS="$2"; shift 2 ;;
        --profile-warmup) need_value "$@"; PROFILE_WARMUP="$2"; shift 2 ;;
        --profile-iterations) need_value "$@"; PROFILE_ITERATIONS="$2"; shift 2 ;;
        --kernel) need_value "$@"; KERNEL="$2"; shift 2 ;;
        --ncu-set) need_value "$@"; NCU_SET="$2"; shift 2 ;;
        --build-dir) need_value "$@"; BUILD_DIR="$2"; shift 2 ;;
        --report-dir) need_value "$@"; REPORT_DIR="$2"; shift 2 ;;
        --arch) need_value "$@"; CUDA_ARCH="$2"; shift 2 ;;
        --nvcc) need_value "$@"; NVCC_BIN="$2"; shift 2 ;;
        --ncu) need_value "$@"; NCU_BIN="$2"; shift 2 ;;
        --no-sudo) USE_SUDO=0; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
        *) SIZES+=("$1"); shift ;;
    esac
done

if ! positive "$WARMUP" || ! positive "$ITERATIONS" ||
   ! nonnegative "$PROFILE_WARMUP" || ! positive "$PROFILE_ITERATIONS"; then
    echo "warmup/iterations must be positive; profile-warmup may be zero" >&2
    exit 1
fi
case "$KERNEL" in
    all|naive|fa1|fa2) ;;
    *) echo "Invalid --kernel: $KERNEL" >&2; exit 1 ;;
esac

if ((${#SIZES[@]} == 0)); then
    SIZES=(64:64:16 128:64:32 256:64:64 64:128:16 128:128:32 256:128:64)
fi
if [[ -z "$REPORT_DIR" ]]; then
    REPORT_DIR="$BUILD_DIR/ncu_reports"
fi
LAUNCHER="$BUILD_DIR/launcher"

if [[ ! -x "$NVCC_BIN" ]]; then
    echo "Error: nvcc not found or not executable: $NVCC_BIN" >&2
    exit 1
fi
if [[ ! -x "$NCU_BIN" ]]; then
    echo "Error: ncu not found or not executable: $NCU_BIN" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR" "$REPORT_DIR"

echo "Compiling launcher"
"$NVCC_BIN" -O3 -std=c++17 -arch="$CUDA_ARCH" -lineinfo -Xptxas=-v \
    "$SCRIPT_DIR/launcher.cu" -o "$LAUNCHER"

PROFILE_REPORT="$REPORT_DIR/flashattn_all"
PROFILE_ARGS=(profile_batch "$KERNEL" "$PROFILE_WARMUP" "$PROFILE_ITERATIONS")

run_ncu() {
    local report="$1"
    shift
    if ((USE_SUDO == 0)) || [[ "$(id -u)" == "0" ]]; then
        "$NCU_BIN" --quiet --set "$NCU_SET" --force-overwrite -o "$report" "$@"
        return
    fi

    printf '%s\n' "$(id -un)" |
        sudo -S -p '' "$NCU_BIN" --quiet --set "$NCU_SET" --force-overwrite -o "$report" "$@"
}

for size in "${SIZES[@]}"; do
    if [[ "$size" != *:*:* ]]; then
        echo "Invalid size '$size'; expected N:HEAD_DIM:Br, for example 256:64:32" >&2
        exit 1
    fi
    n="${size%%:*}"
    rest="${size#*:}"
    d="${rest%%:*}"
    br="${rest##*:}"
    if ! positive "$n" || ! positive "$d" || ! positive "$br"; then
        echo "Invalid size '$size'; expected positive integers in N:HEAD_DIM:Br" >&2
        exit 1
    fi

    if [[ "$KERNEL" != "naive" ]] && { ((n % 64 != 0)) || ((n % br != 0)); }; then
        echo "Invalid size '$size'; FA1 requires N divisible by 64 and Br" >&2
        exit 1
    fi

    echo
    echo "=== N=$n HEAD_DIM=$d Br=$br Bc=64: correctness and benchmark ==="
    benchmark_output="$($LAUNCHER benchmark "$KERNEL" "$n" "$d" "$br" "$WARMUP" "$ITERATIONS")"
    printf '%s\n' "$benchmark_output"

    naive_ms="-"
    fa1_ms="-"
    fa2_ms="-"
    while IFS= read -r line; do
        case "$line" in
            benchmark\ naive\ *) naive_ms="${line##*time_ms=}" ;;
            benchmark\ fa1\ *) fa1_ms="${line##*time_ms=}" ;;
            benchmark\ fa2\ *) fa2_ms="${line##*time_ms=}" ;;
        esac
    done <<< "$benchmark_output"

    PROFILE_ARGS+=("$n" "$d" "$br")
    SUMMARY_ROWS+=("$n|$d|$br|64|$naive_ms|$fa1_ms|$fa2_ms|${PROFILE_REPORT}.ncu-rep")
done

echo
echo "=== Combined Nsight Compute report -> ${PROFILE_REPORT}.ncu-rep ==="
run_ncu "$PROFILE_REPORT" --target-processes all "$LAUNCHER" "${PROFILE_ARGS[@]}"

echo
echo "Reports: $REPORT_DIR"
echo
echo "Benchmark Summary (CUDA event time; profile time is intentionally omitted)"
printf '%-8s %-10s %-6s %-6s %-12s %-12s %-12s %s\n' \
    "N" "HEAD_DIM" "Br" "Bc" "naive_ms" "fa1_ms" "fa2_ms" "NCU report"
printf '%-8s %-10s %-6s %-6s %-12s %-12s %-12s %s\n' \
    "--------" "----------" "------" "------" "------------" "------------" "------------" "----------"
for row in "${SUMMARY_ROWS[@]}"; do
    IFS='|' read -r row_n row_d row_br row_bc row_naive row_fa1 row_fa2 row_report <<< "$row"
    printf '%-8s %-10s %-6s %-6s %-12s %-12s %-12s %s\n' \
        "$row_n" "$row_d" "$row_br" "$row_bc" "$row_naive" "$row_fa1" "$row_fa2" "$row_report"
done
