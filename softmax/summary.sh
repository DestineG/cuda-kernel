#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NCU_BIN="${NCU_BIN:-/usr/local/cuda/bin/ncu}"
REPORT_DIR="${REPORT_DIR:-$SCRIPT_DIR/build/ncu_reports}"
SUMMARY_DIR="${SUMMARY_DIR:-$SCRIPT_DIR/build}"
MODE="${1:-brief}"

usage() {
    echo "Usage: $0 [brief|detail]"
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 1
fi

case "$MODE" in
    brief|detail)
        OUTPUT_FILE="$SUMMARY_DIR/ncu_summary_$MODE.md"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unsupported mode: $MODE" >&2
        usage >&2
        exit 1
        ;;
esac

if [[ ! -x "$NCU_BIN" ]]; then
    echo "Error: ncu not found or not executable: $NCU_BIN" >&2
    exit 1
fi

if [[ ! -d "$REPORT_DIR" ]]; then
    echo "Error: report directory not found: $REPORT_DIR" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required to parse NCU CSV output" >&2
    exit 1
fi

mkdir -p "$SUMMARY_DIR"

python3 - "$MODE" "$NCU_BIN" "$REPORT_DIR" > "$OUTPUT_FILE" <<'PY'
import csv
import glob
import os
import re
import subprocess
import sys

mode, ncu_bin, report_dir = sys.argv[1:4]
name_pattern = re.compile(r"^softmax_ncu_b(\d+)_c(\d+)\.ncu-rep$")

brief_columns = [
    ("batch", "Batch"),
    ("num_classes", "Num Classes"),
    ("kernel", "Kernel"),
    ("gpu__time_duration.sum", "Duration (us)"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "Compute (%)"),
    (
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "Memory (%)",
    ),
]

detail_columns = brief_columns + [
    ("launch__registers_per_thread", "Registers/Thread"),
    ("launch__grid_size", "Grid"),
    ("launch__block_size", "Block"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "Occupancy (%)"),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM (%)"),
    ("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "L1/TEX (%)"),
    ("lts__throughput.avg.pct_of_peak_sustained_elapsed", "L2 (%)"),
    ("l1tex__t_sector_hit_rate.pct", "L1 Hit (%)"),
    ("lts__t_sector_hit_rate.pct", "L2 Hit (%)"),
    ("dram__bytes_read.sum", "DRAM Read (B)"),
    ("dram__bytes_write.sum", "DRAM Write (B)"),
]

columns = brief_columns if mode == "brief" else detail_columns

metric_aliases = {
    "launch__grid_size": ["Grid Size", "launch__grid_size"],
    "launch__block_size": ["Block Size", "launch__block_size"],
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dramc__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "lts__t_sector_hit_rate.pct": [
        "lts__t_sector_hit_rate.pct",
        "lts__average_t_sector_hit_rate_realtime.pct",
    ],
}


def report_key(path):
    match = name_pattern.match(os.path.basename(path))
    if match is None:
        return (sys.maxsize, sys.maxsize, path)
    return (int(match.group(1)), int(match.group(2)), path)


def clean_number(value):
    try:
        number = float(value.replace(",", "").strip())
    except ValueError:
        return value.strip()
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def duration_us(value, unit):
    try:
        number = float(value.replace(",", "").strip())
    except ValueError:
        return value.strip()

    normalized = unit.strip().lower()
    if normalized in {"nsecond", "ns"}:
        number /= 1000.0
    elif normalized in {"msecond", "ms"}:
        number *= 1000.0
    elif normalized in {"second", "s"}:
        number *= 1_000_000.0

    return f"{number:.3f}"


def markdown_cell(value):
    return str(value).replace("|", r"\|").replace("\n", " ")


def parse_report(path, batch, num_classes):
    command = [
        ncu_bin,
        "--import",
        path,
        "--page",
        "raw",
        "--csv",
        "--print-fp",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"failed to import report: {path}")

    rows = list(csv.reader(result.stdout.splitlines()))
    header_index = None

    for index, row in enumerate(rows):
        if "ID" in row and "Kernel Name" in row:
            header_index = index
            break

    if header_index is None or header_index + 1 >= len(rows):
        raise RuntimeError(f"NCU CSV header not found in report: {path}")

    headers = rows[header_index]
    units = rows[header_index + 1]
    id_index = headers.index("ID")
    kernel_index = headers.index("Kernel Name")

    target_metrics = [
        metric
        for metric, _ in columns
        if metric not in {"batch", "num_classes", "kernel"}
    ]

    metric_indices = {}
    for metric in target_metrics:
        aliases = metric_aliases.get(metric, [metric])
        metric_indices[metric] = [
            index
            for index, header in enumerate(headers)
            if any(
                header == alias or header.endswith("." + alias)
                for alias in aliases
            )
        ]

    records = {}
    for row in rows[header_index + 2 :]:
        if len(row) <= max(id_index, kernel_index):
            continue

        launch_id = row[id_index].strip()
        kernel = row[kernel_index].strip()
        if not launch_id or not kernel:
            continue

        record = {
            "batch": str(batch),
            "num_classes": str(num_classes),
            "kernel": kernel,
            "_units": {},
        }

        for metric, indices in metric_indices.items():
            for index in indices:
                if index >= len(row):
                    continue
                value = row[index].strip()
                if not value:
                    continue
                record[metric] = value
                record["_units"][metric] = (
                    units[index].strip() if index < len(units) else ""
                )
                break

        records[(launch_id, kernel)] = record

    return records


reports = []
for path in glob.glob(os.path.join(report_dir, "softmax_ncu_b*_c*.ncu-rep")):
    match = name_pattern.match(os.path.basename(path))
    if match is not None:
        reports.append((path, int(match.group(1)), int(match.group(2))))

reports.sort(key=lambda item: report_key(item[0]))

if not reports:
    raise SystemExit(f"no NCU reports found in {report_dir}")

all_records = []
for path, batch, num_classes in reports:
    parsed = parse_report(path, batch, num_classes)
    for key in sorted(
        parsed,
        key=lambda item: (
            int(item[0]) if item[0].isdigit() else sys.maxsize,
            item[1],
        ),
    ):
        all_records.append(parsed[key])

headers = [label for _, label in columns]
print("| " + " | ".join(headers) + " |")
print("| " + " | ".join("---" for _ in headers) + " |")

for record in all_records:
    cells = []
    for metric, _ in columns:
        value = record.get(metric, "-")
        if value != "-":
            if metric == "gpu__time_duration.sum":
                value = duration_us(value, record["_units"].get(metric, ""))
            elif metric not in {"batch", "num_classes", "kernel"}:
                value = clean_number(value)
        cells.append(markdown_cell(value))
    print("| " + " | ".join(cells) + " |")
PY

echo "Summary written to: $OUTPUT_FILE"
