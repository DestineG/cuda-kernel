#!/bin/bash

BATCH=(512 1024 2048 4096 8192 16384)
NUM_CLASSES=(256 512 1024 2048 4096)

REPORT_DIR="./softmax/build/ncu_reports"
mkdir -p "$REPORT_DIR"

# 提前验证并缓存 sudo 凭据
sudo -v || exit 1

for batch in "${BATCH[@]}"; do
    for num_classes in "${NUM_CLASSES[@]}"; do
        report="${REPORT_DIR}/softmax_ncu_b${batch}_c${num_classes}"

        echo "Profiling: batch=${batch}, num_classes=${num_classes}"

        sudo /usr/local/cuda/bin/ncu --set full --force-overwrite -o "$report" ./softmax/build/softmax profile all "$batch" "$num_classes" || exit 1
    done
done
