# double_V output experiment

Independent H800 experiment; no source repository files are edited. Both variants keep BF16 Q/K/V, FP32 accumulators, the original attention arithmetic, and the original K/V synchronization protocol. The BF16 variant changes only final output conversion/storage and its descriptor/launcher.

Output shared storage aliases the Q/K/V allocation: total dynamic shared memory is identical at matched parameters. BF16 output uses BOX_O_N=64; FP32 uses BOX_O_N=32, so both retain 128-byte swizzle width. Full tiles use TMA; partial query tiles use direct packed global stores. BF16 conversion is round-to-nearest-even.

## Run

```sh
cd /root/autodl-tmp/fa3-double-v-output/flashattn
python tests/run.py all --config tests/configs/full.json --run-root ../runs --name double-v-output
python tests/run.py resume --run /path/to/run
python tests/run.py report --run /path/to/run
python tests/run.py archive --run /path/to/run
```

Four common controls (FP32/BF16 x register redistribution off/on 40/232) run all 157 correctness inputs. A paired pass compares BF16 output with the FP32 result rounded to BF16 at identical parameters. Zero-KV no-write observations are separate. Four sanitizer tools gate default and final selected configurations.

Both variants independently search six tile shapes, producer registers 32/40/48/64, consumer registers 128/160/192/224/232/240/256, and feasible Q/KV/O box rows 64/128. This is staged search, not exhaustive Cartesian search. Per D/causal group, selection uses three representative regimes. All 268 performance inputs compare fresh FP32/BF16 measurements plus official FA3 (automatic split), private PyTorch ATen varlen, and public forced-Flash per-sequence SDPA. Twelve cases repeat in reversed configuration order. Default views have identical tile/register parameters; tuned views use independently selected parameters.

Report: input conditions on the left, hyperparameters in the middle, measured results on the right. Linked multi-select filters, sorting and CSV export; no speedups or relative values. Five CUDA-graph rounds are measured; launcher wall time is separate. Correctness uses CPU FP64 for small validation inputs and explicit GPU FP32 for performance inputs; tolerances match the previous experiment. Numerical failures stop work. Unsupported shared-memory candidates and sanitizer-rejected candidates are retained separately.

Archive includes source, change patch, configuration, exact installed FA3 package, build commands/logs, raw records and reproducible harness. Rebuild custom binaries with harness/runners/reproduce.py; custom shared objects and reference caches are excluded.
