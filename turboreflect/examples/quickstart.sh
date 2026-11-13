#!/usr/bin/env bash
# Builds the first eight LongBench-v2 items for Qwen3-14B, calibrates the deployment,
# runs them as three-stage reflection programs on one GPU and prints their numbers.
# Tasks go to tasks/quickstart/, results to runs/quickstart/.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
config=turboreflect/configs/qwen3-14b.yaml
tasks=tasks/quickstart
settings=(--set layout=continue --set max_model_len=40960)

python -m turboreflect.benchmarks.longbench_v2 build --model Qwen/Qwen3-14B \
    --out "$tasks" --limit 8
python -m turboreflect calibrate --config "$config" --tasks "$tasks" "${settings[@]}"
python -m turboreflect run --config "$config" --tasks "$tasks" --out-dir runs/quickstart \
    "${settings[@]}"
python -m turboreflect report runs/quickstart
