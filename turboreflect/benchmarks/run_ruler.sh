#!/usr/bin/env bash
# Builds RULER, runs it under a run configuration, and exports the answers of the
# generation and refinement stages to preds/ruler/ for RULER's scripts/eval/evaluate.py.
# Run from the root of this repository. Tasks go to tasks/ruler/, results to runs/ruler/.
#
#   bash turboreflect/benchmarks/run_ruler.sh turboreflect/configs/qwen3-14b.yaml ruler_assets
set -euo pipefail

config=$1
assets=$2
tasks=tasks/ruler
run=runs/ruler
preds=preds/ruler

# Each stage stops at the natural end of sequence, on RULER's own prompt.
settings=(
    --set layout=continue
    --set "decode_paths=[quality]"
    --set max_tokens_quality=2048
)

python -m turboreflect.benchmarks.ruler build --assets "$assets" --out "$tasks"
python -m turboreflect calibrate --config "$config" --tasks "$tasks" "${settings[@]}"
python -m turboreflect run --config "$config" --tasks "$tasks" --out-dir "$run" "${settings[@]}"
python -m turboreflect.benchmarks.ruler export "$run" --out "$preds"
