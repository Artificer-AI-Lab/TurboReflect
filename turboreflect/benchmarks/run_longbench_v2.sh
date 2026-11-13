#!/usr/bin/env bash
# Builds LongBench-v2 for a run configuration's model, runs it, and exports the answers
# of the generation and refinement stages to preds/longbench_v2/ for LongBench's
# result.py. Run from the root of this repository. Tasks go to tasks/longbench_v2/,
# results to runs/longbench_v2/. TARGET is the generation-prompt length documents are
# cut to, in the model's tokens (default 32000).
#
#   bash turboreflect/benchmarks/run_longbench_v2.sh turboreflect/configs/qwen3-14b.yaml [TARGET]
set -euo pipefail

config=$1
target=${2:-32000}
tasks=tasks/longbench_v2
run=runs/longbench_v2
preds=preds/longbench_v2

# Each stage stops at the natural end of sequence, on LongBench-v2's own prompt.
settings=(
    --set layout=continue
    --set "decode_paths=[quality]"
    --set max_tokens_quality=2048
)

model=$(python -c 'import sys
from turboreflect.config import read_config_file
print(read_config_file(sys.argv[1])["model"])' "$config")

python -m turboreflect.benchmarks.longbench_v2 build --model "$model" --out "$tasks" \
    --target "$target"
python -m turboreflect calibrate --config "$config" --tasks "$tasks" "${settings[@]}"
python -m turboreflect run --config "$config" --tasks "$tasks" --out-dir "$run" "${settings[@]}"
python -m turboreflect.benchmarks.longbench_v2 export "$run" --out "$preds"
