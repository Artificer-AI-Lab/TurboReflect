# Benchmarks

## Overview

This folder runs public benchmarks through TurboReflect, so that the answers of a reflection program can be scored with each benchmark's own tool. Each benchmark item becomes one program: the benchmark's own prompt is the generation stage, and the feedback and refinement stages follow.

RULER and LongBench-v2 have scripts here. For each, the steps are:

1. `build` writes the benchmark's items as a [task folder](#the-task-folder).
2. `python -m turboreflect calibrate` and `run` run it, as any other tasks.
3. `export` writes the answers of the generation stage and of the whole program in the benchmark's prediction format.
4. The benchmark's own script scores them.

`run_ruler.sh` and `run_longbench_v2.sh` do steps 1 to 3 in one go. Tasks of your own run the same way; see [your own benchmark](#your-own-benchmark). The commands below run from the root of this repository.

## Files

| File | What it does |
|---|---|
| `ruler.py` | `build` writes RULER tasks, `export` writes a run's answers in RULER's prediction format |
| `run_ruler.sh` | builds, runs and exports RULER in one go |
| `longbench_v2.py` | `build` writes LongBench-v2 tasks, `export` writes a run's answers in LongBench's prediction format |
| `run_longbench_v2.sh` | builds, runs and exports LongBench-v2 in one go |
| `common.py` | stage assembly, task files, tokenizer fit and run reading, shared by both |
| `custom/` | building your own benchmark: sample tasks, the format verifier and the shared stage assembly |

## The task folder

`build` writes a folder that `python -m turboreflect run --tasks` takes as it is:

```text
tasks/ruler/
  data/       one task file per item, what the model sees
  rubrics/    the gold answers, which only `export` reads
  prompts.py  tells a run to use the stage assembly in common.py
```

These tasks are one user message with no system message, so they run in layout `continue`. A configuration that states `layout: requote`, as the files in `turboreflect/configs/` do, is refused: remove it, or pass `--set layout=continue`. The `run_*.sh` scripts and the step-by-step commands below pass it.

## Decoding

The `run_*.sh` scripts stop each stage at the natural end of sequence, capped at 2,048 tokens, and sample as the configuration states. For the same settings by hand, add this to the configuration:

```yaml
decode_paths: [quality]
max_tokens_quality: 2048
```

## RULER

Four tasks from the original [RULER](https://github.com/NVIDIA/RULER), 50 items each: `niah_multikey_3` and `niah_multivalue` (multi-key and multi-value needle retrieval), `vt` (variable tracking) and `cwe` (common-word extraction).

The prompt is RULER's task template with its answer prefix. Haystacks are sized with one tokenizer for every model, so that a length means the same on each. `--tiers 128k` builds the longer tier, into a task folder of its own, and `--tokenizer` changes the sizing tokenizer. Items are seeded, so `--seed` and `--per-task` decide the set.

Fetch RULER's haystack material with its own script, and NLTK's sentence tokenizer:

```bash
uv pip install html2text beautifulsoup4
mkdir -p ruler_assets && cd ruler_assets
BASE=https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic/json
curl -sSLO $BASE/PaulGrahamEssays_URLs.txt
curl -sSLO $BASE/download_paulgraham_essay.py
python download_paulgraham_essay.py
curl -sSLO https://media.githubusercontent.com/media/NVIDIA/RULER/main/scripts/data/synthetic/json/english_words.json
cd ..
python -m nltk.downloader punkt_tab
```

Then run everything:

```bash
bash turboreflect/benchmarks/run_ruler.sh turboreflect/configs/qwen3-14b.yaml ruler_assets
```

Or step by step, with the [decoding settings](#decoding) added to `turboreflect/configs/qwen3-14b.yaml`:

```bash
python -m turboreflect.benchmarks.ruler build --assets ruler_assets --out tasks/ruler
python -m turboreflect calibrate --config turboreflect/configs/qwen3-14b.yaml --tasks tasks/ruler --set layout=continue
python -m turboreflect run --config turboreflect/configs/qwen3-14b.yaml --tasks tasks/ruler --set layout=continue --out-dir runs/ruler
python -m turboreflect.benchmarks.ruler export runs/ruler --out preds/ruler
```

`export` writes one `{task}.jsonl` per task into `preds/ruler/generation/` and `preds/ruler/refinement/`. Score each folder with RULER's own script:

```bash
cd /path/to/RULER/scripts/eval
python evaluate.py --data_dir "$OLDPWD/preds/ruler/refinement" --benchmark synthetic
```

## LongBench-v2

All 503 items of the original [LongBench-v2](https://github.com/THUDM/LongBench) with its zero-shot prompt. A document longer than the target is middle-truncated with LongBench-v2's own cut, the first half and the last half of its tokens, on the model's tokenizer; the question and the answer format are never cut. Tasks therefore differ between models, so build them once per model. `--target` sets the generation-prompt length to cut to, in the model's tokens, 32,000 by default.

`build` downloads the original data from Hugging Face, or reads a local `data.json` given with `--source`. `run_longbench_v2.sh` takes the model from the configuration, and its optional second argument is the target length:

```bash
bash turboreflect/benchmarks/run_longbench_v2.sh turboreflect/configs/qwen3-14b.yaml [TARGET]
```

Or step by step, with the [decoding settings](#decoding) added to `turboreflect/configs/qwen3-14b.yaml`, where `--model` is its `model`:

```bash
python -m turboreflect.benchmarks.longbench_v2 build --model Qwen/Qwen3-14B --out tasks/longbench_v2 --target 32000
python -m turboreflect calibrate --config turboreflect/configs/qwen3-14b.yaml --tasks tasks/longbench_v2 --set layout=continue
python -m turboreflect run --config turboreflect/configs/qwen3-14b.yaml --tasks tasks/longbench_v2 --set layout=continue --out-dir runs/longbench_v2
python -m turboreflect.benchmarks.longbench_v2 export runs/longbench_v2 --out preds/longbench_v2
```

`export` writes `generation.jsonl` and `refinement.jsonl` into `preds/longbench_v2/`, with LongBench's own `extract_answer` applied. LongBench's `result.py` reads a `results/` folder in its working directory and prints a row per file:

```bash
ln -s "$PWD/preds/longbench_v2" /path/to/LongBench/results
cd /path/to/LongBench && python result.py && cat result.txt
```

## Exported answers

Both `export` commands take a run directory, or a folder of them, whose newest run they read. They find the gold answers in the task folder the run read; `--tasks` names it if it has moved. `refinement` is what the program delivered at the end; for a two-stage program, the revised response without the review before it.

## Your own benchmark

The [task format](../../docs/turboreflect/task-format.md) shows what a task file holds, how it becomes each stage's prompt, how to write a text task and a video task after the samples in `custom/`, and how to check your files.
