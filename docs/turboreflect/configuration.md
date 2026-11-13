# Configuration reference

A run configuration is a YAML or JSON file. It names the model and the tasks, and chooses the components; the defaults below fill in the rest, and any field the file states is kept over what these would set. Start from your model's file in `turboreflect/configs/`, which states that model's settings ([adding a model](adding-a-model.md) lists them), and set the components.

`run` and `calibrate` take the same flags:

| Flag | Meaning |
|---|---|
| `--config PATH` | the configuration file |
| `--model`, `--tasks`, `--out-dir` | override these fields |
| `--set NAME=VALUE` | override any field; a dotted name reaches into a block, as in `--set pd.delta=0.5`, and the value reads as YAML; repeat per field |

`run` also takes `--dry-run`, which starts no engine: it sizes every prompt with the model's tokenizer and reports whether every task fits `max_model_len`. For tasks with clips it also imports the vLLM model class that the checkpoint's `config.json` names, for the clip placeholder, and runs vLLM's processor over the clips to count their tokens. `calibrate` takes `--force`, which repeats steps that have already run.

`calibrate` and `run` must see the same configuration and flags, apart from `--out-dir` and `--tasks`: calibration results are kept per model, GPU and configuration, and `run` refuses to start without its own. Other tasks reuse them, unless `max_num_seqs` is left unstated: the limit derived from the tasks sets the batches calibration times, so then the tasks count too.

## Components

| Field | Default | Meaning |
|---|---|---|
| `model` | required | a Hugging Face repo id or a local checkpoint |
| `tasks` | required | a task file, a folder of task files, or a list of either; see the [task format](task-format.md) |
| `program` | `three_stage` | `two_stage`, `three_stage` or `multi_round` |
| `context_reuse` | `true` | hold a finished stage's KV cache for its successor |
| `adaptive_draft_length` | `true` | draft with SuffixDecoding and cut drafts by measured verification time |
| `kv_cache_dtype` | `fp8` | `fp8` or `bf16`; vLLM's `auto`, `bfloat16` and `fp8_e4m3` also work, and a model's file states the FP8 spelling it runs with |
| `pd` | absent | run locality-aware P/D; see below |

A `two_stage` program runs generation and a refinement stage that writes its review and then the whole revised response, so that stage may generate up to twice `gen_tokens` (`refinement_gen_tokens`). A `multi_round` program runs generation once, then three rounds of feedback and refinement (`max_rounds`).

To run a `two_stage` program over several rounds, keep `program: two_stage` and state `pipeline_mode: rounds` and `max_rounds`: generation, then that many refinement stages. `max_rounds` alone changes nothing in a `two_stage` program, and `program: multi_round` with a merged `refinement` leaves out the doubled limit.

With `adaptive_draft_length` off, a run does not draft. Stating `suffix_decoding: true` then verifies every draft as proposed.

## The `pd` block

Without a `pd` block, a run is one engine on `tensor_parallel_size` GPUs. A `pd` block runs locality-aware P/D:

| Field | Default | Meaning |
|---|---|---|
| `prefill` | required | how many prefill engines |
| `decode` | required | how many decode engines |
| `max_num_seqs` | required | each engine's sequence limit: how many programs at their longest fit an engine |
| `delta` | `1.0` | the relative decode-latency budget |

A single-engine run derives this limit from the measured KV pool and prints it: `calibrate` or `run` on the same configuration without its `pd` block shows the value to state.

Engines take the node's GPUs in order, prefill first, `tensor_parallel_size` GPUs each. Their NIXL side channels open from port 25600 up. P/D needs `context_reuse`. Any other field of the block is a setting of the engines, such as `kv_lease_duration_s` (600), `request_deadline_s` (7200) and `decoder_kv_blocks_ttl_s` (7200).

Each engine decodes the clips of tasks with `videos` and holds its own copy of their frames. Setting `TURBOREFLECT_FRAME_STORE` in `env` to a node-local directory has the engines of a node decode each clip once, file its frames there and map them from the file. Kimi-K2.6's clips do not use it.

## Generation

| Field | Default | Meaning |
|---|---|---|
| `gen_tokens` | `1024` | tokens each stage generates |
| `max_model_len` | `147456` | the context length; a configuration for tasks with clips states its own |
| `decode_paths` | `[efficiency]` | how a stage stops; see below |
| `max_tokens_quality` | `gen_tokens` | the cap on the `quality` path |
| `temperature`, `top_p`, `top_k` | `0.0`, `1.0`, `0` | sampling; the files in `turboreflect/configs/` state each model's recommended values |
| `seed` | `0` | the sampling seed |
| `seed_per_program` | `false` | sample program `i` under `seed + i` |
| `enable_thinking` | `false` | the model's thinking switch |
| `thinking_kwarg` | `enable_thinking` | the chat-template variable that carries it |

| `decode_paths` entry | A stage stops |
|---|---|
| `efficiency` | after exactly `gen_tokens`, past the end of sequence if need be |
| `quality` | at the end of sequence, capped at `max_tokens_quality`; the output is streamed |
| `terminating` | at the end of sequence, capped at `gen_tokens` |

`efficiency` gives every stage the same work, for measuring speed; use `quality` when the outputs will be scored.

## Prompts

| Field | Default | Meaning |
|---|---|---|
| `feedback` | `goal` | the feedback stage's instruction: `goal` or `mustfix` |
| `refinement` | `goal` | the refinement stage's instruction: `goal` or `full`, and `goal_merged` or `merged` in a `two_stage` program |
| `layout` | `requote` | how a later stage carries the outputs before it; see below |

Under `continue`, a later stage's prompt is the previous prompt and output followed by the next instruction turn. Under `requote`, it keeps the generation prompt up to the end of its task and quotes each output into one user message, followed by the stage's instruction. A task folder with its own `prompts.py` may build only `continue`, which it then runs without being told; stating `layout: requote` for it is refused.

## Engine

| Field | Default | Meaning |
|---|---|---|
| `attention_backend` | vLLM's | vLLM's attention backend |
| `cudagraph_mode` | vLLM's | vLLM's CUDA graph mode |
| `gdn_prefill_backend`, `moe_backend`, `flash_attn_version` | vLLM's | kernel choices |
| `gpu_memory_utilization` | `0.9` | |
| `max_num_batched_tokens` | `16384` | |
| `max_num_seqs` | derived | the sequence limit, derived from the measured KV pool when not stated |
| `tensor_parallel_size` | `1` | |
| `trust_remote_code` | `false` | |
| `language_model_only` | `false` | leave the vision tower of a multimodal checkpoint off the GPU; tasks with clips turn it off |
| `enforce_eager` | vLLM's | |
| `rope_scaling_factor`, `rope_original_max_positions`, `rope_theta` | none; the checkpoint's | a YaRN override, set by `rope_scaling_factor` |
| `limit_mm_per_prompt` | for tasks with clips, the most clips any task has and no images; else vLLM's | the multimodal items a prompt may hold |
| `allowed_local_media_path` | for tasks with clips, the folder that holds them; else vLLM's | where the engine may read clips from |
| `mm_processor_cache_gb` | vLLM's | the size of vLLM's multimodal processor cache; a configuration for tasks with clips states it |
| `mm_processor_kwargs` | vLLM's | the video processor's settings, sent with every stage |
| `env` | none | environment variables, as a mapping or `NAME=VALUE` entries |

## Drafting

| Field | Default | Meaning |
|---|---|---|
| `suffix_num_speculative_tokens` | `16` | the most draft tokens per step |
| `suffix_max_tree_depth` | `32` | the depth of the suffix tree |
| `suffix_max_cached_requests` | `0` | `0` keeps one tree per request; a positive value adds a global tree over that many finished requests |
| `adaptive_draft.explore_every` | `16` | how often a step extends its most cut request to the whole proposal, so the estimates stay measured |
| `adaptive_draft.warmup_steps` | `64` | how many steps with drafts verify them as proposed at the start |

## The run

| Field | Default | Meaning |
|---|---|---|
| `out_dir` | `results` | where run directories go; `--out-dir` sets it |
| `calibration_dir` | `calibration/<model>-<kv dtype>-<digest>` | where calibration results are |
| `concurrency` | every program at once | programs in flight, at least 1; a value above the program count runs every program at once |
| `gpu_hour_usd` | absent | the price of one GPU-hour, for cost in dollars |
| `isolate_programs` | `true` | give each program its own cache salt, so programs share no KV |

## Complete configurations

A file that lists `samples` instead of `tasks` is read as it stands, with no component switches and no defaults. It needs `model`, `samples`, `gen_tokens` and `max_model_len`; `async_scheduling: false`, which the engine trace requires; and `bench_root`, the folder of its [stage assembly](task-format.md#stage-assembly), which relative `samples` are read from. The other fields of `RunConfig` in `turboreflect/config.py` take their own defaults, so a file that leaves out `concurrency` runs one program at a time.
