# The run directory

`python -m turboreflect run --out-dir runs/my_run` writes one directory per run, named `<timestamp>-<fingerprint>`, where the fingerprint covers the configuration and what the task files hold.

```text
runs/my_run/20270101-120000-1a2b3c4d/
  config.json                 the resolved configuration and the KV pool
  provenance.json             the code version, the engine and the devices
  summary.json                the run's totals and reported numbers
  <program_id>.json           one per program
  engine_steps_core.jsonl     one row per engine step
  engine_requests_core.jsonl  one row per request
```

A P/D run has per engine an `engine_trace_<name>/` folder with its own two `.jsonl` files, in place of the two above, and an `engine_log_<name>.txt` with its console output. Each engine's facts are under `pd` in `config.json`, in place of `provenance.json`.

`python -m turboreflect report runs/my_run` prints one row per run and rewrites the reported numbers. It takes run directories, or folders of them, and needs no GPU.

## `summary.json`

`results` holds the run's reported numbers under the key `n<concurrency>`, as in `results.n3`:

| Key | Meaning |
|---|---|
| `makespan_s` | from the batch's release to the last program's end |
| `cost` | `gpus`, `gpu_seconds`, `logical_tokens`, and `usd_per_1m_tokens` when `gpu_hour_usd` is set |
| `decode_latency_ms` | mean time per generated token, across every token |
| `program_e2e_mean_s` | mean program end-to-end latency |
| `programs_per_s`, `output_tokens_per_s`, `total_tokens_per_s` | throughput |
| `prefix_hit_ratio` | cached prompt tokens over prompt tokens |
| `matched_append` | matched prefix and append lengths, per later stage |
| `mean_accepted_length`, `acceptance_rate` | draft acceptance |
| `kv_usage` | peak and mean KV cache usage, per engine |
| `handoff_s` | time from a stage's last token to its successor's first |

A number that a run's files cannot give is null. If computing these numbers fails, the run still finishes and `metrics_error.txt` holds the error.

## A program's file

| Key | Meaning |
|---|---|
| `program_id`, `sample_id`, `decode_path` | which task this program ran, and how |
| `outputs` | the text of every stage: `draft`, `critique` and `revision`; a multi-round program numbers its rounds, as in `critique_2`, and `revision` is what it delivered |
| `artifact` | the text the program delivered |
| `output_token_ids` | the same outputs as token ids |
| `stages` | one row per stage: token counts, cache hits, matched and append lengths, timings and, under P/D, where it ran |
| `e2e_seconds`, `t_start_s`, `t_end_s` | the program's times on the run's clock |

Prompts are not stored: a stage's row holds their sizes, and they can be rebuilt from the task.

## Calibration

`calibrate` writes into `calibration/<model>-<kv dtype>-<digest>/` under the working directory, or `calibration_dir`. The digest covers the configuration's settings and the GPU's name (from `nvidia-smi`), so results measured on another engine or GPU are never read in place of a run's own. It leaves out the tasks, since the step model times synthetic batches and the P/D profile is a function of chunk and cached length, unless `max_num_seqs` is left unstated: the limit derived from the tasks sets the batches the step model times.

| File | Meaning |
|---|---|
| `step_model.json` | the step times adaptive draft length prices drafts with |
| `pd_profile.json` | the costs locality-aware P/D places stages with |

The other files there are what these two were fitted from.
