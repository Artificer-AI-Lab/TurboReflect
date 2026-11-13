# Changes to vLLM

TurboReflect is built on vLLM v0.26.0 (upstream commit `568afb3a13806beb53bb2e6bd518269357b237c0`). The commit "vLLM v0.26.0" in this repository holds that release unchanged, so `git diff <that commit> -- vllm` shows every change made to it.

Fixes backported from later upstream pull requests:

| Pull request | Change |
|---|---|
| vLLM [#48425](https://github.com/vllm-project/vllm/pull/48425), [#53523](https://github.com/vllm-project/vllm/pull/53523) | prefix hits of hybrid models and Mamba prefill truncation under NIXL |
| vLLM [#48860](https://github.com/vllm-project/vllm/pull/48860) | prefix-cache statistics counted once with a KV connector |
| vLLM [#49291](https://github.com/vllm-project/vllm/pull/49291) | conv-state migration of hybrid models under speculative decoding |
| vLLM [#50153](https://github.com/vllm-project/vllm/pull/50153) | NIXL Mamba state pairing for multi-slot block tables |
| vLLM [#50388](https://github.com/vllm-project/vllm/pull/50388), part of [#53781](https://github.com/vllm-project/vllm/pull/53781) | recovery from KV load failures with a hybrid KV cache |
| vLLM [#51065](https://github.com/vllm-project/vllm/pull/51065) | TritonMLA causal multi-token decode |
| vLLM [#51603](https://github.com/vllm-project/vllm/pull/51603) | Mamba alignment before encoder caps |
| vLLM [#51812](https://github.com/vllm-project/vllm/pull/51812) | Qwen GDN gates with speculative tokens |
| vLLM [#52988](https://github.com/vllm-project/vllm/pull/52988) | variable-length decode for adaptive verification on MLA |
| vLLM [#56531](https://github.com/vllm-project/vllm/pull/56531) | speculation-capable rows kept on the speculative path |
| vLLM [#58442](https://github.com/vllm-project/vllm/pull/58442) | the scheduler's skipped waiting queue under KV pressure |
| FlashInfer [#4165](https://github.com/flashinfer-ai/flashinfer/pull/4165) | cuDNN kept out of `bmm_fp8` auto selection on compute capability 12.x |

The other changes are what the components need from the engine: scheduler and KV cache hooks for context reuse, multi-token verification kernels for latent attention and SuffixDecoding trees carried from one stage to the next for adaptive draft length, and the KV pull over NIXL for locality-aware P/D, plus warmups that compile kernels and capture CUDA graphs before the first step.

Outside `turboreflect/`, `docs/turboreflect/` and `vllm/`, the changes are the README, `THIRD_PARTY_NOTICES.md`, `.gitignore`, the install files (`pyproject.toml`, `requirements/` and the FlashInfer version in `docker/`) and the removal of vLLM's repository automation (`.github/` and similar).

## Where each component runs

| Component | Without P/D | With P/D | In `vllm/` |
|---|---|---|---|
| Context reuse | `context_reuse/`: the scheduler and successor priority | `locality_pd/`: a later stage placed on the engine holding its matched prefix (`dispatch.py`), with its KV booked there (`admission.py`); successor priority from `context_reuse/admission.py` | scheduler hooks, hybrid-model state, NIXL holds |
| Adaptive draft length | `adaptive_draft/`, mixed into the engine's scheduler (`metrics/trace.py`) | `adaptive_draft/`, mixed into each engine's scheduler (`locality_pd/append_scheduler.py`) | verification kernels, suffix-tree reuse |
| Locality-aware P/D | | `locality_pd/` | KV pull over NIXL |
| KV cache quantization | `kv_cache_dtype` (`config.py`) | the same | vLLM's FP8 KV cache |
