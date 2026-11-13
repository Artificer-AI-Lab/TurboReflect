# Adding a model

TurboReflect runs on what vLLM runs, so a new model needs a configuration and no code. Start from the file in `turboreflect/configs/` closest to your model; the [configuration reference](configuration.md) has every field.

1. Set `model` and `tasks`, and state what your model needs. A model's file states:

    - the sampling fields: `temperature`, `top_p`, `top_k` and `seed_per_program`, at the values the model recommends;
    - the draft limits: `suffix_num_speculative_tokens` and `suffix_max_tree_depth`;
    - the engine: `attention_backend` and the other kernel choices, `cudagraph_mode`, `tensor_parallel_size`, `trust_remote_code`, `language_model_only` and `env`;
    - `kv_cache_dtype`: how its engines spell FP8, `fp8` or `fp8_e4m3`. The shipped files spell it `fp8_e4m3` for the latent-attention models and `fp8` for the others;
    - `thinking_kwarg`: the chat-template variable of the thinking switch, `enable_thinking` unless stated. Kimi-K2.6's is `thinking`;
    - `rope_scaling_factor`: the YaRN factor, for a model run past its native window. `rope_original_max_positions` and `rope_theta` default to the checkpoint's.

    ```yaml
    model: my-org/my-model
    tasks: my_tasks
    attention_backend: FLASH_ATTN
    kv_cache_dtype: fp8
    ```

2. Calibrate and run. Calibration measures this model on this GPU, so adaptive draft length and locality-aware P/D need nothing else from you:

    ```bash
    python -m turboreflect calibrate --config my_model.yaml
    python -m turboreflect run --config my_model.yaml --out-dir runs/my_model
    ```

## Video

Video tasks run on models with a video encoder. A run puts one line per clip in the prompt: the video placeholder that the vLLM model class for the checkpoint's `architectures` gives (`get_placeholder_str`), which vLLM expands into the clip's tokens. Kimi-K2.6 takes a clip as `vision_chunk` items, which vLLM does not cut for it, so on one engine `turboreflect/program/media.py` cuts each clip as the checkpoint does, one placeholder per chunk. A model whose vLLM class gives no video placeholder cannot run tasks with clips.

A configuration for tasks with clips states its own `max_model_len` and `mm_processor_cache_gb`. `turboreflect/configs/qwen3.8-27b-video.yaml` is one, and the comments of `turboreflect/configs/kimi-k2.6.yaml` list what Kimi-K2.6 needs for clips.
