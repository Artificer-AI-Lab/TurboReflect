# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""KV bytes per token, in the layout the kernels write; importable without vLLM.

- Dense, per layer: `2 * Hkv * D * itemsize` (`FullAttentionSpec.real_page_size_bytes`).
- MLA, per layer: one latent row of `(L + R) * itemsize`, 1152 in bf16 and 576 in fp8,
  as `concat_and_cache_mla` writes it for every dtype but `fp8_ds_mla` (656).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from turboreflect.config import FP8_CACHE_DTYPES


@dataclass(frozen=True)
class Geometry:
    """A model's KV cache, in the terms its layout is written in.

    Under MLA `head_dim` is the latent width (`kv_lora_rank`) and `rope_dim` the
    `k_pe` stored beside it; a dense model has `rope_dim` 0.
    """

    layers: int
    num_kv_heads: int
    head_dim: int
    rope_dim: int = 0

    @property
    def mla(self) -> bool:
        return self.rope_dim > 0


def geometry_from_model_config(model_config: Any, parallel_config: Any) -> Geometry:
    """The running engine's KV geometry, in this GPU's shares of layers and heads."""
    # A hybrid model's other layers hold fixed-size state, not priced per token.
    layers = model_config.get_num_layers_by_block_type(parallel_config, "attention")
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    if not model_config.use_mla:
        return Geometry(layers=layers, num_kv_heads=num_kv_heads, head_dim=head_size)
    text_config = model_config.hf_text_config
    rope_dim = int(getattr(text_config, "qk_rope_head_dim", 0))
    head_dim = int(getattr(text_config, "kv_lora_rank", 0))
    return Geometry(
        layers=layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope_dim=rope_dim,
    )


def _native_itemsize_for(cache_dtype: str, native_itemsize: int) -> int:
    """Bytes per stored element under `cache_dtype`; `KeyError` for other layouts."""
    if cache_dtype == "auto":
        return native_itemsize
    if cache_dtype in FP8_CACHE_DTYPES:
        return 1
    raise KeyError(f"no KV layout for cache dtype {cache_dtype!r}")


def layer_bytes(
    geo: Geometry,
    cache_dtype: str,
    *,
    native_itemsize: int = 2,
) -> int:
    """Bytes one layer holds for one token under `cache_dtype` (`auto` or fp8).

    `native_itemsize` is the model dtype's element size, which `auto` stores.
    """
    itemsize = _native_itemsize_for(cache_dtype, native_itemsize)
    if geo.mla:
        return geo.num_kv_heads * (geo.head_dim + geo.rope_dim) * itemsize
    return 2 * geo.num_kv_heads * geo.head_dim * itemsize


def bytes_per_token(
    geo: Geometry,
    cache_dtype: str,
    *,
    native_itemsize: int = 2,
) -> int:
    """The whole model's KV footprint for one token, across every layer."""
    return geo.layers * layer_bytes(geo, cache_dtype, native_itemsize=native_itemsize)
