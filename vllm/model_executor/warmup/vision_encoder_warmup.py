# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the Triton kernels of a Qwen-style vision tower."""

import itertools
import math
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.utils import StageMissingLayer

logger = init_logger(__name__)

# Triton specializes integer arguments on divisibility by this.
_DIVISIBILITY = 16


def vision_warmup_grids(merge_size: int) -> list[list[int]]:
    """One-item `[t, h, w]` grids reaching every integer specialization.

    The pos-embed kernel takes the grid's h and w, and the rotary kernel the
    item's patch count, each specialized on divisibility by 16 (and on 1).
    Real grids are multiples of the merge size, so these sides reach every
    class a real grid can.
    """
    aligned = math.lcm(_DIVISIBILITY, merge_size)
    sides = sorted({merge_size, 3 * merge_size, aligned})
    return [[1, h, w] for h, w in itertools.product(sides, repeat=2)]


@torch.inference_mode()
def vision_encoder_warmup(model: torch.nn.Module) -> None:
    """Encode each warmup grid once, so a clip's first encode compiles nothing.

    Applies to a vision tower taking flattened patches and `grid_thw`, as the
    Qwen-VL family's does; anything else is left alone.
    """
    visual: Any = getattr(model, "visual", None)
    if isinstance(visual, StageMissingLayer):
        return
    patch_embed: Any = getattr(visual, "patch_embed", None)
    merge_size = getattr(visual, "spatial_merge_size", None)
    if not isinstance(merge_size, int) or not hasattr(
        patch_embed, "temporal_patch_size"
    ):
        return
    grids = vision_warmup_grids(merge_size)
    logger.info("Warming up the vision encoder on %d grids.", len(grids))
    try:
        weight = patch_embed.proj.weight
        patch_dim = (
            patch_embed.proj.in_channels
            * patch_embed.temporal_patch_size
            * patch_embed.patch_size**2
        )
        for grid in grids:
            pixels = torch.zeros(
                math.prod(grid),
                patch_dim,
                device=weight.device,
                dtype=getattr(visual, "dtype", weight.dtype),
            )
            visual(pixels, grid_thw=[grid])
    except Exception:
        logger.warning("Vision encoder warmup failed.", exc_info=True)
