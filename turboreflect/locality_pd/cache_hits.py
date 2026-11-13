# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""How many prompt tokens each engine's prefix cache ought to serve.

Derived only from the token ids the driver holds; the dispatcher reads it to price
each engine's hit before placing a stage. Three boundaries come from vLLM's cache:

  * block hashes cover whole blocks only (`kv_cache_utils.py` `hash_block_tokens`);
  * a request's trailing partial block is never registered (`cache_full_blocks`);
  * `get_computed_blocks` caps a hit at `num_tokens - 1`, leaving a token to sample.

`Ledger` keeps a table per engine, filed only with the stages that engine ran.
Nothing here imports vllm or torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from turboreflect.locality_pd.config import ROLE_D

if TYPE_CHECKING:
    from turboreflect.config import RunConfig

# Token ids `lcp` compares per slice before it compares them one by one.
_LCP_STEP = 1024


class ReuseError(Exception):
    """A ledger was asked for a side it keeps no table for."""


def lcp(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token id sequences.

    Slices of `_LCP_STEP` ids compare in C, so a 128k-token history takes under a
    millisecond.
    """
    if not (isinstance(a, list) and isinstance(b, list)):
        a, b = list(a), list(b)
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i : i + _LCP_STEP] == b[i : i + _LCP_STEP]:
        i += _LCP_STEP
    i = min(i, n)
    while i < n and a[i] == b[i]:
        i += 1
    return i


def predict(
    prompt_ids: Sequence[int], cached_seqs: Sequence[Sequence[int]], unit: int
) -> int:
    """Predict this prompt's cache hit against everything already computed.

    Args:
        cached_seqs: Every prompt plus output the engine already ran in this
            program.
        unit: Tokens per cache key.
    """
    best = 0
    for ids in cached_seqs:
        best = max(best, min(lcp(prompt_ids, ids), len(ids) // unit * unit))

    predicted = best // unit * unit
    cap = max(0, (len(prompt_ids) - 1) // unit * unit)
    return min(predicted, cap)


@dataclass(frozen=True)
class Side:
    """One engine's filing rule: whether it caches, and what it keeps.

    Attributes:
        keeps_output: Whether the generation is filed along with the prompt.
        spec_tokens: Draft tokens per verify step, 0 without speculative decoding.
            A side that drafts files the whole generation.
    """

    role: str
    caching: bool
    keeps_output: bool
    spec_tokens: int = 0


def _split_side(name: str, role: str, caching: bool, spec_tokens: int) -> Side:
    """One engine of a split run, holding only what its role can hold.

    P runs with `max_tokens=1` and files the prompt only. D also files the
    generation it computed, all of it when it drafts. Tokens D received over NIXL
    count in `num_cached_tokens`, not in this prediction.
    """
    return Side(
        name,
        caching,
        keeps_output=role == ROLE_D,
        spec_tokens=spec_tokens if role == ROLE_D else 0,
    )


@dataclass
class Ledger:
    """What each engine's prefix cache should be holding, kept per side.

    One per program, since a program's salt opens a prefix chain no other program
    can reach. Tables hold the token ids the driver submitted, except that a
    program with clips files the engine's expanded ids
    (`turboreflect.program.media`).
    """

    sides: tuple[Side, ...]
    # Out of the repr, since a table holds whole prompts; still compared.
    _rows: dict[str, list[list[int]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        roles = [side.role for side in self.sides]
        self._rows = {role: [] for role in roles}

    @classmethod
    def for_run(cls, cfg: RunConfig) -> Ledger:
        """The ledger for a split run: one table per engine name."""
        spec_tokens = cfg.suffix_num_speculative_tokens if cfg.suffix_decoding else 0
        return cls(
            tuple(
                _split_side(
                    engine.name,
                    engine.role,
                    engine.enable_prefix_caching,
                    spec_tokens,
                )
                for engine in cfg.pd.settings().engines
            )
        )

    def table(self, role: str) -> list[list[int]]:
        """One side's table.

        Raises:
            ReuseError: If this ledger keeps no table for `role`.
        """
        try:
            return self._rows[role]
        except KeyError:
            raise ReuseError(
                f"this ledger keeps no table for role {role!r}; it answers for "
                f"{list(self._rows)}"
            ) from None

    def record(
        self,
        prompt_ids: Sequence[int],
        output_ids: Sequence[int],
        roles: Sequence[str] | None = None,
        decoded: str | None = None,
    ) -> None:
        """File one finished stage against every side whose cache took it.

        A side keeping the output files all but the last token, whose KV never
        exists (`KVCacheManager.allocate_slots` caches at most
        `request.num_tokens`); a side with `spec_tokens` files all of it.

        Args:
            roles: The sides that ran the stage, or `None` for every side.
            decoded: The side that decoded the stage; it files the generation
                whatever its role.
        """
        for side in self.sides:
            if not side.caching or (roles is not None and side.role not in roles):
                continue
            ids = list(prompt_ids)
            if side.keeps_output or side.role == decoded:
                output = list(output_ids)
                ids += output if side.spec_tokens else output[:-1]
            self._rows[side.role].append(ids)

    def predict(self, role: str, prompt_ids: Sequence[int], unit: int) -> int:
        """One side's frozen prediction for the prompt about to be submitted.

        Raises:
            ReuseError: If this ledger keeps no table for `role`.
        """
        return predict(prompt_ids, self.table(role), unit)

    def predict_each(
        self, roles: Sequence[str], prompt_ids: Sequence[int], unit: int
    ) -> dict[str, int]:
        """`predict` per role; may run off the event loop while nothing records."""
        return {role: self.predict(role, prompt_ids, unit) for role in roles}
