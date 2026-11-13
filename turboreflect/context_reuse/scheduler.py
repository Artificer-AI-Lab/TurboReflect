# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors
"""Context reuse for reflection programs: KV retention and admission.

A program's next stage arrives just after the previous one finishes, then waits while
running requests' prefill chunks take free blocks, its own cached prefix among them.
These schedulers hold a finished stage's KV for its successor and keep a waiting
successor's prefix off the free queue, using observed state only. Programs carry their
own cache salt, so a fresh (never preempted) request with a hit is a successor.
"""

import time
from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


class ReserveOnArrivalScheduler(Scheduler):
    """Hold the GPU prefix hit of each fresh waiting request.

    A reservation yields when a running request at least as important cannot grow, or
    one per step when nothing runs and the queue head cannot be admitted. A yielded
    reservation is not taken again for that request.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # request id -> (request, reserved blocks in eviction order)
        self.reserved: dict[str, tuple[Request, list[KVCacheBlock]]] = {}
        self.yielded: set[str] = set()

    def _release(self, request_id: str) -> None:
        _, blocks = self.reserved.pop(request_id)
        self.kv_cache_manager.block_pool.free_blocks(blocks)

    def _release_least_important(self, bound: tuple[int, float] | None) -> bool:
        """Release the reservation with the largest (priority, arrival), if any.

        Only one ordered at or after `bound` qualifies; None allows any.
        """
        if not self.reserved:
            return False
        request_id, (request, _) = max(
            self.reserved.items(),
            key=lambda item: (item[1][0].priority, item[1][0].arrival_time),
        )
        if bound is not None and (request.priority, request.arrival_time) < bound:
            return False
        self._release(request_id)
        self.yielded.add(request_id)
        return True

    def _reserve_arrived(self) -> None:
        waiting = {request.request_id: request for request in self.waiting}
        for request_id in [r for r in self.reserved if r not in waiting]:
            self._release(request_id)
        self.yielded &= waiting.keys()
        manager = self.kv_cache_manager
        if not manager.enable_caching:
            return
        for request_id, request in waiting.items():
            if (
                request_id in self.reserved
                or request_id in self.yielded
                or request.num_preemptions > 0
                or request.skip_reading_prefix_cache
            ):
                continue
            hit_blocks, hit_tokens, _ = manager.coordinator.find_longest_cache_hit(
                request.block_hashes, max(0, request.num_tokens - 1)
            )
            if hit_tokens == 0:
                continue
            blocks = [b for group in hit_blocks for b in group if not b.is_null]
            manager.block_pool.touch(blocks)
            # Released tail first, as a finished request's blocks are.
            self.reserved[request_id] = (request, blocks[::-1])

    def schedule(self, throttle_prefills: bool = False):
        self._reserve_arrived()
        return super().schedule(throttle_prefills=throttle_prefills)

    def _select_preemption_victim(self) -> Request | None:
        victim = super()._select_preemption_victim()
        if victim is not None and self._release_least_important(
            (victim.priority, victim.arrival_time)
        ):
            return None
        return victim

    def _on_waiting_allocation_failure(self) -> None:
        if not self.running:
            self._release_least_important(None)

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        for request_id in list(self.reserved):
            self._release(request_id)
        return super().reset_prefix_cache(reset_running_requests, reset_connector)


@dataclass
class _Hold:
    request: Request
    finished_at: float


class ContextReuseScheduler(ReserveOnArrivalScheduler):
    """Hold at finish, reserve on arrival, admit with headroom, yield by value.

    A finished request keeps its blocks, keyed by its prompt's first block hash, which
    the program's next request shares. The hold ends when that request arrives or its
    value `p_successor * S(age)` falls below `min_value`: the observed share of holds
    whose successor came, times the share of finish-to-arrival gaps longer than the
    age, each with one prior (a successor at `prior_gap_s`). A late successor still
    counts, since an expiry censors a gap.

    A waiting request is admitted only if the free pool, after its whole remaining
    prompt, still covers the running requests' remaining prompts and one decode step
    each; held and reserved blocks are not free. Under pressure holds yield first,
    lowest value first, then reservations; while requests run, a hold older than
    `prior_gap_s` also yields, one per step, when the queue head cannot be admitted.
    """

    prior_gap_s = 5.0
    min_value = 0.05

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.holds: dict[BlockHash, _Hold] = {}
        # Finish times of expired holds whose successor has not arrived.
        self.expired: dict[BlockHash, float] = {}
        self.gaps_s: list[float] = []
        self.num_hold_hits = 0
        self.num_hold_expiries = 0
        self.num_late_successors = 0
        self._arrived_at: dict[str, float] = {}
        self._clock = time.monotonic

    def _p_successor(self) -> float:
        resolved = self.num_hold_hits + self.num_hold_expiries
        came = self.num_hold_hits + self.num_late_successors
        return (came + 1) / (resolved + 1)

    def _survival(self, age_s: float) -> float:
        longer = sum(gap > age_s for gap in self.gaps_s) + (age_s < self.prior_gap_s)
        return longer / (len(self.gaps_s) + 1)

    def _value(self, hold: _Hold, now: float) -> float:
        return self._p_successor() * self._survival(now - hold.finished_at)

    def _drop_hold(self, key: BlockHash) -> None:
        self._free_request_blocks(self.holds.pop(key).request)

    def add_request(self, request: Request) -> None:
        self._arrived_at[request.request_id] = self._clock()
        super().add_request(request)

    def _free_blocks(self, request: Request):
        self._arrived_at.pop(request.request_id, None)
        if not (request.block_hashes and self.kv_cache_manager.enable_caching):
            super()._free_blocks(request)
            return
        key = request.block_hashes[0]
        if key in self.holds:
            self._drop_hold(key)
        self.holds[key] = _Hold(request, self._clock())
        del self.requests[request.request_id]

    def _settle_holds(self) -> None:
        now = self._clock()
        for request in self.waiting:
            key = request.block_hashes[0] if request.block_hashes else None
            if key in self.holds:
                finished_at = self.holds[key].finished_at
                # Reserve-on-arrival has already taken the prefix.
                self._drop_hold(key)
                self.num_hold_hits += 1
            elif key in self.expired:
                finished_at = self.expired.pop(key)
                self.num_late_successors += 1
            else:
                continue
            arrived = self._arrived_at.get(request.request_id, now)
            self.gaps_s.append(max(0.0, arrived - finished_at))
        for key, hold in list(self.holds.items()):
            if self._value(hold, now) < self.min_value:
                self._drop_hold(key)
                self.expired[key] = hold.finished_at
                self.num_hold_expiries += 1

    def schedule(self, throttle_prefills: bool = False):
        self._reserve_arrived()
        self._settle_holds()
        return Scheduler.schedule(self, throttle_prefills=throttle_prefills)

    def _on_waiting_allocation_failure(self) -> None:
        if self.running and self._yield_hold(self.prior_gap_s):
            return
        super()._on_waiting_allocation_failure()

    def _yield_hold(self, min_age_s: float = 0.0) -> bool:
        """Drop the lowest-value hold at least `min_age_s` old, if any."""
        now = self._clock()
        keys = [k for k, h in self.holds.items() if now - h.finished_at >= min_age_s]
        if not keys:
            return False
        key = min(keys, key=lambda k: self._value(self.holds[k], now))
        self._drop_hold(key)
        return True

    def _release_least_important(self, bound: tuple[int, float] | None) -> bool:
        return self._yield_hold() or super()._release_least_important(bound)

    def _admission_reserved_blocks(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int,
        load_kv_async: bool,
    ) -> int:
        reserved = super()._admission_reserved_blocks(
            request, num_new_tokens, num_new_local_computed_tokens, load_kv_async
        )
        if not self.running:
            return reserved
        block_size = self.block_size
        running = sum(self._request_remaining_blocks(r) for r in self.running)
        # A decode step writes its token and its drafts' slots at once.
        decode = len(self.running) * cdiv(
            1 + self.num_spec_tokens + self.num_lookahead_tokens, block_size
        )
        own = (
            self._request_remaining_blocks(request)
            - num_new_local_computed_tokens // block_size
            - cdiv(num_new_tokens, block_size)
        )
        return reserved + running + decode + max(0, own)

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        for key in list(self.holds):
            self._drop_hold(key)
        self.expired.clear()
        return super().reset_prefix_cache(reset_running_requests, reset_connector)
