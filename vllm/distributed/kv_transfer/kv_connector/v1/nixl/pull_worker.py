# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    RELEASED_HOLD_GRACE_S,
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.nixl_utils import nixlXferTelemetry
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# NIXL descriptors one step posts reads for; 0 posts every read at once.
# Posting runs on the step thread, so a long prompt's read posted at once stalls
# every decode on the engine. A read past the budget is posted in slices over
# successive steps, however short its holder's lease.
_MAX_READ_DESCS_PER_STEP = 16384


# Prefixes a notification that read nothing: the reader's own prefix cache
# held all it was offered. Its holder may already have released the blocks
# (the hold is let go at the reader's first token or its end, which a
# notification sent over NIXL can trail), so a late one is only logged.
NOTIF_HIT_PREFIX = "HIT:"


@dataclass
class _ReadXfer:
    """One remote rank's part of a read, posted in slices of its descriptors."""

    local_xfer_side_handle: int
    local_desc_ids: np.ndarray
    remote_xfer_side_handle: int
    remote_desc_ids: np.ndarray
    dst_engine_id: str
    remote_rank: int
    notif_id: bytes
    posted: int = 0


@dataclass
class _SlicedRead:
    """A read posted over several steps.

    The holder releases the blocks on the notification of each remote rank's
    part, so only that part's last slice carries it, as a whole read's one
    transfer does, posted once no other slice of the read is in flight. It
    is recorded as one transfer, over its wall span from `posted_s`, its
    first slice's post, with its slices' post time, bytes and descriptors.
    """

    xfers: deque[_ReadXfer] = field(default_factory=deque)
    failed: bool = False
    posted_s: float | None = None
    post_s: float = 0.0
    total_bytes: int = 0
    descs: int = 0


class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-specific (READ) worker logic."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        # Reads waiting for a step with descriptor budget left, in arrival order.
        self._pending_reads: deque[tuple[str, ReqMeta]] = deque()
        # Reads posted in slices, in arrival order, from their first slice
        # until every slice has completed.
        self._sliced_reads: dict[str, _SlicedRead] = {}
        # Descriptors this step may still post; None outside a budgeted
        # _post_pending_reads.
        self._step_descs_left: int | None = None
        assert vllm_config.kv_transfer_config is not None
        self._max_read_descs_per_step: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "max_read_descs_per_step", _MAX_READ_DESCS_PER_STEP
            )
        )

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        # Before posting reads: posting a backlog can hold this step longer than
        # a heartbeat extends the lease.
        self._send_heartbeats(metadata)

        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            # Remote block IDs are kept logical here; expanded in
            # _read_blocks_for_req using the remote engine's phys ratio.
            remote_engine_id = meta.remote.engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            # always store metadata for failure recovery
            self._recving_metadata[req_id] = meta
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(req_id, remote_engine_id, meta)
                        continue

            # Handshake already completed: queue the read.
            self._pending_reads.append((req_id, meta))

        # Queue reads for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._pending_reads.append(self._ready_requests.get_nowait())
        self._post_pending_reads()

        # Keep around the requests that have been part of a batch. This is
        # needed because async scheduling pushes the misalignment between the
        # moment in which requests expiration is set (P side) and the moment in
        # which blocks are read from D. As P can now more easily lag behind D
        # while processing the next batch, we make sure to only set an
        # expiration for requests that have not been read from D yet.
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        # Remove all requests that are not to be processed (eg aborted).
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            # We should never get an abort after setting an expiry timer
            assert req_id not in self._reqs_to_send

        # Add to requests that are waiting to be read and track expiration.
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time

    def _post_pending_reads(self) -> None:
        """Post queued reads, in order, up to this step's descriptor budget.

        NIXL posts a read's descriptors on the step thread, so a backlog, or
        one long prompt's read, posted at once stalls every decode on this
        engine. Sliced reads post their next slices first; a new read that
        does not fit what is left waits for the next step, and one larger than
        the whole budget is posted in slices from the step it starts.

        A full prefix hit reads nothing and goes at once: its notification is
        all the holder waits for, and held behind reads it lands after the
        request has finished, when the holder may have released its blocks.
        """
        reads: deque[tuple[str, ReqMeta]] = deque()
        for req_id, meta in self._pending_reads:
            if any(len(group) for group in meta.local_physical_block_ids):
                reads.append((req_id, meta))
            else:
                self._read_blocks_for_req(req_id, meta)
        self._pending_reads = reads
        budget = self._max_read_descs_per_step
        if not budget:
            while self._pending_reads:
                self._read_blocks_for_req(*self._pending_reads.popleft())
            return
        self._step_descs_left = budget
        try:
            for req_id, read in list(self._sliced_reads.items()):
                if read.xfers and self._step_descs_left:
                    self._continue_sliced_read(req_id, read)
            while self._pending_reads and self._step_descs_left:
                req_id, meta = self._pending_reads[0]
                descs = self.num_regions * sum(
                    len(group) for group in meta.local_physical_block_ids
                )
                if self._step_descs_left < min(descs, budget):
                    return
                self._pending_reads.popleft()
                self._read_blocks_for_req(req_id, meta)
        finally:
            self._step_descs_left = None

    def _continue_sliced_read(self, req_id: str, read: _SlicedRead) -> None:
        meta = self._recving_metadata[req_id]
        assert meta.remote is not None
        self._engine_last_active[meta.remote.engine_id] = time.perf_counter()
        if self._is_lease_expired(meta):
            logger.warning(
                "Declining the rest of expired remote read for %s from engine %s.",
                req_id,
                meta.remote.engine_id,
            )
            self.xfer_stats.record_kv_expired_req()
            self._handle_failed_transfer(req_id, None)
            return
        self._post_read_slices(req_id, read)

    def _slices_in_flight(self, req_id: str) -> bool:
        """Whether a posted slice of the read is not yet done.

        One whose state cannot be read counts as in flight, and
        `_pop_done_transfers` fails it.
        """
        for handle in self._recving_transfers.get(req_id, ()):
            try:
                if self.nixl_wrapper.check_xfer_state(handle) != "DONE":
                    return True
            except Exception:
                return True
        return False

    def _post_read_slices(self, req_id: str, read: _SlicedRead) -> None:
        """Post the read's next slices, up to the descriptors this step has left.

        A part's last descriptor, which carries its notification, goes only once
        no slice is in flight.
        """
        while read.xfers and self._step_descs_left != 0:
            xfer = read.xfers[0]
            start = xfer.posted
            end = len(xfer.local_desc_ids)
            if self._step_descs_left is not None:
                end = min(end, start + self._step_descs_left)
            last = end == len(xfer.local_desc_ids)
            if last and self._slices_in_flight(req_id):
                end, last = end - 1, False
                if end == start:
                    return
            handle = None
            try:
                handle = self.nixl_wrapper.make_prepped_xfer(
                    "READ",
                    xfer.local_xfer_side_handle,
                    xfer.local_desc_ids[start:end],
                    xfer.remote_xfer_side_handle,
                    xfer.remote_desc_ids[start:end],
                    notif_msg=xfer.notif_id if last else b"",
                )
                if read.posted_s is None:
                    read.posted_s = time.perf_counter()
                self.nixl_wrapper.transfer(handle)
                self._recving_transfers[req_id].append(handle)
            except Exception as e:
                self._log_failure(
                    failure_type="transfer_setup_failed",
                    req_id=req_id,
                    msg="Marking blocks as invalid",
                    error=e,
                    dst_engine_id=xfer.dst_engine_id,
                    remote_rank=xfer.remote_rank,
                )
                self._handle_failed_transfer(req_id, handle)
                return
            if self._step_descs_left is not None:
                self._step_descs_left = max(0, self._step_descs_left - (end - start))
            xfer.posted = end
            if last:
                read.xfers.popleft()

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
        self._engine_last_active[engine_id] = time.perf_counter()

        if self._is_lease_expired(meta):
            if not any(meta.local_physical_block_ids):
                # Nothing to read, and the remote already released the blocks.
                return
            logger.warning(
                "Declining expired remote read for %s from engine %s.",
                req_id,
                engine_id,
            )
            self.xfer_stats.record_kv_expired_req()
            self._handle_failed_transfer(req_id, None)
            return

        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        num_groups = len(local_block_ids)
        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=[
                    list(local_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
                remote_block_ids=[
                    list(remote_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
            )
            for rank in plan.all_source_ranks
        ]

        # D may have to perform multiple reads from different remote ranks.
        # MLA opt: when P TP > D TP, only a single read is executed for
        # the first remote rank (cache is duplicated)..
        if self.use_mla and tp_ratio < 0:
            assert len(read_specs) == 1

        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            # Get side handles.
            if tp_ratio < 0 and not self.use_mla:
                assert remote_block_size == self.block_size
                # Remote tp_size > local tp_size: we must perform multiple
                # reads. Get the memory chunk onto which we will write to.
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][i]
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            # Destination handle: remote_engine_id -> remote_rank -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )

        if self.use_mla and tp_ratio < 0 and read_specs:
            # ..but we still need to notify the other remote ranks that we
            # have the blocks we need so they can update the request state.
            notif_id = f"{meta.remote.request_id}:{self.world_size}".encode()
            if not any(meta.local_physical_block_ids):
                notif_id = NOTIF_HIT_PREFIX.encode() + notif_id
            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != (0, read_specs[0].remote_rank):
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ):
        """
        Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            # TODO (NickLucche) assume HMA is off. Change to handle multiple KV groups.
            assert not self._is_hma_required
            local_block_ids0 = local_block_ids[0] if local_block_ids else []
            remote_block_ids0 = remote_block_ids[0]
            local_block_ids_mapped = self.get_mapped_blocks(
                np.asarray(local_block_ids0), block_size_ratio
            ).tolist()
            if len(local_block_ids_mapped) > len(remote_block_ids0):
                # NOTE:
                # get_mapped_blocks will always expand block_ids for n times.
                # ex:
                # prefill block_ids with block_size as 4:
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                # Local decode block_ids with block_size as 16: [1, 2, 3]
                # expanded decode block_ids with get_mapped_blocks from [1, 2, 3] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                # Then we clip local to align with prefill
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                local_block_ids_mapped = local_block_ids_mapped[
                    : len(remote_block_ids0)
                ]
            local_block_ids = [local_block_ids_mapped] if local_block_ids_mapped else []
            remote_block_ids = [remote_block_ids0]
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes.

        # Number of D TP workers that will read from dst P. Propagate info
        # on notification so that dst worker can wait before freeing blocks.
        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        if len(local_block_ids) == 0:
            # A full prefix cache hit is indicated with an empty list.
            agent_name = self._remote_agents[dst_engine_id][(0, remote_rank)]
            try:
                self.nixl_wrapper.send_notif(
                    agent_name, notif_msg=NOTIF_HIT_PREFIX.encode() + notif_id
                )
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="P worker blocks will be freed after timeout. "
                    "This may indicate network issues.",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()
            return

        assert (
            len(remote_block_ids)
            == len(local_block_ids)
            == len(self.kv_cache_config.kv_cache_groups)
        )
        remote_physical_per_logical = remote_info.remote_physical_blocks_per_logical
        local_block_ids, remote_block_ids = self._apply_prefix_caching(
            local_block_ids, remote_block_ids, remote_physical_per_logical
        )

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        num_descs = len(local_block_descs_ids)
        read = self._sliced_reads.get(request_id)
        left = self._step_descs_left
        # Past what this step has left, or behind a part already sliced.
        if num_descs and (read is not None or (left is not None and num_descs > left)):
            if read is None:
                read = self._sliced_reads[request_id] = _SlicedRead()
            elif read.failed:
                return
            read.xfers.append(
                _ReadXfer(
                    local_xfer_side_handle=local_xfer_side_handle,
                    local_desc_ids=local_block_descs_ids,
                    remote_xfer_side_handle=remote_xfer_side_handle,
                    remote_desc_ids=remote_block_descs_ids,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                    notif_id=notif_id,
                )
            )
            self._post_read_slices(request_id, read)
            return
        if left is not None:
            self._step_descs_left = left - num_descs

        # Prepare transfer with Nixl.
        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )

            # Begin async xfer.
            self.nixl_wrapper.transfer(handle)

            # Use handle to check completion in future step().
            self._recving_transfers[request_id].append(handle)
        except Exception as e:
            # mark all (logical) blocks for this request as invalid
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            self._handle_failed_transfer(request_id, handle)

    def _pop_done_transfers(self, transfers: dict[str, list[int]]) -> set[str]:
        """Also hold a sliced read back until its last slice is done.

        It is then reported done, or failed if a slice failed or the holder's
        lease ran out while its slices were in flight.
        """
        done = super()._pop_done_transfers(transfers)
        for req_id in done & self._sliced_reads.keys():
            read = self._sliced_reads[req_id]
            if read.xfers:
                done.discard(req_id)
                continue
            del self._sliced_reads[req_id]
            meta = self._recving_metadata[req_id]
            if not read.failed and self._lease_left_s(meta) <= 0:
                # Its holder may have let the blocks go under its slices.
                assert meta.remote is not None
                logger.warning(
                    "Read of %s from engine %s completed past the holder's "
                    "lease: marking its blocks invalid.",
                    req_id,
                    meta.remote.engine_id,
                )
                self.xfer_stats.record_kv_expired_req()
                read.failed = True
            if read.failed:
                super()._handle_failed_transfer(req_id, None)
                continue
            assert read.posted_s is not None
            self.xfer_stats.record_sliced_read(
                time.perf_counter() - read.posted_s,
                read.post_s,
                read.total_bytes,
                read.descs,
            )
        return done

    def _record_transfer(self, req_id: str, res: "nixlXferTelemetry") -> None:
        read = self._sliced_reads.get(req_id)
        if read is None:
            super()._record_transfer(req_id, res)
            return
        read.post_s += res.postDuration / 1e6
        read.total_bytes += res.totalBytes
        read.descs += res.descCount

    def _handle_failed_transfer(self, req_id: str, handle: int | None):
        read = self._sliced_reads.get(req_id)
        if read is None:
            super()._handle_failed_transfer(req_id, handle)
            return
        # Slices still in flight write into the request's blocks: it is
        # reported failed once they are done (_pop_done_transfers).
        if handle is not None:
            self.nixl_wrapper.release_xfer_handle(handle)
        read.failed = True
        read.xfers.clear()
        if not self._recving_transfers.get(req_id):
            del self._sliced_reads[req_id]
            super()._handle_failed_transfer(req_id, None)

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.

        Also handles heartbeat notifications ("HB:req1,req2,...") by
        extending the lease on the referenced requests.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                msg = notif.decode("utf-8")

                # Handle heartbeat messages from D-side.
                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                read = not msg.startswith(NOTIF_HIT_PREFIX)
                if not read:
                    msg = msg[len(NOTIF_HIT_PREFIX) :]
                req_id, tp_size = msg.rsplit(":", 1)
                if (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                ):
                    released = self._released_holds.get(req_id)
                    if (
                        read
                        and released is not None
                        and time.perf_counter() - released <= RELEASED_HOLD_GRACE_S
                    ):
                        # Released at its owner's request once the reader had
                        # landed the read, whose notification came later.
                        logger.info(
                            "A read of %s was notified after its hold was "
                            "released at its owner's request.",
                            req_id,
                        )
                    elif read:
                        logger.error(
                            "Potentially invalid KV blocks for "
                            "unrecognized request %s were retrieved by "
                            "a decode worker. They may have expired.",
                            req_id,
                        )
                    else:
                        logger.debug(
                            "A full prefix hit for %s arrived after its blocks "
                            "were released; nothing was read.",
                            req_id,
                        )
                    continue

                # NOTE: `tp_ratio` is the opposite when swapping local<>remote
                n_consumers = int(tp_size)
                tp_ratio = self.transfer_topo.tp_ratio(n_consumers)

                # Number of reads *per producer* to wait for.
                # When remote D TP > local P TP we expect `tp_ratio` reads.
                consumers_per_producer = (
                    -tp_ratio if n_consumers > self.world_size else 1
                )

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if (
                    self.consumer_notification_counts_by_req[req_id]
                    == consumers_per_producer
                ):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids
