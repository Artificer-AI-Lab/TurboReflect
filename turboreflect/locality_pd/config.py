# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""The `pd` block (each engine of a split run: role, devices, port, caching) and
the `pd_method` block (`PolicyConfig`, the placement policy's settings).

`resolve(None)` returns `None`, not a disabled block: `RunConfig.fingerprint()`
hashes every field, so a run without P/D must not gain `"pd": null`, while
`{"enabled": false}` is a distinct statement that does move the fingerprint.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from turboreflect.config import (
    SCHEDULING_FCFS,
    ConfigError,
)
from turboreflect.locality_pd.policy import (
    TPOT_STEP_SHARE,
    TPOT_TOKENS,
    TPOT_WINDOW_S,
    CostPrior,
    ProfileCostModel,
    StepModel,
    load_prior,
)
from turboreflect.locality_pd.runtime import LocalityPDSettings, PolicyRuntime

# vLLM's `VLLM_KV_CACHE_LAYOUT` choices, pinned since a co-located engine defaults
# to NHD and a NIXL engine asks for HND.
LAYOUT_CHOICES = ("NHD", "HND")

ROLE_P = "p"
ROLE_D = "d"

# D's suffix config as overridden on a P of a model with recurrent state. D's draft
# count stays, as it sizes the Mamba conv state and hybrid block size NIXL moves
# slot for slot; the speculation length (factor times prefix match) is 0.
P_SUFFIX_LAYOUT_ONLY = {
    "suffix_decoding_max_spec_factor": 0.0,
    "suffix_decoding_max_tree_depth": 1,
    "suffix_decoding_max_cached_requests": 0,
}

# vLLM's `KVTransferConfig.kv_role` per side. They stand although P also pulls and D
# also exports: at TP1/PP1 only the scheduler reads `kv_role`, to defer a consumer's
# block frees under async scheduling, which no engine here runs
# (`engine._finalize_engine_args`).
KV_ROLES = {ROLE_P: "kv_producer", ROLE_D: "kv_consumer"}

# vLLM's lease margin, restated as this module imports no vLLM: a reader fails a
# read within min(5 s, a quarter of the lease) of the holder's expiry.
D_READ_MARGIN_S = 5.0

# Fields every engine shares; none has a default.
SHARED_FIELDS = (
    "kv_cache_layout",
    "kv_lease_duration_s",
    "request_deadline_s",
)
REQUIRED_WHEN_ENABLED = ("engines", *SHARED_FIELDS)

ENGINE_FIELDS = (
    "name",
    "role",
    "devices",
    "port",
    "max_num_seqs",
    "enable_prefix_caching",
)

# Optional fields: `RunConfig.as_dict` drops absent ones so they do not move the
# fingerprint. An absent scheduling policy is vLLM's FCFS.
ROUTING_FIELDS = (
    "decoder_kv_blocks_ttl_s",
    "kv_recompute_threshold",
    "text_join_pull",
)

SCHEDULING_FIELDS = ("scheduling_policy_p", "scheduling_policy_d")

FIELDS = (
    "enabled",
    *REQUIRED_WHEN_ENABLED,
    *ROUTING_FIELDS,
    *SCHEDULING_FIELDS,
)


def _as_bool(source: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{source} must be true or false, got {value!r}")
    return value


def _as_int(source: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{source} must be an integer, got {value!r}")
    return value


Device = int | str


def _as_device(source: str, value: Any) -> Device:
    """A device index, or a `CUDA_VISIBLE_DEVICES` identifier such as a MIG UUID."""
    if isinstance(value, str):
        if not value or value != value.strip() or "," in value:
            raise ConfigError(f"{source} must be one device identifier, got {value!r}")
        return int(value) if value.isascii() and value.lstrip("-").isdigit() else value
    return _as_int(source, value)


def _as_devices(source: str, value: Any) -> Device | tuple[Device, ...]:
    """One device, or a list of them for a tensor-parallel side."""
    if isinstance(value, list):
        if not value:
            raise ConfigError(f"{source} must name at least one device")
        return tuple(_as_device(f"{source}[{i}]", v) for i, v in enumerate(value))
    return _as_device(source, value)


def devices_of(value: Device | tuple[Device, ...]) -> tuple[Device, ...]:
    return value if isinstance(value, tuple) else (value,)


def _as_float(source: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{source} must be a number of seconds, got {value!r}")
    return float(value)


def _as_str(source: str, value: Any) -> str:
    """A string as written, unstripped: vLLM matches the exported layout exactly."""
    if not isinstance(value, str):
        raise ConfigError(f"{source} must be a string, got {value!r}")
    return value


@dataclass(frozen=True)
class EngineSpec:
    """One engine of a split run.

    Attributes:
        devices: Its `CUDA_VISIBLE_DEVICES`, one per tensor-parallel rank.
        port: Its `VLLM_NIXL_SIDE_CHANNEL_PORT` base; vLLM opens base + DP index, so
            each engine needs its own.
        max_num_seqs: This engine's real concurrency, not the run's program
            concurrency.
    """

    name: str
    role: str
    devices: tuple[Device, ...]
    port: int
    max_num_seqs: int
    enable_prefix_caching: bool


def _as_engine(source: str, entry: Any) -> EngineSpec:
    if not isinstance(entry, Mapping):
        raise ConfigError(
            f"{source} must be an object with fields {list(ENGINE_FIELDS)}, "
            f"got {entry!r}"
        )
    missing = [n for n in ENGINE_FIELDS if n not in entry]
    unknown = sorted(set(entry) - set(ENGINE_FIELDS))
    if missing or unknown:
        raise ConfigError(
            f"{source} must state exactly {list(ENGINE_FIELDS)}, and lacks {missing} "
            f"and adds {unknown} -- no engine field has a default"
        )
    return EngineSpec(
        name=_as_str(f"{source}.name", entry["name"]),
        role=_as_str(f"{source}.role", entry["role"]),
        devices=devices_of(_as_devices(f"{source}.devices", entry["devices"])),
        port=_as_int(f"{source}.port", entry["port"]),
        max_num_seqs=_as_int(f"{source}.max_num_seqs", entry["max_num_seqs"]),
        enable_prefix_caching=_as_bool(
            f"{source}.enable_prefix_caching", entry["enable_prefix_caching"]
        ),
    )


def _as_engines(source: str, value: Any) -> tuple[EngineSpec, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(
            f"{source} must be a non-empty list of engines, each an object with "
            f"fields {list(ENGINE_FIELDS)}"
        )
    return tuple(_as_engine(f"{source}[{i}]", entry) for i, entry in enumerate(value))


_FROM_BLOCK = {
    "enabled": _as_bool,
    "kv_cache_layout": _as_str,
    "kv_lease_duration_s": _as_float,
    "request_deadline_s": _as_float,
    "engines": _as_engines,
    "decoder_kv_blocks_ttl_s": _as_float,
    "kv_recompute_threshold": _as_int,
    "text_join_pull": _as_bool,
    "scheduling_policy_p": _as_str,
    "scheduling_policy_d": _as_str,
}


@dataclass(frozen=True)
class PDSettings:
    """A checked P/D instruction, built only by `PDConfig.settings()` so that no
    code past the checks decides what an absent field means. Fields as `PDConfig`.
    """

    engines: tuple[EngineSpec, ...]
    kv_cache_layout: str
    kv_lease_duration_s: float
    request_deadline_s: float
    decoder_kv_blocks_ttl_s: float | None = None
    kv_recompute_threshold: int | None = None
    text_join_pull: bool = False
    scheduling_policy_p: str = SCHEDULING_FCFS
    scheduling_policy_d: str = SCHEDULING_FCFS

    def names(self, role: str | None = None) -> tuple[str, ...]:
        """Every engine's name in the order the block lists them, or one role's."""
        return tuple(e.name for e in self.engines if role is None or e.role == role)

    def engine(self, name: str) -> EngineSpec:
        """One engine, by name.

        Raises:
            ConfigError: On an unknown name, which would reach vLLM as no transfer.
        """
        for engine in self.engines:
            if engine.name == name:
                return engine
        raise ConfigError(
            f"unknown P/D engine {name!r}; this run has {list(self.names())}"
        )

    def role(self, name: str) -> str:
        return self.engine(name).role

    def devices(self, name: str) -> tuple[Device, ...]:
        return self.engine(name).devices

    def port(self, name: str) -> int:
        return self.engine(name).port

    def prefix_caching(self, name: str) -> bool:
        return self.engine(name).enable_prefix_caching

    def kv_role(self, name: str) -> str:
        return KV_ROLES[self.role(name)]

    def scheduling_policy(self, name: str) -> str:
        if self.role(name) == ROLE_P:
            return self.scheduling_policy_p
        return self.scheduling_policy_d


@dataclass(frozen=True)
class PDConfig:
    """The P/D section of a run: off and silent, or on and fully stated.

    Attributes:
        engines: At least one P and one D, each on a port of its own.
        kv_lease_duration_s: How long P holds a finished request's blocks until D's
            scheduler has it; each D heartbeat then extends it by `lease * 2 // 3`,
            so whole seconds, at least two. Too long a lease leaves a dead decoder's
            requests on P's KV pool.
        request_deadline_s: The driver's wait from handing a request to D until its
            first token, D's queue wait included. vLLM pull mode has no watchdog,
            so without it a request whose KV never arrives hangs and pins P's blocks.
        decoder_kv_blocks_ttl_s: How long D holds a finished request's blocks for P
            to pull (`decoder_kv_blocks_ttl`), unrenewed; not held for a program's
            last stage (`message.DECODER_KV_HOLD_KEY`). Should exceed
            `D_READ_MARGIN_S`.
        kv_recompute_threshold: The fewest tokens P pulls rather than recomputes.
        text_join_pull: Whether a later stage joined as text pulls the whole blocks
            it shares with D's history (`session.truncated_pull`). Absent: true
            under locality-aware P/D (`PolicyConfig.resolve_pd`), else false.
        scheduling_policy_p: vLLM's `scheduling_policy` on every P: `fcfs`
            (absent) or `priority`, which with `stage_priority` `context_reuse_class`
            prefills a program's later stages ahead of new programs.
        scheduling_policy_d: The same on every D.
    """

    enabled: bool
    engines: tuple[EngineSpec, ...] | None
    kv_cache_layout: str | None
    kv_lease_duration_s: float | None
    request_deadline_s: float | None
    decoder_kv_blocks_ttl_s: float | None
    kv_recompute_threshold: int | None
    text_join_pull: bool | None
    scheduling_policy_p: str | None
    scheduling_policy_d: str | None

    @classmethod
    def resolve(cls, block: Mapping[str, Any] | None = None) -> PDConfig | None:
        """Build from a config-file block, or `None` when there was none.

        No environment fallback: a stale `$PD_GPU_D` would silently move a run.

        Raises:
            ConfigError: On a malformed block or one that does not state `enabled`.
        """
        if block is None:
            return None
        if not isinstance(block, Mapping):
            raise ConfigError(f"pd must be an object with fields {list(FIELDS)}")
        if not block:
            raise ConfigError(
                "the pd block is empty -- state the fields "
                f"{list(FIELDS)}, or drop the block entirely to run one engine"
            )
        unknown = sorted(set(block) - set(FIELDS))
        if unknown:
            raise ConfigError(
                f"unknown field(s) in the pd block: {unknown}; expected any of "
                f"{list(FIELDS)}"
            )
        if "enabled" not in block:
            raise ConfigError(
                "pd states "
                + ", ".join(sorted(block))
                + " but not whether it is enabled -- set pd.enabled; whether a "
                "run splits prefill from decode is not something to read off "
                "the presence of a key"
            )
        stated = {
            name: _FROM_BLOCK[name](f"pd.{name}", value)
            for name, value in block.items()
        }
        return cls(**{name: stated.get(name) for name in FIELDS})

    def settings(self) -> PDSettings:
        """The checked instruction; raises when P/D is off, so that a caller cannot
        drift onto the split path by forgetting to ask `enabled`.
        """
        settings = self._checked()
        if settings is None:
            raise ConfigError("pd is disabled: there are no settings to take")
        return settings

    def _checked(self) -> PDSettings | None:
        """Validate and project in one walk so no path skips a check; None if off."""
        if not isinstance(self.enabled, bool):
            raise ConfigError(f"pd.enabled must be true or false, got {self.enabled!r}")

        if not self.enabled:
            stated = [n for n in FIELDS[1:] if getattr(self, n) is not None]
            if stated:
                raise ConfigError(
                    "pd is disabled but still states "
                    + ", ".join(stated)
                    + " -- an off run has one spelling on purpose, so that two "
                    "of them cannot fingerprint differently; drop the fields or "
                    "enable it"
                )
            return None

        missing = [n for n in REQUIRED_WHEN_ENABLED if getattr(self, n) is None]
        if missing:
            raise ConfigError(
                "pd is enabled but does not state "
                + ", ".join(missing)
                + " -- none of these has a default in code; a guessed port or a "
                "guessed deadline is a run that either collides with itself or "
                "hangs for ever without saying so"
            )

        settings = PDSettings(
            engines=self.engines,
            **{name: getattr(self, name) for name in SHARED_FIELDS},
            decoder_kv_blocks_ttl_s=self.decoder_kv_blocks_ttl_s,
            kv_recompute_threshold=self.kv_recompute_threshold,
            text_join_pull=bool(self.text_join_pull),
            scheduling_policy_p=(
                SCHEDULING_FCFS
                if self.scheduling_policy_p is None
                else self.scheduling_policy_p
            ),
            scheduling_policy_d=(
                SCHEDULING_FCFS
                if self.scheduling_policy_d is None
                else self.scheduling_policy_d
            ),
        )
        if settings.kv_cache_layout not in LAYOUT_CHOICES:
            raise ConfigError(
                f"pd.kv_cache_layout must be one of {list(LAYOUT_CHOICES)}, "
                f"spelled exactly, got {settings.kv_cache_layout!r} -- it is "
                "exported as VLLM_KV_CACHE_LAYOUT, which matches these two "
                "strings and nothing near them"
            )
        if settings.kv_lease_duration_s <= 0:
            raise ConfigError(
                "pd.kv_lease_duration_s must be positive, got "
                f"{settings.kv_lease_duration_s}"
            )
        if settings.request_deadline_s <= 0:
            raise ConfigError(
                "pd.request_deadline_s must be positive, got "
                f"{settings.request_deadline_s} -- pull mode has no watchdog of "
                "its own, so a run without a deadline hangs rather than fails"
            )
        return settings


# The engine facts a fitted profile's cost terms depend on (`pd_profile`).
COST_FIELDS = (
    "model",
    "model_dtype",
    "kv_cache_dtype",
    "max_num_batched_tokens",
    "attention_backend",
    "mla_prefill_backend",
    "moe_backend",
    "cudagraph_mode",
    "enforce_eager",
    "tensor_parallel_size",
)


# pd defaults: a later stage pulls or hits its matched prefix's whole blocks on D.
ADAPT_PD = {"text_join_pull": True}

APPEND_TRACE = "turboreflect.locality_pd.append_scheduler.AppendTraceScheduler"


def source_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """The cost-relevant `engine` and `hardware` facts of a run's `provenance.json`."""
    engine = record.get("engine") or {}
    devices = (record.get("hardware") or {}).get("local_devices") or [{}]
    return {
        **{name: engine.get(name) for name in COST_FIELDS},
        "gpu": devices[0].get("name"),
    }


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    profile_path: str
    profile_sha256: str
    tpot_guard_delta: float | None = None
    # `tpot_step_share` None selects the floor rule (`TpotBudget`).
    tpot_window_s: float = TPOT_WINDOW_S
    tpot_step_share: float | None = TPOT_STEP_SHARE
    tpot_weighting: str = TPOT_TOKENS
    role_flex: bool = True

    @classmethod
    def resolve(cls, block: Mapping[str, Any], pd: Any) -> PolicyConfig:
        if set(block) - {f.name for f in dataclasses.fields(cls)}:
            raise ConfigError("unknown P/D method fields")
        if pd is None or not pd.enabled:
            raise ConfigError("P/D requires enabled PD engines")
        prior = cls._resolve_prior(block)
        path, digest = prior.path, prior.sha256
        try:
            method = cls(**{**block, "profile_path": path, "profile_sha256": digest})
            method.make_router()
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"P/D: {exc}") from exc
        return method

    @staticmethod
    def _resolve_prior(block: Mapping[str, Any]) -> CostPrior:
        path = block.get("profile_path")
        if path is None:
            raise ConfigError("P/D requires profile_path")
        if not isinstance(path, str) or not path:
            raise ConfigError("P/D profile paths must be nonempty strings")
        try:
            return load_prior(str(Path(path).expanduser().resolve()))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"P/D prior: {exc}") from exc

    def resolve_pd(self, pd: PDConfig) -> PDConfig:
        """Fill each field the pd block leaves out from `ADAPT_PD`."""
        absent = {
            name: value for name, value in ADAPT_PD.items() if getattr(pd, name) is None
        }
        return dataclasses.replace(pd, **absent)

    def _model(self) -> ProfileCostModel:
        return ProfileCostModel.from_dict(
            json.loads(Path(self.profile_path).read_bytes())
        )

    def prior(self) -> CostPrior:
        """The fit locality-aware P/D prices from, as resolved."""
        model = self._model()
        return CostPrior(
            model,
            StepModel.from_profile(model),
            self.profile_path,
            self.profile_sha256,
        )

    def make_router(self) -> PolicyRuntime:
        assert self.tpot_guard_delta is not None
        return PolicyRuntime(
            self.name,
            LocalityPDSettings(
                self.prior(),
                self.tpot_guard_delta,
                drain_on_p=self.role_flex,
                role_flex=self.role_flex,
            ),
        )

    @property
    def speculates_on_p(self) -> bool:
        """Whether every P runs D's suffix decoding and adaptive draft length too."""
        return self.role_flex

    def scheduler_cls(self) -> str:
        """The traced worker scheduler; every engine serves decodes first."""
        return APPEND_TRACE

    def engine_extra_config(self, settings: Any, name: str) -> dict[str, Any]:
        """An engine's `additional_config.locality_pd` (`AppendScheduler`)."""
        return {
            "role": settings.role(name),
            "profile_path": self.profile_path,
            "tpot_delta": self.tpot_guard_delta,
            "tpot_window_s": self.tpot_window_s,
            "tpot_step_share": self.tpot_step_share,
            "tpot_weighting": self.tpot_weighting,
        }
