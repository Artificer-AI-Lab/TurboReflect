# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Everything that determines a run, in one hashable object.

A YAML or JSON config file names the model, the tasks and the components to enable;
`expand_switches` turns those into run fields, and any field the file states wins. A
file that lists `samples` instead of `tasks` is a complete run configuration, read as
it stands. `--set NAME=VALUE` overrides either kind. The result directory is named
after `fingerprint()`, which covers every stated field.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import runpy
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re

from turboreflect.context_reuse.admission import STAGE_CLASS_STRIDE
from turboreflect.program.stages import MERGED_REFINEMENTS
from turboreflect.program.task import SampleError, file_sha256, video_paths

if TYPE_CHECKING:
    from turboreflect.locality_pd.config import PDConfig, PolicyConfig

# Decode paths: `quality` stops at the end token and streams; `efficiency` generates
# exactly `gen_tokens` with `ignore_eos`; `terminating` is `efficiency` without
# `ignore_eos`. These last two consume the stream in one final yield, so the host does
# no per-token work, and cap each stage at `gen_tokens`, not `max_tokens_quality`.
FINAL_ONLY_PATHS = ("efficiency", "terminating")
DECODE_PATHS = ("quality", *FINAL_ONLY_PATHS)


def strips_stray_reasoning(enable_thinking: bool, decode_path: str) -> bool:
    """Whether `<think>` blocks are cut from what a stage hands the next one.

    Not on `efficiency`: past its ignored end token a `<think>` is not the answer's
    reasoning, and the next stage appends to the ids as generated.
    """
    return not enable_thinking and decode_path != "efficiency"


# A multimodal run's engine fields, passed and fingerprinted only when stated.
MULTIMODAL_FIELDS = (
    "mm_processor_kwargs",
    "limit_mm_per_prompt",
    "mm_processor_cache_gb",
    "allowed_local_media_path",
)

_HF_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


def is_hf_repo_id(model: str) -> bool:
    """Whether `model` names a hub repo rather than a directory."""
    return bool(_HF_REPO_ID.fullmatch(model))


def hf_snapshot_dir(repo_id: str, cache_dir: str | Path | None = None) -> Path | None:
    """The newest hub-cache snapshot of `repo_id` holding `config.json`, or None."""
    if cache_dir is None:
        root = os.environ.get("HF_HUB_CACHE")
        if not root:
            home = os.environ.get("HF_HOME")
            root = (
                str(Path(home) / "hub")
                if home
                else str(Path.home() / ".cache" / "huggingface" / "hub")
            )
    else:
        root = str(cache_dir)
    snapshots = Path(root) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    found = [d for d in snapshots.iterdir() if (d / "config.json").is_file()]
    if not found:
        return None
    return max(found, key=lambda d: d.stat().st_mtime)


# FP8 KV cache; scales default to 1.0 unless the checkpoint provides them.
FP8_CACHE_DTYPES = (
    "fp8",
    "fp8_e4m3",
)

# Unquantized KV that, unlike `auto`, vLLM never replaces with a checkpoint's
# `kv_cache_scheme`. No `float16`: the cache kernels store the activation dtype.
UNQUANTIZED_CACHE_DTYPES = ("bfloat16",)

KV_CACHE_DTYPES = (
    "auto",
    *UNQUANTIZED_CACHE_DTYPES,
    *FP8_CACHE_DTYPES,
)

# The stages in run order; a run may stop early but not skip one. Repeated from
# `program.prompt.STAGES` because that module imports this one.
PIPELINE_STAGES = ("generation", "feedback", "refinement")

# `fixed` walks `stages`; `rounds` runs generation once, then feedback and refinement
# for `max_rounds` rounds. The admission controller learns only whether a program has
# finished a stage before, never its round or stage count.
PIPELINE_MODE_FIXED = "fixed"
PIPELINE_MODE_ROUNDS = "rounds"
PIPELINE_MODES = (PIPELINE_MODE_FIXED, PIPELINE_MODE_ROUNDS)

DEFAULT_MAX_ROUNDS = 4

# How a later stage's prompt carries earlier outputs. Under `continue` it is the
# previous stage's prompt and output plus the next instruction turn. Under `requote`
# the generation prompt runs through its task, then one user message quotes each
# upstream output under a short header before the stage's instruction, so each stage
# matches the previous one through its last quoted output and appends one more.
PROMPT_LAYOUT_CONTINUE = "continue"
PROMPT_LAYOUT_REQUOTE = "requote"
PROMPT_LAYOUTS = (PROMPT_LAYOUT_CONTINUE, PROMPT_LAYOUT_REQUOTE)

# Under `priority` vLLM serves ascending priority, then arrival time, so a lower value
# goes ahead of a request that arrived earlier.
SCHEDULING_FCFS = "fcfs"
SCHEDULING_PRIORITY = "priority"
SCHEDULING_POLICIES = (SCHEDULING_FCFS, SCHEDULING_PRIORITY)

# Ready successors precede new programs; ties follow program order.
STAGE_PRIORITY_NONE = "none"
STAGE_PRIORITY_SUCCESSOR_CLASS = "context_reuse_class"
STAGE_PRIORITIES = (STAGE_PRIORITY_NONE, STAGE_PRIORITY_SUCCESSOR_CLASS)

# Keep an arrived successor's cached prefix resident while it waits.
RESERVE_ON_ARRIVAL_SCHEDULER = (
    "turboreflect.context_reuse.scheduler.ReserveOnArrivalScheduler"
)
# Hold at finish, reserve on arrival, admit with headroom, yield by value.
CONTEXT_REUSE_SCHEDULER = "turboreflect.context_reuse.scheduler.ContextReuseScheduler"
CONTEXT_REUSE_SCHEDULERS = (RESERVE_ON_ARRIVAL_SCHEDULER, CONTEXT_REUSE_SCHEDULER)
SCHEDULER_CLASSES = (None, *CONTEXT_REUSE_SCHEDULERS)
# The adaptive-draft-length form of each base scheduler: one engine's `scheduler_cls`,
# or the worker scheduler locality-aware P/D gives D.
ADAPTIVE_DRAFT_SCHEDULERS = {
    None: "turboreflect.metrics.trace.AdaptiveDraftTraceScheduler",
    CONTEXT_REUSE_SCHEDULER: (
        "turboreflect.metrics.trace.ContextReuseAdaptiveDraftTraceScheduler"
    ),
    "turboreflect.locality_pd.append_scheduler.AppendTraceScheduler": (
        "turboreflect.locality_pd.append_scheduler.AppendAdaptiveDraftTraceScheduler"
    ),
}

_REQUIRED = ("model", "samples", "gen_tokens", "max_model_len")


@dataclass(frozen=True)
class AdaptiveDraftConfig:
    """Adaptive draft length settings (`turboreflect.adaptive_draft.draft_length`).

    The step model that prices verification, and how often drafts are verified as
    proposed to keep its estimates measured.
    """

    step_model_path: str
    explore_every: int = 16
    warmup_steps: int = 64

    @classmethod
    def resolve(cls, value: Any) -> AdaptiveDraftConfig | None:
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ConfigError("adaptive_draft must be an object")
        unknown = sorted(set(value) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise ConfigError(f"unknown adaptive_draft field(s): {unknown}")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ConfigError(f"adaptive_draft: {exc}") from exc


class ConfigError(Exception):
    """Configuration is missing, contradictory, or points at nothing."""


def program_cache_salt(program_total: int | None, run_id: str, program_id: str) -> str:
    """A program's cache salt (`RunConfig.cache_salt`)."""
    if program_total is not None:
        return program_id
    return f"{run_id}:{program_id}"


# The program shapes a config file may name, and the round count of `multi_round`.
PROGRAM_TWO_STAGE = "two_stage"
PROGRAM_THREE_STAGE = "three_stage"
PROGRAM_MULTI_ROUND = "multi_round"
PROGRAMS = (PROGRAM_TWO_STAGE, PROGRAM_THREE_STAGE, PROGRAM_MULTI_ROUND)
MULTI_ROUND_ROUNDS = 3

# What `calibrate` writes into a run's calibration directory.
STEP_MODEL_FILE = "step_model.json"
PD_PROFILE_FILE = "pd_profile.json"

DEFAULT_GEN_TOKENS = 1024
DEFAULT_MAX_MODEL_LEN = 147456

# A run's settings wherever a config file is silent.
RUN_DEFAULTS: dict[str, Any] = {
    "decode_paths": ("efficiency",),
    "feedback": "goal",
    "refinement": "goal",
    "layout": PROMPT_LAYOUT_REQUOTE,
    "max_model_len": DEFAULT_MAX_MODEL_LEN,
    "gpu_memory_utilization": 0.9,
    "max_num_batched_tokens": 16384,
    "async_scheduling": False,
    "isolate_programs": True,
    "enable_thinking": False,
    "tensor_parallel_size": 1,
    "scheduler_reserve_full_isl": True,
    "enable_prefix_caching": True,
    "suffix_max_cached_requests": 0,
    "suffix_num_speculative_tokens": 16,
    "suffix_max_tree_depth": 32,
}
RUN_ENV = {"VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
# A step whose requests schedule no drafts keeps a full decode CUDA graph.
DRAFTING_ENV = {"VLLM_SPEC_SINGLE_TOKEN_FULL_GRAPH": "1"}
PD_ENV = {"VLLM_KV_CACHE_LAYOUT": "HND"}

# Each engine's NIXL side channel opens on its own port, from this one up.
PD_PORT_BASE = 25600
# A `pd` block's own fields; any other field is a field of the engines' block.
PD_SWITCHES = ("prefill", "decode", "delta", "max_num_seqs")
PD_DEFAULT_DELTA = 1.0
PD_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "kv_cache_layout": "HND",
    "kv_lease_duration_s": 600.0,
    "request_deadline_s": 7200.0,
    "scheduling_policy_p": SCHEDULING_PRIORITY,
    "scheduling_policy_d": SCHEDULING_PRIORITY,
    "decoder_kv_blocks_ttl_s": 7200.0,
    "kv_recompute_threshold": 0,
    "text_join_pull": True,
}


# The stage assembly of tasks in the shared format, for a task folder without its own.
SHARED_STAGE_ASSEMBLY = Path(__file__).parent / "benchmarks" / "custom"


def checkpoint_config(model: str) -> dict[str, Any] | None:
    """A checkpoint's `config.json`, or None when it is not on disk."""
    root = Path(model)
    if not root.is_dir() and is_hf_repo_id(model):
        root = hf_snapshot_dir(model) or root
    try:
        raw = json.loads((root / "config.json").read_text())
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def checkpoint_architectures(model: str) -> tuple[str, ...] | None:
    """A checkpoint's `architectures`, or None when it is not on disk."""
    names = (checkpoint_config(model) or {}).get("architectures")
    return tuple(names) if isinstance(names, list) else None


def read_config_file(path: str | Path) -> dict[str, Any]:
    """The fields of a YAML or JSON config file."""
    import yaml

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        if path.suffix in (".yaml", ".yml"):
            loaded = yaml.safe_load(path.read_text())
        else:
            loaded = json.loads(path.read_text())
    except (ValueError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path} does not parse: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must hold a mapping of config fields")
    return loaded


def apply_override(doc: dict[str, Any], entry: str) -> None:
    """Apply one `--set NAME=VALUE` to `doc` in place.

    A dotted NAME reaches into a block; VALUE is read as YAML.
    """
    import yaml

    name, sep, text = entry.partition("=")
    if not sep or not name:
        raise ConfigError(f"--set takes NAME=VALUE, got {entry!r}")
    *blocks, field = name.split(".")
    target = doc
    for block in blocks:
        target = target.setdefault(block, {})
        if not isinstance(target, dict):
            raise ConfigError(f"--set {name}: {block} is not a block")
    target[field] = yaml.safe_load(text)


def config_document(args: argparse.Namespace) -> dict[str, Any]:
    """The config file's fields under the command line's, before any expansion."""
    config_path = getattr(args, "config", None)
    doc = {} if config_path is None else read_config_file(config_path)
    names = {f.name for f in dataclasses.fields(RunConfig)}
    doc.update(
        {
            k: v
            for k, v in vars(args).items()
            if (k in names or k == "tasks") and v is not None
        }
    )
    for entry in getattr(args, "overrides", ()):
        apply_override(doc, entry)
    if "samples" in doc and "tasks" in doc:
        raise ConfigError("a run names its `tasks` or its `samples`, not both")
    return doc


def task_files(tasks: str | Sequence[str]) -> tuple[str, ...]:
    """Every task file `tasks` names, as absolute paths.

    A directory stands for the JSON files of its `data/` folder if any, else its own.
    """
    files: list[Path] = []
    for entry in [tasks] if isinstance(tasks, str) else tasks:
        path = Path(entry).expanduser().resolve()
        if path.is_dir():
            root = path / "data" if (path / "data").is_dir() else path
            found = sorted(root.glob("*.json"))
            if not found:
                raise ConfigError(f"no task files under {root}")
            files += found
        elif path.is_file():
            files.append(path)
        else:
            raise ConfigError(f"no task file or directory at {path}")
    return tuple(str(path) for path in files)


def stage_assembly_root(tasks: str | Sequence[str]) -> str:
    """Where a run's stage assembly (`prompts.py`) is: with its tasks, or shared."""
    for entry in [tasks] if isinstance(tasks, str) else tasks:
        path = Path(entry).expanduser().resolve()
        root = path if path.is_dir() else path.parent
        for folder in (
            root,
            root / "data",
            root.parent if root.name == "data" else root,
        ):
            if (folder / "prompts.py").is_file():
                return str(folder)
    return str(SHARED_STAGE_ASSEMBLY)


def stage_assembly_layouts(root: str) -> tuple[str, ...]:
    """The prompt layouts the stage assembly under `root` builds.

    Run in a namespace of its own, because `load_bench` imports it as `prompts`.
    """
    path = Path(root) / "prompts.py"
    if not path.is_file():
        raise ConfigError(f"no prompts.py under {root}")
    return runpy.run_path(str(path)).get("LAYOUTS", (PROMPT_LAYOUT_CONTINUE,))


def env_mapping(value: Any) -> dict[str, str]:
    """A config's `env`, an object or `NAME=VALUE` entries, as a mapping."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {name: str(entry) for name, entry in value.items()}
    return dict(str(entry).partition("=")[::2] for entry in value)


def engine_kv_cache_dtype(doc: Mapping[str, Any]) -> str:
    """The engine's KV cache dtype for what a config file says, `fp8` when silent."""
    stated = doc.get("kv_cache_dtype", "fp8")
    return "bfloat16" if stated == "bf16" else stated


# Fields that name a task set rather than the engine.
_TASK_SET_FIELDS = ("samples", "bench_root", "program_total", "concurrency")


def gpu_names() -> list[str] | None:
    """The names of the GPUs this process sees, or None without `nvidia-smi`.

    Asked of `nvidia-smi` rather than CUDA, so the driver process initialises no GPU.
    """
    query = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    for command in ([*query, "-i", visible], query) if visible else (query,):
        try:
            out = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=True
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        names = sorted({line.strip() for line in out.splitlines() if line.strip()})
        if names:
            return names
    return None


def calibration_dir(doc: Mapping[str, Any]) -> Path:
    """Where a config's calibration files are: one directory per model, GPU and engine.

    The default name digests the configuration's fields and the GPU, not the tasks:
    the step model times synthetic batches, and the P/D profile is a function of chunk
    and cached length that any tasks measure. The tasks count only when
    `max_num_seqs` is left unstated, since the limit derived from them sets the
    batches the step model times.
    """
    stated = doc.get("calibration_dir")
    if stated is not None:
        return Path(stated).expanduser().resolve()
    fields = expand_switches(doc, calibrated=False)
    fields.pop("out_dir", None)
    if fields.get("max_num_seqs") is None:
        fields["samples"] = [file_sha256(path) for path in fields["samples"]]
    else:
        for name in _TASK_SET_FIELDS:
            fields.pop(name, None)
    fields["gpu"] = gpu_names()
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode())
    name = f"{Path(doc['model']).name}-{engine_kv_cache_dtype(doc)}"
    return (Path("calibration") / f"{name}-{digest.hexdigest()[:12]}").resolve()


def _switch(doc: dict[str, Any], name: str) -> bool:
    value = doc.pop(name, True)
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}")
    return value


def _program_fields(program: str, gen_tokens: int) -> dict[str, Any]:
    """What a program shape sets. A merged refinement reviews, then rewrites in full."""
    if program == PROGRAM_THREE_STAGE:
        return {}
    if program == PROGRAM_TWO_STAGE:
        return {
            "stages": ("generation", "refinement"),
            "refinement_gen_tokens": 2 * gen_tokens,
        }
    if program == PROGRAM_MULTI_ROUND:
        return {
            "pipeline_mode": PIPELINE_MODE_ROUNDS,
            "max_rounds": MULTI_ROUND_ROUNDS,
        }
    raise ConfigError(f"unknown program {program!r}; expected {list(PROGRAMS)}")


def _merged_refinement(refinement: str) -> str:
    """A two-stage program's refinement prompt, which also writes the review."""
    merged = "goal_merged" if refinement == "goal" else refinement
    if merged not in MERGED_REFINEMENTS:
        raise ConfigError(
            "a two_stage program merges feedback into its refinement stage: "
            f"refinement must be goal or one of {sorted(MERGED_REFINEMENTS)}, "
            f"got {refinement!r}"
        )
    return merged


def _video_fields(samples: Sequence[str]) -> dict[str, Any]:
    """The engine settings a run derives from tasks with clips, none for text tasks.

    A model that counts inputs in units other than clips states `limit_mm_per_prompt`.
    """
    try:
        clips = [video_paths(sample) for sample in samples]
    except (OSError, SampleError) as exc:
        raise ConfigError(str(exc)) from exc
    most = max(map(len, clips))
    if not most:
        return {}
    folders = [str(Path(clip).resolve().parent) for task in clips for clip in task]
    return {
        "allowed_local_media_path": os.path.commonpath(folders),
        "limit_mm_per_prompt": {"image": 0, "video": most},
        "language_model_only": None,
    }


def _calibration_file(path: Path, purpose: str) -> str:
    """A calibration file's path, once it is known to exist."""
    if not path.is_file():
        raise ConfigError(
            f"{purpose} needs {path}, which does not exist: run "
            "`python -m turboreflect calibrate` with the same config and flags first"
        )
    return str(path.resolve())


def step_model_block(directory: Path, stated: Mapping[str, Any]) -> dict[str, Any]:
    """Adaptive draft length's settings, pointed at the calibrated step model."""
    block = dict(stated)
    path = block.get("step_model_path") or directory / STEP_MODEL_FILE
    block["step_model_path"] = _calibration_file(Path(path), "adaptive draft length")
    return block


def _pd_engines(
    prefill: int, decode: int, slots: int, tensor_parallel_size: int
) -> list[dict[str, Any]]:
    """Every engine of a P/D run on its own GPUs, the P engines first."""
    names = [f"p{i}" for i in range(prefill)] + [f"d{i}" for i in range(decode)]
    tp = tensor_parallel_size
    return [
        {
            "name": name,
            "role": name[0],
            "devices": index if tp == 1 else list(range(index * tp, (index + 1) * tp)),
            "port": PD_PORT_BASE + index,
            "max_num_seqs": slots,
            "enable_prefix_caching": True,
        }
        for index, name in enumerate(names)
    ]


def _check_pd_block(block: Any) -> None:
    """Raise unless a `pd` block says how many engines to run, and how wide each is."""
    if not isinstance(block, Mapping):
        raise ConfigError(f"pd must be a block of {list(PD_SWITCHES)}, got {block!r}")
    for name in ("prefill", "decode", "max_num_seqs"):
        value = block.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(
                f"pd.{name} must be a positive integer, got {value!r}: prefill and "
                "decode count the engines of each role, and max_num_seqs is each "
                "engine's sequence limit"
            )


def _pd_fields(
    block: Mapping[str, Any],
    method: Mapping[str, Any],
    directory: Path,
    tensor_parallel_size: int,
) -> dict[str, Any]:
    """The `pd`, `pd_method` and top-level fields of a `pd` switch block."""
    engines = _pd_engines(
        block["prefill"], block["decode"], block["max_num_seqs"], tensor_parallel_size
    )
    method = dict(method)
    if "profile_sha256" not in method:
        path = method.get("profile_path") or directory / PD_PROFILE_FILE
        method["profile_path"] = _calibration_file(Path(path), "locality-aware P/D")
        method["profile_sha256"] = file_sha256(path)
    return {
        "scheduling_policy": SCHEDULING_FCFS,
        "pd": {
            **PD_DEFAULTS,
            "engines": engines,
            **{k: v for k, v in block.items() if k not in PD_SWITCHES},
        },
        "pd_method": {
            "name": "adaptive",
            "tpot_guard_delta": block.get("delta", PD_DEFAULT_DELTA),
            **method,
        },
    }


def expand_switches(doc: Mapping[str, Any], calibrated: bool = True) -> dict[str, Any]:
    """The run fields a config file stands for.

    Besides `model` and `tasks`, the switches `program`, `context_reuse`,
    `adaptive_draft_length`, `kv_cache_dtype` and a `pd` block choose what runs and
    `RUN_DEFAULTS` fills the rest; any run field or `env` entry the file states wins. A
    stage assembly without layout `requote` runs `continue`. With
    `adaptive_draft_length` off a run drafts only under `suffix_decoding: true`, which
    verifies every draft as proposed.

    Args:
        calibrated: Whether to read what `calibrate` wrote. Off, the fields are those
            of the one engine a dry run sizes or calibration measures, with no step
            model and no P/D block.
    """
    stated = doc
    doc = dict(doc)
    missing = [name for name in ("model", "tasks") if name not in doc]
    if missing:
        raise ConfigError(f"missing required config field(s): {', '.join(missing)}")
    doc.pop("calibration_dir", None)
    tasks = doc.pop("tasks")
    program = doc.pop("program", PROGRAM_THREE_STAGE)
    reuse = _switch(doc, "context_reuse")
    drafts = _switch(doc, "adaptive_draft_length")
    pd_block = doc.pop("pd", None)
    pd_method = doc.pop("pd_method", None) or {}
    draft_settings = doc.pop("adaptive_draft", None) or {}

    suffix_decoding = doc.get("suffix_decoding", drafts)
    if drafts and not suffix_decoding:
        raise ConfigError("adaptive_draft_length needs suffix_decoding")
    if pd_block:
        _check_pd_block(pd_block)
        if not reuse:
            raise ConfigError("locality-aware P/D runs on context reuse")

    samples = task_files(tasks)
    gen_tokens = doc.get("gen_tokens", DEFAULT_GEN_TOKENS)
    video = _video_fields(samples)
    fields: dict[str, Any] = {
        **RUN_DEFAULTS,
        **video,
        **_program_fields(program, gen_tokens),
        "samples": samples,
        "bench_root": stage_assembly_root(tasks),
        "gen_tokens": gen_tokens,
        "max_tokens_quality": gen_tokens,
        "suffix_decoding": suffix_decoding,
    }
    if reuse:
        fields.update(
            scheduler_cls=CONTEXT_REUSE_SCHEDULER,
            scheduling_policy=SCHEDULING_PRIORITY,
            stage_priority=STAGE_PRIORITY_SUCCESSOR_CLASS,
        )
    fields.update(doc)
    if video:
        # Clips need the vision tower, whatever the file sets for text tasks.
        fields["language_model_only"] = None
    if PROMPT_LAYOUT_REQUOTE not in stage_assembly_layouts(fields["bench_root"]):
        if doc.get("layout") == PROMPT_LAYOUT_REQUOTE:
            raise ConfigError(
                f"layout is requote, but the stage assembly under "
                f"{fields['bench_root']} does not build it: state `layout: continue`"
            )
        fields["layout"] = doc.get("layout", PROMPT_LAYOUT_CONTINUE)
    if program == PROGRAM_TWO_STAGE:
        fields["refinement"] = _merged_refinement(fields["refinement"])
    fields["kv_cache_dtype"] = engine_kv_cache_dtype(doc)
    fields["env"] = {
        **RUN_ENV,
        **(DRAFTING_ENV if suffix_decoding else {}),
        **(PD_ENV if pd_block else {}),
        **env_mapping(doc.get("env")),
    }
    programs = len(samples) * len(fields["decode_paths"])
    fields.setdefault("program_total", programs)
    fields.setdefault("concurrency", programs)
    if not calibrated:
        if pd_block:
            fields["max_num_seqs"] = pd_block["max_num_seqs"]
        return fields
    directory = calibration_dir(stated)
    if drafts:
        fields["adaptive_draft"] = step_model_block(directory, draft_settings)
    if pd_block:
        fields.pop("scheduler_cls", None)
        fields.update(
            _pd_fields(pd_block, pd_method, directory, fields["tensor_parallel_size"])
        )
    return fields


@dataclass(frozen=True)
class RunConfig:
    """One invocation: one engine, one or more programs on top of it.

    An engine field left None keeps vLLM's default.
    """

    model: str
    samples: tuple[str, ...]
    gen_tokens: int
    max_model_len: int

    bench_root: str = ""
    decode_paths: tuple[str, ...] = ("quality",)
    # A prefix of `PIPELINE_STAGES`, or generation and refinement in a two-stage run.
    stages: tuple[str, ...] = PIPELINE_STAGES
    # See `PIPELINE_MODES`; `max_rounds` bounds the driver alone, never the controller.
    pipeline_mode: str = PIPELINE_MODE_FIXED
    max_rounds: int = DEFAULT_MAX_ROUNDS
    # Refinement's cap on the `gen_tokens` paths; None keeps `gen_tokens`. A merged
    # refinement writes its review and the whole revision, so it may need more.
    refinement_gen_tokens: int | None = None
    # Programs in flight at once.
    concurrency: int = 1
    # Declared, a program's id and cache salt depend only on its place in the set;
    # None keeps the concurrency in the ids.
    program_total: int | None = None
    # Queue order and stage priority change no prompt, but decide which prefix is
    # still resident when a program's next stage arrives.
    scheduling_policy: str = SCHEDULING_FCFS
    scheduler_cls: str | None = None
    stage_priority: str = STAGE_PRIORITY_NONE
    max_tokens_quality: int = 4096
    # Greedy by default; sampling avoids the repetition loops long greedy runs enter.
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0  # vLLM's "off"
    # Sample program i under `seed + i` instead of every request under `seed`.
    seed_per_program: bool = False
    enable_prefix_caching: bool = False
    # One salt per program, so concurrent copies of a sample never hit each other.
    isolate_programs: bool = True
    enable_thinking: bool = False
    # The chat-template variable that carries `enable_thinking`.
    thinking_kwarg: str = "enable_thinking"
    gpu_memory_utilization: float | None = None
    max_num_batched_tokens: int | None = None
    # On, vLLM admits a request only if its whole input fits; off, only its first chunk.
    scheduler_reserve_full_isl: bool | None = None
    # Unset, derived from the measured pool (`engine.runner.derive_max_num_seqs`).
    max_num_seqs: int | None = None
    cudagraph_mode: str | None = None
    async_scheduling: bool | None = None
    enforce_eager: bool | None = None
    moe_backend: str | None = None
    # The prefill kernel of gated-delta-net layers in hybrid models.
    gdn_prefill_backend: str | None = None
    # Skip a multimodal checkpoint's vision tower.
    language_model_only: bool | None = None
    kv_cache_dtype: str = "auto"
    flash_attn_version: int | None = None
    attention_backend: str | None = None
    suffix_decoding: bool = False
    suffix_num_speculative_tokens: int = 24
    suffix_max_tree_depth: int = 24
    suffix_max_cached_requests: int = 0
    # Adaptive draft length; None verifies every suffix draft as proposed.
    adaptive_draft: AdaptiveDraftConfig | None = None
    tensor_parallel_size: int | None = None
    trust_remote_code: bool | None = None
    # `MULTIMODAL_FIELDS`; `mm_processor_kwargs` also travels with every stage request.
    mm_processor_kwargs: dict[str, Any] | None = None
    limit_mm_per_prompt: dict[str, Any] | None = None
    mm_processor_cache_gb: float | None = None
    allowed_local_media_path: str | None = None
    # YaRN, set by `rope_scaling_factor`; the other two default to the checkpoint's.
    rope_scaling_factor: float | None = None
    rope_original_max_positions: int | None = None
    rope_theta: float | None = None
    feedback: str = "mustfix"
    refinement: str = "full"
    # See `PROMPT_LAYOUTS`.
    layout: str = PROMPT_LAYOUT_CONTINUE
    # `NAME=VALUE` entries set before the engine starts; a field, so environment
    # switches reach the fingerprint.
    env: tuple[str, ...] = ()
    # On-demand price of one GPU-hour, for the summary's $ per 1M tokens.
    gpu_hour_usd: float | None = None
    # Absent, not disabled, when a run says nothing about it. See `as_dict`.
    pd: PDConfig | None = None
    pd_method: PolicyConfig | None = None
    out_dir: str = "results"
    seed: int = 0

    def __post_init__(self) -> None:
        """Refuse an unknown setting or a concurrency below one; put `env` in order.

        Sorted, two spellings of one `env` set hash alike; `fingerprint()` keeps the
        order of every other sequence.
        """
        for name, known in (
            ("stage_priority", STAGE_PRIORITIES),
            ("pipeline_mode", PIPELINE_MODES),
            ("layout", PROMPT_LAYOUTS),
            ("scheduling_policy", SCHEDULING_POLICIES),
            ("kv_cache_dtype", KV_CACHE_DTYPES),
            ("scheduler_cls", SCHEDULER_CLASSES),
        ):
            value = getattr(self, name)
            if value not in known:
                raise ConfigError(f"unknown {name} {value!r}; expected {list(known)}")
        unknown = [p for p in self.decode_paths if p not in DECODE_PATHS]
        if unknown:
            raise ConfigError(
                f"unknown decode_paths {unknown}; expected any of {list(DECODE_PATHS)}"
            )
        concurrency = self.concurrency
        if isinstance(concurrency, bool) or not isinstance(concurrency, int):
            raise ConfigError(f"concurrency must be an integer, got {concurrency!r}")
        if concurrency < 1:
            raise ConfigError(f"concurrency must be at least 1, got {concurrency}")
        entries = tuple(self.env)
        if all(isinstance(entry, str) for entry in entries):
            entries = tuple(sorted(entries))
        object.__setattr__(self, "env", entries)

    def as_dict(self) -> dict[str, Any]:
        """The fingerprint's payload: every field, less optional ones at their defaults.

        An absent `pd` block is dropped, while `{"enabled": false}` stays: stating P/D
        off is a different config from never mentioning it.
        """
        payload = dataclasses.asdict(self)
        if self.scheduler_cls is None:
            payload.pop("scheduler_cls")
        if self.pd_method is None:
            payload.pop("pd_method")
        if self.pd is None:
            payload.pop("pd")
        else:
            from turboreflect.locality_pd.config import (
                ROUTING_FIELDS,
                SCHEDULING_FIELDS,
            )

            if self.pd.engines is None:
                payload["pd"].pop("engines")
            for name in (*ROUTING_FIELDS, *SCHEDULING_FIELDS):
                if getattr(self.pd, name) is None:
                    payload["pd"].pop(name)
        if self.program_total is None:
            payload.pop("program_total")
        if not self.seed_per_program:
            payload.pop("seed_per_program")
        if self.scheduling_policy == SCHEDULING_FCFS:
            payload.pop("scheduling_policy")
        if self.stage_priority == STAGE_PRIORITY_NONE:
            payload.pop("stage_priority")
        if self.pipeline_mode == PIPELINE_MODE_FIXED:
            payload.pop("pipeline_mode")
        if self.max_rounds == DEFAULT_MAX_ROUNDS:
            payload.pop("max_rounds")
        if self.refinement_gen_tokens is None:
            payload.pop("refinement_gen_tokens")
        if self.layout == PROMPT_LAYOUT_CONTINUE:
            payload.pop("layout")
        if self.kv_cache_dtype == "auto":
            payload.pop("kv_cache_dtype")
        if not self.env:
            payload.pop("env")
        if self.adaptive_draft is None:
            payload.pop("adaptive_draft")
        if not self.suffix_decoding:
            for name in (
                "suffix_decoding",
                "suffix_num_speculative_tokens",
                "suffix_max_tree_depth",
                "suffix_max_cached_requests",
            ):
                payload.pop(name)
        for name in (
            "enforce_eager",
            "moe_backend",
            "gdn_prefill_backend",
            "language_model_only",
            "scheduler_reserve_full_isl",
            "flash_attn_version",
            "attention_backend",
            "tensor_parallel_size",
            "trust_remote_code",
            "gpu_hour_usd",
            *MULTIMODAL_FIELDS,
        ):
            if getattr(self, name) is None:
                payload.pop(name)
        return payload

    def gpu_count(self) -> int:
        """The GPUs this run allocates: every engine's, or the one engine's."""
        if self.pd is None or not self.pd.enabled:
            return self.tensor_parallel_size or 1
        settings = self.pd.settings()
        return len({d for name in settings.names() for d in settings.devices(name)})

    def sample_path(self, sample: str | Path) -> Path:
        """Resolve a `samples` entry against `bench_root` unless it is absolute."""
        p = Path(sample)
        return p if p.is_absolute() else Path(self.bench_root) / p

    def sample_digest(self) -> str:
        """A digest of what the sample files and their clips hold.

        Without it a bench regenerated in place would keep its fingerprint. Empty when
        a sample is missing, so a dry run's config still has a fingerprint.
        """
        digest = hashlib.sha256()
        for path in self.samples:
            digest.update(str(path).encode("utf-8"))
            try:
                resolved = self.sample_path(path)
                digest.update(resolved.read_bytes())
                for index, video in enumerate(video_paths(resolved)):
                    digest.update(f"\nvideo {index}:{file_sha256(video)}".encode())
            except (OSError, SampleError):
                return ""
        return digest.hexdigest()

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        payload += "\nsamples:" + self.sample_digest()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def stage_gen_tokens(self, base: str, decode_path: str) -> int:
        """The most tokens stage `base` (named without its round) may generate."""
        if decode_path not in FINAL_ONLY_PATHS:
            return self.max_tokens_quality
        if base == "refinement" and self.refinement_gen_tokens is not None:
            return self.refinement_gen_tokens
        return self.gen_tokens

    @property
    def two_stage(self) -> bool:
        """Whether one refinement call reviews and revises, with no feedback stage."""
        return self.refinement in MERGED_REFINEMENTS

    def stage_priority_value(
        self,
        *,
        successor: bool = False,
        program_index: int = 0,
    ) -> int:
        """The vLLM priority of one request; successors go before new programs."""
        if self.stage_priority == STAGE_PRIORITY_NONE:
            return 0
        return (0 if successor else 1) * STAGE_CLASS_STRIDE + program_index

    def request_seed(self, program_index: int = 0) -> int | None:
        """The sampling seed of one program's requests; None for a greedy run."""
        if self.temperature <= 0:
            return None
        if self.seed_per_program:
            return self.seed + program_index
        return self.seed

    def program_count(self) -> int:
        return len(self.samples) * len(self.decode_paths)

    def program_slot(self, program_index: int) -> tuple[str, str]:
        """The `samples` entry and decode path of one program, in sample-major order."""
        sample, path = divmod(program_index, len(self.decode_paths))
        return self.samples[sample], self.decode_paths[path]

    def program_id(self, sample_id: str, decode_path: str, n: int) -> str:
        """A program's id, which is also what its cache salt is built on."""
        if self.program_total is not None:
            return f"{sample_id}-{decode_path}-r01"
        return f"{sample_id}-{decode_path}-n{n}"

    def cache_salt(self, run_id: str, program_id: str) -> str:
        """A program's cache salt: the bare id when the run declares `program_total`.

        The id suffices within one engine, as warmups run under ids of their own
        (`engine.runner.warmup_program_id`).
        """
        return program_cache_salt(self.program_total, run_id, program_id)

    def hf_overrides(self) -> dict[str, Any]:
        """The YaRN override that runs the model past its native window, if set.

        `rope_theta` is repeated: the override replaces the whole rope section.
        """
        if self.rope_scaling_factor is None:
            return {}
        checkpoint = checkpoint_config(self.model) or {}
        return {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": self.rope_scaling_factor,
                "original_max_position_embeddings": self.rope_original_max_positions
                or checkpoint["max_position_embeddings"],
            },
            "rope_theta": float(self.rope_theta or checkpoint["rope_theta"]),
        }

    def env_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for entry in self.env:
            name, _, value = str(entry).partition("=")
            out[name] = value
        return out

    def _architectures(self) -> tuple[str, ...] | None:
        return checkpoint_architectures(self.model)

    def has_recurrent_state(self) -> bool:
        """Whether the checkpoint keeps Mamba-type state; False when not on disk."""
        from vllm.model_executor.models.registry import ModelRegistry

        for architecture in self._architectures() or ():
            info = ModelRegistry._try_inspect_model_cls(architecture)
            if info is not None:
                return info.is_hybrid
        return False

    def pd_p_speculates(self) -> bool:
        """Whether P engines run suffix decoding (`PolicyConfig.speculates_on_p`)."""
        return self.suffix_decoding and bool(
            getattr(self.pd_method, "speculates_on_p", False)
        )

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Attach the config flags; SUPPRESS tells unpassed flags from defaults."""
        add = parser.add_argument
        add("--config", default=None, help="a YAML or JSON run configuration")
        add("--model", default=argparse.SUPPRESS, help="the model to run")
        add(
            "--tasks",
            default=argparse.SUPPRESS,
            help="a directory of task files, or one task file",
        )
        add("--out-dir", default=argparse.SUPPRESS, help="where results are written")
        add(
            "--set",
            dest="overrides",
            action="append",
            default=argparse.SUPPRESS,
            metavar="NAME=VALUE",
            help="set any config field, a dotted NAME reaching into a block; repeat "
            "per field",
        )
        return parser

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> RunConfig:
        values = config_document(args)
        if "samples" not in values:
            values = expand_switches(
                values, calibrated=not getattr(args, "dry_run", False)
            )
        unknown = sorted(set(values) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise ConfigError(f"unknown config field(s): {unknown}")

        missing = [k for k in _REQUIRED if k not in values]
        if missing:
            raise ConfigError(
                "missing required config field(s): "
                + ", ".join(missing)
                + " -- supply them in --config or with --set"
            )

        block = values.get("env")
        if isinstance(block, dict):
            values["env"] = [f"{name}={value}" for name, value in block.items()]

        for key in (
            "samples",
            "decode_paths",
            "stages",
            "env",
        ):
            if key in values:
                values[key] = tuple(values[key])

        from turboreflect.locality_pd.config import PDConfig

        values["adaptive_draft"] = AdaptiveDraftConfig.resolve(
            values.get("adaptive_draft")
        )
        values["pd"] = PDConfig.resolve(values.get("pd"))
        from turboreflect.locality_pd.config import PolicyConfig

        block = values.get("pd_method")
        values["pd_method"] = (
            None if block is None else PolicyConfig.resolve(block, values["pd"])
        )
        if values["pd_method"] is not None:
            values["pd"] = values["pd_method"].resolve_pd(values["pd"])
        pd = values["pd"]
        if pd is not None and pd.enabled and values["pd_method"] is None:
            raise ConfigError(
                "a pd block needs a pd_method block: stages are only placed "
                "by a pd_method, there is no placement without one"
            )

        return cls(**values)
