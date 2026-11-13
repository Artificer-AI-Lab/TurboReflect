# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""RULER as reflection programs: build the tasks, export the answers.

    python -m turboreflect.benchmarks.ruler build --assets <dir> --out tasks/ruler
    python -m turboreflect.benchmarks.ruler export runs/ruler --out preds/ruler

The items are RULER's own (arXiv:2404.06654): parameters from its `synthetic.yaml`,
prompts from its `constants.py` templates with `answer_prefix` appended, the form
NeMo-Skills' `rulerv1-ns` hands a chat model. Four of the thirteen tasks:

    niah_multikey_3   one uuid key/value pair among other key/value lines
    niah_multivalue   one word key with four number values in the essay haystack
    vt                one four-hop variable-assignment chain in the noise haystack
    cwe               a numbered word list; ten words appear 30 times, the rest 3

`build` writes `data/ruler_<task>_<idx>_<tier>.json` with the haystack in `context`,
gold answers under `rubrics/`, and `prompts.py`, seeded from `(seed, task, idx)`.
Haystacks are RULER's own (`ESSAY_ASSET`, `WORDS_ASSET`, fetched with its script into
`--assets`), sized with `--tokenizer`, the same for every model.

`export` writes `<out>/generation/` and `<out>/refinement/`, one `<task>.jsonl` each
with the fields RULER's `scripts/eval/evaluate.py` reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

from turboreflect.benchmarks.common import (
    CONTEXT_SLOT,
    EXPORTED_STAGES,
    read_rubric,
    read_run,
    task_folder,
    write_json,
    write_jsonl,
)

ESSAY_ASSET = "PaulGrahamEssays.json"
WORDS_ASSET = "english_words.json"

SIZING_TOKENIZER = "Qwen/Qwen3-14B"

# Haystack tokens per tier (`SIZING_TOKENIZER`); the rendered prompt is longer.
TIER_TARGET_TOKENS: dict[str, int] = {
    "32k": 30_000,
    "128k": 118_000,
}

TIERS = tuple(TIER_TARGET_TOKENS)

# Items per task per tier.
PER_TASK = 50


# RULER's `constants.py` templates, verbatim. `{context}` is left unsubstituted;
# `CONTEXT_SLOT` is that same string. Adapted from https://github.com/NVIDIA/RULER
# (Apache-2.0), see THIRD_PARTY_NOTICES.md.

NIAH_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} "
    "afterwards.\n{context}\nWhat are all the special magic {type_needle_v} "
    "for {query} mentioned in the provided text?"
)
NIAH_ANSWER_PREFIX = (
    " The special magic {type_needle_v} for {query} mentioned in the provided text are"
)

VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the "
    "following text.\n\n{context}\nQuestion: Find all variables that are "
    "assigned the value {query} in the text above."
)
VT_ANSWER_PREFIX = (
    " Answer: According to the chain(s) of variable assignment in the text "
    "above, {num_v} variables are assigned the value {query}, they are: "
)

CWE_TEMPLATE = (
    "Below is a numbered list of words. In these words, some appear more often "
    "than others. Memorize the ones that appear most often.\n{context}\n"
    "Question: What are the 10 most common words in the above list?"
)
CWE_ANSWER_PREFIX = " Answer: The top 10 words that appear most often in the list are:"

# `type_needle_v` stays plural in the needle even where the question is singular.
NEEDLE_TEMPLATE = "One of the special magic {type_needle_v} for {key} is: {value}."

# One filler unit is the whole five-sentence string.
NOISE = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again."
)

# RULER's depth grid: forty evenly spaced percentages.
DEPTHS = tuple(round(100 * i / 39) for i in range(40))

# `synthetic.yaml`: niah_multivalue.
MV_NUM_NEEDLE_V = 4

# `synthetic.yaml`: vt. One chain of four hops, which is five variables.
VT_NUM_HOPS = 4

# `synthetic.yaml`: cwe, and its in-context example for tiers past 4 096 tokens.
CWE_FREQ_CW = 30
CWE_FREQ_UCW = 3
CWE_NUM_CW = 10
CWE_EXAMPLE_WORDS = 40
CWE_EXAMPLE_FREQ_CW = 10
CWE_EXAMPLE_FREQ_UCW = 3

# RULER's vt example: noise grown five units at a time within 500 tokens.
VT_EXAMPLE_NAME_LENGTH = 3
VT_EXAMPLE_MAX_TOKENS = 500
VT_EXAMPLE_NOISE_STEP = 5
VT_NAME_LENGTH = 5


class BuildError(SystemExit):
    """A build that cannot produce the set it was asked for."""


class Tok:
    """Raw-text token counts: what sizes a haystack."""

    def __init__(self, name: str) -> None:
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self._t = tokenizer.backend_tokenizer

    def count(self, text: str) -> int:
        return len(self._t.encode(text, add_special_tokens=False).ids)

    def counts(self, texts: Iterable[str]) -> list[int]:
        return [len(e.ids) for e in self._t.encode_batch_fast(list(texts))]


def resolve_asset(name: str, asset_dir: str) -> Path:
    """One of RULER's haystack files; `BuildError` if it is missing."""
    path = Path(asset_dir).expanduser() / name
    if not path.is_file():
        raise BuildError(f"{path} not found: fetch RULER's own haystack material")
    return path


@dataclass
class Material:
    """RULER's haystack material, read once and shared by every item.

    `cwe_words` is the `wonderwords` pool; `long_words` is the larger list `cwe`
    falls back to once a tier needs more distinct words.
    """

    essay_words: tuple[str, ...]
    adjectives: tuple[str, ...]
    nouns: tuple[str, ...]
    cwe_words: tuple[str, ...]
    long_words: tuple[str, ...]


def load_material(asset_dir: str, seed: int) -> Material:
    """Read the essay corpus and RULER's `wonderwords` pools; `BuildError` without."""
    try:
        import wonderwords  # noqa: PLC0415
    except ImportError as exc:
        raise BuildError(
            "wonderwords is not installed; it is RULER's own word source. "
            "uv pip install wonderwords"
        ) from exc

    essay = json.loads(resolve_asset(ESSAY_ASSET, asset_dir).read_text())["text"]
    essay_words = tuple(re.sub(r"\s+", " ", essay).split(" "))

    get_words = wonderwords.random_word._get_words_from_text_file
    nouns = get_words("nounlist.txt")
    adjectives = get_words("adjectivelist.txt")
    verbs = get_words("verblist.txt")
    cwe_words = sorted(set(nouns + adjectives + verbs))
    random.Random(seed).shuffle(cwe_words)

    long_words = tuple(
        json.loads(resolve_asset(WORDS_ASSET, asset_dir).read_text()).values()
    )
    return Material(
        essay_words=essay_words,
        adjectives=tuple(adjectives),
        nouns=tuple(nouns),
        cwe_words=tuple(cwe_words),
        long_words=long_words,
    )


def sentences_of(words: tuple[str, ...], count: int) -> list[str]:
    """The first `count` essay words, split into sentences by NLTK's `punkt`."""
    from nltk.tokenize import sent_tokenize  # noqa: PLC0415

    if count <= len(words):
        text = " ".join(words[:count])
    else:
        repeats = -(-count // len(words))
        text = " ".join((words * repeats)[:count])
    return sent_tokenize(text.strip())


def item_seed(seed: int, task: str, index: int) -> int:
    """A per-item seed stable across processes, unlike salted `hash()`."""
    digest = hashlib.sha256(f"{seed}:{task}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def a_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def a_number(rng: random.Random, digits: int = 7) -> str:
    return str(rng.randint(10 ** (digits - 1), 10**digits - 1))


def a_word(rng: random.Random, material: Material) -> str:
    """RULER's word key: an adjective and a noun, hyphenated.

    Drawn independently, the same distribution as RULER's sample of the product.
    """
    return f"{rng.choice(material.adjectives)}-{rng.choice(material.nouns)}"


def depths(rng: random.Random, count: int) -> list[int]:
    """`count` of RULER's depth percentages, sorted."""
    return sorted(rng.sample(DEPTHS, count))


def singular(text: str) -> str:
    """RULER's singular rewrite, applied where one value is asked for."""
    rewrites = (("Some", "A"), ("are all", "is"), ("are", "is"), ("answers", "answer"))
    for old, new in rewrites:
        text = text.replace(old, new)
    return text


def fit_to_budget(
    make: Callable[[int], str],
    tok: Tok,
    budget: int,
    first_guess: int,
    tolerance: float = 0.02,
    max_iter: int = 6,
) -> tuple[int, str, int]:
    """`(n, document, tokens)` for the largest `n` whose `make(n)` fits `budget`.

    `make` must be deterministic and monotone in `n`, the filler units around a
    fixed set of needles; proportional correction lands in two or three tries.

    Args:
        tolerance: How far under `budget` an accepted document may land.
        max_iter: Measurements to spend before taking the best seen.

    Raises:
        BuildError: If no `n` fits, i.e. the needles alone exceed the tier.
    """
    n = max(1, first_guess)
    best: tuple[int, str, int] | None = None
    for _ in range(max_iter):
        text = make(n)
        got = tok.count(text)
        if got <= budget:
            if best is None or got > best[2]:
                best = (n, text, got)
            if got >= budget * (1 - tolerance):
                break
            scaled = int(n * budget / max(got, 1))
            n = max(n + 1, scaled)
        else:
            n = max(1, int(n * budget / got * 0.98))
            if best is not None and n <= best[0]:
                break
    if best is None:
        raise BuildError(
            f"nothing fits {budget} tokens -- the needles alone are longer "
            "than the tier"
        )
    return best


def first_guess(tok: Tok, sample_units: list[str], budget: int) -> int:
    """Unit count to start `fit_to_budget` from, from a sample's average cost."""
    total = sum(tok.counts(sample_units)) + len(sample_units)
    average = max(1.0, total / len(sample_units))
    return max(1, int(budget / average))


@dataclass(frozen=True)
class Item:
    """One built item, before it is split into a task and a rubric."""

    document: str
    user_task: str
    gold: list[str]
    query: str


def _insert_into_sentences(
    sentences: list[str], needles: list[str], places: list[int]
) -> str:
    """RULER's essay insertion: needles between sentences, at given depths."""
    cuts = [0] + [int(len(sentences) * depth / 100) for depth in places]
    cuts.append(len(sentences))
    out: list[str] = []
    for i in range(1, len(cuts)):
        out.append(" ".join(sentences[cuts[i - 1] : cuts[i]]))
        if i - 1 < len(needles):
            out.append(needles[i - 1])
    return " ".join(out)


def _niah_multikey_3(
    rng: random.Random, tok: Tok, material: Material, budget: int
) -> Item:
    key = a_uuid(rng)
    value = a_uuid(rng)
    needle = NEEDLE_TEMPLATE.format(type_needle_v="uuids", key=key, value=value)
    place = rng.choice(DEPTHS)
    # Re-seeding the distractor stream per call keeps `make(n)` monotone in `n`.
    filler_seed = rng.getrandbits(64)

    def make(n: int) -> str:
        filler = random.Random(filler_seed)
        units = [
            NEEDLE_TEMPLATE.format(
                type_needle_v="uuids", key=a_uuid(filler), value=a_uuid(filler)
            )
            for _ in range(n)
        ]
        units.insert(min(n, int(n * place / 100)), needle)
        return "\n".join(units)

    sample = random.Random(filler_seed)
    pool = [
        NEEDLE_TEMPLATE.format(
            type_needle_v="uuids", key=a_uuid(sample), value=a_uuid(sample)
        )
        for _ in range(8)
    ]
    _, document, _ = fit_to_budget(make, tok, budget, first_guess(tok, pool, budget))
    # num_needle_q * num_needle_v == 1, so the benchmark asks in the singular.
    template = singular(NIAH_TEMPLATE + NIAH_ANSWER_PREFIX)
    user_task = template.format(type_needle_v="uuid", context=CONTEXT_SLOT, query=key)
    return Item(document, user_task, [value], key)


def _niah_multivalue(
    rng: random.Random, tok: Tok, material: Material, budget: int
) -> Item:
    key = a_word(rng, material)
    values = [a_number(rng) for _ in range(MV_NUM_NEEDLE_V)]
    needles = [
        NEEDLE_TEMPLATE.format(type_needle_v="numbers", key=key, value=v)
        for v in values
    ]
    rng.shuffle(needles)
    places = depths(rng, len(needles))
    cache: dict[int, list[str]] = {}

    def make(n: int) -> str:
        if n not in cache:
            cache[n] = sentences_of(material.essay_words, n)
        return _insert_into_sentences(cache[n], needles, places)

    _, document, _ = fit_to_budget(make, tok, budget, int(budget * 0.72))
    user_task = (NIAH_TEMPLATE + NIAH_ANSWER_PREFIX).format(
        type_needle_v="numbers", context=CONTEXT_SLOT, query=key
    )
    return Item(document, user_task, list(values), key)


def _chain(
    rng: random.Random, name_length: int, value: str
) -> tuple[list[str], list[str]]:
    """RULER's assignment chain: `(variable names, the lines that set them)`."""
    names: list[str] = []
    while len(names) < VT_NUM_HOPS + 1:
        name = "".join(rng.choice(string.ascii_uppercase) for _ in range(name_length))
        if name not in names:
            names.append(name)
    lines = [f"VAR {names[0]} = {value}"]
    for previous, name in zip(names, names[1:]):
        lines.append(f"VAR {name} = VAR {previous} ")
    return names, lines


def _noise_with(lines: list[str], places: list[int], n: int) -> str:
    """RULER's noise haystack: `n` filler units with the chain inserted in order."""
    units = [NOISE] * n
    for offset, (line, depth) in enumerate(zip(lines, places)):
        units.insert(min(len(units), int(n * depth / 100) + offset), line)
    return "\n".join(units).replace(". \n", ".\n")


def _vt_example(rng: random.Random, tok: Tok) -> str:
    """RULER's one in-context example: the same task, small, with its answer."""
    value = str(rng.randrange(10_000, 100_000))
    names, lines = _chain(rng, VT_EXAMPLE_NAME_LENGTH, value)
    places = depths(rng, len(lines))

    def rendered(n: int) -> str:
        return (VT_TEMPLATE + VT_ANSWER_PREFIX).format(
            context=_noise_with(lines, places, n),
            query=value,
            num_v=VT_NUM_HOPS + 1,
        )

    n = VT_EXAMPLE_NOISE_STEP
    while tok.count(rendered(n + VT_EXAMPLE_NOISE_STEP)) <= VT_EXAMPLE_MAX_TOKENS:
        n += VT_EXAMPLE_NOISE_STEP
    return f"{rendered(n)} {' '.join(names)}\n"


def _vt(rng: random.Random, tok: Tok, material: Material, budget: int) -> Item:
    example = _vt_example(rng, tok)
    value = str(rng.randrange(10_000, 100_000))
    names, lines = _chain(rng, VT_NAME_LENGTH, value)
    places = depths(rng, len(lines))

    def make(n: int) -> str:
        return _noise_with(lines, places, n)

    _, document, _ = fit_to_budget(make, tok, budget, first_guess(tok, [NOISE], budget))
    body = (VT_TEMPLATE + VT_ANSWER_PREFIX).format(
        context=CONTEXT_SLOT, query=value, num_v=VT_NUM_HOPS + 1
    )
    return Item(document, example + "\n" + body, list(names), value)


def _numbered(words: list[str]) -> str:
    return " ".join(f"{i + 1}. {word}" for i, word in enumerate(words))


def _cwe_list(
    rng: random.Random,
    pool: list[str],
    distinct: int,
    freq_cw: int,
    freq_ucw: int,
    order: int,
) -> tuple[str, list[str]]:
    """`(the numbered list, the ten common words)`."""
    chosen = rng.sample(pool, distinct)
    common, uncommon = chosen[:CWE_NUM_CW], chosen[CWE_NUM_CW:]
    words = common * freq_cw + uncommon * freq_ucw
    random.Random(order).shuffle(words)
    return _numbered(words), common


def _cwe(rng: random.Random, tok: Tok, material: Material, budget: int) -> Item:
    order = rng.getrandbits(64)
    example_list, example_common = _cwe_list(
        rng,
        list(material.cwe_words),
        CWE_EXAMPLE_WORDS,
        CWE_EXAMPLE_FREQ_CW,
        CWE_EXAMPLE_FREQ_UCW,
        order,
    )
    example = CWE_TEMPLATE.format(
        num_cw=CWE_NUM_CW, context=example_list, query=""
    ) + CWE_ANSWER_PREFIX.format(num_cw=CWE_NUM_CW)
    example = f"{example} {_numbered(example_common)}"

    item_seeds = rng.getrandbits(64)
    common: list[str] = []

    def make(n: int) -> str:
        # RULER's fallback to the larger list, needed by every tier past ~16k.
        pool = (
            list(material.cwe_words)
            if n <= len(material.cwe_words)
            else list(material.long_words)
        )
        if n > len(pool):
            raise BuildError(
                f"cwe: {n} distinct words needed, {WORDS_ASSET} holds "
                f"{len(pool)}; a reused word would pass {CWE_FREQ_UCW} "
                "occurrences and the gold answer would be wrong"
            )
        text, chosen = _cwe_list(
            random.Random(item_seeds), pool, n, CWE_FREQ_CW, CWE_FREQ_UCW, order
        )
        common[:] = chosen
        return text

    floor = CWE_NUM_CW + 1
    n, document, _ = fit_to_budget(make, tok, budget, max(floor, int(budget / 15)))
    make(n)
    body = CWE_TEMPLATE.format(
        num_cw=CWE_NUM_CW, context=CONTEXT_SLOT, query=""
    ) + CWE_ANSWER_PREFIX.format(num_cw=CWE_NUM_CW)
    return Item(document, example + "\n" + body, list(common), "10 most common")


TASK_BUILDERS: dict[str, Callable[..., Item]] = {
    "niah_multikey_3": _niah_multikey_3,
    "niah_multivalue": _niah_multivalue,
    "vt": _vt,
    "cwe": _cwe,
}

TASKS = tuple(TASK_BUILDERS)


def sample_id(task: str, index: int, tier: str) -> str:
    return f"ruler_{task}_{index:02d}_{tier}"


def make_task(name: str, index: int, tier: str, item: Item) -> dict[str, Any]:
    """What the model sees. The gold answers are in `make_rubric`, never here."""
    return {
        "id": sample_id(name, index, tier),
        "system": "",
        "user_task": item.user_task,
        "context": {"document": item.document},
        "rubric": [],
    }


def make_rubric(name: str, index: int, tier: str, item: Item) -> dict[str, Any]:
    """The gold answers, under the task's own id: `cwe`'s differ between tiers."""
    return {
        "id": sample_id(name, index, tier),
        "task": name,
        "index": index,
        "answer": list(item.gold),
        "query": item.query,
    }


def prune(out_dir: Path, rubric_dir: Path, tier: str, keep: set[str]) -> None:
    """Delete this tier's tasks and rubrics whose ids are not in `keep`."""
    for directory in (out_dir, rubric_dir):
        for path in sorted(directory.glob(f"ruler_*_{tier}.json")):
            if path.stem not in keep:
                path.unlink()


def build_tier(
    tier: str,
    *,
    tok: Tok,
    material: Material,
    per_task: int,
    seed: int,
    tasks: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every item of one tier, as `(task, rubric)`."""
    budget = TIER_TARGET_TOKENS[tier]
    out = []
    for name in tasks:
        for index in range(per_task):
            rng = random.Random(item_seed(seed, name, index))
            item = TASK_BUILDERS[name](rng, tok, material, budget)
            out.append(
                (
                    make_task(name, index, tier, item),
                    make_rubric(name, index, tier, item),
                )
            )
        print(f"  {tier} {name}: {per_task} items", flush=True)
    return out


def build(args: argparse.Namespace) -> None:
    """Write the task folder `args.out`."""
    data_dir, rubric_dir = task_folder(Path(args.out))
    tok = Tok(args.tokenizer)
    material = load_material(args.assets, args.seed)
    for tier in args.tiers:
        built = build_tier(
            tier,
            tok=tok,
            material=material,
            per_task=args.per_task,
            seed=args.seed,
            tasks=args.tasks,
        )
        for task, rubric in built:
            write_json(data_dir / f"{task['id']}.json", task)
            write_json(rubric_dir / f"{rubric['id']}.json", rubric)
        prune(data_dir, rubric_dir, tier, {task["id"] for task, _ in built})
        print(f"{tier}: {len(built)} tasks -> {data_dir}")


def export(args: argparse.Namespace) -> None:
    """Write each stage's answers as `{out}/{stage}/{task}.jsonl`.

    `evaluate.py` only prints `input`, so it holds the item's query.
    """
    root, answers = read_run(args.run, args.tasks)
    rubrics = [(read_rubric(root, task_id), task_id) for task_id in sorted(answers)]
    for stage in EXPORTED_STAGES:
        for name in TASKS:
            rows = [
                {
                    "index": rubric["index"],
                    "input": rubric["query"],
                    "pred": answers[task_id][stage],
                    "outputs": rubric["answer"],
                    "others": {},
                }
                for rubric, task_id in rubrics
                if rubric["task"] == name
            ]
            if rows:
                write_jsonl(Path(args.out) / stage / f"{name}.jsonl", rows)
    print(f"{len(answers)} programs -> {args.out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m turboreflect.benchmarks.ruler")
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("build", help="write a task folder")
    make.add_argument(
        "--assets", required=True, help=f"where {ESSAY_ASSET} and {WORDS_ASSET} are"
    )
    make.add_argument("--out", required=True, help="the task folder")
    make.add_argument(
        "--tokenizer", default=SIZING_TOKENIZER, help="what sizes the haystacks"
    )
    make.add_argument("--tiers", nargs="+", choices=TIERS, default=["32k"])
    make.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    make.add_argument("--per-task", type=int, default=PER_TASK)
    make.add_argument("--seed", type=int, default=0)
    make.set_defaults(handler=build)

    write = commands.add_parser("export", help="write a run's answers for evaluate.py")
    write.add_argument("run", help="a run directory, or a folder of them")
    write.add_argument("--out", required=True, help="the prediction folder")
    write.add_argument("--tasks", help="the task folder, if it has moved since the run")
    write.set_defaults(handler=export)

    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
