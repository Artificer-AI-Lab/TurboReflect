# Task format

Collect tasks for your own scenario, write them in the format below and check them with the verifier. Any folder of tasks that passes the verifier runs as a benchmark. `turboreflect/benchmarks/custom/` holds a sample text task, a sample video task and the verifier.

A task is one JSON file, which a run reads with `turboreflect/program/task.py`:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | a unique name for the task |
| `system` | string | yes | the system message |
| `user_task` | string | yes | what the generation stage is asked to do, with every requirement on the answer |
| `context` | object | yes | the documents, shared by every stage, as `label: text`, in the order written |
| `rubric` | list | yes | the criteria the answer is written and judged against |
| `videos` | list of strings | no | video clips, in order, relative to the task file or absolute |

The first line of each document is its heading. Labels are for you and are not rendered.

## The stage prompts

Every stage opens with the same text, built from these fields:

```text
[system]
# Task
<user_task>

# Context
## <first line of the first document>

<the rest of the first document>

## <first line of the next document>

...

# Rubric (the response must satisfy all of these)
1. <first criterion>
2. <second criterion>
```

The generation stage adds its request to complete the task. The feedback stage then quotes the draft and judges it against the rubric, and the refinement stage quotes the draft and the feedback and revises. The model sees nothing of the task but what is above, so a requirement that is not written in `user_task` or `rubric` does not exist for it.

## A text task

`turboreflect/benchmarks/custom/sample/text/river_mills.json` asks a multiple-choice question over two documents and takes a short evidence report as its answer, so the later stages have something to work on.

1. **Put the source material in `context`**, one entry per document, the first line of each its title. Put the question in an entry of its own after the documents, with its options.
2. **State the task in `user_task`**: what to do, what to base it on, and what the deliverable contains. Give the deliverable a fixed last line if the answer will be read by a program, such as `The correct answer is (X)`.
3. **Write the criteria in `rubric`**, two to seven of them, each as `requirement — judge: condition`, where the condition is something a reviewer can check on the text. Open the first with `overall goal:`.

## A video task

`turboreflect/benchmarks/custom/sample/video/channel_opener.json` asks for an edit plan over two clips.

1. **List the clips in `videos`**, in order, as paths relative to the task file.
2. **Describe them in `context`**, in the same order, one line each: `Clip 1 — 3.0 s, 320x180, audio no`. Add an entry of timestamped transcripts if you have them. Do not write the model's video placeholders: a run adds one per clip at the head of `# Context`.
3. **State the brief and the answer format in `user_task`.** The sample's format is an edit plan: `SEGMENT`, `TEXT`, `CAPTION`, `MUSIC` and `TITLE` lines and a final `RUNTIME` line.
4. **Write the criteria in `rubric`**, in the same form as for a text task, so that each can be checked against those lines and the clips.

Video tasks run on models with a video encoder; `turboreflect/configs/qwen3.8-27b-video.yaml` is one such configuration.

## Checking files

```bash
python -m turboreflect.benchmarks.custom.verify_format my_tasks/
```

The verifier reads each file as a run does, and also checks that task ids are unique and that every clip exists. It prints each problem and exits with an error if there is any.

## Task folders

`tasks` in a configuration, and `--tasks`, name a task file, a folder, or, in a configuration, a list of either. A folder stands for the JSON files of its `data/` folder if it has one, else its own, in name order.

```text
my_tasks/
  first.json
  second.json
```

```text
tasks/ruler/
  data/       the task files
  rubrics/    the gold answers, which only `export` reads
  prompts.py  the stage assembly
```

## Stage assembly

A stage assembly turns a task into each stage's prompt. A run takes the `prompts.py` that sits with its tasks, and otherwise the shared one in `turboreflect/benchmarks/custom/`, which builds the [stage prompts](#the-stage-prompts) above and both layouts. The RULER and LongBench-v2 task folders carry a `prompts.py` that uses the assembly in `turboreflect/benchmarks/common.py`, whose generation prompt is the benchmark's own and which builds layout `continue`.

A stage assembly of your own provides `build_messages`, which returns a stage's chat messages. A run calls it with the task, as its `system`, `user_task`, `context` and `rubric`, and `media`, its clip lines, for a task with clips; the stage, `generation`, `feedback` or `refinement`; `draft` and `critique`, the outputs before it; and `feedback` and `refinement`, the configuration's instructions. Under layout `requote` it also passes `layout`, and to a stage that quotes, `quoted`, the `(header, text)` sections to quote. To build layout `requote` as well, it lists both in `LAYOUTS` and provides `requote_user`, `quote_header` and `QUOTE_SEP`.

## Length

Keep each task within the model's context length, leaving room for what the three stages generate. `run --dry-run` reports each stage's prompt length without starting an engine, so it shows whether a task fits before a run.

## Clips

A run with clips sets `allowed_local_media_path` to the folder that holds them and `limit_mm_per_prompt` to the most clips any task has, and leaves the vision tower on. The placeholder it adds for each clip is the one vLLM's model class for the checkpoint gives. The configuration states `max_model_len` and `mm_processor_cache_gb` for the clips; see [adding a model](adding-a-model.md#video).
