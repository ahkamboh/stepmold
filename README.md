# stepmold

**A Python framework for running LLM workflows reliably on small models.**

A small model is not bad at any one step. It is bad at *many steps in a row* — a 2% error
per step compounds to a 33% failure rate over 20 steps, and worse the longer the task runs.
Most agent frameworks answer that by reaching for a bigger model. stepmold answers it by making
each step short, schema-constrained and verified before anything is committed, so errors
stop compounding.

Measured on a 50-node workflow at a 10% per-step error rate: **96.5% end-to-end success,
against 2.0% for the same model with the same prompts and no verification.**

```bash
pip install git+https://github.com/ahkamboh/stepmold
stepmold run mypack --input '{"ticket": "I was charged twice"}'
```

- Zero dependencies. `stepmold` imports nothing outside the Python standard library.
- Any OpenAI-compatible endpoint — llama.cpp-server, vLLM, SGLang, or a hosted API.
- A workflow is a directory of text files, so it can be diffed, reviewed and copied to a
  machine that cannot install anything.
- A step can act, not only decide. A `tool` node calls a function the *host* registered,
  and the call is written down before it is committed, so a resumed run replays it rather
  than doing it twice.

---

## Contents

- [Why](#why) — the problem stepmold exists for
- [Results](#results) — what has been measured
- [Install](#install)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Documentation](#documentation)
- [Observability](#observability)
- [What is built, and what is not](#what-is-built-and-what-is-not)
- [Running the tests](#running-the-tests)

---

## Why

Agent frameworks are runtimes: the model decides what happens next on every step of every
run. That works, and it is why they need a frontier model — the model is doing the planning
continuously, and you pay for planning on every request forever.

stepmold splits that in two.

|                   | Who decides            | How often               |
| ----------------- | ---------------------- | ----------------------- |
| **What to do**    | the workflow's graph    | written once            |
| **How to do it**  | the model               | every step of every run |

The graph decides which step runs and where the result goes. The model only fills in one
short, schema-constrained answer at a time. It never plans, never chooses the next step and
never sees its own rejected output — which is what keeps a small model reliable over a long
task.

That has a useful side effect. Because the model can only emit tokens its schema permits,
an instruction smuggled into the input cannot change the output shape *or* the control
flow. A prompt-injection attempt against a node whose grammar is
`{"category": "billing" | "technical" | "other"}` comes back as one of those three values,
because no token path produces anything else.

## Results

Every number here was produced by a command in this repository. Nothing is estimated.

**Compounding.** A generated N-node chain where each step's correct answer is checkable, run
against a seeded model with a controllable per-step error rate. Both arms see identical
first attempts at every node (blake2b-seeded), so the comparison is paired. 200 seeds per
cell, 2,400 runs.

| Nodes | Error/step | stepmold       | no verification | analytical `(1-p)^N` |
| ----- | ---------- | --------- | --------------- | -------------------- |
| 20    | 2%         | **100.0%** | 68.0%           | 66.8%                |
| 20    | 10%        | **99.0%**  | 16.0%           | 12.2%                |
| 50    | 2%         | **100.0%** | 48.0%           | 36.4%                |
| 50    | 10%        | **96.5%**  | 2.0%            | 0.5%                 |
| 50    | 30%        | **40.0%**  | 0.0%            | 0.0%                 |

The margin over the analytical curve grows with N, which is the signature of attacking
compounding rather than attacking individual errors. Expressed as effective per-step error
at 50 nodes and 10%: **0.0007 for stepmold against 0.0753 without**, for 1.09× the generations.
Across all 2,400 stepmold runs there were **zero silently-wrong answers**; the unverified arm
returns one in 34% of runs at 20 nodes and 10%.

Reproduce: `python3 -m tests.production.test_longhorizon`

**Cost and latency**, `examples/support_triage` against a hosted endpoint, 12 cases:
$0.00100 per case, 0.65s median per call, 8 concurrent calls in 0.83s. Two thirds of
completion tokens were the model's own reasoning — which is why a reasoning model is the
wrong choice for bounded steps. Full table and method in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Install

```bash
pip install git+https://github.com/ahkamboh/stepmold
```

Python 3.9 or newer. No other dependencies, ever — CI fails the build if the dependency
list is not empty.

From a clone, with nothing installed at all:

```bash
git clone https://github.com/ahkamboh/stepmold
cd stepmold
python3 -m stepmold --help
```

## Quickstart

The repository ships seven worked packs. Each runs offline against a scripted model, so this
needs no GPU, no API key and no network. From a clone, use `python3 -m stepmold`; once
installed the bare `stepmold` command does the same thing from anywhere.

```console
$ python3 -m stepmold validate examples/support_triage
support_triage v1: 7 nodes, 5 edges, 12 evalset cases, entry 'classify'
```

Score it against its gold cases:

```console
$ python3 -m stepmold eval examples/support_triage
support_triage: 12/12 cases passed
```

Run one case through it:

```console
$ python3 -m stepmold run examples/support_triage --input '{"ticket": "I was charged twice for order A-1001, $49.99 both times."}'
{"amount_usd": 49.99, "category": "billing", "escalate": false, "order_id": "A-1001", "priority": "p1", "queue": "billing-ops", "sentiment": "frustrated"}
```

`stepmold eval` exits non-zero when a case fails, so a pack can gate a build. Here is a pack
wired to a deliberately wrong model, to show what that looks like:

```console
$ python3 -m stepmold eval tests/fixtures/cli_pack --model fake:fakes/wrong.json
cli_demo: 1/2 cases passed
  FAIL technical case [classify]
    category: expected 'technical', got 'billing'
  failures by node: classify=1
```

Note `[classify]` — the report names the node that caused the failure, not just the field.

Point it at a real model, any OpenAI-compatible server:

```bash
STEPMOLD_API_KEY=... python3 -m stepmold eval examples/support_triage \
  --model 'openai:http://localhost:8000/v1#qwen3-8b'
```

## How it works

A **pack** is a directory of text: a graph, one prompt and one JSON-schema grammar per
generating step, and an evalset of gold cases.

```
mypack/
  manifest.yaml       name, version, entry node, default model
  graph.yaml          the nodes and the edges between them
  prompts/            one .txt per generate node
  grammars/           one .json schema per generate node
  evalset.jsonl       gold cases — the pack's contract
```

There are four kinds of node. `generate` fills one schema-constrained slot, `assert`
branches on a deterministic expression, `tool` takes an action, and `end` stops and
returns. A `tool` node adds no files to the pack: it names a function the **host**
registered, which is what lets a pack stay text. A pack cannot contain an action, only
name one, so it can reach nothing the host did not hand it — an allowlist by construction
rather than a sandbox that has to hold. A name that was never registered is refused when
the pack loads, not at the step that would have called it.

Running it:

1. **Constrain.** The node's schema is sent to the backend as a grammar, so the model's
   output is valid by construction rather than by hope.
2. **Think, then answer.** A node marked `two_stage` generates unconstrained reasoning
   first, then a constrained answer conditioned on it. The reasoning is discarded — it
   never enters state and never reaches a later prompt.
3. **Verify before commit.** Output must parse, satisfy its schema, and satisfy the node's
   optional `assert`. Verification runs against a trial copy of state; only a candidate
   that survives is written back.
4. **Ladder, then divert.** A rejected candidate is re-sampled, then re-sampled with a
   description of what was wrong — never a quote of what the model said — and finally the
   node's `on_fail` edge is taken. A rejected generation is never shown to the model again,
   which is what stops a bad answer conditioning the next one.
5. **Agree, or hand over.** A node can ask to be drawn several times and accepted only
   when the draws match — `samples:` and `agree:`. Draws that all parse and all validate
   but do not match are not a failure, they are the model being inconsistent, so the node
   raises `Unsure` and the pack routes it to `on_unsure` — a human queue, a conservative
   default branch — rather than committing a coin flip. Nothing the model says about its own
   certainty is read. See [Confidence](docs/confidence.md), including the part where this
   is reachable from Python and not yet from a pack file.
6. **Act, at most once.** A `tool` node calls its registered function and commits what it
   returns under the same rules a generation gets: declared `reads`, declared `writes`,
   and no retry ladder, because re-running a tool is a side effect done twice. The moment
   the call returns, the node, its arguments and its result go into the checkpoint with
   the walk still standing on the node — so a run that dies mid-workflow resumes by
   replaying that result instead of sending the second email.
7. **Checkpoint.** State is persisted after each committed node, so a killed run resumes
   instead of restarting.

The evalset is the pack's contract. `stepmold eval` scores every case, names which node caused
each failure, and can assert which branch a case must reach — so a change that quietly
reroutes a workflow fails its tests instead of passing them.

## Documentation

| Document                                     | What is in it                                                   |
| -------------------------------------------- | --------------------------------------------------------------- |
| [Pack format](docs/pack-format.md)            | Every file, every key, every default. Start here to build a pack |
| [Graph and routing](docs/graph.md)            | Node types including `tool`, edges, `when:`, `on_fail`, state and provenance |
| [Confidence](docs/confidence.md)              | The agreement gate, why a self-reported number is not one, and how to read the tier split |
| [Expressions](docs/expressions.md)            | The `assert` language: what it supports and what it refuses       |
| [Testing packs](docs/testing.md)              | Evalsets, scripted models, and scoring offline                    |
| [Compiling a pack](docs/building.md)          | `stepmold build`: the spec format, the loop, and what it cannot do yet |
| [Architecture](docs/ARCHITECTURE.md)          | Why it is built this way                                          |
| [Benchmarks](docs/BENCHMARKS.md)              | Every measurement, with the command that produced it              |

## Observability

Off by default — a library should not configure logging for its host. Turn it on per run:

```bash
stepmold run mypack --input '{...}' --log-level info
```

```
run.start        run_id=6acf57db pack=support_triage version=1 entry=classify
backend.response status=200 duration_ms=779.7 prompt_tokens=303 completion_tokens=52 reasoning_tokens=37
node.ok          node=classify attempts=1 duration_ms=780.1
run.end          end_node=done steps=5 generations=4 failures=0 duration_ms=3414.9
```

`--log-format json` emits one object per line for a log collector. Credentials are redacted
at the formatter, so no key can reach a log through any call site, and a rejected
generation is never logged above DEBUG.

## What is built, and what is not

**Built and tested:** the runtime. Pack loading and validation, the graph walker,
constrained generation, two-stage think-then-answer, verify-before-commit, the retry ladder,
`on_fail` routing, SQLite checkpointing and resume, the evalset runner with per-node blame,
the CLI, an OpenAI-compatible backend, and structured logging.

**Built, and shipped without a worked example:** `tool` nodes and the confidence gate. A
pack can now name an action the host registered, and the walker records the call so a
resumed run replays it rather than repeating it; and a node can be drawn several times and
accepted only on agreement, with disagreement routed to `on_unsure`. Both are tested. What
has not been shown:

- **One example pack uses a tool: `examples/refund_desk`.** It classifies a message, looks
  the order up through a host-registered tool, decides, and either issues the refund
  through a second tool or routes to a human. Run it with
  `python3 -m stepmold eval examples/refund_desk --tools examples/refund_desk/tools.py:registry`.
  The other six decide without acting.
- **The gate is now a pack key.** `samples:` and `agree:` are node keys, and
  `examples/refund_desk` uses them: `approve` draws three times and needs two to match
  before an irreversible refund is issued. A pack that writes `on_unsure:` without
  `samples:` is refused at load, because that edge could never be taken.
- **Nobody has measured what agreement buys.** There is no benchmark here for how much a
  gate raises accuracy inside the auto bucket, or how much of the escalated bucket it
  moves, against a real model. The mechanism is tested; its value is not a number anyone
  in this repository can quote. [Confidence](docs/confidence.md) says how to find out on
  your own workflow, and says the same thing there.

**Built, and newer:** `stepmold build` — the compiler. A frontier model authors a pack once from
a task description and gold examples; a small model then runs it forever. Two of its four
stages use no model at all: schema induction and the offline test model are arithmetic over
the examples.

```console
$ stepmold build ./myspec -o ./mypack --model 'openai:https://host/v1#a-big-model'
```

The loop is what makes it a compiler rather than a generator: it emits a pack, runs
`stepmold eval` against it, and if the pack does not score full marks on its own gold cases it
re-plans with the failing node named. It never edits the evalset — that is the contract —
and a failed attempt never overwrites a working pack. See [Compiling a pack](docs/building.md).

**What that has been shown to do:** recompile `examples/support_triage` from its own gold
cases into a *different* six-node decomposition that scores 12/12 and answers correctly
against a live model. It found the four enums with no model involved, and chose `two_stage`
for the two judgement nodes unprompted.

**What it has not been shown to do:** compile a workflow nobody has built before. The test
above is regeneration, and the task description was written by someone who knew the answer.

**Also open:** three of the seven example packs cannot yet be regenerated — two because a
plan cannot express an `assert` node or an `on_fail` edge, and `refund_desk` because the
compiler has no notion of a `tool` node at all — and whether a small model matches a frontier
model on a real workflow, which needs gold labels that are ground truth and a frontier
baseline on the same cases.

## Running the tests

```console
$ python3 -m pytest -q
```

Around 1,500 tests, no network, no GPU. The suite runs with no dependencies installed — including no pytest, via a stdlib shim. CI
runs it on Python 3.9 through 3.13, checks that `stepmold/` imports nothing outside the standard
library, and builds and installs the wheel into a clean environment.

Roughly 11,000 lines of framework and 20,000 lines of tests. The load-bearing invariants —
that a rejected generation never returns to the model, that nothing is committed unverified,
that the model never chooses the next node — are each guarded by tests verified to fail when
the invariant is deliberately broken.

## License

MIT. Copyright (c) 2026 [Ali Hamza Kamboh](https://github.com/ahkamboh).
