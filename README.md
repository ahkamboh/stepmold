# jig

> A machinist's jig is a custom guide that lets a cheap tool do precision work,
> repeatably, forever. This is that, for language models.

**Compile your agent once with a frontier model. Run it forever on a small one.**

---

## The problem

Companies pay frontier-model prices to do work that contains no thinking — invoice
extraction, ticket triage, CRM updates, compliance checks. The same steps, tens of
thousands of times a month, with no novel reasoning anywhere in them.

They cannot switch to a cheap model, because small models fall apart over multi-step
tasks: per-step error rates compound, tool calls come back malformed, and a model that
sees its own sloppy output in its next prompt gets worse as the run goes on. The two
usual escapes both fail — "use a smaller model" hits that compounding, and "fine-tune
it" needs an ML team, labelled data, GPUs, and a retrain on every base-model update.

jig is a third path: **don't make the model smarter, make the task easier.**

All the thinking moves to compile time, where you pay for it once. What is left at run
time is a state machine: the small model never plans, never chooses what happens next,
and never writes more than one short, schema-constrained, verified step at a time.

Three properties fall out of that, and the last two are what actually close deals:

- **Cost** — the execution tier runs on a small self-hosted model instead of a frontier API.
- **Sovereignty** — a workflow that fits on one modest GPU is a workflow a bank, hospital
  or defence contractor can run at all. For them this is not a discount, it is access.
- **Auditability** — "we prompted a big model and it usually works" cannot be audited. A
  versioned pack, a grammar per step, and an evalset that passes can be.

**Where jig does not help:** open-ended novel tasks, workflows that run a few times a
month, and bad requirements. Repetition is the qualifier — if the plan has to be invented
per request, you want a runtime, not a compiler.

## How it works

A **JigPack** is a directory of text: a graph, one prompt and one JSON Schema per node,
and an evalset. The runtime walks it.

```
examples/support_triage/
  manifest.yaml        name, version, entry node, model
  graph.yaml           nodes + edges — the plan, as data
  prompts/*.txt        one prompt per generate node
  grammars/*.json      one schema per generate node
  evalset.jsonl        the contract: input -> expected output
```

Four ideas do the work:

1. **The graph decides, not the model.** Every node gets one narrow job and a short,
   fresh context. Horizon length becomes a property of the compiler, not of the model.
2. **Think and emit are separate.** Forcing a model to commit to a schema on its first
   token costs quality, so a `two_stage` node reasons freely into a scratchpad first, then
   emits under its grammar — and **the scratchpad is thrown away**, never committed.
3. **Verify before commit.** Output must parse, satisfy its schema, and satisfy the node's
   optional `assert` before it is written to state. On failure: re-sample, re-sample with
   the error attached, then take the node's failure edge. A rejected generation never
   reaches state, the output, or the next prompt — a model that never sees its own bad
   output cannot spiral on it.
4. **The evalset is the source of truth.** Prompts, grammars and graphs are regenerable
   build outputs. The gold examples are the hand-maintained asset, and "v3 passes 50/50"
   is a sentence a buyer can hold you to.

## Install

Python 3 and nothing else. No pip, no virtualenv, no dependencies — that is a hard rule,
because the runtime is meant to drop onto a client box with nothing installed on it.

```console
$ git clone https://github.com/ahkamboh/jig.git
$ cd jig
```

## Quickstart

The example pack is a four-node support-ticket triage workflow. It ships a scripted
stand-in model, so all of this runs offline with no GPU and no network:

```console
$ python3 -m jig validate examples/support_triage
support_triage v1: 7 nodes, 5 edges, 12 evalset cases, entry 'classify'

$ python3 -m jig eval examples/support_triage
support_triage: 12/12 cases passed

$ python3 -m jig run examples/support_triage --input '{"ticket": "I was charged twice for order A-1001, $49.99 both times."}'
{"amount_usd": 49.99, "category": "billing", "escalate": false, "order_id": "A-1001", "priority": "p1", "queue": "billing-ops", "sentiment": "frustrated"}
```

`jig eval` exits non-zero if any case fails, so an evalset is a CI gate rather than a
report. Break the pack and it tells you which node did it:

```console
$ python3 -m jig eval tests/fixtures/cli_pack --model fake:fakes/wrong.json
cli_demo: 1/2 cases passed
  FAIL technical case [classify]
    category: expected 'technical', got 'billing'
  failures by node: classify=1
```

To run against a real model, point `--model` at any OpenAI-compatible server —
llama.cpp-server, vLLM or SGLang:

```console
$ python3 -m jig run examples/support_triage \
    --model openai:http://localhost:8000#qwen3-8b \
    --input '{"ticket": "..."}'
```

## Benchmarks

**TODO: measure.** This table is empty on purpose. Every number that belongs here is a
number nobody has measured yet, and a guessed benchmark in a project whose whole pitch is
verifiability would be self-defeating.

| Metric | jig + small model | Frontier baseline |
| --- | --- | --- |
| Evalset pass rate, same gold cases | TODO: measure | TODO: measure |
| Tokens per run, including think stages | TODO: measure | TODO: measure |
| Batched throughput on the target GPU | TODO: measure | TODO: measure |
| Cost per run | TODO: measure | TODO: measure |

`docs/PLAN.md` sets the go/no-go gate these have to clear. Until someone runs it on a
GPU, the honest claim is the architecture, not the arithmetic.

## What is built

The runtime is complete and tested; the compiler is not written yet.

| | |
| --- | --- |
| Pack format, load + validate | done |
| JSON Schema subset + validation | done |
| Graph walker, conditional edges, step guard | done |
| Two-stage think -> emit codegen | done |
| Verify-before-commit + retry ladder | done |
| Checkpointing and resume (SQLite) | done |
| Evalset runner with per-node attribution | done |
| CLI: `run`, `eval`, `validate` | done |
| Example pack, scored offline | done |
| OpenAI-compatible backend | written, **unverified against a live server** |
| `jig build` — the compiler | not started |

## Running the tests

The suite is plain `unittest`, so it runs under real pytest unchanged. The repo also
vendors a tiny stdlib stand-in so the command works on a machine with no pytest:

```console
$ python3 -m pytest -q
```

Everything is tested against a scripted fake model. No test touches a network, downloads
a model, or needs a GPU.

## Design

`docs/PLAN.md` is the full architecture: the five scaling bugs and their fixes, the cost
model, the language decision, and the roadmap.
