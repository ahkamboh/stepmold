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

A second pack, `examples/lead_qualify/`, qualifies an inbound sales lead. It is the
same idea with a branch in it: a cheap gate node applies the one hard rule (an
enterprise-sized company writing in from a personal mailbox is refused), and a
conditional edge sends a refused lead straight to its ending, so the enrichment,
signal-reading, scoring and routing nodes below the gate never run for it.

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

## Observability

A run used to print its result and nothing else. Every fact you need at 3am — which node
failed, how many retries it burned, what the model returned, how long it took — already
existed inside the runtime; none of it was written down.

Two flags turn it on. Without them jig configures no logging at all and prints exactly
what it printed before, because a library that logs at its host uninvited is a bug.

```console
$ python3 -m jig run examples/support_triage --input '{"ticket": "..."}' --log-level info
13:42:49.979 INFO  jig.graph run.start run_id=ea4d17c0 pack=support_triage version=1 entry=classify resumed=false max_steps=12 inputs=ticket
13:42:49.979 WARNING jig.verify node.rejected node=classify attempt=1 cause=verify reason="output was not valid JSON — return a single JSON object and nothing else" of=3
13:42:49.979 INFO  jig.verify node.retry node=classify attempt=2 of=3 temperature=0.5 seed=1 reason="output was not valid JSON — return a single JSON object and nothing else" rethink=false
13:42:49.980 INFO  jig.graph node.ok run_id=ea4d17c0 node=classify type=generate attempts=2 output=merge duration_ms=0.3
13:42:49.980 INFO  jig.graph node.ok run_id=ea4d17c0 node=extract type=generate attempts=1 output=merge duration_ms=0.0
13:42:49.980 INFO  jig.graph run.end run_id=ea4d17c0 pack=support_triage end_node=done steps=5 generations=5 failures=0 output_keys=7 output_bytes=154 duration_ms=1.0
```

Everything goes to **stderr**, so the JSON result on stdout stays pipeable. `--log-format
json` writes one object per line with stable field names, for anything that ships logs:

```console
$ python3 -m jig run examples/support_triage --input '{"ticket": "..."}' \
    --log-level info --log-format json
{"ts": "2026-08-24T13:42:50.048Z", "level": "INFO", "logger": "jig.graph", "event": "node.ok", "run_id": "ab54abc3", "node": "classify", "type": "generate", "attempts": 2, "output": "merge", "duration_ms": 0.2}
{"ts": "2026-08-24T13:42:50.048Z", "level": "INFO", "logger": "jig.graph", "event": "run.end", "run_id": "ab54abc3", "pack": "support_triage", "end_node": "done", "steps": 5, "generations": 5, "failures": 0, "output_keys": 7, "output_bytes": 154, "duration_ms": 0.8}
```

Against a real endpoint the backend answers the other question — what a run cost:

```console
13:43:00.055 WARNING jig.backend backend.http_error model=qwen3-8b endpoint=http://.../v1/chat/completions status=429 attempt=1 retryable=true duration_ms=1.0
13:43:00.055 INFO  jig.backend backend.backoff endpoint=http://.../v1/chat/completions attempt=1 retry_after=1.0 slept_s=1.0
13:43:01.065 INFO  jig.backend backend.response model=qwen3-8b endpoint=http://.../v1/chat/completions status=200 attempt=2 retries=1 duration_ms=5.1 prompt_tokens=10 completion_tokens=10 reasoning_tokens=0 total_tokens=- finish_reason=stop
```

### What comes out at which level

| Level | Events |
| --- | --- |
| `error` | `run.error`, `backend.failed` — the run stopped, and which node it stopped on |
| `warning` | `node.rejected`, `node.failed`, `backend.http_error`, `lease.refused` |
| `info` | `run.start`, `run.end`, `node.ok`, `node.retry`, `edge.on_fail`, `backend.response`, `backend.backoff`, `run.claimed`, `lease.taken`, `resume.start` |
| `debug` | prompt and state **sizes**, `edge.taken`, `checkpoint.saved`, and the full text of a rejection |

### What never comes out

Logging is a new way for text to leave the runtime, so two rules are enforced at the sink
rather than trusted at each call site, and both are tested in `tests/test_log.py`:

* **No credential, at any level.** Every string a formatter emits goes through the same
  key-shaped filter the backend has always run over upstream error bodies. A planted
  canary key is asserted absent from captured output at `debug`.
* **No rejected model output below `debug`.** `verify.Rejected` keeps two halves: the
  `feedback` says what was wrong and is derived from your own schema; the `detail` may
  quote the generation verbatim. The default path carries only the first. Prompts and
  state never appear at all — sizes, and a digest.

Anything user-controlled is clipped, so a one-megabyte ticket cannot become a
one-megabyte log line.

Embedding jig rather than running the CLI? `jig.log` hangs everything off the `jig`
logger with a `NullHandler`, so `logging.getLogger("jig").setLevel(...)` in your own
application is all it takes; `jig.log.configure` is there if you want jig's formatters.

## Benchmarks

**Measured 2026-08-24** against Cerebras (`https://api.cerebras.ai/v1`, $0.35/1M in,
$0.75/1M out), running `examples/support_triage` — a 4-node pack, 12 evalset cases.
Reproduce with:

```
JIG_API_KEY=... python3 -m jig eval examples/support_triage \
  --model 'openai:https://api.cerebras.ai/v1#gpt-oss-120b#response_format#600'
```

| Metric | gpt-oss-120b | gemma-4-31b |
| --- | --- | --- |
| Evalset agreement | 6 of 12 | 7 of 12 |
| API calls for the run | 60 | TODO: measure |
| Prompt tokens | 18693 | TODO: measure |
| Completion tokens | 7213 | TODO: measure |
| Of which reasoning | 4756 (66%) | not a reasoning model |
| Wall clock | 39.9 s | TODO: measure |
| Cost for 12 cases | $0.01195 | TODO: measure |
| Cost per case | $0.00100 | TODO: measure |

### Read the agreement column carefully — it is not a quality score

The evalset that ships with the example was written to make a scripted `FakeModel` pass.
Its `sentiment` and `priority` labels are one author's opinion, not ground truth, and two
models that share no architecture disagree with the gold **in the same direction** on four
of the twelve cases:

| Ticket | Gold | gpt-oss-120b | gemma-4-31b |
| --- | --- | --- | --- |
| "How do I export my invoices for last year?" | other | billing | billing |
| "Payment failed three times for order C-3003" | p2 | p1 | p1 |
| "Our whole team lost access after the SSO change" | angry | calm | frustrated |
| "I was billed after cancelling... second time" | angry | frustrated | frustrated |

When two independent models agree with each other and disagree with the label, the label
is the outlier. On the first row the models are simply right: an invoice question is
billing. The smaller model also scored *higher* than the larger one, which is another sign
the number is not measuring capability.

So these runs establish three things, and no more than three: the runtime works end to end
against real inference; per-node attribution localised the problem to one node
(`extract`, 4 of 6 failures) in a single line of output; and the cost of running the pack
is now a measured number rather than an estimate.

**Not established:** whether a small model matches a frontier model on this workflow. That
needs an evalset whose labels are ground truth, and a frontier baseline on the same cases.
Both are open. `docs/PLAN.md` §4.1 names the numbers that would close it.

### What running it for real cost us

Two defects survived every mocked test and appeared on first contact with a live endpoint:

* urllib's default `User-Agent` is rejected by Cloudflare-fronted providers with HTTP 403
  error 1010, before the request reaches the model.
* A reasoning model bills its private chain of thought against `max_tokens`. The `classify`
  node budgets 32 tokens — enough for its answer, but the model spent 29 of them thinking
  and returned `finish_reason=length` with `content: null`. Hence `reasoning_reserve`:
  a pack budgets the *answer* and stays portable, and the backend adds the headroom this
  particular model needs to think.

That 66% reasoning share is the number to watch. For a bounded slot-filling node, two
thirds of the output spend bought no output.

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
| Structured logging (`--log-level`, `--log-format=text\|json`) | done |
| Example pack, runs offline against a scripted model | done |
| Second example pack: branching graph, policy asserts | done |
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
