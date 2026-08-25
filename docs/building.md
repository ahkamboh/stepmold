# Compiling a pack

`jig build` takes a description of a task and some gold examples, and writes a pack.

```console
$ jig build ./myspec -o ./mypack --model 'openai:https://host/v1#a-big-model'
```

The model is used **at build time only**. What it produces is a directory of text files
that a small model runs afterwards, forever, with no dependency on the model that wrote it.

---

## The build spec

Two files. That is the whole input.

```
myspec/
  task.md          what the workflow does, in prose
  examples.jsonl   gold cases, one JSON object per line
```

### `task.md`

Plain prose. Say what comes in, what must come out, and any rule a reader would need that
the examples do not show — a threshold, a policy, which cases are urgent. The planner reads
this once, so it is worth being specific about the *decisions* the workflow makes rather
than describing the output format, which the examples already fix.

```markdown
Triage an inbound customer support ticket.

Classify what it is about, extract the order id and amount when present, judge the
customer's sentiment, set a priority, choose the queue that should handle it, and
decide whether it must be escalated.
```

### `examples.jsonl`

The same shape as a pack's `evalset.jsonl`, deliberately — so examples written to test a
pack are the examples that compile one, and a spec can be lifted straight out of a pack
that already exists.

```json
{"input": {"ticket": "I was charged twice for order A-1001, $49.99 both times."},
 "expect": {"category": "billing", "order_id": "A-1001", "amount_usd": 49.99,
            "sentiment": "frustrated", "priority": "p1", "queue": "billing-ops",
            "escalate": false}}
```

| Key | Required | What it is |
| --- | --- | --- |
| `input` | yes | the keys a run is given |
| `expect` | yes | the keys the run must end up producing |
| `name` | no | a label for the report |
| `end` | no | the ending this case must reach — see [testing](testing.md) |
| `rescued` | no | this case is meant to burn a ladder and take `on_fail` |

**These examples are the contract.** The compiler is measured against them and never edits
them. A compiler that adjusted the test until it passed would have laundered its own
failure rather than fixed anything.

Twelve to twenty cases is a workable size. What matters far more than the count is that
every value a field can take appears, and that the labels are actually correct — see
[the labels are the ceiling](#the-labels-are-the-ceiling).

---

## What the compiler does with them

    analyze  ->  induce  ->  write_prompts  ->  script  ->  assemble  ->  verify
                    ^                                                        |
                    +--- re-plan, told which node failed and why -------------+

| Stage | Model? | What it produces |
| --- | --- | --- |
| `analyze` | **no** | field types, and the enums, read off the examples |
| `induce` | yes | the decomposition: nodes, edges, endings |
| `write_prompts` | yes | one prompt per node |
| `script` | **no** | the scripted offline model, from the gold answers |
| `assemble` | **no** | the pack on disk, loaded back to prove it parses |
| `verify` | **no** | `jig eval` against the pack's own gold cases |

Two of those need no model at all. Schema induction is arithmetic over the examples, and
the offline model is arithmetic over the gold answers — if a node writes `category` and the
gold says `billing`, that is what the node must return. Between them they are more than half
the lines of a hand-written pack.

**Enum inference is the most valuable thing here**, because under constrained decoding a
value outside an enum is not merely unlikely — it is unrepresentable. That is also why it is
the most dangerous: an enum wrongly inferred from a set of names would make a seventh name
impossible to emit, forever, with no way to recover at run time. The rule that decides is
documented in `jig/build/analyze.py` and errs toward leaving a field open.

## The loop

An emitted pack is scored against its own gold cases before it is returned. If it does not
pass, the compiler re-plans — and because `jig eval` names the node responsible for each
failure, the next attempt is told *"node `extract` produced sentiment=calm where the gold
says frustrated"* rather than *"try again"*. That per-node attribution is the reason the
loop converges instead of wandering.

Every attempt is built in a scratch directory and moved into place only once it scores full
marks, so a failed compile never leaves a half-written pack where a working one used to be.
`--overwrite` is required to replace an existing pack at all.

```console
$ jig build ./myspec -o ./mypack --model 'openai:https://host/v1#a-big-model'
analyzing 12 gold case(s)
  7 field(s), 1 input(s), 4 enum(s) inferred
attempt 1: planning
  4 node(s): classify_category, extract_order_amount, assess_sentiment, decide_priority_queue_escalate
  attempt 1: node 'decide_priority_queue_escalate''s script key is as long as the think key, so its think call would eat an answer meant for its emit call; shorten the node name
attempt 2: planning
  6 node(s): classify_category, extract_order_amount, assess_sentiment, decide_priority, choose_queue, decide_escalate
  attempt 2: 12/12 cases
./mypack
```

That transcript is a real run. Attempt 1 failed on a genuine constraint, and attempt 2
re-planned into six finer nodes — which is also the decomposition the design wants, since
reliability comes from shorter steps rather than a larger model.

## Flags

| Flag | Default | What it does |
| --- | --- | --- |
| `-o`, `--out` | required | where to write the pack |
| `--model` | required | the planning model; same spec grammar as `jig run` |
| `--name` | the output directory's name | the pack's name |
| `--attempts` | 3 | how many times to re-plan on a failing eval |
| `--overwrite` | off | replace the output directory if it exists |

There is no default model. A compile is the one moment where quietly choosing a model for
someone would be the wrong kind of helpful.

---

## The labels are the ceiling

The compiler optimises toward your examples. If a label is wrong, it will faithfully build a
pack that reproduces the mistake — and constrained decoding will then make that mistake
consistent rather than occasional.

This is not hypothetical. Running two unrelated models over `examples/support_triage`'s
shipped gold set, both disagreed with the label **in the same direction** on four of twelve
cases; on one of them the models were simply right and the label was wrong. Details in
[BENCHMARKS.md](BENCHMARKS.md#3-agreement-with-the-shipped-evalset--and-why-it-is-not-a-quality-score).

Two things follow. Label your examples yourself, from cases whose answer you actually know.
And when you disagree with a compiled pack, check the label before you blame the model —
running the same evalset through two different models is a cheap way to audit it, because
cases where independent models agree with each other and disagree with your gold are the
ones to re-label first.

## What it cannot do yet

* **Express an `assert` node or an `on_fail` edge in a plan.** Two of the six example packs
  cannot be regenerated for this reason: `content_moderation` routes through an assert node,
  and `meeting_actions` asserts over a field its gold set never pins. The compiler reports
  this rather than emitting a pack that quietly scores zero.
* **Refine a prompt.** The loop re-plans the decomposition on failure; it does not tune the
  wording of a prompt that is nearly right.
* **Infer a schema inside an array or object.** A field whose value is a list comes back as
  `{"type": "array"}` with no element schema.
* **Widen an enum beyond what the examples show.** If your cases only ever say USD, EUR and
  GBP, the compiled pack cannot answer JPY. Show every value a field can take.

## Honestly, what has been proven

`examples/support_triage` was recompiled from its own gold cases into a different six-node
decomposition that scores 12/12 offline and answers a real ticket correctly against a live
model, with its four enums derived without a model and `two_stage` chosen unprompted for the
two judgement nodes.

That is regeneration. **Compiling a workflow nobody has built before has not been
demonstrated**, and the task description in that run was written by someone who already knew
the answer. Treat the compiler as working and unproven at the same time, and check what it
gives you.
