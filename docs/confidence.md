# Confidence — agreement, not self-report

A node can be drawn more than once and accepted only when the draws match. That is stepmold's
confidence gate: `samples:` is how many independent answers to draw, `agree:` is how many
of them must be the same, and a node whose draws do not agree raises `Unsure` and is
routed by `on_unsure` instead of committing anything.

This page is about why the gate is shaped that way, what it is worth, and — at some
length, because it is the part that costs people money — what it is not worth.

Everything below was checked against `stepmold/verify.py`, `stepmold/graph.py`, `stepmold/eval.py`,
`stepmold/pack.py` and `stepmold/backends/openai_compat.py`. Every transcript is pasted output,
unedited.

**How to reproduce anything on this page.**

| | |
| --- | --- |
| Where commands run | the root of a stepmold checkout, where `python3 -m stepmold` resolves |
| Probe scripts | saved at that root, and run with `python3 gate.py` |
| Demo packs | created by the shell block in the section that uses them |
| Models | `FakeModel` — scripted, offline. No GPU, no key, no network |
| What differs run to run | the leading timestamp on a log line. Nothing else on this page |

## Read this first

Five things about the gate look different from how they behave.

| Looks like | Actually is | Section |
| --- | --- | --- |
| `samples:` and `agree:` are `graph.yaml` keys | they are, and `examples/refund_desk` uses them. A node with `on_unsure:` and no `samples:` is refused at load, because that edge could never be taken | [The gate in a pack](#the-gate-in-a-pack) |
| the gate scores a confidence | it counts matching draws. The comparison is the whole committed object as canonical JSON — not "the fields that matter" | [What "agree" means](#what-agree-means) |
| `samples: 3` costs three generations | it costs two whenever the first two match, and stops early whenever no group can still reach the threshold | [What it costs](#what-it-costs) |
| `Unsure` is a failure | it is a sibling of `NodeFailed`, not a subclass. Nothing was wrong with the answers; the model was not consistent | [Unsure is not rejected](#unsure-is-not-rejected) |
| an escalated case is reported as `unsure` | `RunResult` publishes no `unsure` list yet, so `stepmold eval` labels the escalation `failed`. The case does leave the auto bucket | [Reading the tier split](#reading-the-tier-split) |

## Why you cannot simply ask the model how sure it is

The obvious design is to have the node emit a `confidence` field and threshold it. Nothing
in stepmold stops you: put `confidence` in the grammar, write `assert: confidence > 0.8`, and
the expression language will enforce it exactly as written, every time, on a number the
model invented.

Three properties of a verbalized confidence make that threshold worse than nothing.

| Property | Consequence for a threshold |
| --- | --- |
| It is generated **after** the answer, conditioned on the answer | it is a continuation of the text, not a measurement of it. The tokens that produced `0.95` were chosen the same way the wrong answer's tokens were |
| It reflects the **prompt's rubric** | rewording "only say high confidence if certain" moves the number without moving a single answer. You are tuning a self-description |
| It is worst in the **0.7–1.0 band** | which is the only band a threshold is ever set in. The cases that clear a high bar and are wrong are exactly the confident-and-wrong cases the gate was built to catch |

So a naive threshold does not filter errors. It filters *hesitation* — and a model that
hesitates is usually a model that was about to be right and said so. The cases it waves
through are the ones that cost you.

The gate below is the opposite arrangement: nothing the model says about itself is read at
all. Two draws that landed on the same answer are evidence about the answer; one draw
claiming to be sure is evidence about nothing.

## The signals, ranked

stepmold has exactly three tiers of signal, and the ranking is not a preference — it is the
order they should be reached for.

| Rank | Signal | What it is | Cost |
| --- | --- | --- | --- |
| 1 | `assert:` on a node | a deterministic expression over the candidate. A fact | zero extra generations |
| 2 | `samples:` / `agree:` | independent draws compared whole | one generation per extra draw |
| 3 | anything the model says about itself | a continuation of the answer | one field, and a false sense of a gate |

There is no fourth.

### Write the assert first

An `assert` is not a cheaper gate — it is a different kind of claim. `amount_usd > 0` is
not a guess that has a 97% chance of holding; it holds. Each assert you can write removes
a whole class of error from the probabilistic tiers, permanently, for no tokens.

The gate cannot substitute for it, because a model that is consistently wrong is
consistent. Save this as `assert_first.py`:

```python
from dataclasses import dataclass

from stepmold.errors import NodeFailed
from stepmold.model import FakeModel
from stepmold.pack import Node
from stepmold.verify import run_node


@dataclass(frozen=True)
class GatedNode(Node):
    samples: int = 1
    agree: int = 0


AMOUNT = {"type": "object", "properties": {"amount_usd": {"type": "number"}},
          "required": ["amount_usd"]}
WRONG = '{"amount_usd": -49.99}'          # a sign error the model is consistent about
state = {"ticket": "charged twice, $49.99 each"}

gate_only = GatedNode(name="extract", type="generate", prompt="Amount in: {ticket}",
                      grammar=AMOUNT, samples=3, agree=2)
print("gate alone  ->", run_node(gate_only, state, FakeModel([WRONG] * 3)))

gated_assert = GatedNode(name="extract", type="generate", prompt="Amount in: {ticket}",
                         grammar=AMOUNT, samples=3, agree=2, retries=0,
                         assert_expr="amount_usd > 0")
try:
    run_node(gated_assert, state, FakeModel([WRONG] * 3))
except NodeFailed as exc:
    print("with assert ->", exc)
```

```console
$ python3 assert_first.py
gate alone  -> {'amount_usd': -49.99}
with assert -> node 'extract' failed after 1 attempt(s): assert failed: amount_usd > 0
```

The gate paid for two generations and confidently committed a negative refund. The assert
caught it on the first one. Every node where you can name a property of a correct answer
is a node that should not be relying on agreement to find a wrong one.

The order to work in, then: constrain what you can in the grammar, assert what the grammar
cannot say, and reach for `samples:` only for what is left — the genuinely judgemental
nodes, where two readings are both well-formed and only one is right.

## The gate, in one node

`gate_for` (`stepmold/verify.py:gate_for`) reads the two keys, defaults them and refuses the
pairs that cannot mean anything. `agree` left unset is a **strict majority** of `samples`,
which is the only default that is a rule rather than a number somebody picked.

Save this as `gateerrors.py`:

```python
from dataclasses import dataclass

from stepmold.pack import Node
from stepmold.verify import GateError, gate_for


@dataclass(frozen=True)
class GatedNode(Node):
    samples: int = 1
    agree: int = 0


def show(**keys):
    node = GatedNode(name="classify", type="generate", **keys)
    try:
        print(keys, "->", gate_for(node))
    except GateError as exc:
        print(keys, "-> GateError:", exc)


show(samples=2)
show(samples=3)
show(samples=4)
show(samples=5)
show(samples=3, agree=1)
show(samples=3, agree=4)
show(samples=1, agree=2)
show(samples=0)
show(samples=True)
```

```console
$ python3 gateerrors.py
{'samples': 2} -> (2, 2)
{'samples': 3} -> (3, 2)
{'samples': 4} -> (4, 3)
{'samples': 5} -> (5, 3)
{'samples': 3, 'agree': 1} -> GateError: node 'classify' asks for agree: 1, which accepts the first answer and never draws the other 2. Use agree: 2 or more, or remove samples.
{'samples': 3, 'agree': 4} -> GateError: node 'classify' asks for agree: 4 out of samples: 3, which no run can satisfy. Raise samples to at least 4, or lower agree.
{'samples': 1, 'agree': 2} -> GateError: node 'classify' asks for agree: 2 but draws only one sample. Add samples: 2 (or more), or remove agree.
{'samples': 0} -> GateError: node 'classify' asks for samples: 0. A node draws at least once — use samples: 1 (or drop the key) for the ordinary single draw.
{'samples': True} -> GateError: node 'classify' has samples: True — it must be a whole number, not bool
```

Two of those are worth dwelling on, because both are a gate that reads as if it works.

* **`agree: 1` is refused**, not honoured. It accepts the first answer and never draws the
  other two, so the pack would carry a gate that never fires and an author who believes it
  does.
* **`samples: True` is refused.** `samples: yes` is an ordinary YAML slip and Python calls
  `True` an `int`, so without this check the node would draw once and the author would
  read the file as asking for a gate.

Nothing is clamped or repaired. `gate_for`'s own comment gives the reason: every reading of
a broken gate is a lie about confidence.

## What "agree" means

Two draws agree when the objects that would be committed are the **same object**, compared
as canonical JSON — sorted keys, no whitespace (`stepmold/verify.py:_canonical`). Not the fields
that matter: at this layer nothing knows which fields those are, and a gate that guesses
wrong here fails in the one direction it must not.

Save as `whole_object.py`:

```python
from dataclasses import dataclass

from stepmold.model import FakeModel
from stepmold.pack import Node
from stepmold.verify import Unsure, run_node


@dataclass(frozen=True)
class GatedNode(Node):
    samples: int = 1
    agree: int = 0


def draws(label, *texts):
    """Draw every `texts` in order and require all of them to agree."""
    node = GatedNode(name="extract", type="generate", prompt="Read: {ticket}",
                     samples=len(texts), agree=len(texts))
    try:
        print("%-22s agreed -> %s" % (label, run_node(node, {"ticket": "t"},
                                                      FakeModel(list(texts)))))
    except Unsure as exc:
        print("%-22s unsure -> agreed=%d of %d, distinct=%d"
              % (label, exc.consensus.agreed, exc.consensus.drawn,
                 exc.consensus.distinct))


draws("key order", '{"a": 1, "b": 2}', '{"b": 2, "a": 1}')
draws("1 against 1.0", '{"amount_usd": 1}', '{"amount_usd": 1.0}')
draws("enum same, amount not",
      '{"category": "billing", "amount_usd": 49.99}',
      '{"category": "billing", "amount_usd": 4999}')
```

```console
$ python3 whole_object.py
key order              agreed -> {'b': 2, 'a': 1}
1 against 1.0          unsure -> agreed=1 of 2, distinct=2
enum same, amount not  unsure -> agreed=1 of 2, distinct=2
```

Three readings of that:

| Case | Result | Why |
| --- | --- | --- |
| field order differs | agrees | canonical JSON sorts keys, so formatting is never mistaken for disagreement |
| `1` against `1.0` | disagrees | stricter than Python's `==`, deliberately. A gate whose job is to notice inconsistency should err towards noticing |
| enum matches, amount does not | disagrees | and this is the case the whole design is for. If the next node is a `tool` that spends the amount, two draws that match on the category are not a confident answer |

If you want agreement on less than the node emits, narrow the node — split the enum into
its own gated node and leave the amount to a node with an `assert`. The grammar is the
pack's declaration of what matters, so the thing being agreed on stays the thing being
committed.

One small surprise in that first line: the object returned is the draw that *completed*
the agreement, not the first draw of it, which is why `{'b': 2, 'a': 1}` came back rather
than `{'a': 1, 'b': 2}`. The two are the same object; only the insertion order differs.

## What it costs

n samples cost at most n generations, and often fewer: the loop stops the moment the
answer cannot change (`stepmold/verify.py:run_node`). Two short-circuits, both visible below.

Save as `gate.py`:

```python
from dataclasses import dataclass

from stepmold.model import FakeModel
from stepmold.pack import Node
from stepmold.verify import Unsure, run_node


@dataclass(frozen=True)
class GatedNode(Node):
    """A Node with the two gate keys on it. See "The gate is not a pack key yet"."""

    samples: int = 1
    agree: int = 0


CATEGORY = {"type": "object",
            "properties": {"category": {"enum": ["billing", "technical"]}},
            "required": ["category"]}

BILLING = '{"category": "billing"}'
TECHNICAL = '{"category": "technical"}'


def node(samples, agree):
    return GatedNode(name="classify", type="generate", prompt="Classify: {ticket}",
                     grammar=CATEGORY, samples=samples, agree=agree)


state = {"ticket": "charged twice for order A-1001"}

agreeing = FakeModel([BILLING, BILLING, BILLING])
print("3/2, all agree      ->", run_node(node(3, 2), state, agreeing),
      "in", agreeing.call_count, "generations")

split = FakeModel([BILLING, TECHNICAL, BILLING])
print("3/2, split 2-1      ->", run_node(node(3, 2), state, split),
      "in", split.call_count, "generations")

consensus = {}
unanimous = FakeModel([BILLING, TECHNICAL, BILLING])
try:
    run_node(node(3, 3), state, unanimous, consensus=consensus)
except Unsure as exc:
    print("3/3, split 2-1      ->", exc)
    print("  consensus:", consensus["classify"])
    print("  closest:  ", exc.value, "(committed by nobody)")
```

```console
$ python3 gate.py
3/2, all agree      -> {'category': 'billing'} in 2 generations
3/2, split 2-1      -> {'category': 'billing'} in 3 generations
3/3, split 2-1      -> node 'classify' is unsure: 1 of 2 draws agreed and 3 had to; 2 generation(s) spent
  consensus: Consensus(node='classify', asked=3, drawn=2, agreed=1, required=3, generations=2, distinct=2)
  closest:   {'category': 'billing'} (committed by nobody)
```

| Line | Asked for | Paid | Why |
| --- | --- | --- | --- |
| 1 | 3 draws | 2 | two matched, so a third could not change the outcome |
| 2 | 3 draws | 3 | the split forced the tie-break draw |
| 3 | 3 draws | 2 | `agree: 3` was already unreachable after two draws differed |

So the ordinary price of `samples: 3, agree: 2` on a node the model is confident about is
**one extra generation**, not two. The price is paid where the model is inconsistent —
which is the node you wanted to spend it on.

Two costs the arithmetic does not show:

* **The ladder is per draw.** A draw that fails verification climbs its own retry ladder,
  so a node with `retries: 2` and `samples: 3` can spend nine generations in the worst
  case. `Consensus.generations` counts the bill; `Consensus.drawn` counts the answers.
* **A rejected draw does not become a dissenting voice.** A draw that spends its whole
  ladder fails the node with `NodeFailed`, because a node that could not produce a valid
  answer has produced no evidence about anything.

## Unsure is not rejected

A rejection means the output was **invalid** — the retry ladder answers that. Disagreement
means every output was **valid** and the model was not consistent, which no re-sample
fixes. So `Unsure` is raised instead, and it is a sibling of `NodeFailed` rather than a
subclass of it: a walker that wants to send both to `on_fail` has to say so.

The walker's routing (`stepmold/graph.py`, the `except Unsure` clause):

| Node declares | Where an unsure node goes |
| --- | --- |
| `on_unsure:` | there. The log line is `edge.on_unsure` |
| only `on_fail:` | there, as the conservative reading of a pack that never considered it. The log line is `edge.on_fail` |
| neither | nothing is committed and the run aborts with `Unsure` |

Nothing the node produced is committed on any of those paths. `Unsure.value` carries the
answer that came closest (the largest group's, ties to the earliest draw) so a human queue
can show it — but committing it is a decision the caller has to make deliberately.

`Consensus` is counts only, and that is on purpose: it is logged, checkpointed and printed
by `stepmold eval`, and a record that holds no model output is safe in all three without anyone
having to remember that it is. `distinct` is the field to watch when tuning — a 2-2 split
is a node with two defensible readings, four different answers is a node that is guessing.

Save as `route.py`:

```python
from dataclasses import dataclass

from stepmold.eval import evaluate
from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import Edge, EvalCase, Node, Pack


@dataclass(frozen=True)
class GatedNode(Node):
    samples: int = 1
    agree: int = 0


CATEGORY = {"type": "object",
            "properties": {"category": {"enum": ["billing", "technical"]}},
            "required": ["category"]}

pack = Pack(
    path=".", name="gate_demo", version=1, entry="classify", model=None,
    nodes={
        "classify": GatedNode(name="classify", type="generate",
                              prompt="Classify: {ticket}", grammar=CATEGORY,
                              output="category", samples=2, on_unsure="human"),
        "human": Node(name="human", type="end", output=["category"]),
        "done": Node(name="done", type="end", output=["category"]),
    },
    edges=[Edge(source="classify", target="done")],
    evalset=[EvalCase(input={"ticket": "charged twice"},
                      expect={"category": "billing"}, name="a split ticket")],
)


def split():
    return FakeModel(['{"category": "billing"}', '{"category": "technical"}'])


result = run(pack, split(), {"ticket": "charged twice"})
print("path:      ", result.path)
print("end_node:  ", result.end_node)
print("failures:  ", [(f.node, f.reason) for f in result.failures])
print("RunResult has an .unsure list:", hasattr(result, "unsure"))

report = evaluate(pack, split)
print(report.tier_summary())
print("escalation kinds:", [(e.node, e.kind) for e in report.cases[0].escalations])
```

```console
$ python3 route.py
path:       ['classify', 'human']
end_node:   human
failures:   [('classify', "node 'classify' is unsure: 1 of 2 draws agreed and 2 had to; 2 generation(s) spent")]
RunResult has an .unsure list: False
gate_demo: 1 case — 0 auto, 1 escalated, 0 failed
  auto         0.0%   accuracy n/a — no case ran without escalating
  escalated  100.0%   at classify=1
  failed       0.0%
escalation kinds: [('classify', 'failed')]
```

Every word of an `Unsure` message is counts and node names — no model output — so unlike a
`NodeFailed` it is safe at any log level whole.

## Independence is not free

The gate measures agreement between draws. If the draws are not independent, it measures
nothing, and it does so silently: identical requests return identical answers, the draws
"agree", and the pack reports full confidence it never earned. **A gate that always agrees
is worse than no gate**, because no gate is at least honest about what it does not know.

stepmold's half of the bargain is to make every generation after the very first ask for
something no other generation of that node asked for (`stepmold/verify.py:sampling_for`). Draw
0 rung 0 asks for nothing, so a pack that runs greedily stays byte-for-byte reproducible
and turning a gate on does not change the answer the node was already giving — the extra
draws are the check, not the answer.

Save as `independence.py`:

```python
import json
import sys
from dataclasses import dataclass

from stepmold.backends.openai_compat import OpenAICompatModel
from stepmold.log import configure
from stepmold.model import FakeModel
from stepmold.pack import Node
from stepmold.verify import run_node, sampling_for

print("what each draw asks for, before any rejection:")
for draw in range(3):
    print("  draw %d, rung 0 ->" % draw, sampling_for(0, draw))

model = OpenAICompatModel(base_url="http://localhost:8000/v1", model="qwen3-8b")
payload = model.build_payload("Classify: ...", None, 64, sampling=sampling_for(0, 1))
print("what goes on the wire for draw 1:",
      json.dumps({key: payload[key] for key in ("model", "temperature", "seed")},
                 sort_keys=True))


@dataclass(frozen=True)
class GatedNode(Node):
    samples: int = 1
    agree: int = 0


configure(level="warning", stream=sys.stdout)
run_node(GatedNode(name="classify", type="generate", prompt="Classify: {ticket}",
                   samples=3, agree=2),
         {"ticket": "charged twice"}, FakeModel(['{"category": "billing"}'] * 3))
```

```console
$ python3 independence.py
what each draw asks for, before any rejection:
  draw 0, rung 0 -> None
  draw 1, rung 0 -> Sampling(temperature=0.5, seed=65536)
  draw 2, rung 0 -> Sampling(temperature=0.5, seed=131072)
what goes on the wire for draw 1: {"model": "qwen3-8b", "seed": 65536, "temperature": 0.5}
11:37:46.693 WARNING stepmold.verify node.samples.blind node=classify samples=3 model=FakeModel reason="backend takes no sampling hint, so extra draws repeat the first"
```

So the request stepmold sends for an extra draw carries **`temperature` and `seed`**, and only
those two (`stepmold/backends/openai_compat.py:build_payload`). The seed rides along because a
server pinned to temperature 0 by policy ignores a temperature it was told to ignore, and
a per-request seed is the only knob left. Every draw gets a seed nothing else in that node
uses — `DRAW_SEED_STRIDE` is `1 << 16`, which is why draw 1 asks for 65536 — so draw 2's
first rung is never the same request as draw 3's.

The temperature is the same for every draw on purpose. A ramp across draws would measure
the ramp rather than the model; a rejection is what makes the ladder climb, and an extra
draw is not a rejection.

### The half stepmold cannot check

That warning fires on a signature, not on behaviour. `accepts_sampling`
(`stepmold/codegen.py`) asks whether the **client object's** `generate` declares a `sampling`
keyword. `FakeModel` does not, so the line above appears. `OpenAICompatModel` does — so
**it never warns**, however the server behind it behaves.

That leaves one gap you have to close yourself:

| Who | Can stepmold tell? |
| --- | --- |
| a client that cannot carry a sampling hint | yes — `node.samples.blind` at WARNING, once per sampled node |
| a server that accepts `temperature` and `seed` and ignores them | **no.** The request is well-formed and the response is a valid answer |

Before you trust a gate, verify your own server honours the hint. Send the same prompt
twice with different `seed` values and confirm the completions differ:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b","temperature":0.5,"seed":65536,
       "messages":[{"role":"user","content":"Name one colour."}]}'
```

Run it again with `"seed":131072`. If the two completions are byte-identical, your server
is not varying, and `samples: 3` on that server buys three copies of one draw at three
times the price. Fix the server, or do not turn the gate on.

(That command needs a live inference server, so it is the one block on this page with no
pasted output — there is nothing offline for it to talk to.)

## Reading the tier split

`stepmold eval --tiers` classifies every case into three buckets (`stepmold/eval.py`) rather than
reporting one pass rate:

```console
$ python3 -m stepmold eval examples/support_triage --tiers
support_triage: 12/12 cases passed
support_triage: 12 cases — 12 auto, 0 escalated, 0 failed
  auto       100.0%   accuracy 100.0% (12/12 correct)
  escalated    0.0%
  failed       0.0%
```

| Bucket | What it means |
| --- | --- |
| `auto` | the run finished with nothing unsure and nothing diverted |
| `escalated` | a node was unsure or ran out of ladder, **and the pack routed it** |
| `failed` | nothing routed it; the run could not finish at all |

The number that matters is not the pass rate. It is the pair **automation rate × auto
accuracy**, and the reason is that a single pass rate averages two populations that are
not comparable: the cases the pack answered alone, and the cases a human is going to
answer.

| Pack | Automated | Accuracy inside the auto bucket | Wrong answers shipped per 1,000 cases |
| --- | --- | --- | --- |
| A | 60% | 99.9% | 0.6 |
| B | 95% | 92% | 76 |

Pack B has the better-looking overall number and ships more than a hundred times as many
unnoticed wrong answers. Pack A hands 400 cases to a person and is the one you would
deploy. One pass rate cannot tell them apart; this is what the split is for. (Those two
rows are arithmetic on hypothetical rates, not a measurement of any pack in this
repository.)

`auto_accuracy` is scored over the auto bucket alone, and is `None` — printed as
`accuracy n/a` — when nothing was automated. A rate over an empty bucket is not a low
score; it is an absent one, and either digit would be a claim nobody measured.

Two details worth knowing before you read a report:

* **The auto bucket is conservative.** A node that was unsure takes its case out of the
  bucket even if the pack chose not to route it, because the auto bucket is a promise
  about answers nobody checked.
* **An unsure escalation is currently labelled `failed`.** `stepmold/eval.py:_unsure` reads
  `run_result.unsure`, and `RunResult` (`stepmold/graph.py`) has no such field — the walker
  records the diversion as an ordinary `Failure` instead. The case is tiered correctly and
  blamed on the right node, as the `route.py` transcript above shows; only the `kind`
  label is wrong. Do not build a report that counts on `kind == "unsure"` yet.

## The gate in a pack

`samples:` and `agree:` are node keys. A `graph.yaml` asks for a gate the same way it
asks for anything else:

```bash
mkdir -p gated/prompts gated/grammars

cat > gated/graph.yaml <<'EOF'
nodes:
  classify:
    type: generate
    output: category
    samples: 3
    agree: 2
    on_unsure: human
  human:
    type: end
    output: [category]
  done:
    type: end
    output: [category]

edges:
  - from: classify
    to: done
EOF

cat > gated/manifest.yaml <<'EOF'
name: gated
version: 1
entry: classify
EOF

printf 'Classify the ticket.\n\nTicket: {ticket}\n' > gated/prompts/classify.txt

cat > gated/grammars/classify.json <<'EOF'
{"type": "object",
 "properties": {"category": {"enum": ["billing", "technical"]}},
 "required": ["category"]}
EOF

printf '{"input": {"ticket": "charged twice"}, "expect": {"category": "billing"}}\n' \
  > gated/evalset.jsonl
```


```console
$ python3 -m stepmold validate gated
gated v1: 3 nodes, 1 edge, 1 evalset case, entry 'classify'
```

Delete the two gate lines, though, and the pack no longer loads — because `on_unsure:` is
then an edge nothing can ever take:

```console
$ python3 - <<'PY'
import pathlib
p = pathlib.Path("gated/graph.yaml")
p.write_text("".join(l for l in p.read_text().splitlines(True)
                     if l.strip().split(":")[0] not in ("samples", "agree")))
PY
$ python3 -m stepmold validate gated
stepmold: pack error: graph.yaml: node 'classify' has an 'on_unsure' edge but draws one sample, so nothing can ever make it unsure and that edge can never be taken. Add 'samples: 3' (or more) to give it a gate, or remove 'on_unsure'.
```

That refusal is the point. Until these keys existed, `on_unsure:` loaded fine on every
pack and could never fire — a destination with no road to it, sitting in the file looking
like a safety net. `pack._check_gates` settles the whole gate at load: an `agree` above
`samples`, an `agree` on a single draw, a gate on a tool node, and this dead edge are all
refused by `stepmold validate` rather than on the draw that trips them.

| To use the gate | How |
| --- | --- |
| from a pack file | `samples:` and `agree:` on a generate node, plus `on_unsure:` for where a disagreement goes. `examples/refund_desk` does this on the node guarding a refund |
| from Python | set the same two fields on a `Node`; `verify.gate_for` reads them either way |
| from `stepmold build` | not yet. The compiler emits neither key |

A pack written before the gate existed behaves identically — a node that names neither key
draws once, and the request it sends is byte-for-byte the one it always sent. `None` and
`1` are deliberately different: the first means the author never asked.

## Limits, in one place

| Limit | Where it bites |
| --- | --- |
| No pack file can turn the gate on | `_NODE_KEYS`, `stepmold/pack.py`. A `graph.yaml` with `samples:` is refused at load |
| `on_unsure:` is unreachable from a pack file | it validates and routes, but nothing in a pack can produce an `Unsure` for it to catch |
| A server that ignores `seed` and `temperature` is undetectable | `accepts_sampling` inspects the client class, never the server's behaviour. Verify it yourself |
| `RunResult` publishes no `unsure` list | so `stepmold eval` labels an unsure escalation `kind="failed"` |
| Agreement is whole-object | you cannot ask for agreement on a subset of the fields. Narrow the node instead |
| No example pack in this repository uses the gate | so there is no worked, end-to-end pack to copy — only the scripts above |
| **Nobody has measured what agreement buys** | there is no benchmark in this repository for how much a gate raises auto accuracy, or how much of the escalated bucket it moves, against a real model. The mechanism is tested; its value is not quantified |

That last row is the one to hold on to. Everything above says what the gate *does*. How
much it is worth on your workflow is an empirical question about your model, and stepmold does
not currently answer it — run your evalset with the gate on and off and compare the tier
splits yourself.

## See also

| Document | What is in it |
| --- | --- |
| [Pack format](pack-format.md) | every key a pack may carry, and what each one defaults to |
| [Graph and routing](graph.md) | node types, edges, `on_fail`, and what it does and does not catch |
| [Expressions](expressions.md) | the `assert` language — the tier-1 signal, and the one to write first |
| [Testing packs](testing.md) | evalsets, scripted models, and scoring offline |
