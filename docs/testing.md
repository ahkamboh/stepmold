# Testing a pack offline

A jig pack is scoreable with no GPU, no network and no API key. Every example pack in
this repo ships a scripted model inside itself, so `jig eval` is a CI gate that needs
nothing but Python:

```
$ python3 -m jig eval examples/incident_triage
incident_triage: 13/13 cases passed
```

That works because jig never calls a model directly — it calls anything satisfying the
`Model` protocol (`jig/model.py:Model`), and `FakeModel` is a drop-in. This page
documents the fake script format, the evalset, and the one thing that will cost you real
time: keeping the two in sync.

## Read this first

| Limit | What it means |
| --- | --- |
| Scoring is **exact equality only** | No substring, no tolerance, no set-equality. `["a","b"] != ["b","a"]`. See `jig/eval.py:_compare`. |
| **Nothing validates the fake script against the evalset** | A missing key is not a load error, and neither is a script file that does not exist. `jig validate` exits 0 on both. It surfaces mid-eval as a per-case failure, or as a raw traceback under `jig run`. |
| An **ordered** script is a positional list over the *whole* eval | Every retry and every think stage consumes an entry. Reordering one evalset line shifts everything after it. |
| A keyed **list** value is positional too | It is one queue per key across the whole eval, so it drifts the same way — just per node instead of globally. Five of the six example packs lean on it. |
| A keyed **string** value answers unlimited times | It cannot run out, so it will happily answer a node you forgot you added. |
| Only a key that names the **case** is drift-proof | `examples/incident_triage` is the one pack here that keys per `(node, case)`. It is the style that survives an edit to `evalset.jsonl`. |

## Following along

Transcripts on this page come from two places.

| Named as | Where it comes from |
| --- | --- |
| `examples/<pack>` | One of the six packs already in this repo. Run from the repo root. |
| `ticket_triage` | **Not in the repo.** A scratch pack you build with the block in [The pack this page uses](#the-pack-this-page-uses). Build it before running anything that names it. |
| `drift`, `netpack`, `tagpack`, `twostage`, `broken`, `noscript`, `stdrift`, `sabotage` | Scratch packs derived by a `cp -r` shown at the point of use. |
| `check_script.py`, `probe_*.py` | Scripts given in full at the point of use. Save them at the repo root. |

Everything is created in the repo root and none of it is tracked; the last section on
this page deletes all of it again.

## The pack this page uses

`ticket_triage`: five nodes (two `generate`, three `end`), one branch, one rescue path.
Paste this whole block at the repo root.

```bash
mkdir -p ticket_triage/prompts ticket_triage/grammars ticket_triage/fakes

cat > ticket_triage/manifest.yaml <<'EOF'
name: ticket_triage
version: 1
entry: classify
model: fake:fakes/script.json
EOF

cat > ticket_triage/graph.yaml <<'EOF'
max_steps: 8

nodes:
  classify:
    type: generate
    max_tokens: 32

  priority:
    type: generate
    max_tokens: 32
    retries: 0
    on_fail: manual

  escalated:
    type: end
    output: [topic, tier]

  logged:
    type: end
    output: [topic, tier]

  manual:
    type: end
    output: [ticket_id]

edges:
  - from: classify
    to: priority

  - from: priority
    to: escalated
    when: {tier: urgent}

  - from: priority
    to: logged
EOF

cat > ticket_triage/prompts/classify.txt <<'EOF'
Task: classify
Ticket: {ticket_id}

Say what this ticket is about: billing, howto, or bug.

Text: {text}
EOF

cat > ticket_triage/prompts/priority.txt <<'EOF'
Task: priority
Ticket: {ticket_id}

Given the topic, say whether this needs a human now (urgent) or not (normal).

Topic: {topic}
Text: {text}
EOF

cat > ticket_triage/grammars/classify.json <<'EOF'
{
  "type": "object",
  "properties": {"topic": {"type": "string", "enum": ["billing", "howto", "bug"]}},
  "required": ["topic"],
  "additionalProperties": false
}
EOF

cat > ticket_triage/grammars/priority.json <<'EOF'
{
  "type": "object",
  "properties": {"tier": {"type": "string", "enum": ["urgent", "normal"]}},
  "required": ["tier"],
  "additionalProperties": false
}
EOF

cat > ticket_triage/evalset.jsonl <<'EOF'
{"name": "double charge", "input": {"ticket_id": "T-1", "text": "I was charged twice for one order"}, "expect": {"topic": "billing", "tier": "urgent"}, "end": "escalated"}
{"name": "how to export", "input": {"ticket_id": "T-2", "text": "how do I export my invoices to csv?"}, "expect": {"topic": "howto", "tier": "normal"}, "end": "logged"}
{"name": "unreadable", "input": {"ticket_id": "T-3", "text": "?????"}, "expect": {"topic": "bug"}, "end": "manual", "rescued": true}
EOF

cat > ticket_triage/fakes/script.json <<'EOF'
{
  "Task: classify\nTicket: T-1": "{\"topic\": \"billing\"}",
  "Task: priority\nTicket: T-1": "{\"tier\": \"urgent\"}",
  "Task: classify\nTicket: T-2": "{\"topic\": \"howto\"}",
  "Task: priority\nTicket: T-2": "{\"tier\": \"normal\"}",
  "Task: classify\nTicket: T-3": "{\"topic\": \"bug\"}",
  "Task: priority\nTicket: T-3": "I am not JSON"
}
EOF
```

Six script keys, because all three cases reach both generate nodes. The last value is
deliberately not JSON: that is what burns `priority`'s ladder and sends case 3 down
`on_fail: manual`.

Green:

```
$ python3 -m jig validate ticket_triage
ticket_triage v1: 5 nodes, 3 edges, 3 evalset cases, entry 'classify'

$ python3 -m jig eval ticket_triage
ticket_triage: 3/3 cases passed
$ echo $?
0
```

It also runs, offline, on whichever scripted ticket you hand it:

```
$ python3 -m jig run ticket_triage --input '{"ticket_id": "T-2", "text": "how do I export my invoices to csv?"}'
{"tier": "normal", "topic": "howto"}
```

### The minimal form

Three things make a pack scoreable offline: `model: fake:<path>` in `manifest.yaml`, the
JSON script at that path, and `evalset.jsonl`. Nothing else in the pack changes.

The script's values are the **raw completion text** the model would have returned. For a
`generate` node that is a JSON string, so the file holds JSON-inside-JSON: every quote in
the answer is escaped, exactly as in `fakes/script.json` above. That is not a quirk of
the format — it is what a real backend returns over the wire.

## The FakeModel script format

`fakes/script.json` is parsed with `json.load` and handed straight to `FakeModel`
(`jig/cli.py:_fake_model`). It must be a non-empty JSON **array** or JSON **object**.

Two rough edges at that boundary, both real:

```
$ printf '{}' > ticket_triage/fakes/empty.json
$ python3 -m jig eval ticket_triage --model fake:fakes/empty.json
jig: /.../ticket_triage/fakes/empty.json is not valid JSON (FakeModel needs at least one scripted response)
```

The file *is* valid JSON. `_fake_model` wraps the construction and the parse in one
`try`, so `FakeModel`'s own `ValueError` is reported under the parse error's message.
Read past the wording: an empty script is refused, not a malformed one. An empty **array**
is refused with the identical sentence — `__post_init__` raises the same `ValueError` on
both branches:

```
$ printf '[]' > ticket_triage/fakes/emptyarr.json
$ python3 -m jig eval ticket_triage --model fake:fakes/emptyarr.json
jig: /.../ticket_triage/fakes/emptyarr.json is not valid JSON (FakeModel needs at least one scripted response)
```

A script that is neither raises `TypeError`, which `jig/cli.py:main` does not catch — so
what reaches the terminal is a full Python traceback, not a `jig:` line (22 lines on
CPython 3.14; the frame list varies by version). First two and last, the rest elided:

```
$ printf '"x"' > ticket_triage/fakes/str.json
$ python3 -m jig eval ticket_triage --model fake:fakes/str.json
Traceback (most recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
[19 lines elided, through jig/cli.py:326 in _fake_model and jig/model.py:71 in __post_init__]
TypeError: FakeModel script must be a list of responses or a dict keyed by prompt substring, not str
```

### Ordered mode — an array

```json
["{\"topic\": \"billing\"}", "{\"tier\": \"urgent\"}"]
```

Responses come back in order, one per `generate` call, regardless of the prompt
(`jig/model.py:FakeModel._next_ordered`). Past the end it raises `ModelExhausted`:

```python
"""probe_ordered.py — an ordered script, one response past the end."""
from jig.model import FakeModel, ModelExhausted

model = FakeModel(["first", "second"])
for prompt in ("anything", "anything at all", "a third time"):
    try:
        print(model.generate(prompt))
    except ModelExhausted as exc:
        print("ModelExhausted: %s" % exc)
print("call_count: %d" % model.call_count)
print("last prompt: %s" % model.calls[-1].prompt)
```

```
$ python3 probe_ordered.py
first
second
ModelExhausted: FakeModel script has 2 responses; call 3 has nothing to return
call_count: 3
last prompt: a third time
```

Note `call_count: 3`. The exhausted call is still recorded in `FakeModel.calls` — the
call is appended before the script is consulted, so `calls[-1].prompt` is the prompt that
had no answer. That is the fastest way to see what you failed to script. It holds in
keyed mode too.

### Keyed mode — an object keyed by prompt substring

```json
{
  "Task: classify\nTicket: T-1": "{\"topic\": \"billing\"}",
  "Task: priority\nTicket: T-1": "{\"tier\": \"urgent\"}"
}
```

The rules, all from `jig/model.py:FakeModel._keyed`:

| Rule | Detail |
| --- | --- |
| Match is plain substring | `key in prompt`. Not a regex, not a glob, not anchored. |
| **Longest matching key wins** | So a specific key overrides a general one. Among equal-length matches the one declared first in the file wins (`max` keeps the earliest maximum) — but do not lean on that. |
| A **string** value answers every matching prompt | Unlimited times. It never runs out. |
| A **list** value is a queue consumed in order | `pop(0)` per matching call; it does run out. |
| No key matches | `ModelExhausted`, quoting the whole prompt. |

```python
"""probe_keyed.py — longest key wins; a list value is a queue."""
from jig.model import FakeModel, ModelExhausted


def ask(model, prompt):
    try:
        print(model.generate(prompt))
    except ModelExhausted as exc:
        print("ModelExhausted: %s" % exc)


longest = FakeModel({
    "Task: classify": "generic",
    "Task: classify\nTicket: T-2": "specific",
})
ask(longest, "Task: classify\nTicket: T-1\n")
ask(longest, "Task: classify\nTicket: T-2\n")
ask(longest, "Task: classify\nTicket: T-2\n")
ask(longest, "Task: priority\nTicket: T-1\n")

print("---")

queue = FakeModel({"Task: classify": ["one", "two"]})
for _ in range(3):
    ask(queue, "Task: classify\nTicket: T-1\n")
```

```
$ python3 probe_keyed.py
generic
specific
specific
ModelExhausted: FakeModel has no scripted response matching prompt: 'Task: priority\nTicket: T-1\n'
---
one
two
ModelExhausted: FakeModel ran out of scripted responses for key 'Task: classify'
```

The queue lives on a copy of the script (`_keyed_script`), so it is per-`FakeModel`
instance and `FakeModel.scripted` still holds the whole original — which is what makes
`check_script.py` below possible.

### Three keying styles, and what each one costs you

Count keys per generate **stage**, not per node: a `two_stage` node with its own
`.think.txt` header needs two. All five of the list-keyed packs below are shaped that way
(`support_triage`: 4 generate nodes, 1 of them two-stage, 5 keys).

| Style | Keys | Drifts when | Used by |
| --- | --- | --- | --- |
| One **string** per stage | one per stage | never — but it also cannot tell two cases apart, so every case reaching that node gets the same answer | the four `flag_*` nodes of `invoice_extract`, each of which exactly one case reaches |
| One **list** per stage | one per stage, each as long as the number of cases reaching it | any evalset reorder: each key is a queue in evalset order | `content_moderation`, `lead_qualify`, `meeting_actions`, `support_triage`, and the five shared stages of `invoice_extract` |
| One key per `(node, case)` | cases reaching each node | never | `incident_triage` |

`invoice_extract` is in two rows because it mixes: list values for the five stages more
than one case reaches (queues of 12, 12, 12, 8, 8), plain strings for the four flag nodes
that one case each reaches.

The middle row is the majority of this repo and the one to watch: a keyed script is not
automatically drift-proof. `examples/support_triage` has 5 keys and 12 cases — five
queues of twelve — and swapping two evalset lines breaks it exactly the way an ordered
script breaks:

```
$ cp -r examples/support_triage stdrift
$ python3 - <<'PY'
lines = open('stdrift/evalset.jsonl').read().splitlines()
lines[0], lines[1] = lines[1], lines[0]
open('stdrift/evalset.jsonl', 'w').write("\n".join(lines) + "\n")
PY
$ python3 -m jig eval stdrift
support_triage: 10/12 cases passed
  FAIL The app crashes on launch every time s [classify]
    category: expected 'technical', got 'billing'
    order_id: expected None, got 'A-1001'
    amount_usd: expected None, got 49.99
    queue: expected 'eng-support', got 'billing-ops'
  FAIL I was charged twice for order A-1001,  [classify]
    category: expected 'billing', got 'technical'
    order_id: expected 'A-1001', got None
    amount_usd: expected 49.99, got None
    queue: expected 'billing-ops', got 'eng-support'
  failures by node: classify=2
```

### Choosing keys

The key must be a substring of the **rendered** prompt — the template in
`prompts/<node>.txt` with `{placeholders}` filled from run state
(`jig/codegen.py:build_prompt`). The convention `ticket_triage` and
`examples/incident_triage` use is a two-line header naming the node and the case:

```
Task: classify
Ticket: {ticket_id}
```

giving keys like `"Task: classify\nTicket: T-1"`. The node name alone disambiguates
nodes; the case id alone disambiguates cases; you need both, because one script serves
every node of every case.

Two things that are *not* safe to key on: anything a previous node wrote (it changes when
you fix an upstream answer), and anything appended by jig itself. A retry appends
`"Your previous answer was rejected: ..."` (`jig/codegen.py:ERROR_BLOCK`) at the end of
the prompt, which is exactly why the header convention puts the key at the front.

A two-stage node's notes land in one of two places, and the difference matters if you key
near them: if the emit template contains a `{scratchpad}` placeholder the notes are
rendered *inline at that position* and nothing is appended; only when it does not does
jig append `"Your notes from thinking this through: ..."` (`SCRATCHPAD_BLOCK`) at the end.
Either way the two-line header at the top is untouched.

### How many calls a node actually makes

Not one per node per case. The count is what an ordered script must match exactly, and
what a keyed list value must be long enough for:

| Situation | Calls | Source |
| --- | --- | --- |
| A plain `generate` node, first draw accepted | 1 | `jig/verify.py:run_node` |
| A rejected draw | +1 per retry rung; `retries` defaults to **2**, so up to 3 | `jig/verify.py:run_node` (`rungs = node.retries + 1`) |
| `two_stage: true` | +1 for the think stage, per rung that re-thinks | `jig/codegen.py:generate_once` |
| An `assert` node (`type: assert`) | 0 — it evaluates an expression | `jig/graph.py`, and `examples/invoice_extract` has five of them |
| An `end` node | 0 | |
| A node the case never reaches | 0 | branch not taken |
| A prompt naming state nobody wrote | 0 — `MissingVariable` skips the ladder | `jig/verify.py:run_node` |

There are exactly three node types: `generate`, `assert`, `end` (`jig/pack.py:NODE_TYPES`).
Only `generate` ever calls the model. Note the separate `assert:` *key*, which
`examples/incident_triage` puts on `intake` and `route`: it is a condition on a generate
node's output, so it adds no call of its own, but a candidate it rejects spends a rung
like any other rejection.

The retry cost is easy to forget. Score the **first case only** of `ticket_triage` —
where `classify` keeps the default `retries: 2` — against an ordered script whose first
entry is junk:

```python
"""probe_retry.py — what one rejected draw costs an ordered script."""
from jig.eval import evaluate
from jig.model import FakeModel
from jig.pack import load_pack

pack = load_pack("ticket_triage")
model = FakeModel([
    "oops not json",
    "{\"topic\": \"billing\"}",
    "{\"tier\": \"urgent\"}",
])
report = evaluate(pack, model, cases=pack.evalset[:1])
print(report.summary())
print("generations spent: %d" % model.call_count)
```

```
$ python3 probe_retry.py
ticket_triage: 1/1 cases passed
generations spent: 3
```

Three generations for a two-node walk, because the rejected draw ate an entry. Note which
way round that runs: the junk entry is what *causes* the retry, so removing it removes the
third generation too — with only the two good answers the same case passes in two
(verified: `1/1 cases passed`, `call_count == 2`).

That is the trap in sizing a script. An author budgets one entry per node and gets two;
the rejection then makes the walk need three. **A script needs one entry per generation,
not per node**, and a node with `retries: 2` can spend three.

For a `two_stage` node **without** a `prompts/<node>.think.txt`, the think prompt is the
emit prompt plus a suffix — so a key on the shared header matches *both* calls and one
string value serves both. Make `priority` two-stage and record every call:

```
$ cp -r ticket_triage twostage
$ sed -i '' 's|^name: ticket_triage|name: twostage|' twostage/manifest.yaml
$ python3 - <<'PY'
path = 'twostage/graph.yaml'
text = open(path).read().replace(
    "  priority:\n    type: generate\n",
    "  priority:\n    type: generate\n    two_stage: true\n")
open(path, 'w').write(text)
PY
```

```python
"""probe_twostage.py — every call a two_stage node makes."""
from jig.cli import resolve_model
from jig.graph import run
from jig.pack import load_pack

pack = load_pack("twostage")
model = resolve_model(None, pack)
run(pack, model, {"ticket_id": "T-1", "text": "I was charged twice for one order"})
for call in model.calls:
    header = " / ".join(call.prompt.splitlines()[:2])
    print("%-30s grammar=%-5s max_tokens=%d"
          % (header, bool(call.grammar), call.max_tokens))
```

```
$ python3 probe_twostage.py
Task: classify / Ticket: T-1   grammar=True  max_tokens=32
Task: priority / Ticket: T-1   grammar=False max_tokens=256
Task: priority / Ticket: T-1   grammar=True  max_tokens=32
```

(Prompt headers flattened to one line; the middle call is the think stage — no grammar,
`think_max_tokens` (default 256) instead of the node's `max_tokens`.)

With a separate `.think.txt` that has its own header (as `examples/incident_triage` does:
`Task: severity notes` vs `Task: severity call`), the two stages need two keys.

## Wiring the fake into the pack

```yaml
# manifest.yaml
model: fake:fakes/script.json
```

| Behaviour | Detail |
| --- | --- |
| The path resolves **inside the pack** | `jig/cli.py:_fake_model` calls `jig/pack.py:_resolve_inside`. Absolute paths, `..`, and symlinks escaping the root are refused. |
| It applies to `--model` too | `--model fake:/etc/hosts` is refused the same way. The containment is a property of the `fake:` scheme, not of where the spec came from. |
| `fake:` needs no `--allow-pack-model` | Only `openai:` chosen by a manifest does. A fake is local and contained, which is what lets a pack ship its own offline model. See `jig/cli.py:resolve_model`. |
| `--model` overrides the manifest | `python3 -m jig eval ticket_triage --model fake:fakes/ordered.json`, or point at a real backend: `--model openai:http://localhost:8000#qwen3-8b`. |

Refusals, exactly as printed. A symlink is resolved before the check, so pointing one out
of the pack is refused like any other escape:

```
$ python3 -m jig eval ticket_triage --model fake:/etc/hosts
jig: fake: /etc/hosts: absolute paths are not allowed in a pack

$ python3 -m jig eval ticket_triage --model fake:../script.json
jig: fake: ../script.json: resolves outside the pack directory

$ ln -s /etc/hosts ticket_triage/fakes/link.json
$ python3 -m jig eval ticket_triage --model fake:fakes/link.json
jig: fake: fakes/link.json: resolves outside the pack directory

$ python3 -m jig eval ticket_triage --model fake:fakes/nope.json
jig: fake model script not found: /.../ticket_triage/fakes/nope.json
```

### A network manifest cannot be evaluated at all

If a pack's manifest names an `openai:` endpoint rather than a fake, `jig eval` refuses
it — and the advice in the refusal does not apply to `eval`:

```
$ cp -r ticket_triage netpack
$ sed -i '' -e 's|^name: ticket_triage|name: netpack|' \
            -e 's|^model: fake:fakes/script.json|model: openai:http://localhost:8000#qwen3-8b|' \
            netpack/manifest.yaml

$ python3 -m jig eval netpack
jig: this pack's manifest selects a network endpoint ('openai:http://localhost:8000#qwen3-8b'). Pass --model to choose the endpoint yourself, or --allow-pack-model to accept the pack's choice.
$ echo $?
1

$ python3 -m jig eval netpack --allow-pack-model
usage: jig [-h] [--version] {validate,run,eval} ...
jig: error: unrecognized arguments: --allow-pack-model
$ echo $?
2
```

`--allow-pack-model` is registered on `jig run` only (`jig/cli.py:build_parser`), so under
`eval` the only way past it is `--model`:

```
$ python3 -m jig eval netpack --model fake:fakes/script.json
netpack: 3/3 cases passed
```

Ship packs with `model: fake:...` in the manifest and point `--model` at a real backend
when you want one — which is what every example pack here does.

### One instance for the whole eval

`jig/cli.py:_fake_model` returns a single `FakeModel`, and `jig/eval.py:evaluate` uses it
for every case. Through `jig eval`, therefore:

* an ordered script is one continuous sequence across all cases, in evalset order;
* a keyed list value is one queue across all cases, in evalset order.

`evaluate` has two escape hatches the CLI never uses, both worth knowing when you are
debugging one case:

| Argument | What it does |
| --- | --- |
| `model` may be a zero-argument **factory** | Each case gets a fresh model, so an ordered script can be per-case rather than global (`jig/eval.py:_model_for`). |
| `cases=` overrides the pack's evalset | Score a subset, or an `EvalCase` you built in memory. There is no CLI equivalent — `jig eval` always runs the whole file. |

`probe_retry.py` above uses `cases=pack.evalset[:1]`; `probe_cases.py` further down builds
a case that is not in the file at all.

## Keeping the script aligned with the evalset

This is the expensive part of authoring a pack, and jig does not help you with it.

**The rule.** A keyed script must cover every `(node, case)` prompt the evalset actually
visits. In a branching graph that is *not* `nodes x cases`: it is, per node, **the number
of cases that reach that node**. `examples/incident_triage` is the proof — 13 cases, 4
generate nodes but 5 generate *stages* (`severity` is `two_stage`), 53 keys:

| Key prefix | Keys | Why |
| --- | --- | --- |
| `Task: intake check` | 13 | every case |
| `Task: cause category` | 10 | 3 cases are malformed and route straight to `dropped` |
| `Task: severity notes` | 10 | think stage of a `two_stage` node |
| `Task: severity call` | 10 | emit stage of the same node |
| `Task: route and summarise` | 10 | |

**Nothing checks this.** `jig/pack.py:load_pack` validates the manifest, the graph, the
grammars and every case's `end` — it never opens the fake script. Neither does
`jig validate`, which exits 0 even when the script file is gone outright:

```
$ cp -r ticket_triage noscript
$ rm noscript/fakes/script.json
$ python3 -m jig validate noscript
ticket_triage v1: 5 nodes, 3 edges, 3 evalset cases, entry 'classify'
$ echo $?
0
$ python3 -m jig eval noscript
jig: fake model script not found: /.../noscript/fakes/script.json
```

A gap shows up only when a run walks into it, and it shows up differently depending on
the command:

| Command | A missing key looks like |
| --- | --- |
| `jig eval` | a per-case failure. `jig/eval.py:_run_case` catches every `Exception`, so the case fails and the rest of the eval continues. |
| `jig run` | an uncaught traceback. `ModelExhausted` is a `RuntimeError`, not a `JigError`, so `jig/cli.py:main` does not catch it. |

Drop one key and see both. Under eval — note that the node blamed is the node that asked,
and the message quotes the full prompt you failed to script:

```
$ python3 - <<'PY'
import json
script = json.load(open("ticket_triage/fakes/script.json"))
del script["Task: priority\nTicket: T-3"]
json.dump(script, open("ticket_triage/fakes/missing.json", "w"), indent=2)
PY

$ python3 -m jig eval ticket_triage --model fake:fakes/missing.json
ticket_triage: 2/3 cases passed
  FAIL unreadable [priority]
    error: ModelExhausted: FakeModel has no scripted response matching prompt: 'Task: priority\nTicket: T-3\n\nGiven the topic, say whether this needs a human now (urgent) or not (normal).\n\nTopic: bug\nText: ?????\n'
  failures by node: priority=1
```

That message is the recipe, not just the error. The same gap under `jig run` is a
traceback, ending:

```
$ python3 -m jig run ticket_triage --model fake:fakes/missing.json --input '{"ticket_id": "T-3", "text": "?????"}'
Traceback (most recent call last):
[frames elided]
  File "/.../jig/model.py", line 103, in _keyed
    raise ModelExhausted(
        "FakeModel has no scripted response matching prompt: %r" % prompt
    )
jig.model.ModelExhausted: FakeModel has no scripted response matching prompt: 'Task: priority\nTicket: T-3\n\nGiven the topic, say whether this needs a human now (urgent) or not (normal).\n\nTopic: bug\nText: ?????\n'
```

### Recipe: build the script from the failures

1. Write the evalset case first. Give every case's input a stable id (`ticket_id`,
   `alert_id`) and put it in every prompt header.
2. Run `python3 -m jig eval <pack>`. The first missing key names itself in the
   `ModelExhausted` text.
3. Copy the first two lines of the quoted prompt as the key; write the answer you want.
4. Repeat. Each round gets one node deeper, because a node only renders its prompt once
   the node before it has committed.

This converges fast, and it is the only method that guarantees your keys match the
prompts as *rendered* rather than as you remember writing them.

### Recipe: keep it aligned afterwards

A script drifts in the direction the eval cannot see: a **stale** key costs nothing and
stays forever. Add a check that fails on unused keys. This one needs nothing but jig:

```python
"""check_script.py <pack> [model-spec] — report script keys the evalset never used."""
import sys
from jig.cli import resolve_model
from jig.eval import evaluate
from jig.pack import load_pack

pack = load_pack(sys.argv[1])
model = resolve_model(sys.argv[2] if len(sys.argv) > 2 else None, pack)
report = evaluate(pack, model)
script = model.scripted
print(report.summary())
if isinstance(script, dict):
    unused = [k for k in script
              if not any(k in call.prompt for call in model.calls)]
    for key in unused:
        print("  UNUSED KEY %r" % key)
    print("  %d of %d keys used, %d model calls"
          % (len(script) - len(unused), len(script), model.call_count))
else:
    print("  ordered script: %d of %d responses consumed"
          % (model.call_count, len(script)))
```

Against the worked example with one leftover key added:

```
$ python3 - <<'PY'
import json
script = json.load(open("ticket_triage/fakes/script.json"))
script["Task: classify\nTicket: T-9"] = "{\"topic\": \"bug\"}"
json.dump(script, open("ticket_triage/fakes/stale.json", "w"), indent=2)
PY

$ python3 check_script.py ticket_triage fake:fakes/stale.json
ticket_triage: 3/3 cases passed
  UNUSED KEY 'Task: classify\nTicket: T-9'
  6 of 7 keys used, 6 model calls

$ python3 check_script.py examples/incident_triage
incident_triage: 13/13 cases passed
  53 of 53 keys used, 53 model calls
```

`FakeModel.calls` (`jig/model.py:Call`) is the whole reason this works: it records every
prompt, grammar and `max_tokens` the run spent.

### Why ordered scripts break

`drift/` is a copy of the worked example with its first two evalset lines swapped and
nothing else changed (the report line still says `ticket_triage` — that is the manifest's
`name:`, not the directory). The ordered script is the same six answers as the keyed one, in the *original*
evalset order — five good draws and the deliberate non-JSON that burns `priority`:

```
$ cp -r ticket_triage drift
$ python3 - <<'PY'
lines = open('drift/evalset.jsonl').read().splitlines()
lines[0], lines[1] = lines[1], lines[0]
open('drift/evalset.jsonl', 'w').write("\n".join(lines) + "\n")
PY

$ cat > drift/fakes/ordered.json <<'EOF'
[
  "{\"topic\": \"billing\"}",
  "{\"tier\": \"urgent\"}",
  "{\"topic\": \"howto\"}",
  "{\"tier\": \"normal\"}",
  "{\"topic\": \"bug\"}",
  "I am not JSON"
]
EOF
```

Same pack, same cases, two scripts:

```
$ python3 -m jig eval drift
ticket_triage: 3/3 cases passed

$ python3 -m jig eval drift --model fake:fakes/ordered.json
ticket_triage: 1/3 cases passed
  FAIL how to export [classify]
    topic: expected 'howto', got 'billing'
    tier: expected 'normal', got 'urgent'
    <ending>: expected 'logged', got 'escalated'
  FAIL double charge [classify]
    topic: expected 'billing', got 'howto'
    tier: expected 'urgent', got 'normal'
    <ending>: expected 'escalated', got 'logged'
  failures by node: classify=2
```

That same file is a clean pass against the *unswapped* pack, which is the whole point —
nothing about the script changed, only the order of two evalset lines:

```
$ cp drift/fakes/ordered.json ticket_triage/fakes/ordered.json
$ python3 -m jig eval ticket_triage --model fake:fakes/ordered.json
ticket_triage: 3/3 cases passed
```

Use an ordered script only for a single-case fixture or a Python-level unit test where
you want to control the exact sequence, including retries.

### When you cannot tell where the responses went

`--log-level` and `--log-format` are on every subcommand (`jig/cli.py:_observability_options`);
the default is `off` and nothing is printed without them. At `info` you get one
`node.retry` line per rung, naming the reason, and a `run.end` line carrying the
generation count — which is the direct answer to "why did my ordered script run out":

```
$ cat > ticket_triage/fakes/retry.json <<'EOF'
["oops not json", "{\"topic\": \"billing\"}", "{\"tier\": \"urgent\"}"]
EOF

$ python3 -m jig run ticket_triage --model fake:fakes/retry.json --log-level info --input '{"ticket_id": "T-1", "text": "I was charged twice for one order"}'
18:02:36.211 INFO  jig.graph run.start run_id=82b5b44d9eb24c229060a7711e3c340d pack=ticket_triage version=1 entry=classify resumed=false max_steps=8 inputs=text,ticket_id
18:02:36.211 WARNING jig.verify node.rejected node=classify attempt=1 cause=verify reason="output was not valid JSON — return a single JSON object and nothing else" of=3
18:02:36.211 INFO  jig.verify node.retry node=classify attempt=2 of=3 temperature=0.5 seed=1 reason="output was not valid JSON — return a single JSON object and nothing else" rethink=false
18:02:36.212 INFO  jig.graph node.ok run_id=82b5b44d9eb24c229060a7711e3c340d node=classify type=generate attempts=2 output=merge duration_ms=0.3
18:02:36.212 INFO  jig.graph node.ok run_id=82b5b44d9eb24c229060a7711e3c340d node=priority type=generate attempts=1 output=merge duration_ms=0.0
18:02:36.212 INFO  jig.graph run.end run_id=82b5b44d9eb24c229060a7711e3c340d pack=ticket_triage end_node=escalated steps=3 generations=3 failures=0 output_keys=2 output_bytes=38 duration_ms=0.7
{"tier": "urgent", "topic": "billing"}
```

(Timestamps, `run_id` and `duration_ms` differ on every run; everything else is stable.)
Logs go to stderr, so the run's JSON on stdout stays pipeable. `--log-format json` gives
one JSON object per line instead.

## evalset.jsonl

One JSON object per line; blank lines are skipped. Loaded by
`jig/pack.py:_load_evalset` into `jig/pack.py:EvalCase`.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `input` | object | yes | The run's inputs. A non-object, or missing, is an `EvalsetError` at load. |
| `expect` | object | yes | Field-by-field expectations, compared exactly. May be `{}`. |
| `name` | string | no | Label in the report. Defaults to `"case N"` where N is the **line number**. |
| `end` | string | no | The end node the run must reach. |
| `rescued` | bool | no | This case is meant to burn a node's ladder and take its `on_fail` edge. Defaults to `false`. |

```json
{"name": "double charge", "input": {"ticket_id": "T-1", "text": "charged twice"}, "expect": {"topic": "billing"}, "end": "escalated"}
```

The `input` / `expect` check is the same code for both keys, and it fires at load, so
`jig validate` catches it:

```
$ cp -r ticket_triage broken
$ cat > broken/evalset.jsonl <<'EOF'
{"name": "fine", "input": {"ticket_id": "T-1", "text": "x"}, "expect": {}}
{"name": "no input", "expect": {"topic": "bug"}}
EOF
$ python3 -m jig validate broken
jig: pack error: evalset.jsonl:2: missing or non-object 'input'
$ echo $?
1
```

```
$ cat > broken/evalset.jsonl <<'EOF'
{"name": "bad expect", "input": {"ticket_id": "T-1", "text": "x"}, "expect": ["topic"]}
EOF
$ python3 -m jig validate broken
jig: pack error: evalset.jsonl:1: missing or non-object 'expect'
```

`name` really does default by line number rather than case index, because
`_load_evalset` enumerates the file handle. A file whose first line is blank reports its
only case as `case 2`:

```
$ printf '\n{"input": {"ticket_id": "T-1", "text": "I was charged twice for one order"}, "expect": {"topic": "howto"}}\n' > broken/evalset.jsonl
$ python3 -m jig eval broken
ticket_triage: 0/1 cases passed
  FAIL case 2 [classify]
    topic: expected 'howto', got 'billing'
  failures by node: classify=1
```

### `end` — because field comparison cannot see routing

`expect` compares values. The branch a run took is not a value, so a pack whose branches
project the same shape can have its **entire routing policy inverted and still score full
marks**. That is not hypothetical: `tests/test_eval_contract.py` reproduces it against a
two-way graph, and it was found by an agent authoring a real pack.

`end` closes it (`jig/eval.py:_compare_ending`). A mismatch is reported under the pseudo
field `<ending>`, as in the routing sabotage further down.

A typo in `end` is caught at **load** time, not silently never-matched
(`jig/pack.py:_check_case_endings`): the name must exist in `graph.yaml` and must be a
node of `type: end`. Either way you get a `PackError` and exit 1 before a single case
runs:

```
$ cat > broken/evalset.jsonl <<'EOF'
{"name": "typo", "input": {"ticket_id": "T-1", "text": "x"}, "expect": {}, "end": "esclated"}
EOF
$ python3 -m jig validate broken
jig: pack error: evalset.jsonl: case 'typo' expects ending 'esclated', which is not a node in graph.yaml

$ cat > broken/evalset.jsonl <<'EOF'
{"name": "not an ending", "input": {"ticket_id": "T-1", "text": "x"}, "expect": {}, "end": "priority"}
EOF
$ python3 -m jig validate broken
jig: pack error: evalset.jsonl: case 'not an ending' expects ending 'priority', but that node is type 'generate', not 'end'
```

This is the one place a mistake in the evalset is caught early. `expect` field names are
not checked against anything, and neither is `rescued`.

`end` is optional, and a case without it still scores on fields alone.

### `rescued` — the on_fail path, as a passing expectation

Without it, a declared rescue path could never be a passing expectation: any run that
recorded a failure failed its case, so the one path an author most wants to prove was the
one the evalset could not score.

`rescued: true` says *this case is supposed to fail a node*. It is checked **both ways**
(`jig/eval.py:_run_case`), so it cannot be used to silence a real failure:

| Case says | Run did | Result |
| --- | --- | --- |
| `rescued: true` | burned a ladder, took `on_fail` | passes — `CaseResult.node` is set to the node that burned, though the human report prints nothing for a passing case (see below) |
| `rescued: true` | completed cleanly | **fails**: "case declares rescued: true but the run completed with no failure" |
| nothing | burned a ladder | **fails** — an undeclared failure is still a failure |
| nothing | completed cleanly | passes |

Row two, verbatim — declare a rescue on the case that sails through:

```
$ cp -r ticket_triage sabotage
$ python3 - <<'PY'
path = "sabotage/evalset.jsonl"
lines = open(path).read().splitlines()
lines[0] = lines[0].replace('"end": "escalated"', '"end": "escalated", "rescued": true')
open(path, "w").write("\n".join(lines) + "\n")
PY

$ python3 -m jig eval sabotage
ticket_triage: 2/3 cases passed
  FAIL double charge [escalated]
    error: case declares rescued: true but the run completed with no failure — either the rescue path is not being exercised, or the flag is wrong
  failures by node: escalated=1
$ rm -rf sabotage
```

Row three — an undeclared failure — is the third sabotage under
[Worked example: break one thing](#worked-example-break-one-thing).

To write one: script the node's answer as something that cannot pass verification (bad
JSON, or valid JSON that violates the grammar), set the node's `retries: 0` so the ladder
is one rung, and point `end` at the node's `on_fail` target.

Note that `rescued` is only checked against the **first** recorded failure
(`run_result.failures[0]`); it is a boolean, not a count, and it does not name which node
was expected to burn. `end` is what pins the rescue to a particular path.

## How scoring works

### Exact match, and nothing else

`jig/eval.py:_compare` uses `!=`. There is no substring match, no numeric tolerance, no
set-equality, no case-insensitivity, no way to supply a custom comparator. A list must be
in the same order. `tagpack` is the smallest pack that shows it:

```bash
mkdir -p tagpack/prompts tagpack/grammars tagpack/fakes

cat > tagpack/manifest.yaml <<'EOF'
name: tagpack
version: 1
entry: tag
model: fake:fakes/script.json
EOF

cat > tagpack/graph.yaml <<'EOF'
max_steps: 4

nodes:
  tag:
    type: generate
    max_tokens: 32

  done:
    type: end
    output: [tags]

edges:
  - from: tag
    to: done
EOF

cat > tagpack/prompts/tag.txt <<'EOF'
Task: tag
Item: {item_id}

List the tags for this item.

Text: {text}
EOF

cat > tagpack/grammars/tag.json <<'EOF'
{
  "type": "object",
  "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
  "required": ["tags"],
  "additionalProperties": false
}
EOF

cat > tagpack/evalset.jsonl <<'EOF'
{"name": "same set, other order", "input": {"item_id": "I-1", "text": "a bug in billing"}, "expect": {"tags": ["a", "b"]}, "end": "done"}
EOF

cat > tagpack/fakes/script.json <<'EOF'
{"Task: tag\nItem: I-1": "{\"tags\": [\"b\", \"a\"]}"}
EOF
```

```
$ python3 -m jig eval tagpack
tagpack: 0/1 cases passed
  FAIL same set, other order [tag]
    tags: expected ['a', 'b'], got ['b', 'a']
  failures by node: tag=1
```

If a field is genuinely order-independent, make the node emit it in a fixed order (say so
in the prompt, or constrain it with an `enum`) rather than hoping the evalset will be
lenient. It will not be.

### What a field is compared against

`jig/eval.py:_actual` looks in the end node's **projected output** first, then the **full
run state**. So `expect` may name an intermediate field the end node does not project —
per-node expectations are the point. In the worked example the `manual` ending projects
only `ticket_id`, yet case 3 expects `topic` and passes, because `topic` is in state.

A key that exists nowhere is reported with a note rather than a value. Here through the
`cases=` override, so no file has to be edited to see it:

```python
"""probe_cases.py — score one ad-hoc case, without touching evalset.jsonl."""
from jig.cli import resolve_model
from jig.eval import evaluate
from jig.pack import EvalCase, load_pack

pack = load_pack("ticket_triage")
case = EvalCase(
    name="wrong input expectation",
    input={"ticket_id": "T-1", "text": "I was charged twice for one order"},
    expect={"ticket_id": "T-9", "nosuchfield": 1},
)
report = evaluate(pack, resolve_model(None, pack), cases=[case])
print(report.summary())
```

```
$ python3 probe_cases.py
ticket_triage: 0/1 cases passed
  FAIL wrong input expectation [<unknown>]
    ticket_id: expected 'T-9', got 'T-1'
    nosuchfield: expected 1, got missing from output
  failures by node: <unknown>=1
```

(The renderer substitutes the note *for* the actual value — hence the slightly odd
"got missing from output".)

### Per-node blame

Every mismatch carries the node that wrote the field, from `RunResult.provenance`. When
several fields are wrong, the case is blamed on the node the walker **reached first**,
not the field listed first in `expect` (`jig/eval.py:_earliest_node`) — key order in a
hand-typed JSON object carries no meaning; the run's visit order does. A node visited
twice in a loop is ranked by its first visit.

Consequences worth knowing:

* A field that came from the run's **inputs** has no provenance, so it cannot be ranked.
  A case whose only mismatches are input fields is blamed `<unknown>`, as above.
* An `<ending>` mismatch is attributed to `path[-2]` — the node that chose the edge.
* A case that fails with an **exception** is blamed on the node that was pending when it
  blew up; `NodeFailed` names its own node.
* A **passing** `rescued` case sets `CaseResult.node` to the node that burned — but
  `Report.summary` skips passing cases entirely, so a green human report shows nothing.
  The node is there in `--json` and on the `Report` object.

`Report.by_node` is the tally of failed cases per blamed node, printed as the last line.

## Reading the report

```
ticket_triage: 2/3 cases passed
  FAIL how to export [classify]
    topic: expected 'howto', got 'billing'
  failures by node: classify=1
```

| Piece | Meaning |
| --- | --- |
| `ticket_triage` | `name:` from `manifest.yaml`, not the directory |
| `2/3 cases passed` | passed / total |
| `FAIL <name> [<node>]` | case name, then the blamed node (`<unknown>` if none) |
| `error: ...` | the case raised, or diverted through `on_fail` without declaring `rescued`, or violated the `rescued` contract. It can appear *alongside* mismatch lines. |
| `<field>: expected X, got Y` | one mismatch; `<ending>` is the pseudo field for `end` |
| `failures by node:` | `Report.by_node`, sorted; omitted when nothing failed |

**Passing cases print nothing at all** — including the burned node of a passing `rescued`
case. `--json` (`jig/cli.py:_report_json`) is the only way to see it from the CLI. It
carries `pack`, `passed`, `failed`, `total`, `by_node`, and per case `name`, `passed`,
`node`, `error`, `expected`, `actual`, `mismatches`:

```
$ python3 -m jig eval ticket_triage --json | python3 -c 'import json,sys
for c in json.load(sys.stdin)["cases"]: print(c["name"], c["passed"], c["node"])'
double charge True None
how to export True None
unreadable True priority
```

`priority` on the last line is the ladder `unreadable` spent on its way to `manual`.
`actual` in the JSON is the end node's projection, so a field asserted from state may not
appear in it.

## `jig run` and `jig eval` are different surfaces

`jig eval` takes `--model`, `--json` and the two log options — nothing else. `jig run`
takes rather more (`jig/cli.py:build_parser`), and none of that applies under `eval`:

| Flag | On | What it does |
| --- | --- | --- |
| `--input '<json>'` | run | The run's inputs, a JSON object. Defaults to `{}`. |
| `--model <spec>` | run, eval | Override the manifest's model. |
| `--allow-pack-model` | run | Accept an `openai:` endpoint the manifest chose. |
| `--state` | run | Print the whole final state instead of the end node's projection. |
| `--store <file>` | run | SQLite file to checkpoint into. |
| `--resume <run-id>` | run | Continue a previous run; needs `--store`. |
| `--run-id <name>` | run | Name this run instead of generating an id. |
| `--json` | eval | Machine-readable report. |
| `--log-level`, `--log-format` | all three | See above. Default `off`. |

```
$ python3 -m jig run ticket_triage --input '{"ticket_id": "T-1", "text": "I was charged twice for one order"}' --state
{"text": "I was charged twice for one order", "ticket_id": "T-1", "tier": "urgent", "topic": "billing"}

$ python3 -m jig run ticket_triage --resume demo-1
jig: --resume needs --store: checkpoints live in the store
```

### Exit codes

| Code | When |
| --- | --- |
| 0 | every case passed |
| 1 | any case failed; or a `PackError` / `JigError` / `ValueError` — including "no model", a fake script outside the pack, an unreadable pack; or an uncaught traceback such as `ModelExhausted` |
| 2 | argparse rejected the command line (`jig eval` with no pack argument, `--allow-pack-model` under `eval`) |

An **empty evalset is not a pass**. `jig/eval.py:evaluate` raises, and the CLI turns it
into exit 1 — but note that `jig validate` is happy with it, so this is a gate `eval`
alone holds:

```
$ : > broken/evalset.jsonl
$ python3 -m jig eval broken
jig: pack 'ticket_triage' has no evalset cases to run — an empty evalset is not a pass
$ echo $?
1
$ python3 -m jig validate broken
ticket_triage v1: 5 nodes, 3 edges, 0 evalset cases, entry 'classify'
$ echo $?
0
```

## Worked example: break one thing

Each sabotage below works on a throwaway copy, so `ticket_triage` itself stays green and
nothing has to be undone. Change one script value first — `T-2`'s classify answer from
`howto` to `billing`:

```
$ rm -rf sabotage && cp -r ticket_triage sabotage
$ python3 - <<'PY'
import json
path = "sabotage/fakes/script.json"
script = json.load(open(path))
script["Task: classify\nTicket: T-2"] = "{\"topic\": \"billing\"}"
json.dump(script, open(path, "w"), indent=2)
PY

$ python3 -m jig eval sabotage
ticket_triage: 2/3 cases passed
  FAIL how to export [classify]
    topic: expected 'howto', got 'billing'
  failures by node: classify=1
$ echo $?
1
```

One field, one node, exit 1. Now break the routing instead — flip the escalation edge,
changing nothing else:

```
$ rm -rf sabotage && cp -r ticket_triage sabotage
$ sed -i '' 's|when: {tier: urgent}|when: {tier: normal}|' sabotage/graph.yaml

$ python3 -m jig eval sabotage
ticket_triage: 1/3 cases passed
  FAIL double charge [priority]
    <ending>: expected 'escalated', got 'logged'
  FAIL how to export [priority]
    <ending>: expected 'logged', got 'escalated'
  failures by node: priority=2
```

Every `expect` field still matches — both endings project `topic` and `tier`, and both
got the right values. Only `end` catches it. Without those three `end:` declarations this
pack would have scored 3/3 with its policy inverted.

And drop `"rescued": true` from case 3, changing nothing else:

```
$ rm -rf sabotage && cp -r ticket_triage sabotage
$ sed -i '' 's|, "rescued": true||' sabotage/evalset.jsonl

$ python3 -m jig eval sabotage
ticket_triage: 2/3 cases passed
  FAIL unreadable [priority]
    error: output was not valid JSON: I am not JSON
  failures by node: priority=1
$ echo $?
1
```

The rescue still happened — the run still ended at `manual` — but an undeclared failure
is a failure.

## Testing jig itself

The repo's own suite is offline for the same reason and by the same mechanism: no test
touches a network or loads a model.

```
$ python3 -m pytest -q
[one long line of dots, elided — a single `x` marks the one expected failure]
----------------------------------------------------------------------
Ran 1079 tests in 25.438s

OK (expected failures=1)
```

(The elapsed time is whatever your machine does; the count and the expected failure are
the parts that mean something.)

`pytest` here is a small stdlib shim in `pytest/`, not the real package. Relevant files:
`tests/test_eval_contract.py` (what `end` and `rescued` are for, tested against
sabotage), and `tests/fixtures/cli_pack/` (a two-node pack with both a correct and a
deliberately wrong fake script).

## Cleaning up

None of the scratch packs on this page belong in the repo:

```
$ rm -rf ticket_triage drift netpack tagpack twostage broken noscript stdrift sabotage
$ rm -f probe_ordered.py probe_keyed.py probe_retry.py probe_twostage.py probe_cases.py check_script.py
```
