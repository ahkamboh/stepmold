# graph.yaml — the walk, the edges, the state

`graph.yaml` is the compiled plan: a map of nodes and a list of edges. `stepmold.graph.run`
walks it one node at a time, and nothing in it ever asks a model where to go. Everything
below was checked against `stepmold/graph.py`, `stepmold/pack.py`, `stepmold/verify.py`, `stepmold/codegen.py`,
`stepmold/state.py`, `stepmold/tools.py`, `stepmold/yamlish.py` and `stepmold/cli.py`.

**How to reproduce anything on this page.**

| | |
| --- | --- |
| Where commands run | the root of a stepmold checkout, where `python3 -m stepmold` resolves |
| Demo packs | created by the shell block in the section that uses them — paste the block, then paste the command |
| Probe scripts | saved next to them, in the same directory, and run with `python3 probe_*.py` |
| Real packs | `examples/` — seven of them, already on disk, no setup |
| What differs run to run | log timestamps, `duration_ms`, and the uuid `run_id` a run gets when `--run-id` is not passed. Nothing else. |

Every transcript below is pasted output, unedited — including the log lines that are
noise.

## Read this first

Four things in this file look more capable than they are. They account for most of the
time people lose on their first pack.

| Looks like | Actually is | Section |
| --- | --- | --- |
| `when:` is an expression language | equality only, against a dotted state path. No `!=`, no `>`, no `not`, no expression of any kind. `when: {amount: "> 500"}` compares the literal string `"> 500"`, and `when: {answer: no}` compares `False` | [`when:`](#when--equality-and-nothing-else) |
| `assert` means one thing | two things. `type: assert` is a **routing node**; `assert:` on a `generate` node is a **verification gate** inside the retry ladder. Different keys, different mechanics | [Node types](#node-types) |
| `on_fail` is the catch-all | it catches exactly two failures from a `generate` node and two more from a `tool` node. A node **without** `on_fail` aborts the whole run, and a backend error, a `DeadEnd`, a `StateCollision`, a missing registry and the step budget are never routed at all | [`on_fail`](#on_fail--what-it-catches-and-what-it-does-not) |
| a pack can ask for the confidence gate | it cannot. The gate is real, and `verify.run_node` runs it — but `samples:` and `agree:` are not node keys (`pack._NODE_KEYS`), so a `graph.yaml` carrying them fails to load and no pack read off disk can ever be `Unsure`. `on_unsure:`, the destination, **is** a key, and today nothing can send a run down it | [Unsure](#unsure--disagreement-is-not-rejection) |

And one thing that is missing rather than misleading: **there is no node type that
computes a value.** Nothing in a graph can write `total = qty * price` into state. See
[No computed-value node](#no-computed-value-node).

## The minimal graph

Two nodes, one edge. This is a complete, valid pack:

```bash
mkdir -p minimal/prompts minimal/grammars minimal/fakes

cat > minimal/graph.yaml <<'EOF'
nodes:
  summarise:
    type: generate
  done:
    type: end
    output: [summary]

edges:
  - from: summarise
    to: done
EOF

cat > minimal/manifest.yaml <<'EOF'
name: minimal
version: 1
entry: summarise
model: fake:fakes/script.json
EOF

cat > minimal/prompts/summarise.txt <<'EOF'
Summarise the text in one short clause, lowercase.

Text: {text}
EOF

cat > minimal/grammars/summarise.json <<'EOF'
{
  "type": "object",
  "properties": {"summary": {"type": "string"}},
  "required": ["summary"],
  "additionalProperties": false
}
EOF

cat > minimal/fakes/script.json <<'EOF'
["{\"summary\": \"the build broke on Friday\"}"]
EOF
```

```
$ python3 -m stepmold validate minimal
minimal v1: 2 nodes, 1 edge, 0 evalset cases, entry 'summarise'

$ python3 -m stepmold run minimal --input '{"text": "The build broke on Friday and nobody noticed until Monday."}'
{"summary": "the build broke on Friday"}
```

A `generate` node needs no keys beyond `type:` — `prompts/<node>.txt` and
`grammars/<node>.json` are found by name (`pack._build_node`). The `fake:` model is a
scripted stand-in (`stepmold/model.py`), which is what lets every example here run offline.

## How a run walks

`stepmold.graph.run` starts at `manifest.yaml`'s `entry` and loops:

1. **Count the step.** Entering any node — including the `end` node — costs one step.
2. **Execute the node.** `generate` renders its prompt, generates, and verifies;
   `tool` calls one of the actions the host registered; `assert` evaluates its
   expression; `end` projects and returns.
3. **Commit**, for a `generate` node that produced a verified value or a `tool` node
   that returned one. Nothing rejected ever reaches state (`verify.run_node`), and a
   tool's return value lands by exactly the same rules as a generation (`graph.commit`).
4. **Choose the next edge**, *after* the commit — so an edge sees the value the node just
   wrote (`graph._next`).
5. Repeat until an `end` node returns a `RunResult`.

Only `end` stops a run normally. There is no implicit termination: a `generate`,
`tool` or `assert` node with no matching outgoing edge raises `DeadEnd`
(`graph._next`).

**The step budget.** `max_steps` in `graph.yaml` caps how many nodes a run may enter;
the default is 100 (`pack.DEFAULT_MAX_STEPS`), it must be a positive integer, and
`run(..., max_steps=N)` overrides it per call. Exceeding it raises `MaxStepsExceeded` —
the run stops, nothing is returned, and `on_fail` does not catch it:

```
stepmold: MaxStepsExceeded: run exceeded max_steps=6 at node 'tick' — the graph is looping
```

Budget the whole walk, not the interesting part: a four-node linear pack costs 4 steps,
because the `end` node is a step too.

**Checkpoints.** When a `store` is passed, a checkpoint is written per node that
completes — but "completes" has one hole in it, and it is not the one you would guess:

| What happened at the node | Checkpoint | `next_node` records |
| --- | --- | --- |
| `generate` committed, an edge matched | yes | the node the edge goes to |
| `generate` committed, **no** edge matched (`DeadEnd`) | yes | the node itself — resume re-runs it |
| `generate` took `on_fail` (spent ladder, or a prompt that would not render) | yes | the `on_fail` target |
| `generate` was `Unsure` and took `on_unsure` — or `on_fail`, when it declares no `on_unsure` | yes | the target it was diverted to |
| `tool` called, **before** the result is committed | yes, when the tool is not `idempotent` | **the node itself**, and the row carries the call — see [Exactly once](#exactly-once--a-tool-call-must-not-happen-twice) |
| `tool` committed, an edge matched | yes — the same step, rewritten | the node the edge goes to, and the recorded call is cleared |
| `tool` committed, **no** edge matched (`DeadEnd`) | only when nothing was recorded; a recorded call already has its row and it is deliberately left standing | the node itself |
| `tool` took `on_fail` (the tool raised, or broke its own contract) | yes | the `on_fail` target |
| `assert` routed, either way | yes | the branch it took |
| `assert` passed but **no** edge matched (`DeadEnd`) | **no** | — nothing is written for that node |
| `end` | yes | `None`, which is how a checkpoint says the run finished |

The gap is real, not a rounding error: `graph.run` wraps the generate branch's `_next`
in `try/except DeadEnd` and checkpoints before re-raising, and the assert branch calls
`_next` bare. Evidence is in
[Checkpoints, resume and replay](#checkpoints-resume-and-replay).

## Node types

| `type` | Does | Can write state | Keys it uses |
| --- | --- | --- | --- |
| `generate` | renders a prompt, generates under the node's grammar, verifies, commits | yes | `output`, `retries`, `assert`, `on_fail`, `on_unsure`, `max_tokens`, `two_stage`, `think_max_tokens`, `prompt`, `grammar` |
| `tool` | calls one action the host registered and commits what it returns | yes | `tool` (required), `output`, `on_fail` |
| `assert` | evaluates `expr` against state and routes | **no** | `expr` (required), `on_fail` |
| `end` | projects `output` out of state and returns | no | `output` (a **list** of state keys) |

Any other `type` is refused at load time (`pack._load_nodes`).

### generate

The only node that produces data. `verify.run_node` runs the ladder: generate, then one
re-sample per `retries` (default 2, so three generations), each rung after the first
carrying a sampling hint and the previous rejection's *safe* feedback. A candidate must
parse as a JSON object, satisfy the node's grammar, and satisfy the node's `assert:`
before `graph.commit` writes it.

`assert:` on a generate node is **verification, not routing**. It is evaluated by
`verify._check_assert` against a *trial* copy of state — the candidate as it would be if
committed — and a failure is a `Rejected`, which spends a rung of the ladder. It is the
full expression language of `stepmold/expr.py`: comparisons, `and`/`or`/`not`, `in`,
arithmetic, indexing, and a fixed helper set (`len`, `lower`, `startswith`, `contains`, …).

```yaml
  emit:
    type: generate
    assert: escalate == (priority == "p0")   # rejected and re-sampled if violated
    on_fail: needs_human                     # ...and diverted if the ladder runs out
```

### assert (the node type)

A deterministic branch. It evaluates `expr` and goes down its normal edges when true, or
to `on_fail` when false. It costs a step and zero generations.

**An assert node cannot write anything into state.** It has no `output` key and
`graph.run` never calls `commit` on its branch — it is a fork, not a computation. But it
is not invisible either. What it leaves behind:

| Where | What it records |
| --- | --- |
| `RunResult.path` | the assert node's own name, followed by the node it branched to |
| the checkpoint, with a `store` | one row for the assert node, whose `next_node` is the branch it took |
| `RunResult.end_node` | the ending — which distinguishes the branches only if they end at *different* `end` nodes |
| `RunResult.state`, `RunResult.provenance` | **nothing at all** |
| `RunResult.failures` | nothing — an assert divert is not a `Failure` |

Here is a pack where both branches converge on the same `end` node, which is the case
that makes `end_node` useless and `path` useful:

```bash
mkdir -p fork/fakes

cat > fork/graph.yaml <<'EOF'
nodes:
  k:
    type: assert
    expr: amount_usd <= 500
    on_fail: over

  under:
    type: assert
    expr: 1 == 1
  over:
    type: assert
    expr: 1 == 1

  z:
    type: end
    output: [amount_usd]

edges:
  - from: k
    to: under
  - from: under
    to: z
  - from: over
    to: z
EOF

cat > fork/manifest.yaml <<'EOF'
name: fork
version: 1
entry: k
model: fake:fakes/unused.json
EOF

cat > fork/fakes/unused.json <<'EOF'
["{}"]
EOF
```

```python
# probe_fork.py
from stepmold.graph import run
from stepmold.pack import load_pack

pack = load_pack("fork")
for amount in (100, 900):
    # No generate node in this pack, so no model is ever called.
    result = run(pack, None, {"amount_usd": amount})
    print("amount_usd=%-4d end_node=%s path=%s state=%s provenance=%s"
          % (amount, result.end_node, result.path, result.state, result.provenance))
```

```
$ python3 probe_fork.py
amount_usd=100  end_node=z path=['k', 'under', 'z'] state={'amount_usd': 100} provenance={}
amount_usd=900  end_node=z path=['k', 'over', 'z'] state={'amount_usd': 900} provenance={}
```

Same `end_node`, same state, different `path`. So:

| You want to know | Read |
| --- | --- |
| which way a fork went, after the fact | `RunResult.path`, or the checkpoint's `next_node` |
| which way a fork went, from the pack's *contract* | `RunResult.end_node` — route the branches to distinct `end` nodes, which is what an evalset case's `end:` field scores (`pack.EvalCase`) |
| which way a fork went, from the run's **output** | nothing. Only a `generate` node can put a value in state, and only state can be projected |

Two more things that surprise people:

* An assert-node divert is **not** recorded in `RunResult.failures`. Only a `generate`
  node's ladder failure is (`graph.run` appends `_failure` on the generate branch only),
  so `failures=0` in the transcripts below is correct even where the run took the
  `on_fail` edge. It also means an evalset case's `rescued: true` cannot describe an
  assert-node divert (`eval` reads `run_result.failures`).
* An assert node with **no** `on_fail` whose expression is false raises `AssertFailed`
  and kills the run.

### tool

Calls one action the host registered and commits what it returns. It is the only node
that can change anything outside the run, and the only one that produces state without
spending a generation.

**A pack never contains an action. It names one.** The functions live on the host's
side, in a `ToolRegistry` the caller passes per run; the node holds a key into it and
nothing else — no import, no dotted path, and deliberately no manifest key, so a pack
you did not write can only reach what you already handed it (`stepmold/tools.py`). At the
command line that is `--tools`, on `run` and `eval` only (`cli._add_tools_option`).

```bash
mkdir -p notify/prompts notify/grammars notify/fakes

cat > notify/graph.yaml <<'EOF'
nodes:
  draft:
    type: generate

  send:
    type: tool
    tool: send_email
    on_fail: apologise

  apologise:
    type: end
    output: [subject]

  done:
    type: end
    output: [subject, receipt]

edges:
  - from: draft
    to: send
  - from: send
    to: done
EOF

cat > notify/manifest.yaml <<'EOF'
name: notify
version: 1
entry: draft
model: fake:fakes/script.json
inputs: [to, incident]
EOF

cat > notify/prompts/draft.txt <<'EOF'
Write a one-line subject for an alert about this incident, lowercase.

Incident: {incident}
EOF

cat > notify/grammars/draft.json <<'EOF'
{
  "type": "object",
  "properties": {"subject": {"type": "string"}},
  "required": ["subject"],
  "additionalProperties": false
}
EOF

cat > notify/fakes/script.json <<'EOF'
["{\"subject\": \"disk full on db-3\"}"]
EOF

cat > notify/evalset.jsonl <<'EOF'
{"name": "alerts ops", "input": {"to": "ops@example.com", "incident": "db-3 disk at 100%"}, "expect": {"subject": "disk full on db-3"}, "end": "done"}
EOF
```

The action itself is the host's file, next to the pack rather than inside it. This one
appends to `/tmp/outbox.txt`, which is the side effect every transcript below counts:

```python
# mailer.py
"""The host's side: the actions this machine is willing to let a pack take."""
from stepmold.tools import ToolRegistry

OUTBOX = "/tmp/outbox.txt"
registry = ToolRegistry()


@registry.register("send_email", reads=["to", "subject"], writes=["receipt"])
def send_email(to, subject):
    """Append one line to the outbox and hand back a receipt."""
    with open(OUTBOX, "a") as outbox:
        outbox.write("%s\t%s\n" % (to, subject))
    with open(OUTBOX) as outbox:
        return {"receipt": "message %d" % len(outbox.readlines())}
```

```
$ python3 -m stepmold validate notify
notify v1: 4 nodes, 2 edges, 1 evalset case, entry 'draft'

$ python3 -m stepmold run notify --tools ./mailer.py --input '{"to": "ops@example.com", "incident": "db-3 disk at 100%"}'
{"receipt": "message 1", "subject": "disk full on db-3"}

$ cat /tmp/outbox.txt
ops@example.com	disk full on db-3
```

What the walker did, in order:

| Step | What happens | Where |
| --- | --- | --- |
| the tool is looked up | by the node's `tool:` key. No registry at all raises `ToolsNotAvailable`; a name the host never registered raises `ToolNotRegistered` | `graph._tool_for` |
| its arguments are built | `{key: state[key] for key in tool.reads}` — exactly the state the tool declared it reads, and nothing else | `tools.Tool.invoke` |
| it is called | once, with those as keyword arguments. A key the tool declared and state does not have is a `ToolContract`, not a dropped argument | `tools.Tool.invoke` |
| what it returned is checked | a dict, or `None` read as `{}`. With `writes:` declared, exactly those keys — a missing one and an extra one are both refused | `tools.Tool._checked` |
| the value is committed | merge mode, or nested under `output:`; `provenance` records the node, and `StateCollision` still refuses a run input | `graph.commit` |
| the edge is chosen | after the commit, exactly as for a `generate` node | `graph._next` |

**No prompt, no grammar, no retry ladder.** A tool is deterministic — same state in, same
call out — so a re-sample of one is just the same call again, and the loader refuses the
keys that would imply otherwise rather than ignoring them (`pack._TOOL_FORBIDDEN_KEYS`):

| Key on a tool node | The loader's reason |
| --- | --- |
| `prompt` | a tool node calls a function, not a model, so there is no prompt to render |
| `grammar` | a tool node's contract is the tool's own `writes`, declared by the host in its registry, not a grammar file in the pack |
| `two_stage`, `max_tokens`, `think_max_tokens` | a tool node never generates |
| `retries` | a re-run tool is a side effect done twice; route the failure with `on_fail` instead of re-attempting it |
| `assert` | `assert:` gates a *generation* before it is committed; a tool node has no retry ladder for a rejection to spend |
| `expr` | `expr` is the assert node's branch condition |

Both refusals below are provoked on a throwaway copy, so the `notify` built above stays
valid for the rest of this page:

```
$ cp -r notify /tmp/notify-retries
$ python3 - <<'PY'
import pathlib
p = pathlib.Path("/tmp/notify-retries/graph.yaml")
p.write_text(p.read_text().replace("    tool: send_email",
                                   "    tool: send_email\n    retries: 2"))
PY
$ python3 -m stepmold validate /tmp/notify-retries
stepmold: pack error: graph.yaml: tool node 'send' carries 'retries'. Those keys belong to a generate or an assert node and nothing would read them here — remove them, or make this a node type that uses them. 'retries': a re-run tool is a side effect done twice; route the failure with `on_fail` instead of re-attempting it.

$ cp -r notify /tmp/notify-toolkey
$ python3 - <<'PY'
import pathlib
p = pathlib.Path("/tmp/notify-toolkey/graph.yaml")
p.write_text(p.read_text().replace("    type: generate",
                                   "    type: generate\n    tool: send_email"))
PY
$ python3 -m stepmold validate /tmp/notify-toolkey
stepmold: pack error: graph.yaml: node 'draft' is type 'generate' but carries 'tool: send_email'. Only a tool node names a tool — set 'type: tool', or drop the key.
```

Committing, nesting and looping are the `generate` node's rules unchanged:

```python
# probe_tool_commit.py — what a tool node commits, and what happens to a tool in a loop.
from stepmold.graph import run
from stepmold.pack import Edge, Node, Pack
from stepmold.tools import ToolRegistry

calls = []
registry = ToolRegistry()


@registry.register("tick", reads=["job"], writes=["count"])
def tick(job):
    calls.append(job)
    return {"count": len(calls)}


loop = Pack(path=".", name="loop", version=1, entry="bump", model=None,
            nodes={"bump": Node(name="bump", type="tool", tool="tick"),
                   "gate": Node(name="gate", type="assert", expr="count < 3",
                                on_fail="z"),
                   "z": Node(name="z", type="end")},
            edges=[Edge("bump", "gate"), Edge("gate", "bump")])
result = run(loop, None, {"job": "sweep"}, tools=registry)
print("in a loop: the tool ran %d times, state %s, path %s"
      % (len(calls), result.state, result.path))

nested = Pack(path=".", name="nested", version=1, entry="a", model=None,
              nodes={"a": Node(name="a", type="tool", tool="tick", output="delivery"),
                     "z": Node(name="z", type="end")},
              edges=[Edge("a", "z")])
result = run(nested, None, {"job": "one"}, tools=registry)
print("output: delivery -> state %s, provenance %s"
      % (result.state, result.provenance))
print("generations spent by a tool node:", result.attempts)
```

```
$ python3 probe_tool_commit.py
in a loop: the tool ran 3 times, state {'job': 'sweep', 'count': 3}, path ['bump', 'gate', 'bump', 'gate', 'bump', 'gate', 'z']
output: delivery -> state {'job': 'one', 'delivery': {'count': 4}}, provenance {'delivery': 'a'}
generations spent by a tool node: {}
```

* **`output:` nests, absence merges** — `{"count": 4}` committed under `output: delivery`
  is `state["delivery"]`, exactly as a generation would be.
* **A tool inside a loop calls every time round.** The record that stops a *resumed* run
  calling twice is cleared the moment the node is left, so the next visit finds nothing
  and calls again (`graph.run`, the `pending_calls[:]` line). That is what a tool in a
  loop is for.
* **`RunResult.attempts` stays empty.** A tool node spends no generations, and `node.ok`
  logs `attempts=0` rather than omitting the field.

Three limits worth knowing before you wire a real action to a pack:

* **`stepmold validate` checks nothing about tools.** It takes no `--tools` flag
  (`cli._add_tools_option` adds it to `run` and `eval` only), and `pack.check_tools` is
  skipped when no registry is passed — so `validate` reports a clean pack whose tool
  names are all wrong. The `validate` line in the first transcript above passed on a
  pack whose tool names it never looked at.
* **A missing registry is discovered at the node, not at the start.** By then the run has
  already spent whatever the earlier nodes cost:

```
$ python3 -m stepmold run notify --log-level info --input '{"to": "ops@example.com", "incident": "db-3 disk at 100%"}'
11:42:47.010 INFO  stepmold.graph run.start run_id=5b3604f061ec4a55bd4d7288e11aa7c2 pack=notify version=1 entry=draft resumed=false max_steps=100 inputs=incident,to
11:42:47.010 INFO  stepmold.graph node.ok run_id=5b3604f061ec4a55bd4d7288e11aa7c2 node=draft type=generate attempts=1 output=merge duration_ms=0.1
11:42:47.010 ERROR stepmold.graph run.error run_id=5b3604f061ec4a55bd4d7288e11aa7c2 pack=notify node=send step=2 error=ToolsNotAvailable reason="node 'send' is a tool node, and this run was given no tools: this pack needs tools; pass tools= to run()" duration_ms=0.5
stepmold: ToolsNotAvailable: node 'send' is a tool node, and this run was given no tools: this pack needs tools; pass tools= to run()
(exit status 1)
```

* **`stepmold eval` really calls the tools.** An evalset is not a dry run: every case that
  reaches a tool node performs the side effect for real. The outbox below is the eval's
  own doing, not the run's:

```
$ rm -f /tmp/outbox.txt
$ python3 -m stepmold eval notify --tools ./mailer.py
notify: 1/1 cases passed

$ cat /tmp/outbox.txt
ops@example.com	disk full on db-3
```

### end

Stops the run. `graph._project` builds the result:

| `output:` | Result |
| --- | --- |
| absent | the **whole state**, inputs included |
| `[a, b]` | `{a: ..., b: ...}` — keys that exist in state; **missing keys are silently dropped** |
| `a` (a bare string) | `{}` — the string is iterated character by character. See below |
| `[a.b]` (a dotted path) | `{}` — `_project` does a **flat** lookup, unlike `when:`. See below |

An `end` node **can only project keys that already exist in state**. It cannot add a
field, cannot rename one, cannot say why it was reached. If three different paths land on
the same `end` node, their outputs are indistinguishable — see the
[worked example](#worked-example), where three genuinely different failures all print an
object of the same two keys.

**The bare-string footgun.** `output: summary` on an `end` node is not a syntax error;
`_project` iterates the string and matches nothing. The CLI refuses the shape before
running (`cli._check_output_shapes`), which is why `stepmold validate` catches it:

```bash
cp -r minimal strung

cat > strung/graph.yaml <<'EOF'
nodes:
  summarise:
    type: generate
  done:
    type: end
    output: summary

edges:
  - from: summarise
    to: done
EOF
```

```
$ python3 -m stepmold validate strung
stepmold: graph.yaml: end node 'done': 'output' must be a list of state keys to project, got 'summary' — write 'output: [summary]' if you meant that one key
```

**But that guard lives in the CLI, not in `pack.load_pack`.** A library caller gets the
silence:

```python
# probe_strung.py
import json

from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import load_pack

pack = load_pack("strung")           # `output: summary` on the end node
model = FakeModel(json.load(open("strung/fakes/script.json")))
result = run(pack, model, {"text": "The build broke on Friday."})

print("end_node:", result.end_node)
print("output  :", result.output)
print("state   :", result.state)
```

```
$ python3 probe_strung.py
end_node: done
output  : {}
state   : {'text': 'The build broke on Friday.', 'summary': 'the build broke on Friday'}
```

**The dotted-path footgun, which nothing catches.** `when:` splits its keys on `.` and
walks into nested values. `output:` on an `end` node does not: `_project` is
`{key: state[key] for key in node.output if key in state}`, a flat lookup. A dotted entry
is a valid list of strings, so `cli._check_output_shapes` passes it, and the projection
comes back empty while the value sits in state:

```bash
mkdir -p dotted/prompts dotted/grammars dotted/fakes

cat > dotted/graph.yaml <<'EOF'
nodes:
  decide:
    type: generate
    output: decision

  done:
    type: end
    output: [decision.action]

edges:
  - from: decide
    to: done
EOF

cat > dotted/manifest.yaml <<'EOF'
name: dotted
version: 1
entry: decide
model: fake:fakes/script.json
EOF

cat > dotted/prompts/decide.txt <<'EOF'
Decide the refund.

  action  - refund or deny
  note    - one short sentence saying why

Ticket: {ticket}
EOF

cat > dotted/grammars/decide.json <<'EOF'
{
  "type": "object",
  "properties": {
    "action": {"type": "string", "enum": ["refund", "deny"]},
    "note": {"type": "string"}
  },
  "required": ["action", "note"],
  "additionalProperties": false
}
EOF

cat > dotted/fakes/script.json <<'EOF'
["{\"action\": \"refund\", \"note\": \"under policy cap\"}"]
EOF
```

```
$ python3 -m stepmold validate dotted
dotted v1: 2 nodes, 1 edge, 0 evalset cases, entry 'decide'

$ python3 -m stepmold run dotted --input '{"ticket": "refund me"}'
stepmold: end node 'done' projected nothing: its 'output' names no key that exists in state (state has: decision, ticket). Fix the node's 'output', or pass --state to print the whole state.

$ python3 -m stepmold run dotted --state --input '{"ticket": "refund me"}'
{"decision": {"action": "refund", "note": "under policy cap"}, "ticket": "refund me"}
```

The run is right, the projection is empty. To get one field out of a committed object,
project the whole object (`output: [decision]`) and pick the field apart in the caller.

That "projected nothing" message is the CLI's second guard: an `end` node whose
projection is empty while state is not stops with exit 1 rather than printing `{}`. A
*partial* miss stays silent, at exit 0 — `output: [summary, reason]` with no `reason` in
state:

```bash
cp -r minimal partial

cat > partial/graph.yaml <<'EOF'
nodes:
  summarise:
    type: generate
  done:
    type: end
    output: [summary, reason]

edges:
  - from: summarise
    to: done
EOF
```

```
$ python3 -m stepmold run partial --input '{"text": "The build broke on Friday."}'
{"summary": "the build broke on Friday"}
```

Use `stepmold run --state` to see everything the run actually held.

## Edges

```yaml
edges:
  - from: classify
    to: escalate
    when: {priority: p0}     # taken only if state["priority"] == "p0"

  - from: classify
    to: normal               # no `when:` — the fallback
```

* Edges are a **list**, and `pack.edges_from` preserves declaration order.
* `graph._next` takes the **first** edge whose `when` matches. First match wins.
* An edge with no `when` (or `when: {}`) always matches — put it **last** or it shadows
  every edge after it.
* No matching edge raises `DeadEnd`. There is no implicit fallthrough.
* `end` nodes may not have outgoing edges; every non-`end` node must have at least one
  (`pack._load_edges`, `pack._check_reachable_targets`).

## `when:` — equality, and nothing else

`when:` is a mapping of *dotted state path* → *expected value*. `graph._matches` is the
entire implementation:

```python
def _matches(when, state):
    if not when:
        return True
    for path, expected in when.items():
        if _lookup(path, state) != expected:
            return False
    return True
```

That is `!=` on a value looked up by splitting the key on `.`. There is no parser, no
operator, no negation. Every pair must match, so multiple pairs are an **AND**; there is
no OR — declare two edges instead.

### What people try, and what happens

Every row below is a run of this probe, against a two-edge graph — `big` when the
condition holds, `small` as the fallback:

```python
# probe_when.py — every row of the `when:` table, run against a real two-edge graph.
# The pack is built in memory so one script can try many mappings; `load_pack`
# builds the same Pack/Node/Edge objects out of graph.yaml. The mapping itself is
# parsed by stepmold's own YAML reader, so what is compared is what a graph.yaml gives.
from stepmold.errors import DeadEnd
from stepmold.graph import run
from stepmold.pack import Edge, Node, Pack
from stepmold.yamlish import parse

STATE = {"kind": "refund", "amount_usd": 900}


def router(edges):
    nodes = {"gate": Node(name="gate", type="assert", expr="1 == 1"),
             "big": Node(name="big", type="end"),
             "small": Node(name="small", type="end")}
    return Pack(path=".", name="probe", version=1, entry="gate", model=None,
                nodes=nodes, edges=edges)


def where(when_yaml, state, fallback=True):
    when = parse("when: %s" % when_yaml)["when"]
    edges = [Edge(source="gate", target="big", when=when)]
    if fallback:
        edges.append(Edge(source="gate", target="small"))
    try:
        # No generate node, so the model is never called and None is enough.
        return run(router(edges), None, dict(state)).end_node
    except DeadEnd as exc:
        return "DeadEnd: %s" % exc


ROWS = [
    ("{kind: refund}", STATE, True),
    ('{amount_usd: "> 500"}', STATE, True),
    ('{kind: "!question"}', STATE, True),
    ("{flag: true}", {"flag": "true"}, True),
    ("{nope: x}", STATE, False),
    ("{nope: null}", STATE, True),
    ("{d.action: refund}", {"d": {"action": "refund"}}, True),
    ("{d.action: refund}", {"d.action": "refund"}, True),
    ("{kind: refund, amount_usd: 900}", STATE, True),
    ("{amount_usd: 900}", {"amount_usd": 900.0}, True),
    ("{d: {action: refund}}", {"d": {"action": "refund"}}, True),
    ("{d: {action: refund}}", {"d": {"action": "refund", "n": 1}}, True),
    ("{answer: no}", {"answer": "no"}, True),
    ('{answer: "no"}', {"answer": "no"}, True),
    ("{tier: off}", {"tier": "off"}, True),
    ("{code: 0700}", {"code": "0700"}, True),
]
for when_yaml, state, fallback in ROWS:
    print("%-32s vs %-40r -> %s" % (when_yaml, state, where(when_yaml, state, fallback)))

# Edge order: first match wins, and an edge with no `when` matches everything.
conditional = Edge(source="gate", target="big", when={"kind": "refund"})
fallback = Edge(source="gate", target="small")
print()
for label, edges in (("fallback declared last ", [conditional, fallback]),
                     ("fallback declared first", [fallback, conditional])):
    print("%s -> %s" % (label, run(router(edges), None, dict(STATE)).end_node))
```

| `when:` written | Intent | Result | Why |
| --- | --- | --- | --- |
| `{kind: refund}` | `kind == "refund"` | `big` | works |
| `{amount_usd: "> 500"}` | `amount_usd > 500` | `small` | compares to the literal string `"> 500"` |
| `{kind: "!question"}` | `kind != "question"` | `small` | compares to the literal string `"!question"` |
| `{flag: true}` vs state `"true"` | truthiness | `small` | `True != "true"`; the YAML scalar's *type* is part of the comparison |
| `{nope: x}`, no fallback | branch on a key a node may not have written | **`DeadEnd`** | a missing path can never equal anything |
| `{nope: null}` | "this key is absent" | `small` | absent is a sentinel, not `None`; `null` means *present and null* |
| `{d.action: refund}` with `d = {"action": "refund"}` | dotted lookup | `big` | works |
| `{d.action: refund}` with a flat key literally named `d.action` | same | `small` | the path is always split on `.`; a key containing a dot is unreachable |
| `{kind: refund, amount_usd: 900}` | AND | `big` | every pair must match |
| `{amount_usd: 900}` vs state `900.0` | number match | `big` | Python `==`, so `900 == 900.0` |
| `{d: {action: refund}}` vs exactly that dict | whole-value match | `big` | `!=` compares the whole value |
| `{d: {action: refund}}` vs `{"action": "refund", "n": 1}` | "contains" | `small` | equality is not a subset test |
| `{answer: no}` vs state `"no"` | the model said "no" | `small` | **`no` is a YAML boolean.** `False != "no"` |
| `{answer: "no"}` vs state `"no"` | same | `big` | quoted, so it stays a string |
| `{tier: off}` vs state `"off"` | the model said "off" | `small` | `off` is a YAML boolean too |
| `{code: 0700}` vs state `"0700"` | a zero-padded code | `small` | `0700` parses as the integer `700` |

```
$ python3 probe_when.py
{kind: refund}                   vs {'kind': 'refund', 'amount_usd': 900}    -> big
{amount_usd: "> 500"}            vs {'kind': 'refund', 'amount_usd': 900}    -> small
{kind: "!question"}              vs {'kind': 'refund', 'amount_usd': 900}    -> small
{flag: true}                     vs {'flag': 'true'}                         -> small
{nope: x}                        vs {'kind': 'refund', 'amount_usd': 900}    -> DeadEnd: no outgoing edge from 'gate' matched the current state
{nope: null}                     vs {'kind': 'refund', 'amount_usd': 900}    -> small
{d.action: refund}               vs {'d': {'action': 'refund'}}              -> big
{d.action: refund}               vs {'d.action': 'refund'}                   -> small
{kind: refund, amount_usd: 900}  vs {'kind': 'refund', 'amount_usd': 900}    -> big
{amount_usd: 900}                vs {'amount_usd': 900.0}                    -> big
{d: {action: refund}}            vs {'d': {'action': 'refund'}}              -> big
{d: {action: refund}}            vs {'d': {'action': 'refund', 'n': 1}}      -> small
{answer: no}                     vs {'answer': 'no'}                         -> small
{answer: "no"}                   vs {'answer': 'no'}                         -> big
{tier: off}                      vs {'tier': 'off'}                          -> small
{code: 0700}                     vs {'code': '0700'}                         -> small

fallback declared last  -> big
fallback declared first -> small
```

### The bare word is the sharpest edge here

The last four rows are one bug, and it is silent: the value in `when:` is parsed by
stepmold's YAML subset (`stepmold/yamlish.py`) *before* it is ever compared, so a bare word that
YAML calls a boolean never equals the string a model emitted.

```python
# probe_scalars.py — how stepmold's YAML subset reads a bare scalar (stepmold/yamlish.py).
from stepmold.yamlish import parse

for text in ["yes", "no", "on", "off", "NO", "Off", "true", "~", "null",
             "0700", "1_000", '"no"', "'no'", "maybe"]:
    print("%-8s -> %r" % (text, parse("v: %s" % text)["v"]))
```

```
$ python3 probe_scalars.py
yes      -> True
no       -> False
on       -> True
off      -> False
NO       -> False
Off      -> False
true     -> True
~        -> None
null     -> None
0700     -> 700
1_000    -> 1000
"no"     -> 'no'
'no'     -> 'no'
maybe    -> 'maybe'
```

| Bare scalar | Becomes | Source |
| --- | --- | --- |
| `true`/`True`/`TRUE`/`yes`/`Yes`/`YES`/`on`/`On`/`ON` | `True` | `yamlish._TRUE` |
| `false`/`False`/`FALSE`/`no`/`No`/`NO`/`off`/`Off`/`OFF` | `False` | `yamlish._FALSE` |
| empty/`~`/`null`/`Null`/`NULL` | `None` | `yamlish._NULL` |
| `0700`, `1_000` | `700`, `1000` | `yamlish._INT` |
| anything else unquoted | the string | `yamlish._scalar` |

If a node's grammar has an enum containing `yes`, `no`, `on` or `off` — a yes/no answer
is the most obvious enum there is — **quote it everywhere it appears in `when:`**. There
is no error, no warning and no failed load; the edge simply never matches, and the run
takes the fallback for the rest of its life.

### What to do instead

**1. Make the model emit the enum, and pin it with `assert:`.** Route on a label, not on
arithmetic. Verify-before-commit means a wrong label never reaches state or an edge:

```bash
mkdir -p bucket/prompts bucket/grammars bucket/fakes

cat > bucket/graph.yaml <<'EOF'
nodes:
  label:
    type: generate
    output: bucket
    retries: 2
    assert: (bucket.tier == "over_cap") == (amount_usd > 500)
    on_fail: unlabelled

  big:
    type: end
    output: [amount_usd, bucket]
  small:
    type: end
    output: [amount_usd, bucket]
  unlabelled:
    type: end
    output: [amount_usd]

edges:
  - from: label
    to: big
    when: {bucket.tier: over_cap}
  - from: label
    to: small
EOF

cat > bucket/manifest.yaml <<'EOF'
name: bucket
version: 1
entry: label
model: fake:fakes/wrong_then_right.json
EOF

cat > bucket/prompts/label.txt <<'EOF'
Label this amount against the 500 USD cap.

  over_cap   - the amount is more than 500
  under_cap  - the amount is 500 or less

Amount: {amount_usd}
EOF

cat > bucket/grammars/label.json <<'EOF'
{
  "type": "object",
  "properties": {"tier": {"type": "string", "enum": ["over_cap", "under_cap"]}},
  "required": ["tier"],
  "additionalProperties": false
}
EOF

cat > bucket/fakes/wrong_then_right.json <<'EOF'
["{\"tier\": \"under_cap\"}", "{\"tier\": \"over_cap\"}"]
EOF

cat > bucket/fakes/always_wrong.json <<'EOF'
["{\"tier\": \"under_cap\"}", "{\"tier\": \"under_cap\"}", "{\"tier\": \"under_cap\"}"]
EOF
```

The model guesses wrong on the first draw; the ladder corrects it and the edge only ever
sees a checked value:

```
$ python3 -m stepmold run bucket --run-id over --log-level info --input '{"amount_usd": 900}'
18:05:08.063 INFO  stepmold.graph run.start run_id=over pack=bucket version=1 entry=label resumed=false max_steps=100 inputs=amount_usd
18:05:08.064 WARNING stepmold.verify node.rejected node=label attempt=1 cause=verify reason="assert failed: (bucket.tier == \"over_cap\") == (amount_usd > 500)" of=3
18:05:08.064 INFO  stepmold.verify node.retry node=label attempt=2 of=3 temperature=0.5 seed=1 reason="assert failed: (bucket.tier == \"over_cap\") == (amount_usd > 500)" rethink=false
18:05:08.064 INFO  stepmold.graph node.ok run_id=over node=label type=generate attempts=2 output=bucket duration_ms=0.3
18:05:08.064 INFO  stepmold.graph run.end run_id=over pack=bucket end_node=big steps=2 generations=2 failures=0 output_keys=2 output_bytes=51 duration_ms=0.8
{"amount_usd": 900, "bucket": {"tier": "over_cap"}}
```

It costs a generation per decision, and a model that never gets it right burns the ladder
and takes `on_fail`:

```
$ python3 -m stepmold run bucket --model fake:fakes/always_wrong.json --run-id burnt --log-level info --input '{"amount_usd": 900}'
18:05:08.128 INFO  stepmold.graph run.start run_id=burnt pack=bucket version=1 entry=label resumed=false max_steps=100 inputs=amount_usd
18:05:08.129 WARNING stepmold.verify node.rejected node=label attempt=1 cause=verify reason="assert failed: (bucket.tier == \"over_cap\") == (amount_usd > 500)" of=3
18:05:08.129 INFO  stepmold.verify node.retry node=label attempt=2 of=3 temperature=0.5 seed=1 reason="assert failed: (bucket.tier == \"over_cap\") == (amount_usd > 500)" rethink=false
18:05:08.129 WARNING stepmold.verify node.rejected node=label attempt=2 cause=verify reason="assert failed: (bucket.tier == \"over_cap\") == (amount_usd > 500)" of=3
18:05:08.129 INFO  stepmold.verify node.retry node=label attempt=3 of=3 temperature=0.8 seed=2 reason="assert failed: (bucket.tier == \"over_cap\") == (amount_usd > 500)" rethink=false
18:05:08.129 WARNING stepmold.verify node.rejected node=label attempt=3 cause=verify reason="assert failed: (bucket.tier == \"over_cap\") == (amount_usd > 500)" of=3
18:05:08.129 WARNING stepmold.graph node.failed run_id=burnt node=label type=generate attempts=3 error=NodeFailed reason="assert failed: (bucket.tier == \"over_cap\") == (amount_usd > 500)" on_fail=unlabelled duration_ms=0.5
18:05:08.129 INFO  stepmold.graph edge.on_fail run_id=burnt node=label to=unlabelled
18:05:08.129 INFO  stepmold.graph run.end run_id=burnt pack=bucket end_node=unlabelled steps=2 generations=3 failures=1 output_keys=1 output_bytes=19 duration_ms=0.8
{"amount_usd": 900}
```

**2. Use an `assert` node for the comparison.** `expr` *is* a real expression language,
so `amount_usd <= 500` belongs there. The node routes on it for free — it just cannot
write the answer into state (see [assert](#assert-the-node-type)).

**3. Compute it before the run.** Run inputs are ordinary state, and `when:` reads them
like anything else. `{"amount_usd": 900, "over_cap": true}` from the caller makes
`when: {over_cap: true}` an exact, free branch.

## Loops

A loop is an edge pointing back at a node the run has already entered. Nothing else
declares one, nothing marks a node as a loop, and there is no loop construct:

```bash
mkdir -p loop/prompts loop/grammars loop/fakes

cat > loop/graph.yaml <<'EOF'
max_steps: 6

nodes:
  tick:
    type: generate

  finished:
    type: end
    output: [round, done]

edges:
  - from: tick
    to: tick
    when: {done: false}
  - from: tick
    to: finished
EOF

cat > loop/manifest.yaml <<'EOF'
name: loop
version: 1
entry: tick
model: fake:fakes/three_rounds.json
EOF

cat > loop/prompts/tick.txt <<'EOF'
You are the tick step. Say which round this is, counting from 1, and whether
the work is done.

Work: {work}
EOF

cat > loop/grammars/tick.json <<'EOF'
{
  "type": "object",
  "properties": {
    "round": {"type": "integer"},
    "done": {"type": "boolean"}
  },
  "required": ["round", "done"],
  "additionalProperties": false
}
EOF

cat > loop/fakes/three_rounds.json <<'EOF'
[
  "{\"round\": 1, \"done\": false}",
  "{\"round\": 2, \"done\": false}",
  "{\"round\": 3, \"done\": true}"
]
EOF

cat > loop/fakes/never_done.json <<'EOF'
{"tick step": "{\"round\": 1, \"done\": false}"}
EOF
```

```
$ python3 -m stepmold run loop --run-id spin --log-level info --input '{"work": "tidy the shed"}'
18:05:34.877 INFO  stepmold.graph run.start run_id=spin pack=loop version=1 entry=tick resumed=false max_steps=6 inputs=work
18:05:34.878 INFO  stepmold.graph node.ok run_id=spin node=tick type=generate attempts=1 output=merge duration_ms=0.1
18:05:34.878 INFO  stepmold.graph node.ok run_id=spin node=tick type=generate attempts=1 output=merge duration_ms=0.0
18:05:34.878 INFO  stepmold.graph node.ok run_id=spin node=tick type=generate attempts=1 output=merge duration_ms=0.0
18:05:34.878 INFO  stepmold.graph run.end run_id=spin pack=loop end_node=finished steps=4 generations=3 failures=0 output_keys=2 output_bytes=26 duration_ms=0.6
{"done": true, "round": 3}
```

Read the price off that transcript before writing one.

| Loop fact | Consequence |
| --- | --- |
| Only a `generate` node can write state ([No computed-value node](#no-computed-value-node)) | **every iteration is a model call.** Three rounds above cost three generations |
| The counter is whatever the model emitted | `round` is not incremented by stepmold. Nothing checks that it went 1, 2, 3 — the fake script above simply said so |
| The exit condition is a `when:` on committed output | the model decides when to stop, by emitting `done: true`. An `assert` node can gate the edge, but it cannot count |
| The only backstop is `max_steps` | there is no iteration limit, no cycle detection, and no timeout |
| `RunResult.attempts` is cumulative per node | a node visited three times, one generation each, reads `{'tick': 3}` — not per visit. The per-visit cost is what `node.ok` logs |

A model that never says `done: true` runs until the budget stops it, and every step of
that is a paid generation:

```
$ python3 -m stepmold run loop --model fake:fakes/never_done.json --run-id runaway --log-level info --input '{"work": "tidy the shed"}'
18:05:34.942 INFO  stepmold.graph run.start run_id=runaway pack=loop version=1 entry=tick resumed=false max_steps=6 inputs=work
18:05:34.942 INFO  stepmold.graph node.ok run_id=runaway node=tick type=generate attempts=1 output=merge duration_ms=0.1
18:05:34.942 INFO  stepmold.graph node.ok run_id=runaway node=tick type=generate attempts=1 output=merge duration_ms=0.0
18:05:34.942 INFO  stepmold.graph node.ok run_id=runaway node=tick type=generate attempts=1 output=merge duration_ms=0.0
18:05:34.942 INFO  stepmold.graph node.ok run_id=runaway node=tick type=generate attempts=1 output=merge duration_ms=0.0
18:05:34.942 INFO  stepmold.graph node.ok run_id=runaway node=tick type=generate attempts=1 output=merge duration_ms=0.0
18:05:34.942 INFO  stepmold.graph node.ok run_id=runaway node=tick type=generate attempts=1 output=merge duration_ms=0.0
18:05:34.942 ERROR stepmold.graph run.error run_id=runaway pack=loop node=tick step=7 error=MaxStepsExceeded reason="run exceeded max_steps=6 at node 'tick' — the graph is looping" duration_ms=0.8
stepmold: MaxStepsExceeded: run exceeded max_steps=6 at node 'tick' — the graph is looping
```

Six generations, no output, exit 1: `MaxStepsExceeded` is not routed to `on_fail` and
nothing is returned. Set `max_steps` to the number of iterations you are willing to pay
for, and note that the run's *other* nodes come out of the same budget.

The library sees the same walk:

```python
# probe_loop.py
import json

from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import load_pack

pack = load_pack("loop")
model = FakeModel(json.load(open("loop/fakes/three_rounds.json")))
result = run(pack, model, {"work": "tidy the shed"})

print("path      :", result.path)
print("steps     :", result.steps)
print("attempts  :", result.attempts)
print("state     :", result.state)
print("provenance:", result.provenance)
```

```
$ python3 probe_loop.py
path      : ['tick', 'tick', 'tick', 'finished']
steps     : 4
attempts  : {'tick': 3}
state     : {'work': 'tidy the shed', 'round': 3, 'done': True}
provenance: {'round': 'tick', 'done': 'tick'}
```

## `on_fail` — what it catches, and what it does not

`on_fail` is a node key (not an edge) naming the node to divert to. `pack._check_reachable_targets`
rejects one pointing at an undefined node.

| Situation | With `on_fail` | Without `on_fail` |
| --- | --- | --- |
| generate: retry ladder spent (bad JSON, schema violation, failed `assert:`) | divert; `Failure(attempts=N)` recorded | **`NodeFailed` — run aborts** |
| generate: prompt names state nobody wrote (`MissingVariable`) | divert; `Failure(attempts=0)` — no generation is spent, the ladder is skipped | **`MissingVariable` — run aborts** |
| generate: backend answered 200 with no text (`EmptyCompletion`) | divert after the ladder — it is treated as a rejection and spends a rung | `NodeFailed` — run aborts |
| generate: independent draws disagreed (`Unsure`) | divert — to `on_unsure` first, when the node declares one | **`Unsure` — run aborts** ([Unsure](#unsure--disagreement-is-not-rejection)) |
| tool node: the tool's own code raised (`ToolFailed`) | divert; `Failure(attempts=0)` — a tool spends no generations | **`ToolFailed` — run aborts** |
| tool node: the tool broke its own declaration (`ToolContract`) | divert; `Failure(attempts=0)` | **`ToolContract` — run aborts** |
| tool node: the tool is not registered, or the run was given no registry | **not caught** — run aborts | run aborts |
| assert node: `expr` is false | divert; **no `Failure` recorded** | **`AssertFailed` — run aborts** |
| assert node: `expr` cannot be evaluated (`ExprError`) | divert; **no `Failure` recorded** | **`ExprError` — run aborts** (not `AssertFailed`: the expression was unanswerable, not false) |
| backend unreachable / 500 (`BackendError`) | **not caught** — run aborts | run aborts |
| no outgoing edge matched (`DeadEnd`) | **not caught** — run aborts | run aborts |
| commit would overwrite a run input (`StateCollision`) | **not caught** — run aborts | run aborts |
| `max_steps` exceeded | **not caught** — run aborts | run aborts |

One probe, one line per row:

```python
# probe_on_fail.py — what `on_fail` catches, and what escapes it.
# The packs are built in memory so one script can vary one node at a time.
from stepmold.errors import RunError
from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import Edge, Node, Pack

GRAMMAR = {
    "type": "object",
    "properties": {"x": {"type": "integer"}, "ticket": {"type": "string"}},
    "required": [],
    "additionalProperties": False,
}


class DeadBackend:
    """A model that cannot be reached at all."""

    def generate(self, prompt, grammar=None, max_tokens=512):
        from stepmold.errors import BackendError

        raise BackendError("connection refused")


def pack_of(node, when=None, max_steps=100, self_loop=False):
    nodes = {
        node.name: node,
        "r": Node(name="r", type="end"),
        "z": Node(name="z", type="end"),
    }
    edges = [Edge(source=node.name, target=node.name if self_loop else "z", when=when)]
    return Pack(path=".", name="probe", version=1, entry=node.name, model=None,
                nodes=nodes, edges=edges, max_steps=max_steps)


def gen(on_fail=None, prompt="say x", assert_expr=None, retries=1):
    return Node(name="a", type="generate", prompt=prompt, grammar=GRAMMAR,
                retries=retries, on_fail=on_fail, assert_expr=assert_expr)


def show(label, pack, model, inputs=None):
    try:
        result = run(pack, model, inputs or {})
    except RunError as exc:
        print("%-28s -> %s: %s" % (label, type(exc).__name__, exc))
    else:
        print("%-28s -> %s" % (label, result.failures or result.end_node))


burnt = lambda on_fail: (pack_of(gen(on_fail=on_fail, assert_expr="x == 99")),
                         FakeModel(['{"x": 1}', '{"x": 1}']))
show("burnt ladder, on_fail", *burnt("r"))
show("burnt ladder, none", *burnt(None))

render = lambda on_fail: (pack_of(gen(on_fail=on_fail, prompt="{absent}")),
                          FakeModel(['{"x": 1}']))
show("render fail, on_fail", *render("r"))
show("render fail, none", *render(None))

false_assert = lambda on_fail: pack_of(
    Node(name="a", type="assert", expr="1 == 2", on_fail=on_fail))
show("assert node false, on_fail", false_assert("r"), None)
show("assert node false, none", false_assert(None), None)

bad_expr = lambda on_fail: pack_of(
    Node(name="a", type="assert", expr="ghost > 1", on_fail=on_fail))
show("assert unevaluable, on_fail", bad_expr("r"), None)
show("assert unevaluable, none", bad_expr(None), None)

show("DeadEnd w/ on_fail",
     pack_of(gen(on_fail="r"), when={"nope": 1}), FakeModel(['{"x": 1}']))
show("backend down w/ on_fail", pack_of(gen(on_fail="r")), DeadBackend())
show("collision w/ on_fail", pack_of(gen(on_fail="r")),
     FakeModel(['{"ticket": "from the node"}']), {"ticket": "from the caller"})
show("max_steps w/ on_fail",
     pack_of(gen(on_fail="r"), max_steps=4, self_loop=True),
     FakeModel(['{"x": 1}'] * 8))
```

```
$ python3 probe_on_fail.py
burnt ladder, on_fail        -> [Failure(node='a', reason='assert failed: x == 99', attempts=2)]
burnt ladder, none           -> NodeFailed: node 'a' failed after 2 attempt(s): assert failed: x == 99
render fail, on_fail         -> [Failure(node='a', reason='prompt needs {absent} but state has nothing', attempts=0)]
render fail, none            -> MissingVariable: prompt needs {absent} but state has nothing
assert node false, on_fail   -> r
assert node false, none      -> AssertFailed: assert node 'a' failed: 1 == 2
assert unevaluable, on_fail  -> r
assert unevaluable, none     -> ExprError: expression references 'ghost', which is not in state
DeadEnd w/ on_fail           -> DeadEnd: no outgoing edge from 'a' matched the current state
backend down w/ on_fail      -> BackendError: connection refused
collision w/ on_fail         -> StateCollision: node 'a' would overwrite 'ticket', which came from the run inputs; give the node its own `output:` key or rename the field in its grammar
max_steps w/ on_fail         -> MaxStepsExceeded: run exceeded max_steps=4 at node 'a' — the graph is looping
```

Notes that matter when you rely on this:

* **The asymmetry is real.** A burnt ladder and an unrenderable prompt take the same edge
  but leave different `attempts` (N vs 0). A `DeadEnd` from the same node does not take
  the edge at all — routing failure is not node failure, and by the time `_next` runs the
  node's output is already committed (`graph.run` checkpoints, then re-raises).
* **`on_fail` is a diversion, not a handler.** The `on_fail` target is an ordinary node,
  executed normally. It may itself fail and take *its own* `on_fail` — chains work.
* Nothing about *why* a node failed reaches the target node's state. `RunResult.failures`
  and the checkpoint hold the reason (for the operator); the graph does not.
* `on_fail` on an `end` node loads and is inert — an `end` node cannot fail.

### A tool node's failures take the same edge

Everything that stops a node producing an output takes the node's `on_fail`, and a tool
node is not an exception: a database that is down and a spent retry ladder are the same
fact to the graph. Two of a tool node's failures are routed and two are not, and the
split is deliberate.

```python
# probe_tool_fail.py — how a tool node's own failures are routed.
# The packs are built in memory so one script can vary one thing at a time.
from stepmold.errors import RunError
from stepmold.graph import run
from stepmold.pack import Edge, Node, Pack
from stepmold.tools import ToolRegistry

MISSING = {}                 # a run whose inputs are empty, distinct from "the default"


def pack_of(node):
    nodes = {node.name: node,
             "rescue": Node(name="rescue", type="end"),
             "z": Node(name="z", type="end")}
    return Pack(path=".", name="probe", version=1, entry=node.name, model=None,
                nodes=nodes, edges=[Edge(source=node.name, target="z")])


def tool_node(on_fail=None, calls="act"):
    return Node(name="a", type="tool", tool=calls, on_fail=on_fail)


def raising():
    """A tool whose own code blows up."""
    registry = ToolRegistry()

    @registry.register("act", reads=["order_id"], writes=["total"])
    def act(order_id):
        raise RuntimeError("the database is down")

    return registry


def returning(value, writes=("total",)):
    """A tool that returns `value`, having declared it writes `writes`."""
    registry = ToolRegistry()

    @registry.register("act", reads=["order_id"], writes=list(writes))
    def act(order_id):
        return value

    return registry


def show(label, node, tools, inputs=None):
    try:
        result = run(pack_of(node), None,
                     {"order_id": 7} if inputs is None else inputs, tools=tools)
    except RunError as exc:
        print("%-30s -> %s: %s" % (label, type(exc).__name__, exc))
    else:
        print("%-30s -> ended at %r, failures %s"
              % (label, result.end_node, result.failures or "none"))


show("tool raised, on_fail", tool_node("rescue"), raising())
show("tool raised, no on_fail", tool_node(), raising())
show("undeclared key, on_fail", tool_node("rescue"), returning({"total": 1, "vat": 0}))
show("nothing to read, on_fail", tool_node("rescue"), returning({"total": 1}), MISSING)
show("not a dict, on_fail", tool_node("rescue"), returning(12))
show("unregistered name, on_fail", tool_node("rescue", calls="ghost"),
     returning({"total": 1}))
show("no registry at all, on_fail", tool_node("rescue"), None)
show("happy path", tool_node("rescue"), returning({"total": 12}))
```

```
$ python3 probe_tool_fail.py
tool raised, on_fail           -> ended at 'rescue', failures [Failure(node='a', reason="tool 'act' on node 'a' failed: RuntimeError: the database is down", attempts=0)]
tool raised, no on_fail        -> ToolFailed: tool 'act' on node 'a' failed: RuntimeError: the database is down
undeclared key, on_fail        -> ended at 'rescue', failures [Failure(node='a', reason="tool 'act' on node 'a' returned undeclared key(s) vat. A tool's `writes` is the contract the graph is built around — widen it deliberately rather than by accident.", attempts=0)]
nothing to read, on_fail       -> ended at 'rescue', failures [Failure(node='a', reason="tool 'act' on node 'a' needs order_id, which state does not have (it has: nothing)", attempts=0)]
not a dict, on_fail            -> ended at 'rescue', failures [Failure(node='a', reason="tool 'act' on node 'a' returned int; a tool must return a dict of the state it writes, or None for nothing", attempts=0)]
unregistered name, on_fail     -> ToolNotRegistered: no tool named 'ghost' on node 'a'. A pack can only call what the host registered (available: act). Register it before the run, or remove the node.
no registry at all, on_fail    -> ToolsNotAvailable: node 'a' is a tool node, and this run was given no tools: this pack needs tools; pass tools= to run()
happy path                     -> ended at 'z', failures none
```

| Failure | Raised by | With `on_fail` | Without |
| --- | --- | --- | --- |
| the tool's own code raised (`ToolFailed`) | `tools.Tool.invoke` | divert; `Failure(attempts=0)` | run aborts |
| the tool broke its own declaration — missing read, wrong type, undeclared key (`ToolContract`) | `tools.Tool.invoke`, `Tool._checked` | divert; `Failure(attempts=0)` | run aborts |
| the pack names a tool the host never registered (`ToolNotRegistered`) | `graph._tool_for` | **not caught** — run aborts | run aborts |
| this run was given no registry at all (`ToolsNotAvailable`) | `graph._tool_for` | **not caught** — run aborts | run aborts |

The two that escape `on_fail` are the two that are facts about the *caller*, not about
this run: a pack naming an action the host never allowed, and a host that started a pack
it cannot let act. Diverting either would finish the workflow with its acting half
missing, which is the one outcome a rescue path must not produce (`stepmold/graph.py`,
`ToolsNotAvailable`).

Two details the transcript is the evidence for:

* **`attempts=0`, always.** A tool spends no generations, so a diverted tool node
  records the honest zero — the same zero an unrenderable prompt records, and for the
  same reason.
* **A `ToolFailed`'s message is not the log's message.** `Failure.reason` holds the
  host function's own exception text; the default-level `node.failed` line carries only
  `tool 'x' raised RuntimeError (detail at DEBUG)`, because a tool raises about the
  thing it was given and that is the caller's data (`graph._safe_reason`).

## Unsure — disagreement is not rejection

A node may be drawn more than once and have its answers compared: `samples:` draws, and
`agree:` of them must match before the answer is accepted (`verify.gate_for`,
`verify.run_node`). When no group of matching draws reaches the threshold, the node does
not fail — it is **unsure**, and that is a different claim.

| | Rejected | Unsure |
| --- | --- | --- |
| What happened | the output was invalid — bad JSON, schema violation, a failed `assert:` | every output was *valid* and the model was not consistent |
| What fixes it | another rung of the ladder | nothing a re-sample can do |
| Where it goes | the ladder, then `on_fail` | `on_unsure`, then `on_fail` |
| The exception | `NodeFailed` | `Unsure` — a `RunError`, **not** a subclass of `NodeFailed`, so `except NodeFailed` does not catch it |

The walker catches `Unsure` ahead of `NodeFailed` on purpose, and routes it by a ladder
of its own (`stepmold/graph.py`, the generate branch):

| The node declares | Where an unsure answer goes | Log line |
| --- | --- | --- |
| `on_unsure: desk` | `desk`, whether or not `on_fail` is also set | `edge.on_unsure` |
| only `on_fail: rescue` | `rescue` — somewhere declared beats nowhere | `edge.on_fail` |
| neither | nothing is committed and the run aborts with `Unsure` | `run.error` |

```python
# probe_unsure.py — where a node goes when its draws disagree.
# `samples`/`agree` are not graph.yaml keys yet (stepmold/pack.py `_NODE_KEYS`), so the node
# is built here with the fields `verify.gate_for` reads. Everything else is the real
# runtime: real draws, real verification, real routing.
import dataclasses
import sys

from stepmold import log
from stepmold.errors import RunError
from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import Edge, Node, Pack

log.configure(level="info", stream=sys.stdout)

GRAMMAR = {
    "type": "object",
    "properties": {"priority": {"type": "string"}},
    "required": ["priority"],
    "additionalProperties": False,
}


@dataclasses.dataclass(frozen=True)
class GatedNode(Node):
    """A generate node carrying the gate's two keys."""

    samples: int = 1
    agree: int = 0


def pack_of(node):
    nodes = {node.name: node,
             "done": Node(name="done", type="end"),
             "desk": Node(name="desk", type="end"),
             "rescue": Node(name="rescue", type="end")}
    return Pack(path=".", name="probe", version=1, entry=node.name, model=None,
                nodes=nodes, edges=[Edge(source=node.name, target="done")])


def classify(**kwargs):
    return GatedNode(name="classify", type="generate", prompt="Classify: {ticket}",
                     grammar=GRAMMAR, samples=3, agree=2, **kwargs)


def show(label, node, script):
    model = FakeModel(script)
    print("\n--- %s" % label)
    try:
        result = run(pack_of(node), model, {"ticket": "the card was charged twice"},
                     run_id=label.replace(", ", "/").replace(" ", "_"))
    except RunError as exc:
        print("%-26s -> %s: %s" % (label, type(exc).__name__, exc))
    else:
        print("%-26s -> ended at %-8r generations=%d state=%s failures=%s"
              % (label, result.end_node, model.call_count, result.state,
                 result.failures or "none"))


THREE_WAYS = ['{"priority": "p0"}', '{"priority": "p1"}', '{"priority": "p2"}']
TWO_AGREE = ['{"priority": "p1"}', '{"priority": "p1"}', '{"priority": "p2"}']

show("agreed, on_unsure", classify(on_unsure="desk"), TWO_AGREE)
show("disagreed, on_unsure", classify(on_unsure="desk", on_fail="rescue"), THREE_WAYS)
show("disagreed, only on_fail", classify(on_fail="rescue"), THREE_WAYS)
show("disagreed, neither", classify(), THREE_WAYS)
```

```
$ python3 probe_unsure.py

--- agreed, on_unsure
11:42:47.181 INFO  stepmold.graph run.start run_id=agreed/on_unsure pack=probe version=1 entry=classify resumed=false max_steps=100 inputs=ticket
11:42:47.182 WARNING stepmold.verify node.samples.blind node=classify samples=3 model=FakeModel reason="backend takes no sampling hint, so extra draws repeat the first"
11:42:47.182 INFO  stepmold.verify node.agreed node=classify agreed=2 of=2 required=2 asked=3 generations=2
11:42:47.182 INFO  stepmold.graph node.ok run_id=agreed/on_unsure node=classify type=generate attempts=2 output=merge duration_ms=0.3
11:42:47.182 INFO  stepmold.graph run.end run_id=agreed/on_unsure pack=probe end_node=done steps=2 generations=2 failures=0 output_keys=2 output_bytes=58 duration_ms=0.7
agreed, on_unsure          -> ended at 'done'   generations=2 state={'ticket': 'the card was charged twice', 'priority': 'p1'} failures=none

--- disagreed, on_unsure
11:42:47.182 INFO  stepmold.graph run.start run_id=disagreed/on_unsure pack=probe version=1 entry=classify resumed=false max_steps=100 inputs=ticket
11:42:47.182 WARNING stepmold.verify node.samples.blind node=classify samples=3 model=FakeModel reason="backend takes no sampling hint, so extra draws repeat the first"
11:42:47.182 WARNING stepmold.verify node.unsure node=classify agreed=1 of=3 required=2 asked=3 distinct=3 generations=3
11:42:47.182 WARNING stepmold.graph node.failed run_id=disagreed/on_unsure node=classify type=generate attempts=3 error=Unsure reason="node 'classify' is unsure: 1 of 3 draws agreed and 2 had to; 3 generation(s) spent" on_fail=rescue duration_ms=0.1
11:42:47.182 INFO  stepmold.graph edge.on_unsure run_id=disagreed/on_unsure node=classify to=desk
11:42:47.182 INFO  stepmold.graph run.end run_id=disagreed/on_unsure pack=probe end_node=desk steps=2 generations=3 failures=1 output_keys=1 output_bytes=40 duration_ms=0.2
disagreed, on_unsure       -> ended at 'desk'   generations=3 state={'ticket': 'the card was charged twice'} failures=[Failure(node='classify', reason="node 'classify' is unsure: 1 of 3 draws agreed and 2 had to; 3 generation(s) spent", attempts=3)]

--- disagreed, only on_fail
11:42:47.182 INFO  stepmold.graph run.start run_id=disagreed/only_on_fail pack=probe version=1 entry=classify resumed=false max_steps=100 inputs=ticket
11:42:47.182 WARNING stepmold.verify node.samples.blind node=classify samples=3 model=FakeModel reason="backend takes no sampling hint, so extra draws repeat the first"
11:42:47.182 WARNING stepmold.verify node.unsure node=classify agreed=1 of=3 required=2 asked=3 distinct=3 generations=3
11:42:47.182 WARNING stepmold.graph node.failed run_id=disagreed/only_on_fail node=classify type=generate attempts=3 error=Unsure reason="node 'classify' is unsure: 1 of 3 draws agreed and 2 had to; 3 generation(s) spent" on_fail=rescue duration_ms=0.1
11:42:47.182 INFO  stepmold.graph edge.on_fail run_id=disagreed/only_on_fail node=classify to=rescue
11:42:47.182 INFO  stepmold.graph run.end run_id=disagreed/only_on_fail pack=probe end_node=rescue steps=2 generations=3 failures=1 output_keys=1 output_bytes=40 duration_ms=0.2
disagreed, only on_fail    -> ended at 'rescue' generations=3 state={'ticket': 'the card was charged twice'} failures=[Failure(node='classify', reason="node 'classify' is unsure: 1 of 3 draws agreed and 2 had to; 3 generation(s) spent", attempts=3)]

--- disagreed, neither
11:42:47.182 INFO  stepmold.graph run.start run_id=disagreed/neither pack=probe version=1 entry=classify resumed=false max_steps=100 inputs=ticket
11:42:47.182 WARNING stepmold.verify node.samples.blind node=classify samples=3 model=FakeModel reason="backend takes no sampling hint, so extra draws repeat the first"
11:42:47.182 WARNING stepmold.verify node.unsure node=classify agreed=1 of=3 required=2 asked=3 distinct=3 generations=3
11:42:47.182 WARNING stepmold.graph node.failed run_id=disagreed/neither node=classify type=generate attempts=3 error=Unsure reason="node 'classify' is unsure: 1 of 3 draws agreed and 2 had to; 3 generation(s) spent" on_fail=- duration_ms=0.1
11:42:47.182 ERROR stepmold.graph run.error run_id=disagreed/neither pack=probe node=classify step=1 error=Unsure reason="node 'classify' is unsure: 1 of 3 draws agreed and 2 had to; 3 generation(s) spent" duration_ms=0.1
disagreed, neither         -> Unsure: node 'classify' is unsure: 1 of 3 draws agreed and 2 had to; 3 generation(s) spent
```

Read the transcript for four things it says quietly:

* **The gate is not free, and it stops early.** `agreed, on_unsure` asked for three draws
  and spent two: the first two matched, and no third draw could change that
  (`node.agreed ... agreed=2 of=2 asked=3 generations=2`). A disagreeing node pays for
  all three.
* **Nothing the unsure node produced reaches state.** `priority` is absent from the state
  of every diverted run. `Unsure.value` carries the answer that came closest, and the
  walker drops it — a caller who decides an unsure answer is good enough for a person to
  look at has to commit it deliberately.
* **The divert *is* recorded in `RunResult.failures`**, unlike an assert-node divert, with
  `attempts` set to the generations the draws cost. Its `reason` is the whole `str(exc)`
  — counts and node names only, no model output, which is why `Unsure` is safe at any log
  level whole (`verify.Unsure`, `graph._failure`).
* **`node.failed` prints `on_fail=rescue` even when the run went to `desk`.** The line is
  logged before the routing decision and reports the node's `on_fail` key; the `edge.*`
  line on the next row is what says where the run actually went.

**The gate's own keys are not pack keys yet.** This is the limit that matters most on
this page: `on_unsure:` is a `graph.yaml` key, validated like any other edge target
(`pack._check_reachable_targets`), but `samples:` and `agree:` are not in
`pack._NODE_KEYS` and are not fields on `pack.Node`. `verify.gate_for` reads them with
`getattr`, so a node loaded from disk always answers "one draw, one answer" —

```
$ cp -r notify /tmp/notify-gate
$ python3 - <<'PY'
import pathlib
p = pathlib.Path("/tmp/notify-gate/graph.yaml")
p.write_text(p.read_text().replace("    type: generate",
                                   "    type: generate\n    samples: 3\n    agree: 2"))
PY
$ python3 -m stepmold validate /tmp/notify-gate
stepmold: pack error: graph.yaml: node 'draft' has unknown key(s): agree, samples
```

— and the probe above builds its node in memory precisely because a pack cannot. Until
those two keys land, `on_unsure:` in a pack on disk is an edge nothing can take.

Three smaller limits, all of them things the code does not do rather than does badly:

* **`Consensus` never leaves the log.** `verify.run_node` will fill a caller's dict with
  the counts (`asked`, `drawn`, `agreed`, `required`, `generations`, `distinct`), but
  `graph.run` does not pass one — there is no `consensus=` in the walker and no field for
  it on `RunResult`. The `node.agreed` and `node.unsure` log lines are the only place
  those numbers are reported.
* **`on_unsure` on any other node type is inert.** The loader accepts it on a `tool` or
  `assert` node — it is checked as a reachable target and never read again, because only
  the generate branch of `graph.run` looks at it.
* **A backend that cannot vary its sampling makes the gate lie**, and stepmold says so at
  WARNING: `node.samples.blind`. Identical requests produce identical answers, the draws
  "agree", and the pack reports a confidence nobody measured. `FakeModel` is exactly such
  a backend — its `generate` takes no `sampling` keyword — which is why that warning is
  on every run in the transcript above. The draws differ there only because the script
  hands back a different answer each call.

## State

One flat dict per run, seeded with the caller's `inputs` and grown by each `generate`
node's commit. Prompts render from it (`{key}`), `when:` reads it, `expr` evaluates
against it.

**Two commit modes** (`graph.commit`):

| Node | Written | Example |
| --- | --- | --- |
| no `output:` — *merge mode* | every top-level key of the generated object drops into state | `{"kind": "refund", "amount_usd": 120}` → `state["kind"]`, `state["amount_usd"]` |
| `output: decision` | the whole object under that one key | → `state["decision"] == {"action": "refund", ...}` |

Merge mode means your **grammar's property names are state keys**. Two nodes with
`{"status": ...}` in their grammars collide. `output:` is the fix, and it is also what
makes dotted `when: {decision.action: refund}` possible — though not dotted
`output:` on an `end` node, which is a flat lookup ([end](#end)).

**Collision rules:**

| Collision | Behaviour |
| --- | --- |
| node overwrites another node's key | **allowed**; `provenance` records the last writer. This is also what a [loop](#loops) does to its own fields on every pass |
| node overwrites a **run input** | **`StateCollision`, run aborts** — in either mode |

```python
# probe_state.py — the three commit collisions, one line each.
from stepmold.graph import StateCollision, run
from stepmold.model import FakeModel
from stepmold.pack import Edge, Node, Pack

GRAMMAR = {
    "type": "object",
    "properties": {"x": {"type": "string"}, "ticket": {"type": "string"}},
    "required": [],
    "additionalProperties": False,
}


def two_nodes(output=None):
    nodes = {
        "a": Node(name="a", type="generate", prompt="a", grammar=GRAMMAR, output=output),
        "b": Node(name="b", type="generate", prompt="b", grammar=GRAMMAR),
        "z": Node(name="z", type="end"),
    }
    edges = [Edge(source="a", target="b"), Edge(source="b", target="z")]
    return Pack(path=".", name="probe", version=1, entry="a", model=None,
                nodes=nodes, edges=edges)


result = run(two_nodes(), FakeModel(['{"x": "from a"}', '{"x": "from b"}']), {})
print("node over node        -> %s  %s" % (result.state, result.provenance))

for label, pack, script in (
    ("node over input", two_nodes(), ['{"ticket": "from a"}']),
    ("output: over an input", two_nodes(output="ticket"), ['{"x": "from a"}']),
):
    try:
        run(pack, FakeModel(script), {"ticket": "from the caller"})
    except StateCollision as exc:
        print("%-21s -> StateCollision: %s" % (label, exc))
```

```
$ python3 probe_state.py
node over node        -> {'x': 'from b'}  {'x': 'b'}
node over input       -> StateCollision: node 'a' would overwrite 'ticket', which came from the run inputs; give the node its own `output:` key or rename the field in its grammar
output: over an input -> StateCollision: node 'a' would overwrite 'ticket', which came from the run inputs; give the node its own `output:` key or rename the field in its grammar
```

`commit` checks every key before writing the first, so a refused commit leaves state
untouched.

**Provenance** (`RunResult.provenance`) maps state key → the node that last wrote it.
Run inputs are deliberately absent from it — that absence *is* how `commit` tells a
caller's value from a node's, so an input and a node-written key are distinguishable
after the fact.

### `scratchpad` is reserved from one side only

`scratchpad` is the name `codegen.think` binds for a two_stage node's notes, and
`pack.RESERVED_STATE_NAMES` lists it. A **node** may not claim it:

```bash
mkdir -p reserved/prompts reserved/grammars reserved/fakes

cat > reserved/graph.yaml <<'EOF'
nodes:
  decide:
    type: generate
    output: scratchpad
  done:
    type: end
    output: [scratchpad]

edges:
  - from: decide
    to: done
EOF

cat > reserved/manifest.yaml <<'EOF'
name: reserved
version: 1
entry: decide
model: fake:fakes/script.json
EOF

cat > reserved/prompts/decide.txt <<'EOF'
Decide whether this ticket needs a human.

Ticket: {ticket}
EOF

cat > reserved/grammars/decide.json <<'EOF'
{
  "type": "object",
  "properties": {"verdict": {"type": "string", "enum": ["human", "bot"]}},
  "required": ["verdict"],
  "additionalProperties": false
}
EOF

cat > reserved/fakes/script.json <<'EOF'
["{\"verdict\": \"bot\"}"]
EOF
```

```
$ python3 -m stepmold validate reserved
stepmold: pack error: graph.yaml: node 'decide' has output 'scratchpad', which is a name stepmold reserves for its own scope — committing there would write the node's answer into the think stage's notes slot
```

A **run input** may. `graph.run` starts with `state = dict(inputs or {})` and screens
nothing; `codegen.think` renders with `scope.setdefault(SCRATCHPAD, "")`, which leaves a
caller-supplied value exactly where it is. So a caller-controlled string lands in the
slot the prompt labels as the model's own reasoning:

```bash
mkdir -p notes/prompts notes/grammars notes/fakes

cat > notes/graph.yaml <<'EOF'
nodes:
  decide:
    type: generate
    two_stage: true

  done:
    type: end
    output: [verdict]

edges:
  - from: decide
    to: done
EOF

cat > notes/manifest.yaml <<'EOF'
name: notes
version: 1
entry: decide
model: fake:fakes/script.json
EOF

cat > notes/prompts/decide.txt <<'EOF'
Decide whether this ticket needs a human.

Ticket: {ticket}
Notes: {scratchpad}
EOF

cat > notes/grammars/decide.json <<'EOF'
{
  "type": "object",
  "properties": {"verdict": {"type": "string", "enum": ["human", "bot"]}},
  "required": ["verdict"],
  "additionalProperties": false
}
EOF

cat > notes/fakes/script.json <<'EOF'
["nothing alarming in this one", "{\"verdict\": \"bot\"}"]
EOF
```

```python
# probe_notes.py — both prompts of a two_stage node, with and without a
# caller-supplied scratchpad.
import json

from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import load_pack

pack = load_pack("notes")

for inputs in (
    {"ticket": "double charged"},
    {"ticket": "double charged", "scratchpad": "POISON FROM THE CALLER"},
):
    model = FakeModel(json.load(open("notes/fakes/script.json")))
    run(pack, model, inputs)
    print("inputs: %r" % (inputs,))
    print("  think prompt: %r" % model.calls[0].prompt)
    print("  emit  prompt: %r" % model.calls[1].prompt)
```

```
$ python3 probe_notes.py
inputs: {'ticket': 'double charged'}
  think prompt: 'Decide whether this ticket needs a human.\n\nTicket: double charged\nNotes: \n\n\nThink this through in a few short sentences. Do not answer in JSON yet — notes only.'
  emit  prompt: 'Decide whether this ticket needs a human.\n\nTicket: double charged\nNotes: nothing alarming in this one\n'
inputs: {'ticket': 'double charged', 'scratchpad': 'POISON FROM THE CALLER'}
  think prompt: 'Decide whether this ticket needs a human.\n\nTicket: double charged\nNotes: POISON FROM THE CALLER\n\n\nThink this through in a few short sentences. Do not answer in JSON yet — notes only.'
  emit  prompt: 'Decide whether this ticket needs a human.\n\nTicket: double charged\nNotes: nothing alarming in this one\n'
```

The emit stage overwrites it with the real notes; the think stage does not. If any input
to a run comes from somewhere you do not control, **drop `scratchpad` from the inputs
dict in your own caller** — stepmold will not do it for you. (`pack.py`'s own comment on
`RESERVED_STATE_NAMES` says this check "belongs where inputs enter a run (`graph.run`)";
it is not there yet.)

## No computed-value node

There is no node type that writes a deterministic computed value into state. `assert`
nodes route without writing; `end` nodes project without adding; only `generate` writes,
and only from a model. This is the single largest gap pack authors hit.

| You want | Do this |
| --- | --- |
| a derived field in the **output** (`total = qty * price`) | compute it in the caller from `RunResult.state`, or pass `--state` and post-process. The graph cannot do it |
| a derived field to **branch on** | precompute it in the run inputs and use `when: {over_cap: true}` |
| a derived field to **branch on**, when only the run knows it | an `assert` node on `expr`, branching to different nodes — free, and the branch is recoverable from `RunResult.path` or the checkpoint, but it never reaches state or the output |
| a derived field that must be **in state** for a later prompt | a `generate` node whose grammar is a tight enum, with `assert:` tying the label to the arithmetic. Costs a generation; verify-before-commit makes the committed value correct or absent |
| a **constant** in the output | put it in the run inputs and project it from the `end` node |
| a counter that increments itself | nothing. A [loop](#loops) counter is a value a model emitted, at one generation per pass |

Do not reach for a `generate` node to do arithmetic a caller could do for free.

## Checkpoints, resume and replay

`stepmold run --store <file.db>` writes a checkpoint after each node that completes (see the
table in [How a run walks](#how-a-run-walks)), and `--resume <run-id>` picks a dead run
up from its last one. The pack below dead-ends on purpose: the only edge out of the
assert node `k` carries a `when:` that can never match.

```bash
mkdir -p snag/prompts snag/grammars snag/fakes

cat > snag/graph.yaml <<'EOF'
nodes:
  a:
    type: generate
  k:
    type: assert
    expr: 1 == 1
  z:
    type: end
    output: [x]

edges:
  - from: a
    to: k
  - from: k
    to: z
    when: {nope: 1}     # never matches: DeadEnd out of the assert node
EOF

cat > snag/manifest.yaml <<'EOF'
name: snag
version: 1
entry: a
model: fake:fakes/script.json
EOF

cat > snag/prompts/a.txt <<'EOF'
Answer with x set to 1.
EOF

cat > snag/grammars/a.json <<'EOF'
{
  "type": "object",
  "properties": {"x": {"type": "integer"}},
  "required": ["x"],
  "additionalProperties": false
}
EOF

cat > snag/fakes/script.json <<'EOF'
["{\"x\": 1}"]
EOF

rm -f /tmp/snag.db      # the transcripts below start from an empty store
```

```python
# probe_checkpoints.py — print a store's whole audit trail.
import sys

from stepmold.state import Store

store = Store(sys.argv[1])
for run_id in sorted(store.runs()):
    for checkpoint in store.history(run_id):
        print("%-16s step=%d node=%-3s next_node=%-6r state=%s"
              % (run_id, checkpoint.step, checkpoint.node,
                 checkpoint.next_node, checkpoint.state))
store.close()
```

```
$ python3 -m stepmold run snag --store /tmp/snag.db --run-id assert_deadend --input '{}'
stepmold: DeadEnd: no outgoing edge from 'k' matched the current state

$ python3 probe_checkpoints.py /tmp/snag.db
assert_deadend   step=1 node=a   next_node='k'    state={'x': 1}
```

**One checkpoint, for `a`. Nothing at all for `k`** — the assert node ran, decided, and
left no record, because `graph.run` only wraps the *generate* branch's routing in
`except DeadEnd: _checkpoint(...)`. Move the same unmatchable `when:` onto the `a -> k`
edge and the generate node does record itself, pointing back at itself so that a resume
re-runs it:

```bash
cat > snag/graph.yaml <<'EOF'
nodes:
  a:
    type: generate
  k:
    type: assert
    expr: 1 == 1
  z:
    type: end
    output: [x]

edges:
  - from: a
    to: k
    when: {nope: 1}     # never matches: DeadEnd out of the generate node
  - from: k
    to: z
EOF
```

```
$ python3 -m stepmold run snag --store /tmp/snag.db --run-id generate_deadend --input '{}'
stepmold: DeadEnd: no outgoing edge from 'a' matched the current state

$ python3 probe_checkpoints.py /tmp/snag.db
assert_deadend   step=1 node=a   next_node='k'    state={'x': 1}
generate_deadend step=1 node=a   next_node='a'    state={'x': 1}
```

Now repair the graph — drop the `when:` — and resume the first run:

```bash
cat > snag/graph.yaml <<'EOF'
nodes:
  a:
    type: generate
  k:
    type: assert
    expr: 1 == 1
  z:
    type: end
    output: [x]

edges:
  - from: a
    to: k
  - from: k
    to: z
EOF
```

```
$ python3 -m stepmold run snag --store /tmp/snag.db --resume assert_deadend --log-level info
18:07:21.874 INFO  stepmold.state resume.start run_id=assert_deadend pack=snag from_step=1 next_node=k
18:07:21.875 INFO  stepmold.state lease.taken run_id=assert_deadend store=/tmp/snag.db seconds=300.0
18:07:21.875 INFO  stepmold.graph run.start run_id=assert_deadend pack=snag version=1 entry=k resumed=true max_steps=100 inputs=x
18:07:21.875 INFO  stepmold.graph run.resumed run_id=assert_deadend pack=snag from_step=1 next_node=k done=1 failures=0
18:07:21.876 INFO  stepmold.graph run.end run_id=assert_deadend pack=snag end_node=z steps=3 generations=1 failures=0 output_keys=1 output_bytes=8 duration_ms=0.8
18:07:21.876 INFO  stepmold.state lease.released run_id=assert_deadend store=/tmp/snag.db
{"x": 1}
```

| Line | What it tells you |
| --- | --- |
| `entry=k` | the resume point is the checkpoint's `next_node`, not the pack's entry |
| no `node.ok` for `a` | `a` was not re-executed. The model was not called again |
| `generations=1` | the *run* total, restored from the checkpoint — not this leg's cost |
| `steps=3` | likewise continued, not restarted |
| `lease.taken` / `lease.released` | one resumer at a time (`state.Store.lease`); a second concurrent one is refused with `ResumeInProgress` |

Resuming a run that already finished replays its checkpoint through `graph.replay` and
calls no model at all, so a supervisor can retry a resume without paying for it twice:

```
$ python3 -m stepmold run snag --store /tmp/snag.db --resume assert_deadend --log-level info
18:07:21.944 INFO  stepmold.state resume.replayed run_id=assert_deadend pack=snag step=3 end_node=z
{"x": 1}
```

Reusing a run id for a *fresh* run in the same store is refused instead:

```
$ python3 -m stepmold run snag --store /tmp/snag.db --run-id assert_deadend --input '{}'
stepmold: RunIdInUse: run id 'assert_deadend' already has checkpoints in this store; resume it, delete it, or choose another id
```

Other things worth knowing before you rely on a store:

| | |
| --- | --- |
| `--resume` without `--store` | refused: `--resume needs --store: checkpoints live in the store` |
| resuming under a different pack | refused with `CheckpointMismatch` — the checkpoint records the pack name and version (`state._check_same_pack`) |
| resuming an id the store never saw | `UnknownRun` |
| resuming a pack with `tool` nodes, without `--tools` | `ToolsNotAvailable` at the tool node. The registry is per call, so a resume needs it exactly as the first attempt did (`state.resume` passes `tools=` through) |
| retention | nothing is deleted on stepmold's own initiative; `Store.prune` and `Store.vacuum` are the operator's tools |

## Exactly once — a tool call must not happen twice

A generation can be thrown away. A sent email cannot. A tool call is the one thing in a
run that retrying cannot undo, so it is the one thing the walker writes down before it
does anything else with it.

**Why replaying state is not enough.** Resume restores the state of the last checkpoint
and continues from its `next_node`, and for a `generate` node that is the whole story: if
the node's output is in state, the node ran; if it is not, running it again costs a
generation and nothing else. A tool call breaks that reasoning in the one place it
matters. State records what a call *returned*; nothing in it records that the call
*happened*. A resumed run standing on a tool node with no `receipt` in state cannot tell
"the mail was never sent" from "the mail was sent and the process died before the commit
landed" — and those two need opposite actions.

So the call itself is recorded, in the checkpoint, next to the state:

| Field | What it holds | Why it is there |
| --- | --- | --- |
| `node` | the node that made the call | what a resumed run matches against (`graph._recorded_call`) |
| `tool` | the registered name it called | named in the mismatch error, and in the log |
| `args` | the exact arguments the tool was handed | a result is only ever the answer to the question it was asked (`graph._replay_call`) |
| `result` | what the tool returned | what a resumed run commits *instead of* calling |

It reaches the store as `tool_calls`, a JSON column on the checkpoint row
(`state.Store.save`), and a resumed run reads it back into the walk (`graph.run`,
`pending_calls`).

### When it is written, and the window that is left open

Read the order carefully, because the guarantee is exactly as strong as this sequence and
no stronger (`stepmold/graph.py`, the `tool` branch):

| # | What the walker does |
| --- | --- |
| 1 | builds the arguments from state |
| 2 | **calls the tool** |
| 3 | appends `{node, tool, args, result}` to the pending list |
| 4 | checkpoints, with `next_node` still **this node** — the walk has not left it |
| 5 | commits the result to state |
| 6 | chooses the outgoing edge |
| 7 | clears the record and checkpoints again, now with `next_node` pointing at the edge's target |

**The record is written after the call returns, not before it.** That is not an oversight
to be documented politely — it is the shape of the promise, and it means stepmold makes a
narrower guarantee than "exactly once" unqualified:

> A call that was **written down** is never made twice. A call that was **not** written
> down may be made twice.

Everything from step 4 onwards is protected: the commit can fail, the edge can dangle,
the machine can lose power, and the resumed run finds the record and replays it. Steps 2
and 3 are not: a process that dies *inside* the tool, or after the tool returned but
before the checkpoint landed, leaves nothing on disk saying the call happened, and the
resumed run calls again. stepmold cannot record a call it never saw return, and it does not
pretend to — there is no pre-call intent record, no two-phase commit, and no
deduplication key handed to the tool. A tool whose second execution would be
catastrophic needs its own idempotency key on the host side; stepmold closes the window from
step 4, not from step 1.

**When it is cleared.** Step 7, the moment an edge out of the node is taken. From there
the committed state is the record that the call happened, and resume will never re-enter
the node. That is also why a tool inside a loop calls every time round.

**It is a lock, not an audit log.** The row is rewritten at the same step, so a run that
finished normally carries no trace of its calls at all:

```python
# probe_rows.py — a store's whole audit trail, tool calls included.
import sys

from stepmold.state import Store

store = Store(sys.argv[1])
for checkpoint in store.history(sys.argv[2]):
    print("step=%d node=%-6s next_node=%-8r tool_calls=%s"
          % (checkpoint.step, checkpoint.node, checkpoint.next_node,
             checkpoint.tool_calls))
store.close()
```

```
$ rm -f /tmp/outbox.txt
$ python3 -m stepmold run notify --store /tmp/rows.db --run-id ok --tools ./mailer.py --input '{"to": "ops@example.com", "incident": "db-3 disk at 100%"}'
{"receipt": "message 1", "subject": "disk full on db-3"}

$ python3 probe_rows.py /tmp/rows.db ok
step=1 node=draft  next_node='send'   tool_calls=[]
step=2 node=send   next_node='done'   tool_calls=[]
step=3 node=done   next_node=None     tool_calls=[]
```

The `tool_calls` column is empty on every row of a healthy run. It is only ever
non-empty on the last row of a run that stopped between steps 4 and 7.

### The worked example: run it, kill it, resume it

The `notify` pack from [tool](#tool), a real store, and a child process that is killed
the instant the call is written down.

```python
# probe_kill.py — kill a run in the window a tool call lives in, and read what the
# crash left on disk. The dying leg is a real child process ending in os._exit.
import os
import subprocess
import sys

from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import load_pack
from stepmold.state import Store

import mailer

DB = "/tmp/notify.db"
INPUTS = {"to": "ops@example.com", "incident": "db-3 disk at 100%"}


class Guillotine(Store):
    """A real store that kills the process the moment a tool call is written down."""

    def save(self, **kwargs):
        Store.save(self, **kwargs)
        if kwargs.get("tool_calls"):
            print("killed at node %r, next_node=%r"
                  % (kwargs["node"], kwargs["next_node"]), flush=True)
            os._exit(9)          # no unwinding, no cleanup: as close to SIGKILL as it gets


if len(sys.argv) > 1:                                    # the child: crash on purpose
    run(load_pack("notify", tools=mailer.registry),
        FakeModel(['{"subject": "disk full on db-3"}']), INPUTS,
        run_id="alert1", store=Guillotine(DB), tools=mailer.registry)
    raise SystemExit("the child was supposed to die")

for path in (mailer.OUTBOX, DB):
    if os.path.exists(path):
        os.remove(path)
child = subprocess.run([sys.executable, __file__, "child"])
print("child exit status:", child.returncode)
print("outbox:", open(mailer.OUTBOX).read().strip())
store = Store(DB)
for checkpoint in store.history("alert1"):
    print("step=%d node=%-6s next_node=%-8r state=%s\n    tool_calls=%s"
          % (checkpoint.step, checkpoint.node, checkpoint.next_node,
             sorted(checkpoint.state), checkpoint.tool_calls))
store.close()
```

```
$ python3 probe_kill.py
killed at node 'send', next_node='send'
child exit status: 9
outbox: ops@example.com	disk full on db-3
step=1 node=draft  next_node='send'   state=['incident', 'subject', 'to']
    tool_calls=[]
step=2 node=send   next_node='send'   state=['incident', 'subject', 'to']
    tool_calls=[{'args': {'subject': 'disk full on db-3', 'to': 'ops@example.com'}, 'node': 'send', 'result': {'receipt': 'message 1'}, 'tool': 'send_email'}]
```

Three facts in that output:

* The child died at `send` with `next_node='send'` — pointing at itself, because the walk
  had not left the node.
* The mail was sent: one line in the outbox.
* The step-2 checkpoint's state holds `incident`, `subject` and `to` and **not**
  `receipt`. The record was written before the commit, so the crash caught the run in
  the state where the call had happened and its result had not landed. This is the exact
  case that state alone cannot describe.

Now resume it with the ordinary CLI — same pack, same store, same registry:

```
$ python3 -m stepmold run notify --store /tmp/notify.db --resume alert1 --tools ./mailer.py --log-level info
11:42:58.792 INFO  stepmold.state resume.start run_id=alert1 pack=notify from_step=2 next_node=send
11:42:58.793 INFO  stepmold.state lease.taken run_id=alert1 store=/tmp/notify.db seconds=300.0
11:42:58.793 INFO  stepmold.graph run.start run_id=alert1 pack=notify version=1 entry=send resumed=true max_steps=100 inputs=incident,subject,to
11:42:58.793 INFO  stepmold.graph run.resumed run_id=alert1 pack=notify from_step=2 next_node=send done=2 failures=0
11:42:58.794 INFO  stepmold.graph node.ok run_id=alert1 node=send type=tool attempts=0 tool=send_email replayed=true output=merge duration_ms=0.0
11:42:58.794 INFO  stepmold.graph run.end run_id=alert1 pack=notify end_node=done steps=4 generations=1 failures=0 output_keys=2 output_bytes=56 duration_ms=0.9
11:42:58.795 INFO  stepmold.state lease.released run_id=alert1 store=/tmp/notify.db
{"receipt": "message 1", "subject": "disk full on db-3"}

$ cat /tmp/outbox.txt
ops@example.com	disk full on db-3
```

| Line | What it tells you |
| --- | --- |
| `entry=send` | the resume point is the checkpoint's `next_node`, which the crash left pointing at the tool node |
| `node.ok ... tool=send_email replayed=true` | the tool was **not** called. `replayed` is the field to grep for |
| `attempts=0` | as ever for a tool node — and the `generations=1` on `run.end` is the draft node's, restored from the checkpoint |
| the outbox, unchanged | one line, and the receipt in the output is `message 1`, the recorded result rather than a fresh one |

### What turns the promise off

Four ways a second call gets through, three of them things a host chooses. The driver
below crashes the same pack at the same instant under four different hosts:

```python
# probe_exactly_once.py — kill a run in the window a tool call lives in, resume it, and
# count the side effects. Four hosts, one kill, one pack.
#
#   recorded    the ordinary tool, in a store that can hold the record
#   idempotent  the same tool declared idempotent=True, so nothing is recorded
#   old-store   the ordinary tool, in a store whose save() predates `tool_calls`
#   mid-call    the ordinary tool, killed inside itself: after the send, before the return
#
# The crashing leg runs in a real child process that ends in os._exit, so nothing is
# unwound and nothing is flushed on the way out.
import os
import subprocess
import sys

from stepmold import log
from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import load_pack
from stepmold.state import Store, resume
from stepmold.tools import ToolRegistry

OUTBOX = "/tmp/probe_outbox.txt"
DB = "/tmp/probe_notify.db"
MODES = ("recorded", "idempotent", "old-store", "mid-call")
INPUTS = {"to": "ops@example.com", "incident": "db-3 disk at 100%"}


def registry_for(mode, killing=False):
    """The host's side. `killing` is true only in the leg that must crash."""
    registry = ToolRegistry()

    @registry.register("send_email", reads=["to", "subject"], writes=["receipt"],
                       idempotent=(mode == "idempotent"))
    def send_email(to, subject):
        with open(OUTBOX, "a") as outbox:
            outbox.write("%s\t%s\n" % (to, subject))
        if killing and mode == "mid-call":
            os._exit(9)                 # the send happened; the return never did
        return {"receipt": "message %d" % len(open(OUTBOX).readlines())}

    return registry


class DocStore(Store):
    """A real store, killed the instant the walk tries to leave the tool node."""

    def __init__(self, path, killing=False):
        Store.__init__(self, path)
        self.killing = killing

    def save(self, run_id, step, node, next_node, state, path, provenance, failures,
             output=None, pack=None, attempts=None, tool_calls=None):
        if self.killing and node == "send" and next_node == "done":
            os._exit(9)
        return Store.save(self, run_id=run_id, step=step, node=node,
                          next_node=next_node, state=state, path=path,
                          provenance=provenance, failures=failures, output=output,
                          pack=pack, attempts=attempts, tool_calls=tool_calls)


class OldStore(DocStore):
    """The same store, written before the walker had a tool call to record.

    `graph._store_records_tool_calls` reads the signature, so dropping the keyword is
    all it takes to be one.
    """

    def save(self, run_id, step, node, next_node, state, path, provenance, failures,
             output=None, pack=None, attempts=None):
        return DocStore.save(self, run_id=run_id, step=step, node=node,
                             next_node=next_node, state=state, path=path,
                             provenance=provenance, failures=failures, output=output,
                             pack=pack, attempts=attempts)


def store_for(mode, killing=False):
    kind = OldStore if mode == "old-store" else DocStore
    return kind(DB, killing=killing)


def model():
    return FakeModel(['{"subject": "disk full on db-3"}'])


log.configure(level="warning", stream=sys.stdout)

if len(sys.argv) > 1:                                       # the child: crash on purpose
    mode = sys.argv[1]
    tools = registry_for(mode, killing=True)
    run(load_pack("notify", tools=tools), model(), INPUTS,
        run_id=mode, store=store_for(mode, killing=True), tools=tools)
    raise SystemExit("the child was supposed to die")

print("%-11s %-9s %-9s %s" % ("mode", "crash", "resume", "receipt"), flush=True)
for mode in MODES:
    for path in (OUTBOX, DB):
        if os.path.exists(path):
            os.remove(path)
    subprocess.run([sys.executable, __file__, mode])
    sent_before = len(open(OUTBOX).readlines())
    tools = registry_for(mode)
    store = store_for(mode)
    result = resume(load_pack("notify", tools=tools), model(), mode, store, tools=tools)
    store.close()
    print("%-11s %-9s %-9s %s"
          % (mode, "%d sent" % sent_before,
             "%d sent" % len(open(OUTBOX).readlines()), result.output["receipt"]),
          flush=True)
```

```
$ python3 probe_exactly_once.py
mode        crash     resume    receipt
recorded    1 sent    1 sent    message 1
idempotent  1 sent    2 sent    message 2
11:42:47.693 WARNING stepmold.graph tool.unrecorded run_id=old-store node=send tool=send_email store=OldStore reason="store.save takes no tool_calls, so a resumed run cannot know this call already happened"
11:42:47.697 WARNING stepmold.graph tool.unrecorded run_id=old-store node=send tool=send_email store=OldStore reason="store.save takes no tool_calls, so a resumed run cannot know this call already happened"
old-store   1 sent    2 sent    message 2
mid-call    1 sent    2 sent    message 2
```

| Mode | Sends | Why |
| --- | --- | --- |
| `recorded` | 1 | the record was on disk before the crash, so the resume replayed it |
| `idempotent` | 2 | `idempotent=True` skips the bookkeeping entirely — nothing is recorded, so the resumed run calls again |
| `old-store` | 2 | the store's `save` takes no `tool_calls`, so the record had nowhere to go |
| `mid-call` | 2 | the process died inside the tool, before the record could be written — the open window |

And a fifth, which is not in the table because there is nothing to resume: **a run with no
`store` keeps no record at all.** That is legal and unremarkable — no store means no
resume means no repeat — but it also means the exactly-once machinery needs `store` *and*
`tools` together to do anything (`graph.run`).

**`idempotent=True` is a promise the author makes, not a hint.** It says: calling this
tool twice with the same arguments has the same effect as calling it once. stepmold takes it
literally — it skips the record, and a resumed run calls the tool again. It is not "this
tool is fast", "this tool is probably safe" or "this tool mostly reads". A `PUT` to a
fixed key qualifies; an `INSERT`, a payment and an email do not.

**A store that cannot hold the record says so, once per run.** Losing a diagnostic to an
old store is a shrug; losing this is a promise broken, so the walker logs
`tool.unrecorded` at WARNING rather than degrading in silence
(`graph._store_records_tool_calls`) — the two lines in the transcript above, one from the
crashing process and one from the resume. It still does not refuse the run: a host that
brought its own store has chosen this.

### When the record and the run come apart

A replayed result is only the answer to the arguments it was computed from, so the walker
compares them before committing it. They cannot differ on a faithful resume — the state
and the record came out of the same row — so a difference means something outside stepmold has
moved. Neither choice left is safe, and the run stops instead of choosing:

```python
# probe_mismatch.py — resuming a recorded call into state that would ask a different
# question. The checkpoint is edited by hand, which is the only way to get here: on a
# faithful resume the state and the record came out of the same row.
import dataclasses

from stepmold.graph import run
from stepmold.model import FakeModel
from stepmold.pack import load_pack
from stepmold.state import Store

import mailer

store = Store("/tmp/notify.db")
checkpoint = store.latest("alert1")
print("recorded call:", checkpoint.tool_calls)
tampered = dataclasses.replace(
    checkpoint, state=dict(checkpoint.state, to="someone-else@example.com"))
try:
    run(load_pack("notify", tools=mailer.registry), FakeModel(['{"subject": "x"}']),
        run_id="alert1", store=None, resume_from=tampered, tools=mailer.registry)
except Exception as exc:
    print("%s: %s" % (type(exc).__name__, exc))
store.close()
```

`probe_kill.py` above left the interrupted checkpoint this reads, and the resume in
between consumed it — so re-run the kill first, or `alert1` has nothing pending and the
recorded call comes back empty:

```
$ python3 probe_kill.py >/dev/null
$ python3 probe_mismatch.py
recorded call: [{'args': {'subject': 'disk full on db-3', 'to': 'ops@example.com'}, 'node': 'send', 'result': {'receipt': 'message 1'}, 'tool': 'send_email'}]
ToolReplayMismatch: run resumed into node 'send' holding a call to 'send_email' that already happened, but state no longer matches the arguments it was made with (to differ). The recorded result answers a question this run is no longer asking, and calling again would repeat a side effect that already took place — so the run stops instead of choosing between them. Read the run's checkpoints before resuming it again.
```

The comparison is canonical JSON, not `==`, for the reason `state._dict_delta` gives:
`True == 1` in Python, and two argument sets a tool would treat as different must not
compare equal (`graph._canonical`). The error names *which* argument moved and never what
it moved to — an argument's value is the caller's data, and it is already in the
checkpoint where it is allowed to be (`graph._moved_args`).

## Worked example

A four-path refund router: one `generate` node that classifies, one `assert` node that
compares, one `generate` node that decides, three endings.

```bash
mkdir -p refund_router/prompts refund_router/grammars refund_router/fakes

cat > refund_router/graph.yaml <<'EOF'
max_steps: 20

nodes:
  classify:
    type: generate
    max_tokens: 64

  small_enough:
    type: assert
    expr: amount_usd <= 500
    on_fail: manual_review

  decide:
    type: generate
    output: decision
    retries: 1
    assert: decision.action == "refund" or decision.action == "deny"
    on_fail: manual_review

  refunded:
    type: end
    output: [kind, amount_usd, decision]
  denied:
    type: end
    output: [kind, amount_usd, decision]
  manual_review:
    type: end
    output: [kind, amount_usd, decision]

edges:
  - from: classify
    to: small_enough
    when: {kind: refund}
  - from: classify
    to: manual_review          # fallback, declared last

  - from: small_enough
    to: decide

  - from: decide
    to: refunded
    when: {decision.action: refund}
  - from: decide
    to: denied                 # fallback, declared last
EOF

cat > refund_router/manifest.yaml <<'EOF'
name: refund_router
version: 1
entry: classify
model: fake:fakes/refund.json
EOF

cat > refund_router/prompts/classify.txt <<'EOF'
Classify this support ticket.

  kind        - refund, question, or other
  amount_usd  - the dollar amount at stake, 0 if none is named

Ticket: {ticket}
EOF

cat > refund_router/prompts/decide.txt <<'EOF'
Decide the refund.

  action  - refund or deny
  note    - one short sentence saying why

Kind: {kind}
Amount: {amount_usd}
Ticket: {ticket}
EOF

cat > refund_router/grammars/classify.json <<'EOF'
{
  "type": "object",
  "properties": {
    "kind": {"type": "string", "enum": ["refund", "question", "other"]},
    "amount_usd": {"type": "number"}
  },
  "required": ["kind", "amount_usd"],
  "additionalProperties": false
}
EOF

cat > refund_router/grammars/decide.json <<'EOF'
{
  "type": "object",
  "properties": {
    "action": {"type": "string", "enum": ["refund", "deny"]},
    "note": {"type": "string"}
  },
  "required": ["action", "note"],
  "additionalProperties": false
}
EOF

cat > refund_router/fakes/refund.json <<'EOF'
[
  "{\"kind\": \"refund\", \"amount_usd\": 120}",
  "{\"action\": \"refund\", \"note\": \"under policy cap\"}"
]
EOF

cat > refund_router/fakes/big.json <<'EOF'
["{\"kind\": \"refund\", \"amount_usd\": 900}"]
EOF

cat > refund_router/fakes/broken.json <<'EOF'
[
  "{\"kind\": \"refund\", \"amount_usd\": 120}",
  "sorry, I cannot do that",
  "{\"action\": \"escalate\", \"note\": \"not in the enum\"}"
]
EOF

cat > refund_router/fakes/question.json <<'EOF'
["{\"kind\": \"question\", \"amount_usd\": 0}"]
EOF
```

```
$ python3 -m stepmold validate refund_router
refund_router v1: 6 nodes, 5 edges, 0 evalset cases, entry 'classify'
```

Each path is one scripted model: `fakes/<name>.json` answers the nodes in order.

**Path 1 — the happy path.** Both generate nodes land first try; the dotted `when:`
matches:

```
$ python3 -m stepmold run refund_router --model fake:fakes/refund.json --run-id refund --log-level info --input '{"ticket": "t"}'
18:05:55.246 INFO  stepmold.graph run.start run_id=refund pack=refund_router version=1 entry=classify resumed=false max_steps=20 inputs=ticket
18:05:55.246 INFO  stepmold.graph node.ok run_id=refund node=classify type=generate attempts=1 output=merge duration_ms=0.0
18:05:55.246 INFO  stepmold.graph node.ok run_id=refund node=decide type=generate attempts=1 output=decision duration_ms=0.1
18:05:55.246 INFO  stepmold.graph run.end run_id=refund pack=refund_router end_node=refunded steps=4 generations=2 failures=0 output_keys=3 output_bytes=99 duration_ms=0.5
{"amount_usd": 120, "decision": {"action": "refund", "note": "under policy cap"}, "kind": "refund"}
```

Four steps for four nodes — `refunded` is a step. The passing `assert` node logs nothing
at INFO (it is DEBUG when it passes, `graph.run`).

**Path 2 — the assert node diverts.** `amount_usd` is 900:

```
$ python3 -m stepmold run refund_router --model fake:fakes/big.json --run-id big --log-level info --input '{"ticket": "t"}'
18:05:55.306 INFO  stepmold.graph run.start run_id=big pack=refund_router version=1 entry=classify resumed=false max_steps=20 inputs=ticket
18:05:55.306 INFO  stepmold.graph node.ok run_id=big node=classify type=generate attempts=1 output=merge duration_ms=0.1
18:05:55.306 INFO  stepmold.graph node.assert run_id=big node=small_enough type=assert passed=false expr="amount_usd <= 500" to=manual_review duration_ms=0.0
18:05:55.307 INFO  stepmold.graph run.end run_id=big pack=refund_router end_node=manual_review steps=3 generations=1 failures=0 output_keys=2 output_bytes=37 duration_ms=0.5
{"amount_usd": 900, "kind": "refund"}
```

`failures=0` — an assert divert is not a `Failure`.

**Path 3 — the ladder burns and `on_fail` catches it.** `decide` returns unparseable
text, then an out-of-enum action:

```
$ python3 -m stepmold run refund_router --model fake:fakes/broken.json --run-id broken --log-level info --input '{"ticket": "t"}'
18:05:55.367 INFO  stepmold.graph run.start run_id=broken pack=refund_router version=1 entry=classify resumed=false max_steps=20 inputs=ticket
18:05:55.367 INFO  stepmold.graph node.ok run_id=broken node=classify type=generate attempts=1 output=merge duration_ms=0.0
18:05:55.367 WARNING stepmold.verify node.rejected node=decide attempt=1 cause=verify reason="output was not valid JSON — return a single JSON object and nothing else" of=2
18:05:55.367 INFO  stepmold.verify node.retry node=decide attempt=2 of=2 temperature=0.5 seed=1 reason="output was not valid JSON — return a single JSON object and nothing else" rethink=false
18:05:55.367 WARNING stepmold.verify node.rejected node=decide attempt=2 cause=verify reason="schema: action: value is not one of 'refund', 'deny'" of=2
18:05:55.367 WARNING stepmold.graph node.failed run_id=broken node=decide type=generate attempts=2 error=NodeFailed reason="schema: action: value is not one of 'refund', 'deny'" on_fail=manual_review duration_ms=0.2
18:05:55.367 INFO  stepmold.graph edge.on_fail run_id=broken node=decide to=manual_review
18:05:55.368 INFO  stepmold.graph run.end run_id=broken pack=refund_router end_node=manual_review steps=4 generations=3 failures=1 output_keys=2 output_bytes=37 duration_ms=0.7
{"amount_usd": 120, "kind": "refund"}
```

`retries: 1` bought two generations at `decide`; `generations=3` is the run total.

**Path 4 — the fallback edge.** `kind` is `question`, so `when: {kind: refund}` misses
and the un-conditioned edge takes it straight to the ending:

```
$ python3 -m stepmold run refund_router --model fake:fakes/question.json --run-id question --log-level info --input '{"ticket": "t"}'
18:05:55.431 INFO  stepmold.graph run.start run_id=question pack=refund_router version=1 entry=classify resumed=false max_steps=20 inputs=ticket
18:05:55.431 INFO  stepmold.graph node.ok run_id=question node=classify type=generate attempts=1 output=merge duration_ms=0.1
18:05:55.431 INFO  stepmold.graph run.end run_id=question pack=refund_router end_node=manual_review steps=2 generations=1 failures=0 output_keys=2 output_bytes=37 duration_ms=0.4
{"amount_usd": 0, "kind": "question"}
```

**The lesson in the transcript.** Paths 2, 3 and 4 are three different stories — over the
cap, model failed twice, not a refund at all — and all three print an object of the
*same two keys*:

```
{"amount_usd": 900, "kind": "refund"}
{"amount_usd": 120, "kind": "refund"}
{"amount_usd": 0, "kind": "question"}
```

`decision` is simply missing from each, silently, because `_project` drops keys that are
not in state. Nothing in the graph can add a `reason` field. What distinguishes them,
from outside the run, is `RunResult.end_node` — so give each reason its own `end` node
(`over_cap`, `undecided`, `not_a_refund`) and let the ending carry the meaning. That is
also what an evalset case's `end:` scores. Inside the run, `RunResult.path` already holds
the whole story; it is just not part of the printed output.

## The packs on disk

`examples/` ships seven full packs, each with its own evalset and its own scripted model,
and they are the best next read after this page:

| Pack | Shows |
| --- | --- |
| `support_triage` | the plain linear shape: classify → extract → priority → emit, one `two_stage` node, one `assert:` with `on_fail` |
| `incident_triage` | routing on a normalised severity, plus a malformed-input ending |
| `invoice_extract` | four `assert` **nodes** doing the arithmetic a model must never do, each with its own failure ending, and one deliberately without `on_fail` |
| `lead_qualify` | a cheap gate node that decides whether four expensive nodes run at all |
| `meeting_actions` | a conditional node that is skipped in the common case |
| `content_moderation` | two locks in series: a generate `assert:` and an `assert` node re-reading committed state |

```
$ python3 -m stepmold eval examples/support_triage
support_triage: 12/12 cases passed
```

## Field reference

**Node keys** (`pack._NODE_KEYS`; anything else is a load error)

| Key | Types | Default | Meaning |
| --- | --- | --- | --- |
| `type` | all | — | `generate` \| `tool` \| `assert` \| `end`, required |
| `output` | generate, tool | none (merge mode) | one state key (a string) to commit the object under |
| `output` | end | none (whole state) | **list** of state keys to project, matched flat — a dot is part of the name, not a path |
| `on_fail` | generate, tool, assert | none → abort | node to divert to (accepted on an `end` node, where it is inert) |
| `on_unsure` | generate | none → falls back to `on_fail` | node to divert to when independent draws disagree. Accepted on any node type; only the generate branch reads it |
| `tool` | tool | — | required; the registered name this node calls. Refused on any other node type |
| `expr` | assert | — | required; the routing expression |
| `assert` | generate | none | verification expression, checked before commit |
| `retries` | generate | 2 | re-samples after the first attempt (must be an integer >= 0) |
| `max_tokens` | generate | 512 | emit budget |
| `two_stage` | generate | false | unconstrained think pass before the constrained emit |
| `think_max_tokens` | generate | 256 | think budget |
| `prompt` | generate | `prompts/<node>.txt` | emit template path, inside the pack |
| `grammar` | generate | `grammars/<node>.json` | schema path, inside the pack |
| `description` | all | — | documentation, ignored at run time |

There is no key here for the confidence gate. `samples:` and `agree:` are read by
`verify.gate_for` but are not in `pack._NODE_KEYS`, so a `graph.yaml` carrying either
fails to load; see [Unsure](#unsure--disagreement-is-not-rejection). And a tool node
refuses `prompt`, `grammar`, `two_stage`, `retries`, `max_tokens`, `think_max_tokens`,
`assert` and `expr` by name, each with its reason (`pack._TOOL_FORBIDDEN_KEYS`).

**The think template has no key of its own.** `two_stage: true` turns the stage on;
which text it renders is decided by the filesystem (`pack._build_node`):

| On disk | Think stage renders |
| --- | --- |
| `prompts/<node>.think.txt` exists | that file |
| it does not | the node's **emit** prompt plus a fixed suffix: `"\n\nThink this through in a few short sentences. Do not answer in JSON yet — notes only."` (`codegen.DEFAULT_THINK_SUFFIX`) |

The think path is built from the **node name** and nothing else. A `prompt:` override
moves the emit template only — a node named `decide` with `prompt: prompts/emit_v2.txt`
still looks for `prompts/decide.think.txt`. Either template may contain `{scratchpad}`:
in the think stage it renders empty (or, unscreened, whatever the caller passed — see
[`scratchpad`](#scratchpad-is-reserved-from-one-side-only)), and in the emit stage it
holds the notes. A prompt with no `{scratchpad}` gets the notes appended in a labelled
block instead (`codegen.build_prompt`).

**Edge keys** (`pack._EDGE_KEYS`): `from`, `to` (both required, both must name defined
nodes), `when` (a mapping or absent), `description`.

**Graph keys**: `nodes` (required, non-empty mapping), `edges` (list), `max_steps`
(positive int, default 100).

**Checked at load** (`pack.load_pack`): unknown node/edge keys; unknown node type; an
`assert` node with no `expr`; a `tool` node with no `tool:`, or carrying a generate-only
key; a `tool:` key on a node that is not a tool node; `output: scratchpad`; a
`from`/`to`/`on_fail`/`on_unsure` naming an undefined node; an outgoing edge from an `end` node; a non-`end` node with **no** outgoing
edge; `entry` not in `nodes`; every `end:` in `evalset.jsonl` naming a real `end` node.
Unreachable nodes are **not** flagged.

**Checked only when a registry is passed** (`pack.check_tools`, from
`load_pack(path, tools=registry)`): that every tool node names a registered tool, and
that everything the tool declares it `reads` is written by some earlier node or declared
as a run input. `stepmold validate` has no `--tools` flag, so it never runs either check.

**Checked by the CLI only** (`cli._check_output_shapes`, run by `validate`, `run` and
`eval`): `output` written in the other node type's shape. Library callers do not get
this — and neither check catches a dotted `output:` entry, which is a valid list of
strings.

## Errors

| Error | Raised by | Meaning |
| --- | --- | --- |
| `DeadEnd` | `graph._next` | no outgoing edge matched — usually a `when:` that can never be true, or a missing fallback |
| `DanglingEdge` | `graph._node` | an edge or `on_fail` names a node that is not defined (normally caught at load) |
| `MaxStepsExceeded` | `graph.run` | the step budget ran out; suspect a loop |
| `NodeFailed` | `verify.run_node` | a generate node's ladder is spent and it has no `on_fail` |
| `MissingVariable` | prompt render | a prompt names state nobody wrote, and the node has no `on_fail` |
| `AssertFailed` | `graph.run` | an `assert` node's expression was false and it has no `on_fail` |
| `ExprError` | `stepmold.expr` | an expression is unsupported or names something absent |
| `StateCollision` | `graph.commit` | a node's commit would overwrite a run input |
| `BackendError` | the backend | the model could not be reached; never routed to `on_fail` |
| `Unsure` | `verify.run_node` | a node's independent draws disagreed and it declares neither `on_unsure` nor `on_fail`. A `RunError`, **not** a `NodeFailed` |
| `ToolsNotAvailable` | `graph._tool_for` | the walk reached a `tool` node and the run was given no registry; never routed to `on_fail` |
| `ToolNotRegistered` | `tools.ToolRegistry.get` | the pack names a tool the host did not register — at load when `tools=` was passed, otherwise at the node; never routed to `on_fail` |
| `ToolFailed` | `tools.Tool.invoke` | the tool's own code raised; routed to `on_fail` |
| `ToolContract` | `tools.Tool.invoke` | the tool was missing one of its declared `reads`, or returned something its `writes` did not describe; routed to `on_fail` |
| `ToolReplayMismatch` | `graph._replay_call` | a run resumed into a recorded call whose arguments no longer match state ([Exactly once](#exactly-once--a-tool-call-must-not-happen-twice)) |
| `RunIdInUse` | `graph.run`, `state.Store.save` | a fresh run reused a run id that already has checkpoints |
| `CheckpointMismatch`, `UnknownRun`, `ResumeInProgress`, `StoreBusy` | `stepmold.state` | resume-time and store-level failures ([Checkpoints](#checkpoints-resume-and-replay)) |

`RunIdInUse` needs one clause of its own, because it is easy to think you are protected
by it when you are not. `graph.run` performs its check **only** when a `store` is
passed, a `run_id` is passed, and it is not a resume. `stepmold run --run-id refund` with no
`--store` — which is eight of the commands on this page — can never raise it, and run ids
are not unique across runs unless a store is enforcing them. The store raises its own
`RunIdInUse` (different wording) when two runs race for the same id at their first
checkpoint (`state.Store._claim`).

All of these subclass `RunError` (`stepmold/errors.py`, plus `graph.StateCollision`, the
`stepmold.tools` and `graph` tool errors, and the `stepmold.state` ones). Every one raised *during* the walk is logged as `run.error` on its way
out; `graph.run`'s `RunIdInUse` check happens before the walk starts, so it is logged as
nothing at all — the run has not even printed `run.start`.
