# Expressions

jig packs carry two kinds of expression string: a node-level `assert:` and an `assert`
node's `expr:`. Both are parsed and walked by `jig/expr.py`. Neither is ever `eval()`-ed.

An expression is a **deterministic check over run state** — the part of a workflow that a
model must never be trusted to do in its head. `abs(subtotal + tax_amount - total_amount)
< 0.01` is arithmetic; `escalate == (priority == "p0")` is a policy invariant. Both are
free, both are exact, and neither costs a token.

Every command in this document was run. The `jig run` blocks all use the five-file demo
pack in the next section, or that pack with one line changed — the change is named above
each block. Build it first and everything below reproduces. The `python3 - <<'PY'` blocks
are run from the repository root, so `import jig` resolves — and so are the `jig run` and
`jig validate` blocks. jig is not installed here, so `python3 -m jig` only resolves from
the clone. Write the demo pack, then `cd` back to the repository before running anything
below it.

---

## The demo pack

Five files, no GPU, no network: the manifest points at a scripted `fake:` model. Paste
this whole block.

```sh
mkdir -p /tmp/jig-expr-demo/prompts /tmp/jig-expr-demo/grammars /tmp/jig-expr-demo/fakes
cd /tmp/jig-expr-demo   # written from here; return to the clone before running anything

cat > manifest.yaml <<'EOF'
name: expr_demo
version: 1
entry: triage
model: fake:fakes/script.json
EOF

cat > graph.yaml <<'EOF'
max_steps: 8

nodes:
  triage:
    type: generate
    max_tokens: 32
    retries: 2
    # Placement 1 -- a verification. Evaluated against a trial copy of state
    # before anything is committed. False rejects the candidate and spends a rung.
    assert: escalate == (priority == "p0")
    on_fail: needs_human

  gate:
    # Placement 2 -- a routing decision over state that is already committed.
    # No model call, no rungs. False takes `on_fail`.
    type: assert
    expr: escalate and priority == "p0"
    on_fail: done

  escalated:
    type: end
    output: [priority, escalate]

  done:
    type: end
    output: [priority, escalate]

  needs_human:
    type: end
    output: [ticket]

edges:
  - from: triage
    to: gate
  - from: gate
    to: escalated
EOF

cat > prompts/triage.txt <<'EOF'
You are the triage step of a support-ticket workflow.

Answer with a priority (p0, p1 or p2) and whether to escalate.
Escalate only when the priority is p0.

Ticket: {ticket}
EOF

cat > grammars/triage.json <<'EOF'
{
  "type": "object",
  "properties": {
    "priority": {"type": "string", "enum": ["p0", "p1", "p2"]},
    "escalate": {"type": "boolean"}
  },
  "required": ["priority", "escalate"],
  "additionalProperties": false
}
EOF

cat > fakes/script.json <<'EOF'
{
  "card declined twice": [
    "{\"priority\": \"p1\", \"escalate\": true}",
    "{\"priority\": \"p0\", \"escalate\": true}"
  ],
  "export a CSV": [
    "{\"priority\": \"p2\", \"escalate\": false}"
  ],
  "vague": [
    "{\"priority\": \"p1\", \"escalate\": true}",
    "{\"priority\": \"p2\", \"escalate\": true}",
    "{\"priority\": \"p1\", \"escalate\": true}"
  ]
}
EOF
```

```
$ python3 -m jig validate /tmp/jig-expr-demo
expr_demo v1: 5 nodes, 2 edges, 0 evalset cases, entry 'triage'
```

Three details in those files are not guessable, and each one costs an afternoon:

| Detail | Why it is that way | What happens if you get it wrong |
| --- | --- | --- |
| `prompts/triage.txt` writes `{ticket}`, single braces | `jig/render.py` substitutes `{name}`; `{{` and `}}` are escapes for literal braces | `{{ticket}}` renders the literal text `{ticket}`, the fake matches no key, and the run dies with `jig.model.ModelExhausted: FakeModel has no scripted response matching prompt: '...Ticket: {ticket}\n'` (prompt elided) and a raw traceback, not a jig error message |
| `fakes/script.json` is a **mapping**, not a list | keyed mode: the key must be a substring of the rendered prompt, longest match wins, and each key's list is consumed in order — so different tickets can draw different answers | an ordered list answers calls 1, 2, 3… regardless of ticket, so the three runs below cannot be produced |
| `"vague"` has three responses | that node's ladder is three attempts (`retries: 2`), and this run burns all of them | `jig.model.ModelExhausted: FakeModel ran out of scripted responses for key 'vague'`, again as a traceback |

Everything below runs against this pack. Where a block runs a different path —
`/tmp/jig-expr-typo`, `/tmp/jig-expr-gate`, `/tmp/jig-expr-nested` and so on — that
directory is `cp -r /tmp/jig-expr-demo <path>` plus the single edit named above the block.

`run_id`, the `HH:MM:SS.mmm` prefix and `duration_ms` differ on every run. `--run-id
<name>` is why some blocks show `run_id=typo` instead of a hex string; every other field
in the pasted output is reproducible.

---

## The trap: `assert` means two different things

Same word, two placements, **different failure semantics**. Pack authors report this as
the single biggest surprise in the format. Read this table before anything else.

| | `assert:` on a `generate` node | a node of `type: assert` |
| --- | --- | --- |
| YAML key | `assert:` | `expr:` |
| It is a | **verification** of one candidate output | **routing decision** over committed state |
| Runs against | a *trial* copy of state with the candidate merged in | real, already-committed state |
| Model call | it is checking one | none, ever |
| Costs | a retry rung each time it fails | nothing |
| On false | candidate discarded, re-sample; ladder spent → `on_fail`, else `NodeFailed` | take `on_fail`, else `AssertFailed` kills the run |
| On unevaluable | **same as false** — a `Rejected`, burns the whole ladder | `on_fail` if declared, else the `ExprError` escapes and kills the run |
| Source | `verify.py:_check_assert` | `graph.py` (the `node.type == "assert"` branch) |

The practical consequence, and it is expensive: **a typo in a `generate` node's `assert:`
spends every retry the node allows before it fails.** A missing name is not a graph error
there; it is a rejected candidate, once per rung.

How many rungs is that? `retries:` defaults to `2` (`pack.py:DEFAULT_RETRIES`), and the
ladder is `retries + 1` attempts, which is the `of=3` in every log line below. The demo
pack writes `retries: 2` explicitly, but deleting that line changes nothing: a node that
says nothing about retries still costs three generations on a typo. `retries: 0` is the
only way to pay one (`of=1`).

The demo pack with `triage`'s assert changed to `escalate == (severity == "p0")` —
nothing writes `severity`:

```
$ python3 -m jig run /tmp/jig-expr-typo --input '{"ticket": "vague"}' --log-level info --run-id typo
17:18:21.633 INFO  jig.graph run.start run_id=typo pack=expr_demo version=1 entry=triage resumed=false max_steps=8 inputs=ticket
17:18:21.634 WARNING jig.verify node.rejected node=triage attempt=1 cause=verify reason="assert 'escalate == (severity == \"p0\")' could not be evaluated: expression references 'severity', which is not in state" of=3
17:18:21.634 INFO  jig.verify node.retry node=triage attempt=2 of=3 temperature=0.5 seed=1 reason="assert 'escalate == (severity == \"p0\")' could not be evaluated: expression references 'severity', which is not in state" rethink=false
17:18:21.634 WARNING jig.verify node.rejected node=triage attempt=2 cause=verify reason="assert 'escalate == (severity == \"p0\")' could not be evaluated: expression references 'severity', which is not in state" of=3
17:18:21.644 INFO  jig.verify node.retry node=triage attempt=3 of=3 temperature=0.8 seed=2 reason="assert 'escalate == (severity == \"p0\")' could not be evaluated: expression references 'severity', which is not in state" rethink=false
17:18:21.661 WARNING jig.verify node.rejected node=triage attempt=3 cause=verify reason="assert 'escalate == (severity == \"p0\")' could not be evaluated: expression references 'severity', which is not in state" of=3
17:18:21.673 WARNING jig.graph node.failed run_id=typo node=triage type=generate attempts=3 error=NodeFailed reason="assert 'escalate == (severity == \"p0\")' could not be evaluated: expression references 'severity', which is not in state" on_fail=needs_human duration_ms=40.1
17:18:21.674 INFO  jig.graph edge.on_fail run_id=typo node=triage to=needs_human
17:18:21.674 INFO  jig.graph run.end run_id=typo pack=expr_demo end_node=needs_human steps=2 generations=3 failures=1 output_keys=1 output_bytes=19 duration_ms=40.6
{"ticket": "vague"}
```

Three generations, on a spelling mistake. (Reproduce with
`cp -r /tmp/jig-expr-demo /tmp/jig-expr-typo` and that one substitution in `graph.yaml`.
The `vague` ticket is the one whose fake script has three responses to burn.)

The mirror-image trap on an `assert` node is worse, because it is silent. An expression
that cannot be evaluated is routed **exactly like one that was false**. The demo pack with
`gate`'s `expr:` changed to `severity == "high"`:

```
$ python3 -m jig run /tmp/jig-expr-gate --input '{"ticket": "how do I export a CSV"}' --log-level info --run-id gate
17:18:28.198 INFO  jig.graph run.start run_id=gate pack=expr_demo version=1 entry=triage resumed=false max_steps=8 inputs=ticket
17:18:28.198 INFO  jig.graph node.ok run_id=gate node=triage type=generate attempts=1 output=merge duration_ms=0.1
17:18:28.198 INFO  jig.graph node.assert run_id=gate node=gate type=assert passed=false expr="severity == \"high\"" to=done duration_ms=0.0
17:18:28.198 INFO  jig.graph run.end run_id=gate pack=expr_demo end_node=done steps=3 generations=1 failures=0 output_keys=2 output_bytes=37 duration_ms=0.6
{"escalate": false, "priority": "p2"}
```

`severity` is not in state at all — the run reported `passed=false` and took the failure
edge with a clean exit 0. Raising the log level does not help: at `--log-level debug` that
same run emits exactly one line naming `node=gate` — the `passed=false` one above. Nothing
anywhere records that the expression was unanswerable rather than false.

Drop the `on_fail` from `gate` and the truth comes out:

```
$ python3 -m jig run /tmp/jig-expr-gate-loud --input '{"ticket": "how do I export a CSV"}'
jig: ExprError: expression references 'severity', which is not in state
exit=1
```

That is why `examples/invoice_extract/graph.yaml` gives its `inputs_check` node no
`on_fail`: a guard that cannot distinguish "false" from "unanswerable" is not a guard.

> **Rule of thumb.** Give an `assert` node an `on_fail` only when a *false* result is a
> real, expected outcome you have a node for. For a precondition — "this run has the
> inputs it needs" — leave `on_fail` off so a broken expression is loud.

A false expression on an `assert` node with no `on_fail` stops the run too, with a
different exception:

```
$ python3 -m jig run /tmp/jig-expr-loudfalse --input '{"ticket": "how do I export a CSV"}'
jig: AssertFailed: assert node 'gate' failed: escalate and priority == "p0"
exit=1
```

---

## Minimal form

```yaml
nodes:
  emit:
    type: generate
    max_tokens: 128
    assert: escalate == (priority == "p0")   # verification
    on_fail: needs_human

  currency_check:
    type: assert                             # routing
    expr: currency in ["USD", "EUR", "GBP"]
    on_fail: flag_currency
```

`type: assert` **requires** `expr:`; a node without one is refused at load
(`pack.py:_build_node`). A `generate` node's `assert:` is optional.

---

## Scope: what names an expression can see

The scope is a flat dict — run inputs plus everything committed so far, exactly as state.
There is no `self`, no module namespace, no import, and no attribute access on anything
but a mapping. There *is* one namespace behind state: the helper functions are bound as
bare names too, so `len` with no state key called `len` evaluates to the function object
rather than failing. That fallback and its consequences are in **Name resolution order**
below, and it is the one silent-pass hole in the language.

| Placement | Scope is |
| --- | --- |
| `assert` node (`expr:`) | committed state as it stands |
| `generate` node, no `output:` ("merge") | `dict(state)` **updated with the candidate's own fields at top level** |
| `generate` node with `output: name` | `dict(state)` plus `state["name"] = <the whole candidate>` |

Two consequences of the merge-mode trial scope, both real (`verify.py:_check_assert`):

* The candidate **shadows** the previous value of any key it rewrites. If state had
  `priority: "p2"` and the candidate says `"p0"`, the assert sees `"p0"`. You cannot
  compare a candidate against the old value of the same key from inside the assert.
* Nothing is committed either way. A rejected candidate never touches real state.

The trial scope is exactly `dict(state)` plus the candidate, so it can be built by hand:

```
$ python3 - <<'PY'
from jig.expr import evaluate
from jig.errors import ExprError

state = {"ticket": "card declined", "priority": "p2"}
candidate = {"priority": "p0", "escalate": True}

merge = dict(state)
merge.update(candidate)              # a generate node with no `output:`
named = dict(state)
named["triage"] = candidate          # a generate node with `output: triage`

print("merge trial scope:", merge)
print('  escalate == (priority == "p0") ->', evaluate('escalate == (priority == "p0")', merge))
print('  priority == "p2"               ->', evaluate('priority == "p2"', merge))
print("state itself is untouched:", state)
print("named trial scope:", named)
print('  triage.escalate == (triage.priority == "p0") ->',
      evaluate('triage.escalate == (triage.priority == "p0")', named))
try:
    evaluate('escalate == true', named)
except ExprError as exc:
    print('  escalate == true ->', exc)
PY
merge trial scope: {'ticket': 'card declined', 'priority': 'p0', 'escalate': True}
  escalate == (priority == "p0") -> True
  priority == "p2"               -> False
state itself is untouched: {'ticket': 'card declined', 'priority': 'p2'}
named trial scope: {'ticket': 'card declined', 'priority': 'p2', 'triage': {'priority': 'p0', 'escalate': True}}
  triage.escalate == (triage.priority == "p0") -> True
  escalate == true -> expression references 'escalate', which is not in state
```

`verify.py:_check_assert` turns that last `ExprError` into
`Rejected: assert 'escalate == true' could not be evaluated: <the message>`. There is a
whole-pack run of the `output: triage` form under **Complete worked example**.

`scratchpad` is reserved (`pack.py:RESERVED_STATE_NAMES`); a node may not commit to it, so
do not expect a think stage's notes in scope.

### A missing name

Always an `ExprError` — never `None`, never false. What happens next is the placement
difference in the table above. The message names the identifier but **not** the available
keys: in merge mode the state keys *are* the rejected candidate's field names, and putting
those in front of the model on the next rung is the self-conditioning spiral jig exists to
prevent.

The key list exists, on `exc.detail` (`expr.py:_error`) — but **nothing in jig ever reads
it.** `verify.py` builds its `Rejected` from `str(exc)`, `graph.py` re-raises or discards
the `ExprError`, and no log line at any level carries `detail`. (The `node.rejected.detail`
and `node.failed.detail` lines at `--log-level debug` are a different attribute,
`verify.Rejected.detail`, and they print the same redacted sentence.) So do not go looking
in the logs for the missing key list. It is reachable from Python only:

```
$ python3 - <<'PY'
from jig.expr import evaluate
from jig.errors import ExprError

state = {
  "priority": "p0", "escalate": True,
  "subtotal": 100.0, "tax_amount": 8.25, "total_amount": 108.25,
  "currency": "EUR", "due_date": None, "today": "2026-08-24",
  "tags": ["urgent", "paid"], "owner": None,
  "classification": {"category": "billing", "confidence": 0.9},
  "action_items": [{"task": "ship", "owner": "ali"},
                   {"task": "review", "owner": None}],
}
try:
    evaluate("missing_key == 1", state)
except ExprError as exc:
    print("str   :", exc)
    print("detail:", exc.detail)
PY
str   : expression references 'missing_key', which is not in state
detail: expression references 'missing_key', which is not in state (state has: action_items, classification, currency, due_date, escalate, owner, priority, subtotal, tags, tax_amount, today, total_am...)
```

The clip at `total_am...` is `_SHOW_LIMIT`, 120 characters. Long state lists are truncated
even here.

---

## Grammar

Everything below is derived from the whitelist in `jig/expr.py`. Anything not listed is
refused by node-type name.

### Literals

| Form | Notes |
| --- | --- |
| `true` / `false` / `null` / `none` | jig's own spellings (`_LITERALS`) |
| `True` / `False` / `None` | also work — Python constants reach `ast.Constant` |
| `1`, `-2`, `3.14`, `"text"`, `'text'` | numbers and strings |
| `[a, b]`, `(a, b)`, `{a, b}` | **all three produce a list.** `(1, 2)` evaluates to `[1, 2]` |
| `{"k": v}` | dict. `{**other}` is refused by name |

### Operators

| Category | Supported | Refused |
| --- | --- | --- |
| Arithmetic | `+` `-` `*` `/` `//` `%`, unary `-` `+` | `**` (`Pow`), all bitwise (`&` `\|` `^` `<<` `>>` `~`) |
| Comparison | `==` `!=` `<` `<=` `>` `>=` | |
| Membership | `in`, `not in` | |
| Identity | `is`, `is not` | see the gotcha below |
| Boolean | `and`, `or`, `not` | |
| Chained | `0 < len(tags) <= 5` | |
| Conditional | `a if cond else b` | |
| Indexing | `tags[0]`, `row["key"]`, `today[4]`, `tags[-1]` | slices (`tags[0:1]`) |
| Dotted | `classification.category` — mapping keys only | |

### Helpers

The complete set, and the only callables that exist. Positional arguments only —
`node.keywords` is refused outright, so `round(1.55, ndigits=1)` is
`ExprError: jig expression helpers take positional arguments only`.

| Helper | Behaviour |
| --- | --- |
| `len` `abs` `min` `max` `sum` `round` `sorted` `any` `all` | the Python builtins |
| `str` `int` `float` `bool` | the Python constructors |
| `lower(v)` `upper(v)` `strip(v)` | `str(v)` first, then the string method |
| `startswith(v, prefix)` `endswith(v, suffix)` | `str(v)` first |
| `contains(haystack, needle)` | `needle in haystack` — argument order is haystack-first |

The error message lists them for you:

```
open("/etc/passwd") -> ExprError: 'open' is not a jig expression helper (allowed: abs, all,
any, bool, contains, endswith, float, int, len, lower, max, min, round, sorted, startswith,
str, strip, sum, upper)
```

There are no methods. `priority.upper()` is not a method call — the walker flattens
`priority.upper` to a dotted path, finds it is not a helper name, and refuses it. Write
`upper(priority)`.

### Name resolution order

A bare name goes through `expr.py:_name` in this order, and every step of it has a
consequence worth knowing:

| # | Checked | Consequence |
| --- | --- | --- |
| 1 | `_LITERALS`: `true`, `false`, `null`, `none` | a state key with one of those four names is **unreachable** |
| 2 | names starting with `__` | refused: `name '__x' is not allowed in a jig expression` |
| 3 | `state` | a state key shadows a helper of the same name |
| 4 | `_HELPERS` | a bare helper name evaluates to **the function object**, which is truthy |
| — | otherwise | `ExprError: expression references 'x', which is not in state` |

A call (`_call`) never consults state at all — it looks only at `_HELPERS`. So a state key
called `len` shadows the bare name but not the call:

```
$ python3 - <<'PY'
from jig.expr import evaluate, is_true

print("state {'len': 5}: bare `len`   ->", evaluate("len", {"len": 5, "tags": ["a", "b"]}))
print("state {'len': 5}: `len(tags)`  ->", evaluate("len(tags)", {"len": 5, "tags": ["a", "b"]}))
print("state {}:         bare `len`   ->", evaluate("len", {}))
print("state {}:         is_true(len) ->", is_true("len", {}))
print("state {'none': 5}: `none`      ->", evaluate("none", {"none": 5}))
print("state {'false': 5}: `false`    ->", evaluate("false", {"false": 5}))
print("state {'false': 5}: is_true    ->", is_true("false", {"false": 5}))
print("state {'none': 'x'}: `none == null` ->", evaluate("none == null", {"none": "x"}))
PY
state {'len': 5}: bare `len`   -> 5
state {'len': 5}: `len(tags)`  -> 2
state {}:         bare `len`   -> <built-in function len>
state {}:         is_true(len) -> True
state {'none': 5}: `none`      -> None
state {'false': 5}: `false`    -> False
state {'false': 5}: is_true    -> False
state {'none': 'x'}: `none == null` -> True
```

Two failure modes fall out of that table, and neither one produces an error message.

**A field named `true`, `false`, `null` or `none` cannot be read.** Step 1 wins before
state is consulted. This bites exactly where it hurts most: in merge mode the candidate's
own field names *are* state keys, so a model field called `none` reads as `null` inside
the very assert that exists to check it — and `null == null` is `True`. Rename the field.

**A bare name that resolves to a helper always passes.** `expr: len` is `True`. An
`assert:` truncated to `all` while editing is `True`. There is no arity check, no
type check and no warning: an expression that is accidentally a bare name is a check
that has been switched off. This is the one shape this language has no defence against.

The demo pack with `gate`'s `expr:` replaced by the single word `all`, run on the ticket
that produced `passed=false` earlier:

```
$ python3 -m jig run /tmp/jig-expr-barename --input '{"ticket": "how do I export a CSV"}' --log-level info --run-id barename
17:21:18.883 INFO  jig.graph run.start run_id=barename pack=expr_demo version=1 entry=triage resumed=false max_steps=8 inputs=ticket
17:21:18.884 INFO  jig.graph node.ok run_id=barename node=triage type=generate attempts=1 output=merge duration_ms=0.1
17:21:18.884 INFO  jig.graph run.end run_id=barename pack=expr_demo end_node=escalated steps=3 generations=1 failures=0 output_keys=2 output_bytes=37 duration_ms=0.5
{"escalate": false, "priority": "p2"}
```

A p2 ticket with `escalate: false` ended at `escalated`. Note there is no `node.assert`
line at all: a *passing* assert node logs at DEBUG (`graph.py`, `INFO if not passed else
DEBUG`), so at INFO a gate that never fails is invisible. At `--log-level debug` it reads
`node.assert ... passed=true expr=all to=escalated`.

---

## Refused, on purpose

A pack is data. `graph.yaml` travels between machines, and a compiler — not a human —
writes most of it. An expression string that reached `eval()` would be a remote code
execution hole in a format designed to run untrusted, generated packs. So the walker
whitelists and refuses by name.

| Written | Result |
| --- | --- |
| `[x for x in tags]` | `Comprehension is not allowed in a jig expression (in '[x for x in tags]')` |
| `lambda x: x` | `Lambda is not allowed in a jig expression (in 'lambda x: x')` |
| `__import__("os")` | `'__import__' is not a jig expression helper (allowed: ...)` |
| `open("/etc/passwd")` | `'open' is not a jig expression helper (allowed: ...)` |
| `priority.upper()` | `'priority.upper' is not a jig expression helper (allowed: ...)` |
| `n ** 2` | `Pow is not allowed in a jig expression (in 'n ** 2')` |
| `n & 1` | `BitAnd is not allowed in a jig expression (in 'n & 1')` |
| `tags[0:1]` | `Slice is not allowed in a jig expression (in 'tags[0:1]')` |
| `{**other}` | `** unpacking is not allowed in a jig expression (in '{**other}')` |
| `x = 1` | `could not parse expression 'x = 1' (invalid syntax)` |
| any name starting with `__` | `name '__x' is not allowed in a jig expression` |

The quoted text in those messages is always the source string you passed, interpolated
by the walker (`%r % source`), never a canonical form. Write `{**d}` and the message says
`(in '{**d}')`.

There is no import, no assignment, no attribute access on anything but a mapping, no
function definition, no `eval`, no `exec`, no walrus, no f-string, no starred unpacking.

**Every** failure arrives as `jig.errors.ExprError`. That is load-bearing, not tidiness:
`verify._check_assert` and the graph walker each catch exactly `ExprError`. A raw
`TypeError`, `RecursionError` or `MemoryError` would sail past both handlers and kill the
run instead of failing the node — which is why the budgets below exist.

---

## Short-circuit: the guard idiom

`and` / `or` stop at the first operand that decides the result (`expr.py:_bool_op`), which
is what makes the standard null guard work at all:

```
expression: name is not null and len(name) > 2
{'name': None}    -> False
{'name': 'hi'}    -> False
{'name': 'hello'} -> True

without the guard, on {'name': None}:
ExprError: helper len() failed in 'len(name) > 2'
```

`examples/invoice_extract/graph.yaml` leans on this: `due_date == null or due_date >=
today` cannot compare `None` to a string, because it never gets there.

They also return **the operand, not a bool**, exactly like Python:

```
state = {'priority': 'p0', 'tags': ['urgent', 'paid']}

evaluate('priority or "none"', state) -> 'p0'
evaluate('tags and 0', state)         -> 0
is_true('tags and 0', state)          -> False
```

Both call sites go through `is_true`, so the truthiness coercion is what actually decides
pass/fail. `expr: tags` passes when `tags` is non-empty.

### `is` is identity, and it will lie to you

```
state = {'note': None, 'priority': 'p0', 'n': 3}

note is null        -> True
priority is "p0"    -> False      # state's string is a different object
n is 3              -> True       # small-int caching, today, on this build
```

**Use `is` / `is not` against `null` only.** For everything else use `==` / `!=`.

---

## Budgets

| Budget | Value | Constant |
| --- | --- | --- |
| Nesting depth | 100 levels below the root | `expr.py:_MAX_DEPTH` |
| Sequence repetition (`*`) | 1,000,000 elements | `expr.py:_MAX_REPEAT` |
| Diagnostic value clipping | 120 characters | `expr.py:_SHOW_LIMIT` |

Measured at the boundary, with `abs(abs(...abs(1)...))` and `not not ... true`:

```
100 nested abs() -> 1
101 nested abs() -> ExprError (depth)
100 nested not   -> True
101 nested not   -> ExprError (depth)

len("ab" * 500000) -> 1000000
len("ab" * 500001) -> ExprError: repetition in 'len("ab" * 500001)' would build a sequence that is too large (limit 1000000)
```

Every AST level counts, including the ones you did not write down as operators: the same
100 `abs(` wrapped around `-1` instead of `1` is 101 levels and is refused, because unary
minus is a level of its own.

The depth message quotes the whole expression, so it is as long as the expression was
(middle elided here):

```
expression 'not not not not ... not not true' is nested more than 100 levels deep, which jig refuses to evaluate
```

### Parentheses are not free

Grouping parens produce no AST node, so they cost nothing against `_MAX_DEPTH` — but
CPython's *parser* has its own ceiling, and it is much lower than the walker's:

```
200 nested parens -> 1
201 nested parens -> ExprError: could not parse expression '(((((((((((( ... ))))))))))))))' (too many nested parentheses)
```

(That message carries all 201 parentheses; the middle is elided above.)

**201, not 2000.** That refusal comes from `ast.parse` and is converted in
`expr.py:evaluate`, so it still arrives as an `ExprError` — but a generated expression
that stacks `(((...` for a routing table can hit it, and 200 is the whole budget.

---

## The hard limit: no iteration

There is no `for`, no comprehension, no `map`, and no lambda to pass to one. **You cannot
write "every item in this array has an owner."** This is the limitation pack authors hit
most.

`all()` and `any()` exist, but they test the *truthiness of the elements themselves*, which
on an array of objects is always true:

```
action_items = [{"task": "ship", "owner": "ali"}, {"task": "review", "owner": None}]

all(action_items)                -> True      # every dict is non-empty. Says nothing about owners.
contains(action_items, "ship")   -> False     # membership, not a search
sorted(action_items)             -> ExprError: helper sorted() failed in 'sorted(action_items)'
```

### What to do instead

**Make the model emit the aggregate, then assert the aggregate is consistent.** The model
is doing the iteration; the expression is checking its arithmetic. Three shapes that work:

| Want to check | Ask the grammar for | Assert |
| --- | --- | --- |
| the list is not silently truncated | an integer `action_count` | `action_count == len(action_items)` |
| every item has an owner | an integer `unowned_count` | `unowned_count == 0` |
| every item has an owner, flat | a parallel array `owners` of scalars | `all(owners)` |

Real results, against a state carrying that array plus the three counted fields —
`{"action_items": <above>, "action_count": 2, "unowned_count": 1, "owners": ["ali", None]}`:

```
len(action_items) == action_count -> True
unowned_count == 0                -> False
all(owners)                       -> False
any(owners)                       -> True
```

`examples/meeting_actions/graph.yaml` ships the first shape:
`assert: action_count == len(action_items) and (unowned == "none" or action_count > 0)` —
a truncated array is the classic silent failure of a small model, and this is how it is
caught. The second shape is the general answer: if a per-item property matters, make it a
counted field, then check the count.

If neither fits, the check wants to be a node in the graph, not a one-liner in YAML.

---

## Writing them in `graph.yaml`

jig parses its own YAML subset (`jig/yamlish.py`). Behaviour that matters here, all
verified against `yamlish.parse`:

| Written | Parsed as |
| --- | --- |
| `assert: escalate == (priority == "p0")` | `'escalate == (priority == "p0")'` |
| `expr: currency in ["USD", "EUR"]` | `'currency in ["USD", "EUR"]'` — kept as a string |
| `expr: tally == {"a": 1}` | `'tally == {"a": 1}'` |
| `expr: score > 3 # a comment` | `'score > 3'` — **the `#` and everything after it is stripped** |
| `expr: "score > 3 # not a comment"` | `'score > 3 # not a comment'` |

So: an unquoted `#` inside an expression is eaten as a comment. Quote the whole value if
you need one.

For long expressions use a folded scalar, which is what `examples/lead_qualify` does:

```yaml
    assert: >-
      disqualified == (segment == "enterprise" and email_domain_kind == "free_mail")
      and (disqualify_reason == "free_mail_enterprise") == disqualified
```

`>-` joins the lines with single spaces: `expr: >-\n  a == 1\n  and b == 2` becomes
`'a == 1 and b == 2'`.

### `jig validate` does not check your expression

Nothing parses expression text at load time. A pack with syntactically broken expressions
validates clean and fails at run time, one rung at a time. The demo pack with `triage`'s
assert replaced by `assert: "escalate ==== broken("`:

```
$ python3 -m jig validate /tmp/jig-expr-broken
expr_demo v1: 5 nodes, 2 edges, 0 evalset cases, entry 'triage'
validate exit=0

$ python3 -m jig run /tmp/jig-expr-broken --input '{"ticket": "vague"}' --log-level warning --run-id broken
17:19:23.708 WARNING jig.verify node.rejected node=triage attempt=1 cause=verify reason="assert 'escalate ==== broken(' could not be evaluated: could not parse expression 'escalate ==== broken(' (invalid syntax)" of=3
17:19:23.708 WARNING jig.verify node.rejected node=triage attempt=2 cause=verify reason="assert 'escalate ==== broken(' could not be evaluated: could not parse expression 'escalate ==== broken(' (invalid syntax)" of=3
17:19:23.708 WARNING jig.verify node.rejected node=triage attempt=3 cause=verify reason="assert 'escalate ==== broken(' could not be evaluated: could not parse expression 'escalate ==== broken(' (invalid syntax)" of=3
17:19:23.708 WARNING jig.graph node.failed run_id=broken node=triage type=generate attempts=3 error=NodeFailed reason="assert 'escalate ==== broken(' could not be evaluated: could not parse expression 'escalate ==== broken(' (invalid syntax)" on_fail=needs_human duration_ms=0.8
{"ticket": "vague"}
run exit=0
```

Three model calls and a clean exit 0, for an expression that could never have parsed.
Check an expression yourself before shipping it:

```
$ python3 -c "
from jig.expr import evaluate
print(evaluate('escalate == (priority == \"p0\")', {'priority': 'p0', 'escalate': True}))
"
True
```

---

## `when:` is not this language

Edges route on `when:`, which is **a mapping of dotted path to expected value, compared
with `!=`, and nothing more** (`graph.py:_matches`, `graph.py:_lookup`). It is not an
expression, it has no operators, and it does not go through `jig/expr.py`.

```yaml
  - from: emit
    to: escalated
    when: {priority: p0}          # equality only
```

The expected value is whatever `yamlish` coerced the scalar to, and the comparison is
plain `!=` against the value in state — no coercion at match time:

| Written | Expected value is | Matches state value |
| --- | --- | --- |
| `when: {priority: p0}` | the string `'p0'` | `'p0'` exactly — `'P0'` does not match |
| `when: {needs_review: true}` | the bool `True` | `True` only |
| `when: {count: 3}` | the int `3` | `3` only — the string `'3'` does not match |
| `when: {order_id: null}` | `None` | a key that **is** in state holding `None` |

Multiple keys are ANDed. A path that does not resolve returns a private `_MISSING`
sentinel, which compares unequal to everything, so the edge simply does not match — and
that has one consequence worth stating plainly: `when: {k: null}` matches a key that was
*written as null*, and can never match a key that was **never written at all**. "Route
when this key is absent" is not expressible in `when:`.

There is no `>`, no `in`, no `not`. If you need any of that, put an `assert` node in the
graph and route on its `on_fail`. `pack.py` refuses a `when:` that is not a mapping.

---

## Worked expressions

Every row below was run through `jig.expr.evaluate` against this state:

```python
{
  "priority": "p0", "escalate": True,
  "subtotal": 100.0, "tax_amount": 8.25, "total_amount": 108.25,
  "currency": "EUR", "due_date": None, "today": "2026-08-24",
  "tags": ["urgent", "paid"], "owner": None,
  "classification": {"category": "billing", "confidence": 0.9},
  "action_items": [{"task": "ship", "owner": "ali"},
                   {"task": "review", "owner": None}],
}
```

| Expression | Result |
| --- | --- |
| `escalate == (priority == "p0")` | `True` |
| `classification.category` | `'billing'` |
| `classification.confidence >= 0.8` | `True` |
| `abs(subtotal + tax_amount - total_amount) < 0.01` | `True` |
| `currency in ["USD", "EUR", "GBP"]` | `True` |
| `due_date == null or due_date >= today` | `True` |
| `owner is not null and len(owner) > 2` | `False` |
| `0 < len(tags) <= 5` | `True` |
| `today[4] == "-"` | `True` |
| `action_items[1]["owner"] is null` | `True` |
| `lower("P0") == priority` | `True` |
| `all(action_items)` | `True` (truthiness of the dicts — **not** a per-item check) |
| `(1, 2)` | `[1, 2]` (tuples and sets become lists) |
| `classification.missing` | `ExprError: expression references 'classification.missing', which is not a mapping key in state` |
| `tags.owner` | `ExprError: expression references 'tags.owner', which is not a mapping key in state` |
| `tags[9]` | `ExprError: cannot index by the value 'tags[9]' asks for` |
| `int("p0")` | `ExprError: helper int() failed in 'int("p0")'` |
| `1 / 0` | `ExprError: cannot evaluate '1 / 0' against these values` |
| `priority > 1` | `ExprError: cannot compare these values in 'priority > 1'` |
| `[x for x in tags]` | `ExprError: Comprehension is not allowed in a jig expression (in '[x for x in tags]')` |

Note what the error strings do **and do not** say. `tags[9]` does not print `9`, and
`int("p0")` does not print `p0` — those live on `exc.detail` (`expr.py:_error`), because
`str(exc)` becomes the feedback the model sees on the next rung and the offending value is
usually the model's own rejected output. As with a missing name, `detail` reaches no log:
read it from Python or not at all.

---

## Complete worked example

The demo pack from the top of this document, run three times. Both placements fire.

**Run 1 — the model breaks the invariant once, then gets it right.** The first draw says
`p1` with `escalate: true`; the assert rejects it before anything is committed and the
ladder re-samples.

```
$ python3 -m jig run /tmp/jig-expr-demo --input '{"ticket": "card declined twice"}' --log-level info
17:17:59.419 INFO  jig.graph run.start run_id=e4a436ebd2934e12acd5ec4b59761b3a pack=expr_demo version=1 entry=triage resumed=false max_steps=8 inputs=ticket
17:17:59.420 WARNING jig.verify node.rejected node=triage attempt=1 cause=verify reason="assert failed: escalate == (priority == \"p0\")" of=3
17:17:59.420 INFO  jig.verify node.retry node=triage attempt=2 of=3 temperature=0.5 seed=1 reason="assert failed: escalate == (priority == \"p0\")" rethink=false
17:17:59.420 INFO  jig.graph node.ok run_id=e4a436ebd2934e12acd5ec4b59761b3a node=triage type=generate attempts=2 output=merge duration_ms=0.2
17:17:59.420 INFO  jig.graph run.end run_id=e4a436ebd2934e12acd5ec4b59761b3a pack=expr_demo end_node=escalated steps=3 generations=2 failures=0 output_keys=2 output_bytes=36 duration_ms=0.6
{"escalate": true, "priority": "p0"}
```

Note `reason=` — the feedback the model is shown on rung 2 is the *expression text*, which
is pack-authored and therefore safe. It is never the rejected output.

**Run 2 — a clean p2 ticket. The `assert` node is false and routes, costing nothing.**

```
$ python3 -m jig run /tmp/jig-expr-demo --input '{"ticket": "how do I export a CSV"}' --log-level info
17:18:07.330 INFO  jig.graph run.start run_id=e61122a6f69949788b45aaac520189bc pack=expr_demo version=1 entry=triage resumed=false max_steps=8 inputs=ticket
17:18:07.331 INFO  jig.graph node.ok run_id=e61122a6f69949788b45aaac520189bc node=triage type=generate attempts=1 output=merge duration_ms=0.1
17:18:07.331 INFO  jig.graph node.assert run_id=e61122a6f69949788b45aaac520189bc node=gate type=assert passed=false expr="escalate and priority == \"p0\"" to=done duration_ms=0.0
17:18:07.331 INFO  jig.graph run.end run_id=e61122a6f69949788b45aaac520189bc pack=expr_demo end_node=done steps=3 generations=1 failures=0 output_keys=2 output_bytes=37 duration_ms=0.6
{"escalate": false, "priority": "p2"}
```

`generations=1`: the routing decision was free.

**Run 3 — the model never satisfies the invariant. The ladder is spent, then `on_fail`.**

```
$ python3 -m jig run /tmp/jig-expr-demo --input '{"ticket": "vague"}' --log-level info
17:18:07.397 INFO  jig.graph run.start run_id=c4c52b9024e1471e838fcc0e40c6c09e pack=expr_demo version=1 entry=triage resumed=false max_steps=8 inputs=ticket
17:18:07.398 WARNING jig.verify node.rejected node=triage attempt=1 cause=verify reason="assert failed: escalate == (priority == \"p0\")" of=3
17:18:07.398 INFO  jig.verify node.retry node=triage attempt=2 of=3 temperature=0.5 seed=1 reason="assert failed: escalate == (priority == \"p0\")" rethink=false
17:18:07.398 WARNING jig.verify node.rejected node=triage attempt=2 cause=verify reason="assert failed: escalate == (priority == \"p0\")" of=3
17:18:07.398 INFO  jig.verify node.retry node=triage attempt=3 of=3 temperature=0.8 seed=2 reason="assert failed: escalate == (priority == \"p0\")" rethink=false
17:18:07.398 WARNING jig.verify node.rejected node=triage attempt=3 cause=verify reason="assert failed: escalate == (priority == \"p0\")" of=3
17:18:07.398 WARNING jig.graph node.failed run_id=c4c52b9024e1471e838fcc0e40c6c09e node=triage type=generate attempts=3 error=NodeFailed reason="assert failed: escalate == (priority == \"p0\")" on_fail=needs_human duration_ms=0.4
17:18:07.398 INFO  jig.graph edge.on_fail run_id=c4c52b9024e1471e838fcc0e40c6c09e node=triage to=needs_human
17:18:07.398 INFO  jig.graph run.end run_id=c4c52b9024e1471e838fcc0e40c6c09e pack=expr_demo end_node=needs_human steps=2 generations=3 failures=1 output_keys=1 output_bytes=19 duration_ms=0.8
{"ticket": "vague"}
```

Three generations, one clean escalation to a human, and nothing wrong ever reached state.
That is the whole point of the language.

### The same pack with `output: triage`

The nested scope, end to end. Copy the pack (`cp -r /tmp/jig-expr-demo
/tmp/jig-expr-nested`) and edit `graph.yaml`: add `output: triage` to the `triage` node,
rewrite both expressions against the nested path, and change the `escalated` and `done`
projections to `[triage]`, since `priority` and `escalate` are no longer top-level keys.

```yaml
  triage:
    type: generate
    max_tokens: 32
    retries: 2
    output: triage
    assert: triage.escalate == (triage.priority == "p0")
    on_fail: needs_human

  gate:
    type: assert
    expr: triage.escalate and triage.priority == "p0"
    on_fail: done

  escalated:
    type: end
    output: [triage]

  done:
    type: end
    output: [triage]
```

```
$ python3 -m jig run /tmp/jig-expr-nested --input '{"ticket": "card declined twice"}' --log-level info --run-id nested
17:19:18.320 INFO  jig.graph run.start run_id=nested pack=expr_demo version=1 entry=triage resumed=false max_steps=8 inputs=ticket
17:19:18.321 WARNING jig.verify node.rejected node=triage attempt=1 cause=verify reason="assert failed: triage.escalate == (triage.priority == \"p0\")" of=3
17:19:18.321 INFO  jig.verify node.retry node=triage attempt=2 of=3 temperature=0.5 seed=1 reason="assert failed: triage.escalate == (triage.priority == \"p0\")" rethink=false
17:19:18.321 INFO  jig.graph node.ok run_id=nested node=triage type=generate attempts=2 output=triage duration_ms=0.6
17:19:18.321 INFO  jig.graph run.end run_id=nested pack=expr_demo end_node=escalated steps=3 generations=2 failures=0 output_keys=1 output_bytes=48 duration_ms=1.1
{"triage": {"escalate": true, "priority": "p0"}}
```

Same two rungs, same routing, one key out instead of two; in the ladder itself the
only structural change is `output=triage` on the `node.ok` line. Leave one path un-nested (write
`escalate` where you meant `triage.escalate`) and it is not a graph error: it is
`expression references 'escalate', which is not in state`, three times over.

---

## Checklist

- [ ] Every name in the expression is a run input or something an earlier node commits.
- [ ] For a `generate` node with `output: name`, paths are written `name.field`.
- [ ] The expression is not a bare name — `expr: len` and `assert: all` pass forever.
- [ ] No field is called `true`, `false`, `null` or `none`. Such a key reaches state and
      the run output normally; what it cannot do is be *read from an expression*, where
      the name resolves to the literal instead.
- [ ] `is` / `is not` appear only against `null`.
- [ ] Any operand that can be `null` is guarded before it is passed to a helper.
- [ ] No unquoted `#` in the YAML value.
- [ ] An `assert` node's `on_fail` exists only because *false* is an outcome you handle.
- [ ] Anything per-item is checked through a count the model emits, not through `all()`.
- [ ] The expression was run once against a sample state before it shipped.
