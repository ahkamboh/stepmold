# Night log

Append one dated block per loop iteration: task, what changed, tests, blockers.

---
## 2026-08-24 — T0 (unplanned prerequisite): stdlib test harness

**Why this exists.** The build oracle is `python3 -m pytest -q`, but no pytest is installed
on this machine for any interpreter (3.9 / 3.12 / 3.14 all report "No module named pytest"),
and the hard rule forbids `pip install`. Per the standing instruction — *if a task seems to
need a dependency, implement the minimal piece yourself in stdlib* — I vendored a tiny
pytest stand-in instead of installing one.

**Files changed:** `pytest/__init__.py`, `pytest/_compat.py`, `pytest/_shim.py`,
`pytest/__main__.py`, `conftest.py`, `tests/__init__.py`, `tests/test_smoke.py`

**Tests:** 3 passing (the harness self-test).

**Decisions I made alone — please review:**
1. **The whole suite is written as plain `unittest.TestCase`**, not pytest-style bare
   functions. That is deliberate: `unittest` is stdlib, and the real pytest collects
   `unittest.TestCase` natively, so every test in this repo runs identically under real
   pytest with zero changes. Nothing in the suite depends on the shim.
2. **The shim defers to the real pytest.** `pytest/_compat.py` looks for a genuine pytest
   on `sys.path` (excluding the repo root) and, if it finds one, installs it into
   `sys.modules` and steps aside entirely. So `pip install pytest` later just works — the
   shim never shadows it. If you would rather delete the `pytest/` directory outright once
   you have real pytest, you can: nothing else imports it.
3. `conftest.py` at the root only puts the repo root on `sys.path`; it is there for the
   real pytest's benefit as much as the shim's.

**Surprising:** `python3` here is 3.14.5, not the 3.12 that `docs/PLAN.md` §7 locks in.
Everything written tonight is kept 3.9-compatible in syntax so the version choice stays open.

## 2026-08-24 — T1: Model protocol + FakeModel

**Files changed:** `jig/__init__.py`, `jig/model.py`, `tests/test_model.py`

**Tests:** 18 passing total (15 new).

`Model` is a `runtime_checkable` Protocol with one method,
`generate(prompt, grammar=None, max_tokens=512) -> str`. `FakeModel` implements it in two
modes (ordered list, or dict keyed by prompt substring) and raises `ModelExhausted` when the
script runs dry.

**Decisions I made alone:**
1. **`FakeModel` records every call** (`model.calls` → `Call(prompt, grammar, max_tokens)`,
   plus `model.call_count`). Not in the T1 spec, but T5 requires asserting "exactly 2 model
   calls" and T6 requires proving a rejected generation never reached state — both need this
   record, and bolting it on later would have meant reopening T1.
2. **Keyed mode: longest matching key wins.** With plain "first match" the result depends on
   dict order, which is a coin-flip for the test author. Longest-match lets a specific key
   (`"emit ticket"`) override a general one (`"emit"`) predictably.
3. **A keyed value may be a list, consumed in order.** T10 scripts one 4-node workflow across
   12 evalset cases; a single string per node cannot express "different answer per case", and
   substring keys alone can't separate the cases. A plain string key stays inexhaustible
   (answers every matching prompt); a list key is a queue.
4. `ModelExhausted` covers both "ordered script ran out" and "no key matched" — one error
   class, distinct messages, rather than two near-identical exceptions.


## 2026-08-24 — T2: Pack format (load + validate)

**Files changed:** `jig/yamlish.py`, `jig/pack.py`, `tests/test_yamlish.py`,
`tests/test_pack.py`, `tests/fixtures/` (1 valid pack + 8 malformed).

**Tests:** 67 passing total (49 new: 27 yamlish, 22 pack).

**The big decision: I wrote a YAML parser.** `TASKS.md` specifies `manifest.yaml` and
`graph.yaml`, the stdlib has no YAML parser, and `pip install` is forbidden. Per the
standing rule (*implement the minimal piece yourself in stdlib*), `jig/yamlish.py` is a
~330-line subset parser: block mappings, block sequences, nesting, comments, single-line
flow collections (`[a, b]`, `{k: v}`), quoted/plain scalars, null/bool/int/float/str.
Anchors, aliases, tags, block scalars (`|`, `>`) and multi-document files each raise a
clear `YamlError` with a line number rather than being silently mis-parsed. It has its
own 27-test file because a hand-rolled parser under the whole pack format is exactly the
sort of thing that fails quietly.

**Conflicts / notes:**
- `docs/PLAN.md` says `graph.json`; `TASKS.md` says `graph.yaml`. Followed TASKS.md as
  instructed. The loader accepts YAML only; JSON support would be two lines if you want it.
- PLAN.md §7 names Pydantic as the single source of truth for node contracts. Not
  available (stdlib rule), so grammars are plain JSON Schema files on disk and T3 will
  validate against them by hand.

**Decisions I made alone — please review:**
1. **Node fields for later tasks are already in the schema** (`two_stage`, `retries`,
   `on_fail`, `assert`, `max_tokens`, `think_max_tokens`, `output`). T2 only had to load
   them, but T4–T6 all need them and reopening the format later would churn every fixture.
   They are loaded and validated now, and used from T4 on.
2. **A generate node writes to state via `output:`** — `output: classification` nests the
   emitted object under that key; omitting `output` merges the object's keys into state at
   the top level. On an `end` node, `output:` instead means "project only these keys into
   the run result".
3. **Edge conditions are `when: {key: value}` equality maps, evaluated in declaration
   order, first match wins; an edge with no `when` is the fallback.** No expression
   language in edges — assert nodes and node-level `assert` are where expressions belong
   (T6), and keeping edges dumb keeps the graph readable in a diff.
4. **`model:` in the manifest is a URI-ish string** (`fake:fakes/script.json`,
   `openai:http://host:8000|model`), doubling as the "model hint" T2 asks for and the
   default the T9 CLI resolves when `--model` is not given. That is what will let
   `jig eval examples/support_triage` run offline in CI (T10).
5. **`evalset.jsonl` is optional at load time** but strictly validated when present; `jig
   eval` will be the thing that insists on it.
6. **Extra validations beyond the five TASKS.md asks for**: unknown node/edge keys,
   `end` nodes with outgoing edges, non-end nodes with no outgoing edge, `on_fail`
   pointing at an undefined node, malformed `evalset.jsonl`. Nine malformed fixtures, not
   five — the failure modes are cheap to catch at load time and expensive at run time.

**Surprising:** `python3` is **3.11.15** in this container, not the 3.14.5 the T0 entry
recorded. Code stays 3.9-compatible so this keeps not mattering.

**Process note — where I am pushing.** The instruction says "push to origin main", but
this session's harness pins development to the branch
`claude/autonomous-jig-build-62hlu8` and forbids pushing elsewhere. I am pushing every
commit there. The branch is a straight-line continuation of `main`, so
`git merge --ff-only claude/autonomous-jig-build-62hlu8` on main will fast-forward it
with no conflicts.

## 2026-08-24 — T3: Schema/grammar layer

**Files changed:** `jig/grammar.py`, `tests/test_grammar.py`, `jig/pack.py` (wiring),
`tests/test_pack.py`, `tests/fixtures/bad_grammar_schema/`.

**Tests:** 89 passing total (22 new).

`schema_to_grammar(schema)` returns `{"kind": "json_schema", "schema": <deep copy>}` —
the backend-neutral struct that gets handed to `Model.generate` and that T11 translates
into `response_format`. It deep-copies so a backend can never mutate a pack's schema.
`validate_against(schema, obj)` implements the subset by hand: `type` (incl. union types
like `["string", "null"]`), `properties`, `required`, `enum`, `items`,
`additionalProperties: false`. Errors carry a dotted `path` (`ticket.id`, `tags[1]`) as an
attribute, not just in the message, so T8 can attribute a failure without regex-ing text.

**Decisions I made alone — please review:**
1. **Unknown schema keywords are an error, not ignored.** `{"requried": [...]}` raises
   `SchemaError`. A silently-ignored constraint is a constraint you think you have and
   don't — and this is a verification system.
2. **Grammars are schema-checked at pack load time** (`jig/pack.py` now calls
   `check_schema`, raising a new `GrammarError`). This was not asked for in T2 or T3, but
   it is the difference between `jig validate` catching a typo and a node failing at run
   time. Tenth malformed fixture added for it.
3. **`items` is in the subset** even though TASKS.md's list stops at
   `additionalProperties`. An array-of-strings field is unavoidable in a real pack (T10's
   ticket has tags), and validating the array but not its contents would be a hole.
4. **Booleans are not integers/numbers**, matching JSON rather than Python — `True` fails
   `{"type": "integer"}`. Python's `isinstance(True, int)` is a genuine trap here.
5. `ValidationError` subclasses `ValueError` and `SchemaError` does too, so a caller can
   catch either without importing jig internals.

## 2026-08-24 — T4: Graph walker

**Files changed:** `jig/graph.py`, `jig/errors.py`, `jig/render.py`, `jig/expr.py`,
`tests/test_graph.py`, `tests/test_render.py`, `tests/test_expr.py`.

**Tests:** 150 passing total (61 new: 31 graph, 10 render, 20 expr).

`run(pack, model, inputs, run_id=None, max_steps=None) -> RunResult`. The loop is
deliberately dumb: execute node, commit output to state, pick the first edge whose `when`
matches. Nothing asks the model where to go — that is the whole "the small model never
plans" property from PLAN.md §3, and it is why a graph is auditable.

**Decisions I made alone — please review:**
1. **Three support modules, not one fat walker.** `errors.py` (the run-time exception
   hierarchy, in a leaf module so graph/codegen/verify/eval can all import it without a
   cycle), `render.py` (prompt templating), `expr.py` (assert expressions). Each is small
   and separately tested; the walker itself is ~120 lines.
2. **`render` is not `str.format`.** Prompts contain literal JSON constantly (`{"a": 1}`),
   which `str.format` would read as a field, and `str.format` also does attribute and
   index access — more machinery than a prompt deserves. `render` does one thing: look a
   dotted name up in state and render it (non-strings as JSON, so `true`/`null`, not
   `True`/`None`). `{{` and `}}` are literal braces so a prompt can show an example object.
3. **`expr.py` parses with `ast` and walks a whitelist — it never calls `eval()`.** Packs
   are compiler-generated data; a pack that can `eval()` is remote code execution. Allowed:
   names from state, dotted mapping lookup, comparisons, and/or/not, `in`, arithmetic,
   indexing, literals, and 16 named helpers (`len`, `lower`, `startswith`, ...). Refused by
   name: lambdas, comprehensions, method calls, attribute access on non-mappings, dunders,
   assignment. `true`/`false`/`null` are accepted as literals so an expression reads like
   the YAML around it.
4. **`assert` node semantics:** expression true -> follow edges normally; false -> jump to
   `on_fail` if declared, else raise `AssertFailed`. `on_fail` is a *node name*, not an
   edge, so a failure path does not need its own conditional edge.
5. **Edge `when` keys may be dotted** (`{c.category: billing}`), and a key that is missing
   from state simply does not match rather than raising. A missing key is the normal case
   on the first pass through a loop; raising there would make every loop need a guard edge.
6. **Provenance is tracked as state is written** (`{state_key: node_name}`) and returned on
   the `RunResult`. T8 needs it to attribute an eval failure to a node, and reconstructing
   it afterwards would be guesswork.
7. **`end` nodes with `output: [keys]` project just those keys into `RunResult.output`;**
   without it the output is the whole state. `RunResult.state` always holds everything, so
   nothing is lost — projection is about what the caller sees.
8. **Non-object generations are rejected** — a node must emit a JSON object, not a bare
   string or array, because state is a mapping. This currently raises `NodeFailed` on the
   first attempt; T6 turns that into the retry ladder.

## 2026-08-24 — T5: Two-stage codegen (think -> emit)

**Files changed:** `jig/codegen.py`, `jig/graph.py` (now delegates), `tests/test_codegen.py`.

**Tests:** 167 passing total (17 new).

`generate_once(node, state, model, error=None, scratchpad=None) -> Attempt`. Two-stage
nodes make exactly 2 calls (unconstrained think capped at `think_max_tokens`, then
constrained emit); single-stage nodes make 1. `Attempt.scratchpad` is returned to the
caller and **never written to state** — `graph.commit` only ever writes `Attempt.text`.
Two tests pin that: the scratchpad string is absent from the final state *and* from a
later node's prompt.

**Decisions I made alone — please review:**
1. **A re-sample re-rolls the emit stage only, reusing the scratchpad** (that is what the
   `scratchpad=` argument is for). PLAN.md §3 says the ladder is cheap-first, and the emit
   half is the cheap half. The cost: if the *thinking* is what was wrong, re-emitting
   repeats the mistake. If measurement later shows retries failing for that reason, the
   fix is a `rethink_on_retry: true` node flag — deliberately not built on speculation.
2. **Scratchpad placement is prompt-cache-aware.** By default the notes are appended after
   the rendered node prompt, and a correction after that — stable content first, volatile
   last, per PLAN.md §2's prefix-ordering rule. A prompt that contains `{scratchpad}`
   explicitly overrides that and places it wherever the author wants.
3. **The think stage renders `{scratchpad}` as empty** when the default think template is
   derived from an emit prompt that references it. Found by a test failure, not by
   reading — there are no notes yet at think time, so empty is the honest value.
4. **The walker no longer knows how many calls a node costs.** `graph.execute_generate` is
   now three lines that call `codegen`. Keeping the call-count policy in one module is what
   will let T6 wrap the ladder around it without touching the walk.

## 2026-08-24 — T6: Verify-before-commit + retry ladder

**Files changed:** `jig/verify.py`, `jig/graph.py` (ladder + `Failure` record),
`tests/test_verify.py`, `tests/test_graph.py` (one test updated, see below).

**Tests:** 193 passing total (26 new).

`verify(node, text, state)` parses, schema-validates, and evaluates the node's optional
`assert` — against a **trial copy** of state, so a candidate that fails is discarded whole
and the real state never saw it. `run_node` is the ladder: attempt, plain re-sample,
re-sample with the rejection appended, then `on_fail` or `NodeFailed`. Three tests pin the
anti-self-conditioning property directly: the rejected value is absent from final state,
absent from the output, and absent from the next node's prompt.

**Decisions I made alone — please review:**
1. **The first re-sample deliberately carries no error text**, exactly as TASKS.md
   describes the ladder. Cheap-first: a different sample often just works, and appending a
   rejection costs tokens and biases the model toward the field it just got wrong. The
   error text appears from rung 3 on.
2. **`extract_json` is forgiving about *finding* JSON, and verification stays merciless
   about what is in it.** It tries the raw text, then a ``` fence, then the first balanced
   `{...}` span (string-aware, so a `}` inside a string does not truncate it). With a real
   constrained decoder none of that fires; without one (llama.cpp in loose modes, any
   server that ignores `response_format`) it is the difference between working and not.
3. **A diverted failure is recorded on `RunResult.failures`**, not swallowed. When a node
   exhausts its ladder and `on_fail` sends the run somewhere else, the run "succeeds" —
   and without a record, T8 could never attribute that. `Failure(node, reason, attempts)`.
4. **`retries` counts re-samples, not attempts.** Default 2 = the three-rung ladder.
   `retries: 0` means one attempt and no ladder, which is what you want for a node whose
   failure should route immediately.
5. **A node-level `assert` sees the candidate placed exactly where it would be committed**
   — merged at top level, or under `output:` — so the expression reads the same as one in
   a downstream `assert` node.
6. **I changed one T4 test.** `test_a_non_object_generation_is_rejected` scripted a single
   bad response and expected an immediate `NodeFailed`; that was pinning the T4 placeholder
   behaviour, which T6 was always going to replace. It now scripts three and asserts the
   ladder spent all three. No other test needed touching.

**No frontier fallback.** PLAN.md §3 rules it out for v1 and I did not add one — a silent
escalation to a big model would make every cost number in the README a lie.
