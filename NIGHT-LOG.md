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
