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

