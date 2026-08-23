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

