# jig — build queue

Work tasks **strictly in order**. One task per loop iteration. A task is DONE only when its
tests pass under `python3 -m pytest -q`. Mark `[x]` and commit before moving on.

**Hard rule for this queue: everything is tested against `FakeModel`. No GPU, no network, no
model downloads.** Real-backend work is T11 and is code-only (not executed).

---

- [x] **T1 — Model protocol + FakeModel**
  `jig/model.py`. Define `Model` protocol: `generate(prompt: str, grammar: dict | None, max_tokens: int) -> str`.
  Implement `FakeModel(scripted: list[str] | dict[str, str])` returning canned responses in order,
  or keyed by a substring match on the prompt. Raise on exhaustion.
  DONE: `tests/test_model.py` covers ordered mode, keyed mode, exhaustion error.

- [x] **T2 — Pack format: load + validate**
  `jig/pack.py`. A JigPack is a directory: `manifest.yaml` (name, version, entry node, model hint),
  `graph.yaml` (nodes + edges), `prompts/<node>.txt`, `grammars/<node>.json`, `evalset.jsonl`.
  Implement `load_pack(path) -> Pack` with clear validation errors (missing node, dangling edge,
  prompt/grammar file absent, no entry node).
  DONE: `tests/test_pack.py` — one valid fixture pack loads; five malformed fixtures each raise a
  distinct, specific error. Put fixtures in `tests/fixtures/`.

- [x] **T3 — Schema/grammar layer**
  `jig/grammar.py`. `schema_to_grammar(schema: dict) -> dict` (pass-through struct for now, backend
  translates later) and `validate_against(schema, obj) -> None | raises`. Pure-stdlib JSON Schema
  subset: type, required, properties, enum, additionalProperties=false. No external deps.
  DONE: `tests/test_grammar.py` — valid objects pass, each violation type raises with the field name.

- [x] **T4 — Graph walker**
  `jig/graph.py`. Execute nodes from entry, following edges. Node types: `generate`, `assert`, `end`.
  State is a dict passed between nodes; prompts render with `{var}` substitution from state.
  Edges may be conditional on state values.
  DONE: `tests/test_graph.py` — linear 3-node path, a conditional branch, a loop-with-max-iterations
  guard, dangling-edge error. Uses FakeModel only.

- [x] **T5 — Two-stage codegen (think → emit)**
  `jig/codegen.py`. For nodes with `two_stage: true`: first an unconstrained `think` generation
  (capped tokens, result kept only in a scratchpad var, never in final output), then a constrained
  `emit` conditioned on the scratchpad. Single-stage nodes skip straight to emit.
  DONE: `tests/test_codegen.py` — two-stage node makes exactly 2 model calls and the scratchpad does
  NOT appear in committed state; single-stage makes 1.

- [x] **T6 — Verify-before-commit + retry ladder**
  `jig/verify.py`, wired into the walker. Output must pass (a) grammar validation, (b) optional
  `assert` expression declared on the node. On failure: re-sample → re-sample with the error text
  appended → follow the node's `on_fail` edge (or raise `NodeFailed`).
  **Critical: failed output must never enter downstream state** (anti-self-conditioning).
  DONE: `tests/test_verify.py` — passes first try; recovers on retry 2; exhausts to `on_fail`;
  asserts that a rejected generation is absent from final state.

- [x] **T7 — Checkpointing**
  `jig/state.py`. Persist state after each committed node, keyed by run id. Backend: stdlib
  `sqlite3` (no third-party driver). `resume(run_id)` continues from the last committed node.
  DONE: `tests/test_state.py` — kill mid-run (simulate by raising in node 3), resume, verify nodes
  1–2 are not re-executed and the run completes.

- [x] **T8 — Evalset runner**
  `jig/eval.py`. Load `evalset.jsonl` (`{"input": {...}, "expect": {...}}`), run each case through
  the pack, score exact-match on the declared output fields. Report pass/fail per case + totals +
  per-node failure counts (this is the per-node signal the design calls for).
  DONE: `tests/test_eval.py` — a pack scoring 3/4 reports exactly that, and attributes the failure
  to the correct node.

- [x] **T9 — CLI**
  `jig/cli.py` + `python3 -m jig`. Commands: `jig run <pack> --input '<json>'`,
  `jig eval <pack>`, `jig validate <pack>`. Stdlib `argparse` only. Exit code 1 on eval failure.
  DONE: `tests/test_cli.py` — invokes via `subprocess`, asserts output text and exit codes.

- [ ] **T10 — Example pack: support_triage**
  `examples/support_triage/` — a real 4-node workflow (classify → extract fields → decide priority →
  emit ticket JSON), with prompts, grammars, and a 12-case evalset. Wire a FakeModel script so
  `jig eval examples/support_triage` passes end to end in CI.
  DONE: `tests/test_example.py` — the example scores 12/12 with the scripted FakeModel.

- [ ] **T11 — Real backend adapter (CODE ONLY, DO NOT RUN)**
  `jig/backends/openai_compat.py`. `OpenAICompatModel(base_url, model, api_key=None)` speaking the
  OpenAI-compatible `/v1/chat/completions` used by llama.cpp-server, vLLM, and SGLang. Pass grammars
  through as `response_format`/`grammar` per backend flag. Use stdlib `urllib.request` — no `requests`.
  DONE: `tests/test_backend.py` mocks the HTTP layer only; **never make a real network call.**

- [ ] **T12 — README**
  Root `README.md`: the §0 problem statement from `docs/PLAN.md` compressed, a 15-line quickstart
  using the example pack, and the benchmark table **left empty with a `TODO: measure` marker** —
  do NOT invent numbers.
  DONE: README exists, quickstart commands are copy-pasteable and actually work.
