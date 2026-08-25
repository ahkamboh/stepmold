# stepmold — the agent compiler for small models

> **A machinist's stepmold is a custom guide that lets a cheap tool do precision work, repeatably, forever.**
> stepmold is that for LLMs: the big model builds the stepmold once; a small model runs inside it thousands of times.

- Working name: **`stepmold`** (npm-safe fallback: `agentstepmold`)
- One-liner: *"Compile your agent once with a frontier model. Run it forever on an 8B."*
- Thesis: agentic cost is wasted on **runtime** thinking for **repeated** workflows. Move all thinking to **compile time** (paid once), leave only constrained execution at runtime (paid per call, at small-model prices).

---

## 0. The problem statement

**Companies pay frontier-model prices to do work that contains no thinking — and can't switch to cheap models because small models fall apart over multi-step tasks.**

The two existing escapes both fail:
- *"Use a smaller model"* → error compounding (2%/step = 33% failure at 20 steps), invalid tool calls, self-conditioning decay.
- *"Fine-tune it"* → needs an ML team, labeled data, GPUs, and a retrain on every base-model update. Most companies can't.

stepmold is the third path: **don't make the model smarter, make the task easier.** All thinking moves to compile time; the small model only ever does one short, constrained, verified step.

### Problems it solves
1. **High-volume repetitive workflows** — invoice extraction, ticket triage, CRM updates, lead qual, compliance checks. Same N steps, 50k×/month, zero novel reasoning. → 30–100× on the execution tier.
2. **"We can't send this data anywhere"** (banks, health, legal, gov, defense) — today their options are *no AI* or *a GPU cluster for a 70B*. stepmold makes an 8B on one modest GPU reliable enough to use. **Not a discount — access they didn't have.** Strongest wedge.
3. **"Works 85% of the time, don't know why"** — versioned pack + grammars + 50-test evalset turns "usually works" into "v3 passes 50/50". (Shopify: DSPy-optimized small model was ~2× *more reliable* than the GPT-5 prompt it replaced.)
4. **"We can't prove what our AI did"** — regulated buyers can't audit *"we prompted Claude"*; they can audit a versioned artifact. Nobody ships this today.
5. **Vendor lock-in** — model swap = `stepmold build --model X` gated on the evalset, not a rewrite.
6. **Offline/edge** — static binary + pack + local model, no internet.

<!-- a section on commercial strategy is kept out of the public repo -->
### What stepmold does NOT solve
Open-ended novel tasks (plan must be invented per request); workflows run ~10×/month (build never amortizes — just call an API); bad requirements (50 bad examples compile a bad stepmold); research/exploration.
**Boundary: stepmold is for workflows repeated enough that "compile once" pays off. Repetition is the qualifier.**

<!-- a section on commercial strategy is kept out of the public repo -->
## 1. The core inversion

Every existing framework (LangChain, smolagents, CrewAI) is a **runtime**: the model decides what happens next, on every step, on every run. That's why they need frontier models — and why cost scales linearly with usage *at frontier prices*.

stepmold is a **compiler**. It splits an agent's life into two phases:

| Phase | Model | Frequency | Cost |
|---|---|---|---|
| **stepmold build** (compile) | Frontier (Opus-class) | Once per workflow + on regression | ~$5–50, one-time |
| **stepmold run** (execute) | Small (4B–14B, self-hosted) | Every request | ~$0.15/M tokens |

The compile step produces a **StepmoldPack** — a versioned, cached bundle of five artifacts. The runtime never improvises; it executes the pack.

```
stepmold build  ──►  StepmoldPack v3
                ├── graph.json        # the workflow state machine (the "plan")
                ├── prompts/          # DSPy/GEPA-optimized per-node prompts
                ├── grammars/         # precompiled xgrammar automata per node
                ├── evalset/          # ~50 gold examples (the CONTRACT)
                └── snapshot.fc       # Firecracker memory snapshot of the tool env

stepmold run    ──►  small model + StepmoldPack on one shared vLLM/SGLang instance
```

**The eval set is the source of truth, not the prompts.** Prompts, grammars, and graphs are all regenerable build outputs. The ~50 gold examples per workflow are the only hand-maintained asset. Model changed? `stepmold build` again. This is CI/CD for agents.

---

## 2. The five scaling bugs and their engineered fixes

### Bug 1 — Grammar compilation cost explodes at scale
Naive constrained decoding recompiles schemas per request; complex schemas measured at 3.6×–8.2× latency overhead.

**Fix (in stepmold):**
- Backend on **XGrammar-2**: <40µs/token overhead (bitwise mask ops), grammar automaton compiled once in 20–50ms, then cached. Per-request overhead on repeated schemas drops **to ~zero after warmup** — and stepmold's whole model is repeated schemas.
- **Compile at build time, not request time**: `stepmold build` runs the grammar compiler over every node schema and ships the compiled automata inside the StepmoldPack. Runtime never compiles anything.
- XGrammar-2's **Cross-Grammar Cache** reuses substructures across grammars (shared `reasoning` field, shared enums) and **TagDispatch** handles dynamic tool selection — 6× faster tool-calling compilation than prior SOTA.
- Rule enforced by the compiler: **one tool per generation node**. Never a union grammar of 40 tools; the graph decides which tool, the grammar only shapes its arguments. Keeps every automaton tiny.

### Bug 2 — The constraint tax (valid JSON, wrong answer)
Research ("The Constraint Tax", "When Correct Isn't Usable"): quality loss comes mostly from forcing early commitment, not from the decoder. Fix is known: decouple reasoning from formatting.

**Fix (in stepmold):** **two-stage generation is the compiler's default codegen**, not an option the developer must remember:
1. `think`: short **unconstrained** generation (capped tokens, discarded from final output)
2. `emit`: **constrained** extraction against the node's grammar, conditioned on the scratchpad

The compiler auto-inserts stage 1 only on nodes whose eval-set error rate justifies it (measured during `stepmold build`), so trivial nodes stay single-stage and cheap. Alternative codegen for simple nodes: `reasoning` field ordered before `answer` in the schema.

### Bug 3 — Prompt optimization is expensive and model-coupled
DSPy compile = 100–500 LLM calls, $20–50, 10–30 min per pipeline; the output is coupled to one checkpoint and silently degrades when you swap models.

**Fix (in stepmold):**
- Optimization runs **inside `stepmold build`**, per node, against the evalset — treated exactly like a compiler optimization pass (GEPA as the default optimizer; the Shopify result — GPT-5 task → small Qwen, ~75× cheaper, ~2× more reliable — is the proof this pass pays for itself).
- **Per-node signal, not end-to-end**: because the graph gives every node its own input/output contract, stepmold can localize which node fails — solving DSPy's "black box end-to-end only" limitation structurally.
- **Model swap = rebuild, not rewrite.** `stepmold build --model qwen3-8b` recompiles all prompts against the new model and gates on the evalset. Build artifacts are content-hashed and cached: only nodes whose (schema, evalset slice, model) changed get re-optimized. Incremental compilation, like any real compiler.
- Amortization math: $50 build ÷ 1M runs = **$0.00005/run**. Rounding error.

### Bug 4 — Sandbox economics (code-as-action doesn't scale)
One Docker sandbox per agent works in demos; 500 concurrent agents = 500 cold-starting containers.

**Fix (in stepmold):** three execution tiers, chosen per node at compile time:
- **Tier 0 — whitelisted namespace (default, ~90% of nodes):** the node can only call functions declared in the StepmoldPack manifest; runs in-process, zero isolation cost. Most "agentic" client work is tool orchestration, not arbitrary code.
- **Tier 1 — warm Firecracker pool:** for real code execution. Pre-warmed microVM **snapshots restore in 28–150ms** (E2B/Manus/Perplexity run this pattern in production); pool is reset-and-reused, sized by Little's law from measured node concurrency.
- **Tier 2 — cold isolated microVM:** untrusted/long-running code only.
- The **environment snapshot ships in the StepmoldPack** (`snapshot.fc`): deps pre-installed, tools pre-imported — so tier-1 restore lands in a ready interpreter, not a booting one.

### Bug 5 — Graph authoring is linear human cost + serving sprawl
Every workflow needs a hand-built graph (agencies die on this), and naive "one model config per workflow" eats one GPU per workflow.

**Fix (in stepmold):**
- **Graph induction at compile time**: `stepmold build` gives the frontier model the task description + gold examples and has it *propose* the state machine; the evalset gates acceptance; a human approves the diff once. Big model as **design-time compiler**, never a runtime dependency. (Same shape as the proven distill pattern: large model produces traces/structure once → small model executes; GRPO-on-large → SFT-distill beats training the small model directly.)
- **Behavior-as-artifact serving**: all workflows share **one base model on one vLLM/SGLang instance**. A workflow is a StepmoldPack (prompts + grammars + graph), not a model copy. 30 workflows = 30 packs, 1 GPU.
- **Prefix caching (RadixAttention) as a compiler target**: the compiler orders every node prompt as `[shared system block][workflow block][node block][live variables]` — most-stable-first — so the KV cache hit rate is engineered, not accidental. Agent loops re-send the same prefix thousands of times; this is where self-hosted throughput 2–3×'s.
- LoRAs are **explicitly out of v1** (no fine-tune, per design). If ever added: vLLM multi-LoRA serves dozens of adapters per GPU, but note ForkKV's finding — distinct adapters break prefix-cache sharing — one more reason behavior-in-artifacts beats behavior-in-weights for scaling.

---

## 3. Runtime reliability (what stops error compounding without a big model)

- **Graph-bounded horizons**: the small model never plans; it fills one slot per node with short, fresh context. Horizon length is a compiler property, not a model property (MAKER showed million-step near-zero-error via extreme decomposition).
- **Verify-before-commit**: each node's output must pass its grammar (guaranteed) + optional deterministic assert declared in the manifest. Failed output never enters downstream context — this directly attacks the self-conditioning effect.
- **Retry ladder, cheap-first**: re-sample (temp bump) → re-sample with error appended → mark node failed per graph's failure edge. No silent frontier fallback in v1 — keeps the cost story honest; an optional escalation edge can be a v2 flag the client opts into.
- **Checkpointed state**: graph state persisted per node (SQLite/Postgres), so long workflows resume instead of restart.

<!-- a section on commercial strategy is kept out of the public repo -->
<!-- a section on commercial strategy is kept out of the public repo -->
## 5. What stepmold is NOT

- Not a router/cascade (no per-request model choice; the frontier model is compile-time only)
- Not RAG, not fine-tuning (v1 is strictly inference-time; ART/RULER-style RL is a far-future `stepmold train` if ever)
- Not for open-ended novel tasks — stepmold targets **defined, repeatable workflows**, which is exactly where the money is being burned today

## 6. MVP roadmap

1. **M0 — hardcoded pack (1–2 wk):** one real workflow, hand-written graph + grammars, SGLang + XGrammar-2 + Qwen 8B, two-stage codegen, tier-0 exec. Metric: pass-rate vs Opus baseline on the evalset, cost per run.
2. **M1 — `stepmold build` (2–3 wk):** evalset format, GEPA pass per node, content-hashed incremental artifact cache, `stepmold build --model` swap gate.
3. **M2 — graph induction (2 wk):** frontier model proposes graph from task+examples, human-approved diff, evalset gate.
4. **M3 — scale layer (3–4 wk):** shared serving with prefix-ordered prompts, Firecracker warm pool (tier 1), per-node tracing/replay, `stepmold run --serve` multi-pack.
5. **M4 — polish:** StepmoldPack registry, OSS release (MIT, same playbook as scrolltape/agent-eyes), docs with the cost benchmark as the headline.

Go/no-go gate after M0: small-model pass rate within ~2–3 pts of frontier on the chosen workflow at <1/30th cost. If not, the workflow needs decomposition (fix the graph), not a bigger model.

## 7. Language & tech stack (decided)

**Python 3.12** for the whole framework. Rationale: stepmold is orchestration glue — all performance-critical work already lives in C++/CUDA/Rust engines we call, and every mandatory dependency is Python-first (DSPy/GEPA has no non-Python equivalent; XGrammar's primary API is its Python binding). Runtime is I/O-bound on GPU inference → asyncio scales it.

| Component | Choice |
|---|---|
| `stepmold build` compiler | Python 3.12 (DSPy/GEPA, xgrammar bindings, content-hash cache) |
| `stepmold run` executor | Python + asyncio (graph walker, retry ladder) → SGLang over HTTP |
| CLI | Typer + Rich |
| StepmoldPack format | **Language-agnostic**: JSON/YAML manifest + artifacts (enables future TS runtime SDK) |
| Node contracts | Pydantic models → JSON Schema → grammars (single source of truth) |
| State/checkpoints | SQLite default, Postgres option |
| Firecracker pool daemon | Rust/Go, M3+ only; E2B API / Docker subprocess until then |
| Packaging | `uv` + pyproject, PyPI as `stepmold` (fallback `agentstepmold`) |

Rule: Python everywhere code touches models; the pack format belongs to no language.

### 7.1 Language decision, long form (Rust/Go evaluated)

stepmold has two halves with **opposite** requirements — that's why there's no single answer:

| | `stepmold build` (compiler) | `stepmold run` (runtime) |
|---|---|---|
| Work | DSPy/GEPA, graph induction, grammar precompile | HTTP→SGLang, graph walk, retry, checkpoint |
| Frequency | once per workflow | millions of times |
| Bound by | LLM calls (minutes) | I/O wait on GPU |
| Language | **Python — locked** (DSPy/GEPA exists nowhere else) | **open — Go or Rust beat Python** |

The language-agnostic StepmoldPack format is what makes a non-Python runtime legal later without touching the compiler. Deliberate.

**Rust vs Go for the runtime → Go.** The runtime is HTTP + state machine, not CPU-bound. Rust's edges (no GC, deterministic latency, memory control) buy ~nothing when every step blocks ~200ms on a GPU, and cost months of solo dev. Go: single static binary, 18MB image, sub-second deploy, 30–60MB RAM for hundreds of concurrent sessions, written in ~a week. Also: grammar enforcement runs **server-side in SGLang**, so the runtime needs no xgrammar/llguidance bindings — removing Rust's one real technical advantage here.

**Rust earns exactly one component:** the Firecracker warm-pool daemon (tier-1 sandbox, M3+) — Firecracker is Rust, isolation is security-critical, genuine hot loop. Matches the 2026 production pattern (Python builds → Go serves → Rust sandboxes).

**Business driver (the operator on-prem):** shipping `stepmold run` as one static binary (`scp` it) vs. a Python env (3.12 + uv + ~40 deps + CUDA version matching) across 30 client boxes is the difference between a product and a support burden. This — not throughput — is the real reason to port.

**Sequenced call:**
1. M0–M2: **Python for everything.** Do not split while the runtime spec is still moving.
2. M3: port `stepmold run` to **Go** if on-prem deploy pain is real; keep Python runtime as reference impl + dev mode.
3. `stepmold build` stays Python permanently.
4. Rust only if the Firecracker pool daemon ships.

Rejected: TypeScript (fine future client SDK, poor daemon, no compiler libs), Mojo (immature; bottleneck was never Python compute), C/Zig (hot path is already C++/CUDA inside xgrammar/vLLM/llama.cpp — you inherit C, you don't write it).

Note for grammar backend choice: llguidance (Rust, ~50µs/tok) beats XGrammar on **unique/dynamic** schemas; XGrammar wins on **repeated** schemas via caching. stepmold is repeated-schema by architecture → **XGrammar-2 stays the pick**.

### 7.2 Size / compile-speed constraints (stated priority)

**Key insight: the compiler never ships.** gcc doesn't ship with the binary it compiled; `stepmold-build` doesn't ship with the agent it compiled. Python's weight is a build-time cost on hardware we control. The client receives one static binary + one pack.

```
BUILD MACHINE (ours)                    CLIENT SERVER
stepmold-build (Python, DSPy/GEPA,  ──pack──► stepmold (Go static, <10MB)
xgrammar, 500MB+ deps)                   + StepmoldPack (<500KB)
                                         no Python, no CUDA deps
```

| Artifact | Target | Method |
|---|---|---|
| `stepmold` runtime binary | **<10MB** | Go, `CGO_ENABLED=0`, `-ldflags="-s -w"` |
| StepmoldPack | **<500KB** | all text: prompts, JSON grammars, graph |
| Cold start | **<20ms** | pack parsed once at boot |
| Full compile | **<3s** | dependency budget below |

Build: `CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o stepmold ./cmd/stepmold`
(`CGO_ENABLED=0` → static, no libc, cross-compile to any client box via `GOOS`/`GOARCH`.)

**Runtime dependency budget — hard rule, ~2 direct deps:**
- Allowed: Go stdlib (`net/http`, `encoding/json`, `crypto/tls`, `flag`) + `modernc.org/sqlite` (pure Go, no cgo) + optionally one tiny CLI lib.
- Banned: ORMs, DI frameworks, reflection-heavy config, anything with a transitive tree. Every dep = compile seconds + binary bytes.

**Where "fast" actually comes from.** stepmold's own CPU work is microseconds; each node then waits ~200ms on the GPU — so language speed is irrelevant to end-to-end latency. The real levers, in order:
1. HTTP keep-alive + connection pooling to SGLang (no per-node TLS handshake)
2. Prefix-ordered prompts → engineered KV-cache hits (2–3× throughput)
3. Pack loaded once at boot, never re-parsed
4. Pipeline next-node prompt assembly while current node drains
5. Batch concurrent runs into one SGLang instance (GPU occupancy is the throughput number)

With those done, Go and Rust perform identically — which is precisely why Go: same small fast binary, without Rust's minutes-long compiles (directly against the "easily compile" goal).

**Roadmap revision:** size/compile/speed is now a top-3 constraint, so port the runtime to Go at **M2, not M3** — before surface area grows. Porting 800 lines is a weekend; porting 5,000 is not.

## 8. Risks

- **Evalset quality is the ceiling** — 50 bad examples compile a bad stepmold. Mitigation: evalset linter + frontier-model review pass during build.
- **XGrammar/SGLang API churn** — pin versions in the pack; artifacts declare their backend version.
- **"Compile step" adoption friction** — devs expect `pip install && run`. Mitigation: `stepmold init` does M2's induction interactively so the first build feels like magic, not homework.
- **Name collision check before publish** — `stepmold` is short; verify npm/PyPI/GitHub availability, fall back to `agentstepmold`.

<!-- a section on commercial strategy is kept out of the public repo -->
## Research base

- XGrammar-2: <40µs/token, 20–50ms one-time grammar compile, cross-grammar cache, 6× tool-call compile speedup — arxiv.org/html/2601.04426, github.com/mlc-ai/xgrammar, blog.mlc.ai/2026/05/04/xgrammar-2
- Constraint tax / two-stage fix — arxiv.org/pdf/2605.26128, arxiv.org/pdf/2605.02363, aidancooper.co.uk/constrained-decoding
- DSPy/GEPA economics + Shopify 75× case — dspy.ai/getting-started/gepa-optimization, natebjones.com/prompts-and-guides/products/dspy-guide, arxiv.org/pdf/2507.03620
- Firecracker snapshot pools 28–150ms, E2B/Manus/Perplexity in production — dev.to/adwitiya (28ms), addozhang.medium.com sandbox survey, firecracker snapshot docs
- Prefix caching + multi-LoRA limits (ForkKV) — docs.vllm.ai automatic prefix caching, arxiv.org/pdf/2604.06370, spheron.network multi-LoRA guide
- Error compounding / self-conditioning / MAKER decomposition — arxiv.org/pdf/2509.09677, arxiv.org/html/2604.11978, Cognizant MAKER
- Small-model agentic bases (if needed later): xLAM-2, Arch-Agent, Qwen — github.com/SalesforceAIResearch/xLAM, github.com/katanemo/Arch-Function
