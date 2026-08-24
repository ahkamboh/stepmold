# Overnight cloud build — how to run it

Open **claude.ai/code** → new session → connect to repo **`ahkamboh/jig`**, branch `main`.
Then copy everything in the block below and paste it as the first message.

Once it's running, the laptop can be off — it runs on Anthropic's infrastructure.

---

## The prompt (copy from here)

```
You are building `jig` autonomously overnight. I am asleep and cannot answer questions — make reasonable decisions yourself and record them in NIGHT-LOG.md.

START: read TASKS.md (the ordered build queue) and docs/ARCHITECTURE.md (the architecture) in full before writing any code. T1 is already complete — begin at T2.

YOUR ORACLE: passing tests are the only proof of progress. Never mark a task done without them. Note that pytest is NOT installed; a stdlib test harness already exists in the repo — use it, do not install anything.

Work TASKS.md strictly in order, one task at a time. For each task:
1. Write its tests FIRST, then the implementation.
2. Iterate until the ENTIRE test suite is green, not just the new tests.
3. Mark the task [x] in TASKS.md.
4. Append a dated block to NIGHT-LOG.md: task id, files changed, test count, and any decision you made on your own or anything surprising.
5. Commit with a conventional-commit message, then push to origin main. Commit and push ONLY when the full suite is green — every pushed commit must be a working state.
6. If a task fails 3 attempts in a row: write a BLOCKED entry in NIGHT-LOG.md with exactly what is stuck and what you tried, mark it [!] instead of [x], commit, push, and move on. Do not burn the night on one task.

HARD RULES:
- Python 3 standard library ONLY. No pip install, no venv, no new dependencies. If a task seems to need one, implement the minimal piece yourself in stdlib and note it.
- No network calls and no model downloads at test time. Everything tests against the FakeModel from T1. T11 is CODE ONLY — write it, mock the HTTP layer in its test, never make a real request.
- Never invent benchmark, accuracy, or cost numbers in code, comments, docs, or README. Write `TODO: measure`. The project's credibility depends on measured numbers only.
- No GitHub issues, PRs, or releases. Commit to main and push, nothing else.
- Follow docs/ARCHITECTURE.md for architecture. If ARCHITECTURE.md and TASKS.md conflict, follow TASKS.md and note it.
- Keep code small and readable — plain functions and dataclasses, no class hierarchies, no frameworks, no metaprogramming.

FINISH: when every task is [x] or [!], append a SUMMARY to NIGHT-LOG.md: what's complete, what's blocked and why, decisions I should review, and what to look at first. Commit and push it.
```

## Copy to here

---

## Checking progress from your phone

GitHub app → `ahkamboh/jig` → Commits.
Each commit = one task finished with green tests. **4 commits at handoff**, up to ~15 by morning.
Tap `NIGHT-LOG.md` to read what it did and any decisions it made.

Gaps between commits are normal — some tasks (graph walker, verify-and-retry) are much
bigger than others. Forty quiet minutes does not mean it is stuck.

## In the morning

```bash
cd ~/Documents/GitHub/jig && git pull && cat NIGHT-LOG.md && python3 -m pytest -q
```

Read the log first, then verify the tests yourself rather than trusting the log's claim.
Any task marked `[!]` is where it got blocked — that is where you pick up.

## Note

Everything tonight is tested against `FakeModel`. By morning the framework skeleton is
mechanically proven, but it does **not** tell you whether a real 8B passes an evalset —
that is the M0 gate and needs a live model plus supervision. Tonight builds the harness
so that measurement takes an hour instead of a week.
