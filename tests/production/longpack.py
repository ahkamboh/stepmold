"""A generated N-node pack, and a model that fails like a small model does.

docs/ARCHITECTURE.md §0 makes one quantitative claim and the whole product rests on it:

    "Use a smaller model" -> error compounding (2%/step = 33% failure at 20 steps)

and §3 claims jig removes that structurally — short bounded steps, a grammar per node,
verify-before-commit, and a rejected generation that never re-enters context. Nobody had
ever measured it. This module is the apparatus; `test_longhorizon.py` is the experiment.

Two pieces:

* `build_pack` writes a real JigPack to disk — an N-link arithmetic chain where every
  node's correct output is a single integer computed from the previous one. That
  checkability is the point: a wrong answer is unambiguous, so "did the run succeed" is
  a fact rather than a judgement. The same chain is emitted under different `Arm`s
  (strict schema or permissive, semantic assert or none, retries or none), so jig's
  defences can be switched off one at a time and the difference measured.

* `FlakyModel` is a `jig.model.Model` that fails on purpose with a *seeded* per-call
  probability, in the six ways a small model actually fails: prose instead of JSON,
  prose wrapped around good JSON, truncated JSON, a wrong-typed field, an extra field,
  and — the one no schema can catch — a plausible integer that is simply wrong.

The fault draw is a hash of `(seed, node, attempt)`, not a sequential RNG. That is
deliberate and load-bearing: a paired comparison needs node 7's first attempt to
misbehave *identically* whether or not the arm under test allows retries. A sequential
RNG would desynchronise the moment one arm made a different number of calls, and the
"jig vs baseline" number would be measuring stream drift as much as jig.

stdlib only, like everything else in jig. Nothing here touches a network.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from jig.model import Call
from jig.pack import load_pack

__all__ = [
    "ARMS",
    "Arm",
    "Emission",
    "FAULT_MIX",
    "FlakyModel",
    "POISON_MARK",
    "STEP_MOD",
    "build_pack",
    "chain_answer",
    "expected_final",
    "node_schema",
    "step_value",
]


# --------------------------------------------------------------------- the chain

# The recurrence every node computes. Chosen for three properties: it stays inside three
# digits so prompts never grow with N (the horizon claim would be untestable if they
# did), it is cheap to express in jig's assert language, and it mixes enough that a
# corrupted link never coincidentally re-joins the correct chain.
STEP_MUL = 7
STEP_ADD = 13
STEP_MOD = 1000


def step_value(previous):
    """One link of the chain."""
    return (previous * STEP_MUL + STEP_ADD) % STEP_MOD


def chain_answer(seed_value, links):
    """Apply `step_value` `links` times — the analytically correct answer."""
    value = seed_value
    for _ in range(links):
        value = step_value(value)
    return value


def expected_final(seed_value, n):
    """What `v{n}` must be at the end of an n-node run started from `v0`."""
    return chain_answer(seed_value, n)


# ----------------------------------------------------------------------- the arms


@dataclass(frozen=True)
class Arm:
    """One configuration of jig's defences, so they can be removed one at a time.

    `naive` is the closest thing this harness has to ARCHITECTURE.md's "naive 8B, free-running
    loop": the model's output is committed if it is a JSON object at all. It is not a
    *straw* baseline — it still gets jig's graph decomposition and its fresh per-node
    context, because those are structural and cannot be switched off without writing a
    different runtime. So every number below understates jig's total contribution and
    isolates exactly the part this experiment is about: verification and the ladder.
    """

    name: str
    strict_schema: bool   # a real grammar per node, or `{"type": "object"}`
    check_answer: bool    # node-level `assert:` — the only thing that catches a lie
    retries: int          # re-samples after the first attempt
    two_stage: bool = False
    on_fail_end: bool = False  # divert a spent node to an end node instead of raising


NAIVE = Arm("naive", strict_schema=False, check_answer=False, retries=0)
GRAMMAR_ONLY = Arm("grammar_only", strict_schema=True, check_answer=False, retries=0)
VERIFY_ONLY = Arm("verify_only", strict_schema=True, check_answer=True, retries=0)
JIG = Arm("jig", strict_schema=True, check_answer=True, retries=2)
JIG_LADDER_5 = Arm("jig_r5", strict_schema=True, check_answer=True, retries=5)
JIG_TWO_STAGE = Arm("jig_two_stage", strict_schema=True, check_answer=True, retries=2,
                    two_stage=True)
JIG_ON_FAIL = Arm("jig_on_fail", strict_schema=True, check_answer=True, retries=2,
                  on_fail_end=True)

ARMS = (NAIVE, GRAMMAR_ONLY, VERIFY_ONLY, JIG)


# ------------------------------------------------------------------ pack emission

_PROMPT = """\
Step {step} of a deterministic arithmetic chain. You do one link and nothing else.

Compute: v{step} = (v{prev_step} * %d + %d) mod %d

NODE=n{step}
INPUT={{v{prev_step}}}

Answer with {{{{"v{step}": <integer>}}}} and nothing else.
""" % (STEP_MUL, STEP_ADD, STEP_MOD)


def build_pack(root, n, arm):
    """Write an `n`-node chain pack under `root` and return the loaded `Pack`.

    The directory is named for `(n, arm)` so a caller can build every configuration
    once into one scratch root and reuse the loaded packs across thousands of trials —
    `Pack` is frozen and a run never edits it.
    """
    path = os.path.join(root, "%s_n%d" % (arm.name, n))
    if not os.path.isdir(path):
        _emit(path, n, arm)
    return load_pack(path)


def _emit(path, n, arm):
    os.makedirs(os.path.join(path, "prompts"))
    os.makedirs(os.path.join(path, "grammars"))

    _write(path, "manifest.yaml", "\n".join([
        "name: longchain_%s_n%d" % (arm.name, n),
        "version: 1",
        "entry: n1",
        "description: >-",
        "  A generated %d-node arithmetic chain, arm %r. Every node's correct" % (n, arm.name),
        "  output is a single integer derived from the previous node's, so a wrong",
        "  answer is unambiguous and end-to-end success is a fact, not a judgement.",
        "",
    ]))

    lines = ["# generated by tests/production/longpack.py — do not hand-edit", ""]
    # +3 buys the end node and a little slack; a chain that needs more than that is
    # looping, which is exactly what max_steps is for.
    lines.append("max_steps: %d" % (n + 3))
    lines.append("")
    lines.append("nodes:")
    for step in range(1, n + 1):
        lines.append("  n%d:" % step)
        lines.append("    type: generate")
        lines.append("    max_tokens: 32")
        if arm.two_stage:
            lines.append("    two_stage: true")
            lines.append("    think_max_tokens: 64")
        lines.append("    retries: %d" % arm.retries)
        if arm.check_answer:
            lines.append("    assert: v%d == (v%d * %d + %d) %% %d"
                         % (step, step - 1, STEP_MUL, STEP_ADD, STEP_MOD))
        if arm.on_fail_end:
            lines.append("    on_fail: gave_up")
        lines.append("")
    lines.append("  done:")
    lines.append("    type: end")
    lines.append("    output: [v%d]" % n)
    lines.append("")
    if arm.on_fail_end:
        # A separate end node, so "the run gave up here" is distinguishable from "the
        # run finished" by end_node alone — no need to inspect the output to tell.
        lines.append("  gave_up:")
        lines.append("    type: end")
        lines.append("    output: [v0]")
        lines.append("")
    lines.append("edges:")
    for step in range(1, n + 1):
        lines.append("  - from: n%d" % step)
        lines.append("    to: %s" % ("n%d" % (step + 1) if step < n else "done"))
    lines.append("")
    _write(path, "graph.yaml", "\n".join(lines))

    for step in range(1, n + 1):
        _write(path, os.path.join("prompts", "n%d.txt" % step),
               _PROMPT.format(step=step, prev_step=step - 1))
        _write(path, os.path.join("grammars", "n%d.json" % step),
               json.dumps(node_schema(step, strict=arm.strict_schema), indent=2))


def node_schema(step, strict=True):
    """The schema node `n{step}` emits under."""
    if not strict:
        # The permissive arm still declares a schema, because `generate` nodes require
        # one — but it constrains nothing beyond "is an object", which is what an
        # unconstrained free-running loop effectively enforces.
        return {"type": "object"}
    field_name = "v%d" % step
    return {
        "type": "object",
        "properties": {field_name: {"type": "integer"}},
        "required": [field_name],
        "additionalProperties": False,
    }


def _write(path, relative, text):
    with open(os.path.join(path, relative), "w") as handle:
        handle.write(text)


# ------------------------------------------------------------------- the model


# Every faulted generation carries this outside its JSON, so a test can search the whole
# run for it. It is the tracer dye for ARCHITECTURE.md §3's "a rejected generation never re-enters
# context": if it ever shows up in a later prompt, the invariant is broken.
POISON_MARK = "POISON-9f3a"

# How a small model fails, and how often. `wrong_value` carries the most weight on
# purpose: it is the only fault a grammar cannot see, and a mix that under-weights it
# would flatter jig. The rest split between format failures (which a grammar catches
# for free) and shape failures (which a schema catches).
FAULT_MIX = (
    ("wrong_value", 0.35),     # valid JSON, valid schema, plausible integer, wrong
    ("prose_only", 0.15),      # chatty answer with no JSON in it at all
    ("truncated", 0.15),       # ran out of tokens mid-object
    ("schema_type", 0.15),     # right field, wrong type ("four hundred and six")
    ("extra_key", 0.10),       # correct answer plus an invented field
    ("prose_wrapped", 0.10),   # correct JSON, buried in prose
)

FAULT_NAMES = tuple(name for name, _ in FAULT_MIX)

_INPUT = re.compile(r"^INPUT=(.*)$", re.MULTILINE)
_NODE = re.compile(r"^NODE=(n\d+)$", re.MULTILINE)
_SCRATCH_HINT = re.compile(r"WORKED IT OUT: (-?\d+)")


@dataclass
class Emission:
    """One generation this model produced, and why it looks the way it does."""

    node: str
    attempt: int
    stage: str          # "think" or "emit"
    fault: Optional[str]
    text: str


@dataclass
class FlakyModel:
    """A `Model` that fails with a seeded probability, the way a small model does.

    `p` is the per-*attempt* probability of a bad generation. Faults are drawn
    independently per attempt, which is the optimistic reading of a retry ladder — see
    `stubborn=True` for the pessimistic one, where the model makes the same mistake
    every time it is asked (what a greedy decoder actually does when the only thing the
    retry changed is a line of feedback appended to the prompt).
    """

    seed: int
    p: float
    stubborn: bool = False
    calls: List[Call] = field(default_factory=list, init=False, repr=False)
    emissions: List[Emission] = field(default_factory=list, init=False, repr=False)
    attempts: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    think_attempts: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    # ---------------------------------------------------------------- protocol

    def generate(self, prompt, grammar=None, max_tokens=512):
        self.calls.append(Call(prompt=prompt, grammar=grammar, max_tokens=max_tokens))
        node = _match(_NODE, prompt) or "?"
        step = int(node[1:]) if node[1:].isdigit() else 0
        correct = self._answer_for(prompt, step)

        # `grammar is None` is how the think stage identifies itself: `codegen.think`
        # never passes one, `codegen.emit` always does for a node that has a schema.
        if grammar is None:
            return self._think(node, step, correct)

        attempt = self.attempts.get(node, 0)
        self.attempts[node] = attempt + 1

        # A two-stage node that already reasoned its way to a number follows its own
        # notes. That is not this model being difficult: it is what conditioning on a
        # scratchpad means, and it is the whole reason ARCHITECTURE.md Bug 2 wants the think
        # stage in the first place.
        hinted = _match(_SCRATCH_HINT, prompt)
        if hinted is not None:
            value = int(hinted)
            return self._record(
                node, attempt, "emit",
                None if value == correct else "wrong_value",
                json.dumps({"v%d" % step: value}),
            )

        fault = self._fault_for(node, attempt)
        return self._record(node, attempt, "emit", fault,
                            self._render(fault, step, correct, node, attempt))

    # ------------------------------------------------------------------ stages

    def _think(self, node, step, correct):
        """The unconstrained stage. Its number is what the emit stage will obey.

        Counted separately from the emit attempts, so that a `think` which is re-run
        draws a fresh fault. Today `verify.run_node` never re-runs it — that is the
        defect this harness exists to measure — but the counter has to be here, or the
        measurement would be an artefact of the model rather than of jig.
        """
        attempt = self.think_attempts.get(node, 0)
        self.think_attempts[node] = attempt + 1
        fault = self._fault_for(node + "#think", attempt)
        value = _plausible_wrong(correct, node) if fault else correct
        text = ("Let me take this step by step. v%d follows from the input, so\n"
                "WORKED IT OUT: %d\n" % (step, value))
        return self._record(node, attempt, "think", fault, text)

    def _record(self, node, attempt, stage, fault, text):
        self.emissions.append(Emission(node=node, attempt=attempt, stage=stage,
                                       fault=fault, text=text))
        return text

    # ------------------------------------------------------------------ faults

    def _fault_for(self, node, attempt):
        """Which fault (if any) this attempt produces — a pure function of the seed.

        Two independent draws off different salts: one decides *whether* the attempt is
        bad, one decides *how*. Keeping them separate means changing the fault mix does
        not reshuffle which attempts fail, so two sweeps stay comparable.
        """
        # A stubborn model's draw ignores the attempt number, so every re-sample of a
        # node reproduces the identical mistake. Nothing in jig's ladder can move it:
        # `jig/verify.py` says so itself — with no sampling parameter in the `Model`
        # protocol, a re-sample changes only the appended feedback.
        index = 0 if self.stubborn else attempt
        if _draw(self.seed, node, index, "fail") >= self.p:
            return None
        return _weighted(_draw(self.seed, node, index, "kind"))

    def _answer_for(self, prompt, step):
        """The correct answer given whatever the prompt actually says the input is.

        Note "whatever the prompt says", not "whatever the run started with". A model
        computes from the number in front of it, so once an unverified arm commits a
        corrupted link every later node is confidently wrong about a wrong input. That
        is compounding, and reproducing it faithfully is the point of the experiment.
        """
        raw = (_match(_INPUT, prompt) or "").strip()
        try:
            previous = int(raw)
        except ValueError:
            # A previous node committed something that is not a number. A real model
            # asked to multiply `"four hundred and six"` still answers *something*;
            # this answers something deterministic.
            previous = int(hashlib.blake2b(raw.encode("utf-8"),
                                           digest_size=4).hexdigest(), 16) % STEP_MOD
        return step_value(previous)

    def _render(self, fault, step, value, node, attempt):
        """Turn a (fault, value) pair into the bytes a model would have emitted."""
        name = "v%d" % step
        good = json.dumps({name: value})
        if fault is None:
            return good
        tag = "%s-%s-%d" % (POISON_MARK, node, attempt)
        if fault == "wrong_value":
            # No poison tag: this generation is *accepted* by every arm without an
            # assert, so its bytes legitimately reach later prompts as state. Tagging it
            # would make the tracer-dye test assert something false.
            return json.dumps({name: _plausible_wrong(value, node)})
        if fault == "prose_only":
            return ("I think the answer here is roughly %d, but let me flag %s "
                    "before you rely on it." % (value, tag))
        if fault == "truncated":
            return '{"%s": %d' % (name, value)
        if fault == "schema_type":
            # "about 406", not "406": a stringified integer round-trips back to the
            # right number at the next node, which would let a type error be harmless.
            # A model that ignores `type: integer` writes prose into the field.
            return json.dumps({name: "about %d" % value}) + "  <!-- %s -->" % tag
        if fault == "extra_key":
            return json.dumps({name: value, "confidence": "high", "note": tag})
        if fault == "prose_wrapped":
            return ("Sure — here is the step %s you asked for:\n\n%s\n\nHope that "
                    "helps!" % (tag, good))
        raise AssertionError("unknown fault %r" % fault)

    # ------------------------------------------------------------------ reading

    @property
    def call_count(self):
        return len(self.calls)

    def rejected_texts(self):
        """Every generation that no arm of this experiment could have committed."""
        return [e.text for e in self.emissions
                if e.fault is not None and e.fault != "wrong_value"]

    def wasted_calls(self):
        """Generations spent on attempts after the first, per node."""
        return sum(max(0, count - 1) for count in self.attempts.values())


def _plausible_wrong(correct, node):
    """A different integer in the same range — the lie a schema cannot see."""
    offset = 1 + int(hashlib.blake2b(node.encode("utf-8"),
                                     digest_size=2).hexdigest(), 16) % (STEP_MOD - 2)
    return (correct + offset) % STEP_MOD


def _draw(seed, node, attempt, salt):
    """A uniform [0, 1) that depends only on its arguments."""
    key = "%d|%s|%d|%s" % (seed, node, attempt, salt)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _weighted(draw):
    total = 0.0
    for name, weight in FAULT_MIX:
        total += weight
        if draw < total:
            return name
    return FAULT_MIX[-1][0]


def _match(pattern, text):
    found = pattern.search(text)
    return found.group(1) if found else None
