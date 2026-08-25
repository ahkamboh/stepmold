"""Verify-before-commit, and the retry ladder.

This module is the reason a small model can be trusted here at all (docs/ARCHITECTURE.md §3).
Two rules, and the second one is the load-bearing one:

1. **Nothing is committed until it is verified.** Output must parse, satisfy the node's
   schema, and satisfy the node's optional `assert` expression.
2. **A rejected generation is never shown to the model again.** Not in the run's state,
   not in a later node's prompt, and not in this node's own retry prompt. Verification
   runs against a *trial* copy of state; only a candidate that survives is written back.
   The retry prompt may say *what was wrong* (`Rejected.feedback`, derived from the
   pack's schema) but never *what the model said*. That is the structural fix for
   self-conditioning decay — a model that never sees its own bad output cannot spiral
   on it.

   The full text, offending value and all, is retained in `RunResult.failures` and in
   the checkpoint. That is deliberate: an operator debugging a pack needs it, and bytes
   on disk cannot condition a model. `tests/test_invariants.py` enforces both halves.

The ladder is: generate, then re-sample once for each retry the node allows, then the
node's `on_fail` edge (or `NodeFailed`). No frontier-model fallback, by design — ARCHITECTURE.md
keeps the cost story honest.

Every rung after the first changes two things, because a re-sample is only worth its
tokens if it is a *different draw*:

* **a sampling hint** (`codegen.Sampling`, `RESAMPLE_TEMPERATURES` below). ARCHITECTURE.md §3
  writes the first rung as a plain re-sample "(temp bump)", and for a long time nothing
  at this layer could bump anything: the protocol had no sampling parameter, so against
  a greedy backend rung 1 re-issued a byte-identical request and got the byte-identical
  rejection. The hint is optional in the protocol and passed only to a model that
  declares it (`codegen._generate`), so a backend that cannot vary its sampling is
  called exactly as before and still gets the feedback rung.
* **the rejection, as feedback** — what was wrong, never what was said.

And it drops one thing: on a `two_stage` node, the scratchpad behind a rejected answer.
The think stage is what produced the reasoning the emit stage obeyed, so re-emitting from
those same notes is not a re-sample at all — `tests/production/test_longhorizon.py`
measured a two-stage arm sitting exactly on the naive analytical curve because of it. A
rejection makes the notes suspect, so the next rung re-thinks. A rung spent for any other
reason (a 200 that carried no text — see `EmptyCompletion`) keeps them: nothing about
those notes was ever judged, and re-thinking would be a call spent to learn nothing.

`EmptyCompletion` also turns that content-less 200 into a rejection rather than an abort,
so the ladder answers it and the node's `on_fail` edge still means what the pack says.

Rule 2 has one more downstream now that jig logs (`jig.log`). A log file cannot condition
a model, so `Rejected.detail` is allowed there — but a log an operator ships to a
collector is still somewhere a customer's ticket should not turn up by accident, so the
detail is DEBUG-only and every other level carries `feedback`, the half already derived
from the pack's own schema. `NodeFailed` carries both for the same reason.

## The confidence gate — agreement, not self-report

A node may ask to be drawn more than once and have the answers compared:

    classify:
      type: generate
      samples: 3          # draw three independent answers
      agree: 2            # accept when this many match; otherwise the node is unsure

This exists because the obvious alternative does not work. A number the model *says*
about its own answer is generated after the answer is already on the page: it reflects
the rubric in the prompt, not whether the answer is right, and it is overconfident
exactly where being wrong costs the most. jig's usable signals, best first, are a
deterministic `assert` (a fact, not a guess), then agreement across independent draws,
and only then anything the model claims about itself. `verify` is the first. This is the
second. There is no third.

**What agreement means.** Two draws agree when the objects that would be committed are
the same object — compared as canonical JSON (`_canonical`: sorted keys, no whitespace).
Not "the fields that matter": at this layer nothing knows which fields those are, and
guessing wrong is the one failure that must not happen here — two draws that match on
the enum and differ on the amount are *not* a confident answer when the next node is a
tool that spends the amount. The node's own grammar is already the pack's declaration of
what matters; an author who wants agreement on less should narrow it, or split the node,
so that the thing being agreed on stays the thing being committed. Canonical text is also
the cheap answer: one string per draw, hashable, O(draws) grouping, and stable across key
order. It is stricter than `==` in exactly one place — `1` and `1.0` are different draws
where Python would call them equal — and that is the direction to be strict in for a gate
whose entire job is to notice that the model was not consistent.

**Unsure is not rejected.** A rejection means the output was invalid, and the ladder
below answers it. Disagreement means every output was *valid* and the model was not
consistent, which no re-sample fixes and which deserves a different destination — a
human queue, a cheaper safe branch, a second opinion. So it raises `Unsure`, a sibling of
`NodeFailed` rather than a subclass of it: a walker that wants to route both to `on_fail`
can, but it has to say so, and inheritance would have made that choice silently.
`Unsure` carries the counts (`Unsure.consensus`), the answer that came closest
(`Unsure.value`), and nothing is committed by this module either way.

**Cost.** n samples cost n generations, so the loop stops the moment the answer cannot
change: as soon as one group reaches `agree`, and as soon as the draws left cannot lift
any group to `agree`. A node with `samples: 3, agree: 2` whose first two draws match
spends two generations, not three.

**The ladder is per draw.** A draw that fails verification is a rejection, not a
disagreement: it climbs the rungs exactly as it does today, and a draw that spends the
whole ladder fails the node (`NodeFailed`) rather than counting as one dissenting voice —
a node that could not produce a valid answer has not produced evidence about anything.
Each draw starts its ladder clean, with no feedback and no scratchpad from the draw
before it, because a draw conditioned on another draw's rejection is not independent, and
agreement between draws that were conditioned on each other measures nothing.

Every generation after the very first also carries a distinct sampling hint (see
`sampling_for`) — against a backend that varies nothing, two draws are one draw charged
twice and would agree by construction. That is a false confidence signal rather than a
missing one, so a sampled node in front of a backend that cannot hear the hint says so at
WARNING (`node.samples.blind`).
"""

import json
from dataclasses import dataclass

from .codegen import Sampling, accepts_sampling, generate_once
from .errors import BackendError, ExprError, NodeFailed, RunError
from .expr import is_true
from .grammar import ValidationError, validate_against
from .log import DEBUG, INFO, WARNING, clip, event, get_logger

_log = get_logger("verify")

__all__ = [
    "Consensus",
    "DRAW_SEED_STRIDE",
    "DRAW_TEMPERATURE",
    "EmptyCompletion",
    "GateError",
    "RESAMPLE_TEMPERATURES",
    "Rejected",
    "Unsure",
    "extract_json",
    "gate_for",
    "run_node",
    "sampling_for",
    "verify",
]

# What each re-sample rung asks for. The climb is deliberate: rung 1 wants a different
# answer, not a wild one, and only a ladder that has already spent two rungs has evidence
# that the neighbourhood of the greedy answer is the problem. Deeper ladders stay at the
# top of the range rather than inventing values nobody has measured.
RESAMPLE_TEMPERATURES = (0.5, 0.8, 1.0)

EMPTY_COMPLETION_FEEDBACK = (
    "your previous answer arrived empty — return a single JSON object and nothing else"
)

# What an *extra draw* asks for. Not a rung of the ladder: nothing was rejected, so there
# is nothing to climb away from, and a temperature ramp across draws would measure the
# ramp rather than the model. Same temperature every time, different seed every time —
# the bottom of the measured range, because a draw taken to check another draw should be
# drawn the way the node normally is.
DRAW_TEMPERATURE = RESAMPLE_TEMPERATURES[0]

# Spacing between one draw's seeds and the next's. Every generation a node spends gets a
# seed nothing else in that node uses, which is what stops draw 2's first rung and draw
# 3's first rung from being the same request answered twice and counted as agreement.
# Draw 0 keeps `seed == rung` exactly as before, so an unsampled node's requests are
# byte-for-byte the ones it has always made.
DRAW_SEED_STRIDE = 1 << 16


class Rejected(ValueError):
    """One candidate output failed verification. Not fatal — the ladder continues.

    Two halves, and keeping them apart is what makes rule 2 above true:

    * ``str(exc)`` is the full detail, including the offending text. It goes to logs,
      `RunResult.failures` and the checkpoint — diagnostics can never self-condition a
      model, and an operator debugging a pack needs to see what came back.
    * ``feedback`` is the only half the *model* is ever shown. It says what was wrong
      without quoting what the model said.
    """

    def __init__(self, detail, feedback=None):
        ValueError.__init__(self, detail)
        self.detail = detail
        self.feedback = detail if feedback is None else feedback


class EmptyCompletion(BackendError):
    """A 200 that carried no text — a bad *sample*, not a backend that is down.

    The endpoint answered, and this draw simply had nothing in `message.content`: a
    reasoning model that spent its budget thinking, an empty choice, a filtered
    completion. That is the case a re-sample exists for, so `run_node` turns it into a
    `Rejected` and spends a rung instead of aborting the run past the node's `on_fail`.

    A backend that is unreachable, or returning 500s, is a different thing and still
    stops the run: no rung can fix it, and burning the ladder against a dead endpoint
    only delays the error.

    `empty_content` is the flag `run_node` actually reads, so a backend can mark an
    existing `BackendError` instead of raising this class — `jig.backends.openai_compat`
    keeps its own diagnostic message either way.
    """

    empty_content = True


@dataclass(frozen=True)
class Consensus:
    """What one visit to a sampled node found out — the reporting half of the gate.

    Counts only, deliberately. This record is meant to be logged, checkpointed and
    printed by `jig eval`, and every one of those is somewhere a customer's data must not
    turn up by accident; a record that holds no model output is safe in all three without
    anyone having to remember that it is. The value itself is *returned* when the node
    agreed, and carried on `Unsure.value` when it did not.

    * `asked` — what the pack asked for (`samples:`).
    * `drawn` — what it actually paid for. Lower than `asked` whenever the answer could
      no longer change: a group reached the threshold, or the draws left could not lift
      one to it.
    * `agreed` — the size of the largest group of matching draws.
    * `required` — the threshold (`agree:`, or a strict majority of `asked`).
    * `generations` — model calls this visit spent, rejected ones included. `drawn`
      counts answers; this counts the bill.
    * `distinct` — how many different answers the draws produced. It is the shape of a
      disagreement rather than its size, and it is what an author tuning `samples` needs:
      a 2-2 split is a node with two defensible readings, four different answers is a
      node that is guessing.
    """

    node: str
    asked: int
    drawn: int
    agreed: int
    required: int
    generations: int
    distinct: int = 1

    @property
    def unsure(self):
        """True when no group of matching draws reached the threshold."""
        return self.agreed < self.required


class Unsure(RunError):
    """A node's draws were all valid and did not agree.

    Distinct from `NodeFailed` on purpose (see this module's docstring): nothing was
    wrong with any of these answers, so the ladder has nothing to fix and the node's
    `on_fail` edge is not obviously where this should go. What a pack usually wants is a
    different destination — a human queue, a cheaper safe branch — which is why this is a
    signal a walker routes on rather than a failure it absorbs.

    For whoever wires the routing: catch `Unsure` alongside `NodeFailed`, and route it to
    the node's `on_unsure` if the pack declares one, its `on_fail` if it does not, and
    let it escape when it declares neither. `node`, `reason`, `feedback` and `attempts`
    are named the same as on `NodeFailed` so one failure-recording path can hold both.

    Every word of `str(exc)` is counts and node names — no model output — so unlike a
    `NodeFailed` this one is safe at any log level whole. `value` is the answer that came
    closest (the largest group's, ties going to the earliest draw). Nothing here is
    committed: a caller that decides an unsure answer is good enough for a human to look
    at has to commit it deliberately.
    """

    def __init__(self, node, consensus, value=None):
        self.node = node
        self.consensus = consensus
        self.value = value
        self.attempts = consensus.generations
        self.reason = (
            "%d of %d draws agreed and %d had to; %d generation(s) spent"
            % (consensus.agreed, consensus.drawn, consensus.required,
               consensus.generations)
        )
        # Both halves are the same text because both halves are safe. `feedback` exists
        # so a walker written against `NodeFailed` can read either exception the same way.
        self.feedback = self.reason
        RunError.__init__(self, "node %r is unsure: %s" % (node, self.reason))


class GateError(RunError):
    """A node's `samples`/`agree` pair does not describe a gate that can do anything.

    Raised rather than quietly repaired. Every reading of a broken gate is a lie about
    confidence — clamping `agree: 5, samples: 3` down to 3 silently weakens the check the
    author asked for, and accepting `agree: 1` alongside `samples: 3` gives a pack a gate
    that never fires and an author who believes it does. `jig validate` should catch these
    before a run starts; this is the backstop for a pack that arrived another way.
    """


def sampling_for(rung, draw=0):
    """The sampling hint for rung `rung` of draw `draw`, or None for the very first one.

    Draw 0, rung 0 asks for nothing on purpose: the first generation is the one the
    operator configured, so a pack that runs greedily stays reproducible and a run that
    never needed the ladder is byte-for-byte what it always was. Turning a node's gate on
    does not change that generation either — the extra draws are the check, not the
    answer.

    Every other generation the node spends asks for something no other generation of that
    node asked for. Across rungs that is the ladder climbing away from a rejection; across
    draws it is the same temperature with a different seed, because an extra draw is not a
    retry (`DRAW_TEMPERATURE`) — and because two identical requests are one draw charged
    twice, which on a sampled node would be counted as the model agreeing with itself.
    """
    if rung <= 0 and draw <= 0:
        return None
    # The seed rides along for a server pinned to temperature 0 by policy: greedy
    # decoding ignores a temperature it was told to ignore, and a per-request seed is
    # the only knob left.
    seed = rung + draw * DRAW_SEED_STRIDE
    if rung <= 0:
        return Sampling(temperature=DRAW_TEMPERATURE, seed=seed)
    index = min(rung, len(RESAMPLE_TEMPERATURES)) - 1
    return Sampling(temperature=RESAMPLE_TEMPERATURES[index], seed=seed)


def gate_for(node):
    """The `(samples, agree)` this node asks for — defaulted, checked, and cheap.

    A node that says nothing draws once and needs one answer, which is exactly what every
    pack written before this feature existed does. Read with `getattr` because the keys
    are optional in the node record as well as in the pack: a `Node` without them is a
    node that never asked for a gate.

    An `agree` left unset on a sampled node is a strict majority of the samples, which is
    the only default that is a *rule* rather than a preference — anything else has to
    pick a number, and a number picked here would be a confidence threshold nobody
    measured.
    """
    samples = _whole("samples", getattr(node, "samples", None), node)
    if samples is None:
        samples = 1
    if samples < 1:
        raise GateError(
            "node %r asks for samples: %d. A node draws at least once — use samples: 1 "
            "(or drop the key) for the ordinary single draw." % (node.name, samples)
        )
    required = _whole("agree", getattr(node, "agree", None), node)
    if samples == 1:
        # Nothing to compare a lone draw with, so `agree` cannot mean anything here. Say
        # so rather than ignoring it: a pack that set one and not the other has a gate its
        # author believes in and the runtime does not.
        if required is not None and required > 1:
            raise GateError(
                "node %r asks for agree: %d but draws only one sample. Add samples: %d "
                "(or more), or remove agree." % (node.name, required, required)
            )
        return 1, 1
    if required is None or required == 0:
        return samples, samples // 2 + 1
    if required < 2:
        raise GateError(
            "node %r asks for agree: %d, which accepts the first answer and never draws "
            "the other %d. Use agree: 2 or more, or remove samples."
            % (node.name, required, samples - 1)
        )
    if required > samples:
        raise GateError(
            "node %r asks for agree: %d out of samples: %d, which no run can satisfy. "
            "Raise samples to at least %d, or lower agree."
            % (node.name, required, samples, required)
        )
    return samples, required


def _whole(key, value, node):
    """`value` as a non-negative whole number, or None when the node did not set it."""
    if value is None:
        return None
    # `isinstance(True, int)` is True in Python, and `samples: yes` is a plausible slip in
    # a YAML file. A boolean here is a key that was misunderstood, not a count.
    if not isinstance(value, int) or isinstance(value, bool):
        raise GateError(
            "node %r has %s: %r — it must be a whole number, not %s"
            % (node.name, key, value, type(value).__name__)
        )
    if value < 0:
        raise GateError("node %r has %s: %d — it cannot be negative"
                        % (node.name, key, value))
    return value


def run_node(node, state, model, attempts=None, consensus=None):
    """Generate, verify, and return the object to commit.

    Raises `NodeFailed` when no draw could be verified, and `Unsure` when the draws were
    all valid and did not agree. Those are different outcomes and this module keeps them
    different; see the module docstring for why, and for what a walker should do with the
    second one.

    `retries` is the number of *re-samples* after the first attempt, so the default of 2
    buys three generations — per draw. `samples` (default 1) is how many verified answers
    the node wants compared, and `agree` (default: a strict majority) is how many of them
    must match. A node that sets neither draws once and behaves exactly as it always has,
    down to the bytes of the request.

    `attempts` is an optional dict the caller owns; the generations this node spends are
    added to it under the node's name, whether the node ends up succeeding or failing.
    That is how a finished run can say it needed three attempts at node 40 — see
    `graph.RunResult.attempts`.

    `consensus` is the same arrangement for the gate: an optional dict the caller owns,
    which gets a `Consensus` under the node's name whenever the node drew more than once.
    A node absent from it drew once and had nothing to compare — the same convention
    `attempts` uses for a node that spent no rungs. A node inside a loop leaves its most
    recent visit there, the way `provenance` records the most recent writer.

    A prompt that names state nobody wrote raises `MissingVariable` out of here
    untouched, before any generation is spent: no re-sample can fix a template, so the
    ladder is skipped rather than burned. The walker routes it to the node's `on_fail`
    exactly as it routes `NodeFailed` — a node that cannot produce a verified output is
    what that edge is for.
    """
    samples, required = gate_for(node)
    if samples > 1 and not accepts_sampling(model) and _log.isEnabledFor(WARNING):
        # The failure this catches is silent by nature: identical requests produce
        # identical answers, the draws "agree", and the pack reports high confidence it
        # never measured. An operator has to be told, because the run will look fine.
        event(_log, WARNING, "node.samples.blind", node=node.name, samples=samples,
              model=type(model).__name__,
              reason="backend takes no sampling hint, so extra draws repeat the first")

    spent = 0           # generations this visit has paid for, rejections included
    counts = {}         # canonical form -> how many draws produced it
    answers = {}        # canonical form -> the object itself, from the first draw of it
    drawn = 0
    for draw in range(samples):
        try:
            value, cost = _ladder(node, state, model, draw)
        except NodeFailed as exc:
            # A draw that could not be verified is a rejection, and a node that cannot
            # produce a valid answer has produced no evidence about anything — so this
            # ends the node rather than counting as one dissenting voice.
            _tally(attempts, node.name, exc.attempts)
            if not spent:
                # Nothing came before it: this is the single-draw failure the ladder has
                # always raised, handed on exactly as it was built.
                raise
            raise NodeFailed(exc.node, exc.reason, attempts=spent + exc.attempts,
                             feedback=exc.feedback)
        spent += cost
        drawn += 1
        _tally(attempts, node.name, cost)
        if samples == 1:
            # The overwhelming majority of nodes. One draw commits itself; there is
            # nothing to compare it with, and none of the bookkeeping below is paid for.
            return value

        key = _canonical(value)
        counts[key] = counts.get(key, 0) + 1
        answers.setdefault(key, value)
        if _log.isEnabledFor(DEBUG):
            event(_log, DEBUG, "node.sample", node=node.name, draw=drawn, of=samples,
                  matched=counts[key], needed=required)
        if counts[key] >= required:
            # Short-circuit: enough draws already match, and the ones not yet taken
            # cannot change that. `samples: 3, agree: 2` costs two generations when the
            # first two match, which is the common case a gate is worth paying for.
            record = _report(consensus, node, samples, drawn, counts[key], required,
                             spent, len(counts))
            # INFO, once per sampled node rather than once per generation: a gate that
            # quietly costs three generations on every node is the same kind of
            # unnoticed bill the retry rung is logged for. Counts only, so the line is
            # safe wherever an operator ships it.
            event(_log, INFO, "node.agreed", node=node.name, agreed=record.agreed,
                  of=drawn, required=required, asked=samples, generations=spent)
            return value
        if max(counts.values()) + (samples - drawn) < required:
            # The other short-circuit: no group can still reach the threshold, so the
            # remaining draws would cost tokens to confirm an answer already known.
            break

    # Ties go to the earliest draw: `max` keeps the first key it saw at that count and
    # dicts preserve insertion order, so the answer reported as closest is the same one
    # every time rather than whichever the dict happened to hold first.
    leader = max(counts, key=lambda key: counts[key])
    record = _report(consensus, node, samples, drawn, counts[leader], required, spent,
                     len(counts))
    event(_log, WARNING, "node.unsure", node=node.name, agreed=record.agreed, of=drawn,
          required=required, asked=samples, distinct=record.distinct,
          generations=spent)
    raise Unsure(node.name, record, value=answers[leader])


def _report(consensus, node, samples, drawn, agreed, required, spent, distinct):
    """Build this visit's `Consensus`, and leave it in the caller's dict if it kept one."""
    record = Consensus(node=node.name, asked=samples, drawn=drawn, agreed=agreed,
                       required=required, generations=spent, distinct=distinct)
    if consensus is not None:
        consensus[node.name] = record
    return record


def _canonical(value):
    """The comparison key for one draw: the committed object as canonical JSON.

    Sorted keys so that two draws that emitted the same fields in a different order are
    one answer, and no whitespace so that formatting is never mistaken for disagreement.
    Nothing here can fail: `verify` only ever returns a dict that came out of `json.loads`
    and survived the JSON-shape check, so it is round-trippable by construction.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ladder(node, state, model, draw=0):
    """One verified draw and what it cost, or `NodeFailed` when the rungs run out.

    This is the retry ladder, and it is per draw. A sampled node runs it once per draw,
    each time from rung 0 with no feedback and no scratchpad carried over: a draw
    conditioned on another draw's rejection is not an independent draw, and agreement
    between draws that saw each other's mistakes is not evidence.
    """
    rungs = node.retries + 1
    scratchpad = None
    reason = "no attempts were made"
    feedback = None
    spent = 0
    for rung in range(rungs):
        # `feedback` is None on the first attempt and the previous rejection after that;
        # `sampling_for` makes every rung after the first an independent draw wherever
        # the backend can manage one — and every draw after the first, for the same
        # reason at the other scale.
        sampling = sampling_for(rung, draw)
        # `rung and ...` short-circuits on the first attempt, so the happy path through
        # this loop — the overwhelming majority of generations — asks the logging layer
        # nothing at all. The first attempt is not silent: `codegen` logs `node.emit`
        # with the prompt size, and `graph` logs `node.ok` when it lands. There is no
        # event here that only restates one of those.
        if rung and _log.isEnabledFor(INFO):
            # The rung an operator is paying for. INFO rather than DEBUG because a pack
            # that quietly burns two extra generations on node 40 of every run is the
            # single most expensive thing that can go unnoticed here, and `reason` is
            # the model-safe half by construction — it is the same text the next prompt
            # is allowed to carry.
            event(_log, INFO, "node.retry", node=node.name, attempt=rung + 1, of=rungs,
                  temperature=sampling.temperature if sampling else None,
                  seed=sampling.seed if sampling else None, reason=feedback,
                  rethink=bool(node.two_stage and scratchpad is None))
        try:
            candidate = generate_once(
                node, state, model, error=feedback, scratchpad=scratchpad,
                sampling=sampling,
            )
        except BackendError as exc:
            if not getattr(exc, "empty_content", False):
                raise
            # The endpoint answered and this draw simply had no text in it. That is a
            # rejection — one rung of the ladder — not a reason to abort the run past
            # the node's declared `on_fail`. The backend's own diagnostic is the detail
            # an operator needs, so it is kept verbatim; the model is told only that its
            # answer arrived empty.
            spent += 1
            reason, feedback = str(exc), EMPTY_COMPLETION_FEEDBACK
            # The backend's diagnostic names the endpoint and the finish reason and
            # quotes no generation, so it is safe at WARNING as it stands. `clip` runs
            # over text of unbounded length, which is exactly the kind of work that must
            # not happen when nobody is listening — hence the guard rather than trusting
            # `event` to throw the result away.
            if _log.isEnabledFor(WARNING):
                event(_log, WARNING, "node.rejected", node=node.name, attempt=spent,
                      cause="empty_completion", reason=EMPTY_COMPLETION_FEEDBACK,
                      detail=clip(reason))
            # Nothing about a two-stage node's notes was judged here, so the next rung
            # keeps them: `codegen.generate_once` hands back the notes of the attempt
            # that never got an answer.
            scratchpad = getattr(exc, "scratchpad", None) or scratchpad
            continue
        spent += 1
        try:
            value = verify(node, candidate.text, state)
        except Rejected as exc:
            # `reason` keeps the whole story for diagnostics; `feedback` is the only
            # part that may be shown to the model on the next rung. The notes behind a
            # rejected answer are discarded with it: the reasoning is what the emit
            # stage obeyed, so it is part of what was just judged.
            reason = str(exc)
            feedback = exc.feedback
            scratchpad = None
            # The split that rule 2 of this module's docstring is about, spelled out at
            # the one place both halves are in scope. `feedback` says what was wrong and
            # is derived from the pack's own schema, so it goes out at WARNING where an
            # operator shipping logs to a collector will see it. `detail` may quote the
            # generation verbatim, so it is DEBUG-only: someone who asked for DEBUG asked
            # to read what came back, and a log file cannot condition a model — but a
            # default-level line that carried it would put rejected model output into
            # every collector jig is ever pointed at.
            if _log.isEnabledFor(WARNING):
                event(_log, WARNING, "node.rejected", node=node.name, attempt=spent,
                      cause="verify", reason=feedback, of=rungs)
            if _log.isEnabledFor(DEBUG):
                event(_log, DEBUG, "node.rejected.detail", node=node.name,
                      attempt=spent, detail=clip(reason))
            continue
        return value, spent
    # The count rides on the exception rather than being tallied here, because the caller
    # is the one that knows whether earlier draws of the same visit have already been
    # charged for.
    raise NodeFailed(node.name, reason, attempts=spent, feedback=feedback)


def _tally(attempts, name, spent):
    """Add this visit's generations to the run's per-node count.

    Added rather than assigned: a node inside a loop is visited more than once, and what
    an operator is reading this for is what the node cost the run — and a sampled node
    adds every draw's ladder to the same total, because every one of them is on the bill.
    """
    if attempts is not None and spent:
        attempts[name] = attempts.get(name, 0) + spent


def verify(node, text, state):
    """Return the object `text` commits to, or raise `Rejected` saying why not."""
    value = extract_json(text)
    if not isinstance(value, dict):
        raise Rejected(
            "output must be a JSON object, got %s" % type(value).__name__
        )
    # Always, even with no schema. `validate_against` runs a JSON-shape check (no NaN or
    # Infinity, nothing nested past the ceiling) before it looks at the schema, and shape
    # is not a schema question — a value JSON cannot represent is bad output whether or not
    # a node declared constraints. Guarding this with `if node.grammar:` skipped it for a
    # free-form node, which the pack format documents `{}` as being: NaN then committed,
    # printed as a bare NaN no strict reader can parse, and killed a checkpointed run AFTER
    # the commit — the exact hazard checking-before-commit exists to prevent. An empty
    # schema constrains nothing, so free-form nodes stay free.
    try:
        validate_against(node.grammar or {}, value)
    except ValidationError as exc:
        raise Rejected("schema: %s" % exc, feedback="schema: %s" % exc.safe_text)
    if node.assert_expr:
        _check_assert(node, value, state)
    return value


def _check_assert(node, value, state):
    """Evaluate the node's assert against state *as it would be if committed*.

    The trial scope is a copy. A candidate that fails here is discarded whole, so the
    real state never saw it.
    """
    trial = dict(state)
    if node.output:
        trial[node.output] = value
    else:
        trial.update(value)
    try:
        passed = is_true(node.assert_expr, trial)
    except ExprError as exc:
        raise Rejected("assert %r could not be evaluated: %s" % (node.assert_expr, exc))
    if not passed:
        raise Rejected("assert failed: %s" % node.assert_expr)


def extract_json(text):
    """Pull an object out of a generation.

    With a real constrained decoder the text is already exact JSON. Backends without
    grammar support (and llama.cpp's looser modes) still wrap it in prose or a fence, so
    this is forgiving about *finding* the object — and `verify` stays merciless about
    what is in it.
    """
    if not isinstance(text, str):
        raise Rejected("model returned %s, not text" % type(text).__name__)
    for attempt in _candidates(text.strip()):
        try:
            return json.loads(attempt)
        except ValueError:
            continue
        except RecursionError:
            # Deeply nested output exhausts the decoder's own stack. On CPython before
            # 3.12 this escapes json.loads as RecursionError rather than ValueError, so
            # without this it is not a Rejected, bypasses the retry ladder and the node's
            # on_fail edge, and kills the run. A model that emits 10,000 nested arrays is
            # a bad generation, not a broken runtime — reject it like any other.
            raise Rejected(
                "output nested too deeply to parse (%d bytes)" % len(text),
                feedback="output was nested too deeply — return a flat JSON object",
            )
    raise Rejected(
        "output was not valid JSON: %s" % _clip(text),
        feedback="output was not valid JSON — return a single JSON object and nothing else",
    )


def _candidates(text):
    """Every span worth parsing, best first: the whole answer, a fence, then objects.

    The objects come out **last one first**, and that ordering is a security property
    rather than a preference. A small model restates the format it was given, or quotes
    the ticket it was handed, and *then* answers; taking the first balanced `{...}` in
    the text hands the node whatever the model was merely repeating — user-controlled
    text, schema-validated and committed, which is the one thing verify-before-commit
    exists to prevent. What a model says last is its answer; everything before it is
    preamble.

    Deliberately *not* "the object that validates against the node's schema". That rule
    reads better until the model's own answer is the imperfect one: the echo then
    outranks it, and a rejection the ladder could have fixed becomes a silent commit of
    quoted input. Selection stays blind to the schema; earlier spans are reached only
    when the later ones do not parse at all, which is how a valid object behind a
    braces-in-prose span is still recovered.
    """
    yield text
    fenced = _unfence(text)
    if fenced is not None:
        yield fenced
    for span in reversed(_objects(text)):
        yield span


def _unfence(text):
    if not text.startswith("```"):
        return None
    body = text[3:]
    newline = body.find("\n")
    if newline == -1:
        return None
    body = body[newline + 1:]
    end = body.rfind("```")
    return body[:end].strip() if end != -1 else body.strip()


def _objects(text):
    """Every balanced top-level `{...}` span, in order, ignoring braces inside strings."""
    spans = []
    depth = 0
    quote = False
    start = -1
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                quote = False
        elif char == '"':
            # Only quotes *inside* a span can hide a brace; prose before one is not JSON
            # and its apostrophes and quotes must not swallow the object that follows.
            if depth:
                quote = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start:index + 1])
        index += 1
    return spans


def _clip(text, limit=120):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."
