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
"""

import json

from .codegen import Sampling, generate_once
from .errors import BackendError, ExprError, NodeFailed
from .expr import is_true
from .grammar import ValidationError, validate_against
from .log import DEBUG, INFO, WARNING, clip, event, get_logger

_log = get_logger("verify")

__all__ = [
    "EmptyCompletion",
    "RESAMPLE_TEMPERATURES",
    "Rejected",
    "extract_json",
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


def sampling_for(rung):
    """The sampling hint for ladder rung `rung`, or None for the first attempt.

    Rung 0 asks for nothing on purpose: the first draw is the one the operator
    configured, so a pack that runs greedily stays reproducible and a run that never
    needed the ladder is byte-for-byte what it always was.
    """
    if rung <= 0:
        return None
    index = min(rung, len(RESAMPLE_TEMPERATURES)) - 1
    # The seed rides along for a server pinned to temperature 0 by policy: greedy
    # decoding ignores a temperature it was told to ignore, and a per-request seed is
    # the only knob left.
    return Sampling(temperature=RESAMPLE_TEMPERATURES[index], seed=rung)


def run_node(node, state, model, attempts=None):
    """Generate, verify, and return the object to commit — or raise `NodeFailed`.

    `retries` is the number of *re-samples* after the first attempt, so the default of 2
    buys three generations.

    `attempts` is an optional dict the caller owns; the generations this node spends are
    added to it under the node's name, whether the node ends up succeeding or failing.
    That is how a finished run can say it needed three attempts at node 40 — see
    `graph.RunResult.attempts`.

    A prompt that names state nobody wrote raises `MissingVariable` out of here
    untouched, before any generation is spent: no re-sample can fix a template, so the
    ladder is skipped rather than burned. The walker routes it to the node's `on_fail`
    exactly as it routes `NodeFailed` — a node that cannot produce a verified output is
    what that edge is for.
    """
    rungs = node.retries + 1
    scratchpad = None
    reason = "no attempts were made"
    feedback = None
    spent = 0
    for rung in range(rungs):
        # `feedback` is None on the first attempt and the previous rejection after that;
        # `sampling_for` makes every rung after the first an independent draw wherever
        # the backend can manage one.
        sampling = sampling_for(rung)
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
        _tally(attempts, node.name, spent)
        return value
    _tally(attempts, node.name, spent)
    raise NodeFailed(node.name, reason, attempts=spent, feedback=feedback)


def _tally(attempts, name, spent):
    """Add this visit's generations to the run's per-node count.

    Added rather than assigned: a node inside a loop is visited more than once, and what
    an operator is reading this for is what the node cost the run.
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
    if node.grammar:
        try:
            validate_against(node.grammar, value)
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
