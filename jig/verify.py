"""Verify-before-commit, and the retry ladder.

This module is the reason a small model can be trusted here at all (docs/PLAN.md §3).
Two rules, and the second one is the load-bearing one:

1. **Nothing is committed until it is verified.** Output must parse, satisfy the node's
   schema, and satisfy the node's optional `assert` expression.
2. **A rejected generation never enters state.** Not into the run's state, not into a
   later node's prompt, not even into the retry prompt's history. Verification runs
   against a *trial* copy of state; only a candidate that survives is written back. That
   is the structural fix for self-conditioning decay — a model that never sees its own
   bad output cannot spiral on it.

The ladder is cheap-first: plain re-sample, then re-sample with the rejection appended,
then the node's `on_fail` edge (or `NodeFailed`). No frontier-model fallback, by design —
PLAN.md keeps the cost story honest.
"""

import json

from .codegen import generate_once
from .errors import ExprError, NodeFailed
from .expr import is_true
from .grammar import ValidationError, validate_against

__all__ = ["Rejected", "extract_json", "run_node", "verify"]


class Rejected(ValueError):
    """One candidate output failed verification. Not fatal — the ladder continues."""


def run_node(node, state, model):
    """Generate, verify, and return the object to commit — or raise `NodeFailed`.

    `retries` is the number of *re-samples* after the first attempt, so the default of 2
    gives the three-rung ladder TASKS.md describes.
    """
    attempts = node.retries + 1
    scratchpad = None
    reason = "no attempts were made"
    for attempt in range(attempts):
        # Rung 1 is a plain re-sample: cheap, and a different sample often just works.
        # Only from rung 2 do we spend tokens telling the model what went wrong.
        error = reason if attempt >= 2 else None
        candidate = generate_once(
            node, state, model, error=error, scratchpad=scratchpad
        )
        scratchpad = candidate.scratchpad
        try:
            return verify(node, candidate.text, state)
        except Rejected as exc:
            reason = str(exc)
    raise NodeFailed(node.name, reason, attempts=attempts)


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
            raise Rejected("schema: %s" % exc)
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
    candidate = text.strip()
    for attempt in (candidate, _unfence(candidate), _first_object(candidate)):
        if attempt is None:
            continue
        try:
            return json.loads(attempt)
        except ValueError:
            continue
    raise Rejected("output was not valid JSON: %s" % _clip(text))


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


def _first_object(text):
    """The first balanced `{...}` span, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    quote = False
    index = start
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                quote = False
        elif char == '"':
            quote = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
        index += 1
    return None


def _clip(text, limit=120):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."
